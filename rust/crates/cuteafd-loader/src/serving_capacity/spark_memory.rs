//! Idle GB10 unified-memory capacity. Never substitute CUDA free or MemFree.
use anyhow::{bail, Context, Result};
use std::sync::OnceLock;

pub const INVENTORY: &str = include_str!("spark_memory.json");
pub const SERVING_HOSTS: [&str; 6] = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"];

pub fn measured_bytes(host: &str) -> Result<u64> {
    static TABLE: OnceLock<serde_json::Value> = OnceLock::new();
    let table = TABLE.get_or_init(|| serde_json::from_str(INVENTORY).expect("checked-in Spark inventory"));
    let host = host.split(':').next().unwrap_or(host);
    anyhow::ensure!(!host.is_empty(), "missing physical Spark host identity");
    let host = (0..6).find(|rank| host == format!("10.55.0.{}", rank + 1)
        || host == format!("10.55.1.{}", rank + 1)).map_or(host, |rank| SERVING_HOSTS[rank]);
    anyhow::ensure!(table["hosts"][host]["serving"] != false,
        "Spark {host} is build-only (serving_capacity/spark_memory.json)");
    let kib = table["hosts"][host]["mem_available_kib"].as_u64()
        .with_context(|| format!("missing idle MemAvailable measurement for Spark {host} in serving_capacity/spark_memory.json; supply --spark-memory-cap"))?;
    let bytes = kib.checked_mul(1024).context("Spark MemAvailable bytes overflow")?;
    if bytes == 0 { bail!("empty MemAvailable measurement for Spark {host}"); }
    Ok(bytes)
}

pub fn serving_bytes() -> Vec<u64> {
    SERVING_HOSTS.iter().map(|host| measured_bytes(host).expect("measured serving pool")).collect()
}

static LIVE: OnceLock<Vec<u64>> = OnceLock::new();

/// Per-rank Spark capacity: idle `MemAvailable` plus the GB10 display carveout
/// a validated spark-attic would lend this host's routed-expert arena
/// (`carveout_bytes`, absent or 0 where none is reachable; #2546). The
/// carveout lies outside `MemAvailable`, so the two never overlap; the worker
/// maps it only into its routed-expert arena, so it never backs another owner.
pub fn sample_bytes(samples: &str) -> Result<Vec<u64>> {
    Ok(sample_parts(samples)?.into_iter().map(|(available, carveout)| available.saturating_add(carveout)).collect())
}

/// `(MemAvailable, carveout)` per rank; the carveout counts only while enabled.
pub fn sample_parts(samples: &str) -> Result<Vec<(u64, u64)>> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(samples)?;
    anyhow::ensure!(!rows.is_empty() && rows.len() <= 8, "Spark samples require 1..8 ranks");
    rows.iter().map(|row| {
        let bytes = row["bytes"].as_u64().filter(|bytes| *bytes > 0)
            .context("missing positive idle MemAvailable sample")?;
        anyhow::ensure!(row["boot_id"].as_str().is_some_and(|id| !id.is_empty())
            && row["age_seconds"].as_f64().is_some_and(|age| age >= 0.0),
            "Spark sample requires boot_id and age_seconds");
        let carveout = match row.get("carveout_bytes") {
            None | Some(serde_json::Value::Null) => 0,
            Some(value) => value.as_u64().context("carveout_bytes must be a byte count")?,
        };
        anyhow::ensure!(carveout % (2 << 20) == 0, "carveout_bytes must be whole 2 MiB slices");
        Ok((bytes, if cuteafd_ffi_attic_enabled() { carveout } else { 0 }))
    }).collect()
}

/// `--spark-carveout off` (process-wide) plans without the carveout.
fn cuteafd_ffi_attic_enabled() -> bool { CARVEOUT_ENABLED.load(std::sync::atomic::Ordering::SeqCst) }
static CARVEOUT_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
pub fn set_carveout_enabled(enabled: bool) { CARVEOUT_ENABLED.store(enabled, std::sync::atomic::Ordering::SeqCst); }

pub fn install_samples(samples: &str) -> Result<()> {
    let bytes = sample_bytes(samples)?;
    if let Some(current) = LIVE.get() {
        anyhow::ensure!(*current == bytes, "Spark samples already installed with different capacities");
        return Ok(());
    }
    LIVE.set(bytes).map_err(|_| anyhow::anyhow!("Spark samples already installed"))
}

pub fn live_bytes(ranks: usize) -> Result<Option<Vec<u64>>> {
    LIVE.get().map(|bytes| {
        anyhow::ensure!(bytes.len() >= ranks, "missing live Spark rank measurement: need {ranks}, have {}", bytes.len());
        Ok(bytes[..ranks].to_vec())
    }).transpose()
}

pub fn apply_rank_budgets(budgets: &mut [crate::placement::planner::Budget], bytes: &[u64]) -> Result<()> {
    for budget in budgets {
        if let Some(rank) = budget.locations.iter().find_map(|location| match location.device {
            crate::placement::planner::Device::Spark(rank) => Some(rank as usize), _ => None,
        }) {
            let capacity = *bytes.get(rank).filter(|bytes| **bytes > 0)
                .with_context(|| format!("missing measured capacity for Spark rank {rank}"))?;
            budget.physical_bytes = capacity;
            budget.logical_bytes = capacity;
        }
    }
    Ok(())
}

/// Resolve every physical role; non-expert ranks require a live sample or endpoint identity.
pub fn runtime_physical_bytes(peers: &[String], ranks: usize) -> Result<Vec<u64>> {
    if let Some(bytes) = cuteafd_core::memory_policy::MemoryPolicy::current().spark_cap {
        return Ok(vec![bytes; ranks]);
    }
    if let Some(bytes) = live_bytes(ranks)? { return Ok(bytes); }
    anyhow::ensure!(peers.len() >= ranks, "missing host identity for physical Spark rank {}", peers.len());
    runtime_bytes(&peers[..ranks])
}

/// Runtime coordinators consume the same measured capacities as their launch plan.
pub fn runtime_bytes(peers: &[String]) -> Result<Vec<u64>> {
    let explicit = cuteafd_core::memory_policy::MemoryPolicy::current().spark_cap;
    if let Some(bytes) = explicit { return Ok(vec![bytes; peers.len()]); }
    if let Some(bytes) = LIVE.get() {
        anyhow::ensure!(bytes.len() >= peers.len(), "missing live Spark rank measurement");
        return Ok(bytes[..peers.len()].to_vec());
    }
    peers.iter().map(|peer| measured_bytes(peer)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_samples_are_per_rank_and_missing_measurements_fail_closed() {
        let samples = r#"[{"bytes":123,"boot_id":"a","age_seconds":0},
            {"bytes":456,"boot_id":"b","age_seconds":17}]"#;
        assert_eq!(sample_bytes(samples).unwrap(), vec![123, 456]);
        for invalid in ["[]", r#"[{"bytes":0,"boot_id":"a","age_seconds":0}]"#,
            r#"[{"bytes":123,"age_seconds":0}]"#, r#"[{"bytes":123,"boot_id":"a"}]"#] {
            assert!(sample_bytes(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn carveout_is_added_per_rank_and_validated() {
        let samples = r#"[{"bytes":1000,"boot_id":"a","age_seconds":0,"carveout_bytes":2143289344},
            {"bytes":2000,"boot_id":"b","age_seconds":1},{"bytes":3000,"boot_id":"c","age_seconds":1,"carveout_bytes":0}]"#;
        assert_eq!(sample_parts(samples).unwrap(), vec![(1000, 2143289344), (2000, 0), (3000, 0)]);
        assert_eq!(sample_bytes(samples).unwrap(), vec![2143290344, 2000, 3000]);
        for invalid in [r#"[{"bytes":1,"boot_id":"a","age_seconds":0,"carveout_bytes":3}]"#,
            r#"[{"bytes":1,"boot_id":"a","age_seconds":0,"carveout_bytes":"x"}]"#] {
            assert!(sample_bytes(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn runtime_unknown_hosts_and_short_rank_inventory_fail_closed() {
        assert!(runtime_bytes(&["unknown:19441".into()]).is_err());
        assert!(runtime_physical_bytes(&["rhea:19441".into()], 2).is_err());
        let mut budgets = vec![crate::placement::families::muse_spark::spark_budget(1, 100)];
        assert!(apply_rank_budgets(&mut budgets, &[123]).is_err());
        apply_rank_budgets(&mut budgets, &[123, 456]).unwrap();
        assert_eq!(budgets[0].physical_bytes, 456);
    }

    #[test]
    fn exact_per_host_measurements_and_missing_fail_closed() {
        assert_eq!(measured_bytes("dodo").unwrap(), 121_171_588 * 1024);
        assert_eq!(measured_bytes("10.55.0.2:19441").unwrap(), measured_bytes("dodo").unwrap());
        assert_ne!(measured_bytes("ostrich").unwrap(), measured_bytes("rhea").unwrap());
        assert!(measured_bytes("unknown").is_err());
        assert!(measured_bytes("kakapo").is_err());
        assert!(measured_bytes("cassowary").is_err());
    }
}
