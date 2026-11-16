//! Idle GB10 unified-memory capacity. Never substitute CUDA free or MemFree.
use anyhow::{bail, Context, Result};
use std::sync::OnceLock;

pub const INVENTORY: &str = include_str!("spark_memory.json");
pub const SERVING_HOSTS: [&str; 6] = ["ostrich", "dodo", "emu", "kiwi", "rhea", "moa"];

pub fn measured_bytes(host: &str) -> Result<u64> {
    static TABLE: OnceLock<serde_json::Value> = OnceLock::new();
    let table = TABLE.get_or_init(|| serde_json::from_str(INVENTORY).expect("checked-in Spark inventory"));
    let host = host.split(':').next().unwrap_or(host);
    let host = (0..6).find(|rank| host == format!("10.55.0.{}", rank + 1)
        || host == format!("10.55.1.{}", rank + 1)).map_or(host, |rank| SERVING_HOSTS[rank]);
    let kib = table["hosts"][host]["mem_available_kib"].as_u64()
        .with_context(|| format!("missing idle MemAvailable measurement for Spark {host}; supply --spark-memory-cap"))?;
    let bytes = kib.checked_mul(1024).context("Spark MemAvailable bytes overflow")?;
    if bytes == 0 { bail!("empty MemAvailable measurement for Spark {host}"); }
    Ok(bytes)
}

pub fn serving_bytes() -> Vec<u64> {
    SERVING_HOSTS.iter().map(|host| measured_bytes(host).expect("measured serving pool")).collect()
}

static LIVE: OnceLock<Vec<u64>> = OnceLock::new();

pub fn sample_bytes(samples: &str) -> Result<Vec<u64>> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(samples)?;
    anyhow::ensure!(!rows.is_empty() && rows.len() <= 8, "Spark samples require 1..8 ranks");
    rows.iter().map(|row| {
        let bytes = row["bytes"].as_u64().filter(|bytes| *bytes > 0)
            .context("missing positive idle MemAvailable sample")?;
        anyhow::ensure!(row["boot_id"].as_str().is_some_and(|id| !id.is_empty())
            && row["age_seconds"].as_f64().is_some_and(|age| age >= 0.0),
            "Spark sample requires boot_id and age_seconds");
        Ok(bytes)
    }).collect()
}

pub fn install_samples(samples: &str) -> Result<()> {
    let bytes = sample_bytes(samples)?;
    if let Some(current) = LIVE.get() {
        anyhow::ensure!(*current == bytes, "Spark samples already installed with different capacities");
        return Ok(());
    }
    LIVE.set(bytes).map_err(|_| anyhow::anyhow!("Spark samples already installed"))
}

pub fn live_bytes(ranks: usize) -> Option<Vec<u64>> {
    LIVE.get().filter(|bytes| bytes.len() >= ranks).map(|bytes| bytes[..ranks].to_vec())
}

pub fn apply_live_budgets(budgets: &mut [crate::placement::planner::Budget]) {
    if cuteafd_core::memory_policy::MemoryPolicy::current().spark_cap.is_some() { return; }
    if let Some(bytes) = LIVE.get() {
        for budget in budgets {
            if let Some(rank) = budget.locations.iter().find_map(|location| match location.device {
                crate::placement::planner::Device::Spark(rank) => Some(rank as usize), _ => None,
            }) {
                budget.physical_bytes = bytes.get(rank).copied().unwrap_or(0);
                budget.logical_bytes = budget.physical_bytes;
            }
        }
    }
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
    fn exact_per_host_measurements_and_missing_fail_closed() {
        assert_eq!(measured_bytes("dodo").unwrap(), 121_171_588 * 1024);
        assert_eq!(measured_bytes("10.55.0.2:19441").unwrap(), measured_bytes("dodo").unwrap());
        assert_ne!(measured_bytes("ostrich").unwrap(), measured_bytes("rhea").unwrap());
        assert!(measured_bytes("unknown").is_err());
    }
}
