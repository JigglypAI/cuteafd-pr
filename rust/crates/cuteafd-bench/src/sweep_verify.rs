//! The verify-shape sweep (`cuteafd bench smoke` entry `sweep_verify`, wip-cards
//! `--verify-shapes`): warmed, verify-only timing of exact per-request row shapes
//! through the server's production decode step, by the `verify_timing` probe.
//!
//! One load: an isolation reference (a greedy speculation-off decode whose rows
//! are recorded, its prompt retained in the prefix cache), one untimed probe
//! pass over every shape (captures any graph a shape needs), the measurement
//! window, the timed probe, then the reference again restored from the cache.
//! The restore must reproduce the reference's tokens and row hashes byte for
//! byte: the probe may not change any state a later request reads.
use crate::client::{Chat, Client};
use crate::panels::common::plain;
use crate::text::corpus;
use anyhow::{bail, Context, Result};
use cuteafd_api::openai::probe::verify_timing::{VerifyShape, VerifyTiming, VerifyTimingRecord};
use cuteafd_api::openai::probe::ProbeSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Isolation reference decode length (greedy rows recorded).
const REFERENCE_TOKENS: u64 = 16;

fn default_context() -> usize { 1024 }
fn default_repeats() -> usize { 20 }

/// The smoke entry / wip-cards option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyOptions {
    pub shapes: Vec<VerifyShape>,
    /// Prefilled history per request before its verify rows (served tokens).
    #[serde(default = "default_context")]
    pub context: usize,
    #[serde(default = "default_repeats")]
    pub repeats: usize,
}

impl VerifyOptions {
    pub fn timing(&self, warmups: usize, repeats: usize) -> VerifyTiming {
        VerifyTiming { shapes: self.shapes.clone(), context: self.context, warmups, repeats }
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.repeats > 0, "verify sweep repeats must be at least 1");
        self.timing(3, self.repeats).validate()
    }

    /// `4x8;5,4x7;8x8@64` (shapes separated by `;`), the wip-cards spelling.
    pub fn parse_shapes(text: &str) -> Result<Vec<VerifyShape>, String> {
        text.split(';').map(str::trim).filter(|s| !s.is_empty()).map(str::parse).collect()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VerifySweep {
    pub options: Option<VerifyOptions>,
    pub record: VerifyTimingRecord,
    /// The untimed pass's record (graph captures for new shapes land here).
    pub warm: VerifyTimingRecord,
    /// GPU throttle report over the timed probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<Value>,
    /// The restored reference reproduced every token and row hash.
    pub isolation: Isolation,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Isolation {
    pub passed: bool,
    pub reference_tokens: usize,
    pub restored_cached_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_difference: Option<usize>,
}

fn reference_spec(cold: bool) -> ProbeSpec {
    ProbeSpec { cold, no_speculation: true, pair_exempt: true, record_rows: REFERENCE_TOKENS as usize, top_k: 1,
        ..ProbeSpec::default() }
}

fn rows_of(chat: &Chat) -> Result<Vec<(u32, String)>> {
    let probe = chat.probe.as_ref().context("isolation reference: no probe record")?;
    if let Some(error) = &probe.error { bail!("isolation reference probe: {error}"); }
    Ok(probe.generated.iter().copied().zip(probe.rows.iter().map(|row| row.hash.clone())).collect())
}

/// Compares the reference's rows with the restored ones.
pub fn isolation(before: &[(u32, String)], after: &[(u32, String)], cached: u64) -> Isolation {
    let first_difference = (0..before.len().max(after.len())).find(|&i| before.get(i) != after.get(i));
    Isolation { passed: first_difference.is_none() && !before.is_empty() && cached > 0, reference_tokens: before.len(),
        restored_cached_tokens: cached, first_difference }
}

fn timing_probe(client: &Client, body: &Value, timing: VerifyTiming) -> Result<VerifyTimingRecord> {
    let chat = client.chat(body.clone(), Some(ProbeSpec { cold: true, no_speculation: true, pair_exempt: true,
        verify_timing: Some(timing), ..ProbeSpec::default() })).context("verify timing probe")?;
    let probe = chat.probe.context("verify timing: no probe record (server predates verify_timing)")?;
    if let Some(error) = probe.error { bail!("{error}"); }
    probe.verify_timing.context("verify timing: the server's family does not run verify_timing probes")
}

/// Runs the sweep; `on_timed` opens the measurement window just before the timed probe.
pub fn run(client: &Client, options: &VerifyOptions, on_timed: crate::sweep::OnTimed) -> Result<VerifySweep> {
    options.validate()?;
    let timing = options.timing(3, options.repeats);
    // The probe tokenizes this prompt and takes its context rows from it: corpus text well
    // past the probe's needs (tokens per reference token vary by model; trimmed server-side).
    let body = plain(client, &format!("Summarize in one word.\n\n{}", corpus(4343, timing.prompt_tokens() * 2 + 512)), 1);
    let reference = plain(client, &format!("Continue the code.\n\n{}", corpus(4444, 2048)), REFERENCE_TOKENS);
    let before = rows_of(&client.chat(reference.clone(), Some(reference_spec(false))).context("isolation reference")?)?;
    let warm = timing_probe(client, &body, options.timing(3, 0)).context("untimed verify pass")?;
    if let Some(open) = on_timed { open()?; }
    let mark = client.thermal_mark();
    let record = timing_probe(client, &body, timing)?;
    let thermal = client.thermal_since(mark);
    let restored = client.chat(reference, Some(reference_spec(false))).context("isolation restore")?;
    let isolation = isolation(&before, &rows_of(&restored)?, restored.timing.cached_tokens);
    for sample in &record.samples {
        eprintln!("verify {} R={} B={} (natural {}): verify {:.3} ms, select {:.3} ms, graphed={}",
            sample.shape, sample.real_rows, sample.bucket, sample.natural_bucket,
            sample.verify_median_ms().unwrap_or(f64::NAN), sample.select_median_ms().unwrap_or(f64::NAN), sample.graphed);
    }
    let sweep = VerifySweep { options: Some(options.clone()), record, warm, thermal, isolation };
    if !sweep.isolation.passed {
        bail!("verify timing changed server state: restored reference differs ({:?})", sweep.isolation);
    }
    Ok(sweep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_validate_and_build_probes() {
        let shapes = VerifyOptions::parse_shapes("4x8; 5,4x7;8x8@64").unwrap();
        assert_eq!(shapes.iter().map(|s| s.to_string()).collect::<Vec<_>>(), ["4x8", "5,4x7", "8x8@64"]);
        let options: VerifyOptions = serde_json::from_value(serde_json::json!({"shapes": ["4x8", "5,4x7"]})).unwrap();
        assert_eq!((options.context, options.repeats), (1024, 20));
        options.validate().unwrap();
        assert_eq!(options.timing(3, 0).repeats, 0);
        assert!(VerifyOptions { repeats: 0, ..options.clone() }.validate().is_err());
        assert!(VerifyOptions::parse_shapes("4x8;0").is_err());
        assert!(serde_json::from_value::<VerifyOptions>(serde_json::json!({"shapes": ["4x8"], "typo": 1})).is_err());
    }

    #[test]
    fn isolation_needs_identical_rows_and_a_real_restore() {
        let rows = |n: u32| (0..n).map(|i| (i, format!("{i:x}"))).collect::<Vec<_>>();
        assert!(isolation(&rows(4), &rows(4), 100).passed);
        let mut changed = rows(4);
        changed[2].1 = "ff".into();
        assert_eq!(isolation(&rows(4), &changed, 100).first_difference, Some(2));
        assert!(!isolation(&rows(4), &rows(3), 100).passed);
        assert!(!isolation(&rows(4), &rows(4), 0).passed, "a cold rerun is not a restore");
        assert!(!isolation(&[], &[], 100).passed);
    }
}
