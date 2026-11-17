//! Identical, untimed policy conditioning, retained once per serving instance.
use crate::{client::Client, panels::common};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::sync::Mutex;

pub const SCHEMA: &str = "cuteafd.bench.prelude/1";
pub const ROUNDS: usize = 2;

#[derive(Default)]
pub(crate) struct Prelude(Mutex<Option<Value>>);

impl Prelude {
    pub(crate) fn ensure(&self, instance: &str, tokens: u64, run: impl FnOnce() -> Result<Value>) -> Result<Value> {
        let mut receipt = self.0.lock().map_err(|_| anyhow::anyhow!("prelude lock poisoned"))?;
        if let Some(receipt) = receipt.as_ref() { return Ok(receipt.clone()); }
        anyhow::ensure!((1..=crate::sweep::MAX_TOKENS).contains(&tokens), "prelude tokens must be 1..={}", crate::sweep::MAX_TOKENS);
        let mut value = run()?;
        value["schema"] = json!(SCHEMA);
        value["server_instance"] = json!(instance);
        value["tokens"] = json!(tokens);
        value["rounds"] = json!(ROUNDS);
        value["complete"] = json!(true);
        *receipt = Some(value.clone());
        Ok(value)
    }
}

fn counters(client: &Client) -> Option<common::DraftCounters> { common::draft_counters(client) }

pub(crate) fn run(client: &Client, tokens: u64) -> Result<Value> {
    let mut requests = Vec::new();
    let mut final_widths = serde_json::Map::new();
    for round in 0..ROUNDS {
        for (content, prompt) in crate::baseline::CONTENT {
            client.check()?;
            let text = format!("[cuteafd-bench-prelude-v1-{round}-{content}] {prompt}");
            let before = counters(client);
            let console_before = client.decode_counters();
            let chat = client.chat(common::plain(client, &text, tokens), None)
                .with_context(|| format!("untimed prelude round {} {content}", round + 1))?;
            let after = counters(client);
            let console_after = client.decode_counters();
            let cycles = chat.usage["speculation"]["cycles"].as_u64().or_else(|| console_before.as_ref().zip(console_after.as_ref()).and_then(|(a,b)|
                Some(b["verification_rounds"].as_u64()?.saturating_sub(a["verification_rounds"].as_u64()?))));
            let width = console_before.as_ref().zip(console_after.as_ref()).and_then(|(a,b)| {
                let passes = b["draft_verify_passes"].as_f64()? - a["draft_verify_passes"].as_f64()?;
                let rows = b["verified_drafts"].as_f64()? - a["verified_drafts"].as_f64()?;
                (passes > 0.).then(|| rows/passes)
            });
            final_widths.insert(content.into(), json!(width));
            requests.push(json!({"round": round + 1, "content": content, "prompt": text,
                "requested_tokens": tokens, "emitted_tokens": chat.timing.completion_tokens,
                "cycles": cycles, "mean_chosen_width": width,
                "opening": chat.timing.opening,
                "cycles_unavailable": cycles.is_none().then_some("server console counters unavailable"),
                "width_unavailable": width.is_none().then_some("no speculative passes or console counters unavailable"),
                "acceptance_length": common::acceptance_length(before, after)}));
        }
    }
    let policy = client.stats().ok().and_then(|stats| stats.get("draft_policy").cloned()
        .or_else(|| stats.get("speculation").cloned()));
    Ok(json!({"requests":requests,"final_mean_width_by_content":final_widths,"policy_state":policy,
        "conditioning":"fixed identical traffic, not a convergence guarantee"}))
}

pub fn validate(receipt: &Value) -> Result<()> {
    anyhow::ensure!(receipt["schema"] == SCHEMA && receipt["complete"] == true
        && receipt["server_instance"].as_str().is_some_and(|s| !s.is_empty())
        && receipt["requests"].as_array().is_some_and(|r| r.len() == ROUNDS * 3),
        "timed benchmark requires a complete warm prelude receipt from this server instance");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prelude_is_idempotent_and_failed_conditioning_never_authorizes_timing() {
        let prelude = Prelude::default();
        assert!(prelude.ensure("server",192,|| anyhow::bail!("warm failure")).is_err());
        assert!(validate(&Value::Null).is_err());
        let receipt = prelude.ensure("server",192,|| Ok(json!({"requests":vec![json!({});6]}))).unwrap();
        validate(&receipt).unwrap();
        assert_eq!(prelude.ensure("server",320,|| panic!("second call must not run")).unwrap(),receipt);
    }

    #[test]
    fn receipts_embed_in_reports_and_sweeps_and_every_timed_path_is_after_conditioning() {
        let prelude = Prelude::default();
        let receipt = prelude.ensure("instance",192,|| Ok(json!({"requests":vec![json!({});6]}))).unwrap();
        let mut report = crate::sample::report(false);
        report.prelude = Some(receipt.clone());
        assert_eq!(serde_json::to_value(&report).unwrap()["prelude"], receipt);
        let sweep = crate::sweep::Sweep { prelude: Some(receipt.clone()), ..Default::default() };
        assert_eq!(serde_json::to_value(&sweep).unwrap()["prelude"],receipt);
        let runner = include_str!("runner.rs");
        let body = runner.split("    fn run_plan(").nth(1).unwrap();
        assert!(body.find("self.ensure_prelude").unwrap() < body.find("crate::baseline::run").unwrap());
        assert!(body.find("self.ensure_prelude").unwrap() < body.find("panel.run(&ctx)").unwrap());
        for profile in crate::profiles::builtin() {
            assert!(!profile.name.is_empty()); // Every profile shares run_plan's ordering.
        }
        let sweep = include_str!("sweep.rs").split("fn run_configured(").nth(1).unwrap();
        assert!(sweep.find("client.warm_prelude").unwrap() < sweep.find("sweep.prefill = prefill").unwrap());
        assert!(sweep.find("client.warm_prelude").unwrap() < sweep.find("measure(").unwrap());
        let smoke = include_str!("smoke.rs");
        assert!(smoke.find("client.warm_prelude").unwrap() < smoke.find("let run_sweep =").unwrap());
    }
}
