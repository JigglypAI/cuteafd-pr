//! Deterministic launch defaults. Live capacities and availability are explicit inputs.
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolved<T> {
    pub value: T,
    pub provenance: String,
}

pub fn selected<T>(explicit: Option<T>, configured: Option<T>, automatic: T, rule: &str) -> Resolved<T> {
    match (explicit, configured) {
        (Some(value), _) => Resolved { value, provenance: "explicit".into() },
        (_, Some(value)) => Resolved { value, provenance: "config".into() },
        _ => Resolved { value: automatic, provenance: format!("auto: {rule}") },
    }
}

pub fn glmf_index_cache(explicit: Option<&str>, configured: Option<&str>, tp: usize) -> Resolved<String> {
    selected(explicit.filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        configured.filter(|_| explicit.is_none()).filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        if tp == 1 { "compact" } else { "keys" }.into(), "compact on TP1; keys otherwise")
}

pub fn glmf_replay_records(explicit: Option<&str>, configured: Option<&str>, tp: usize,
    auto_pool: bool, spark_experts: bool) -> Resolved<String> {
    selected(explicit.filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        configured.filter(|_| explicit.is_none()).filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        if tp == 1 && auto_pool && spark_experts { "shared" } else { "own" }.into(),
        "shared on TP1 with automatic pool and admitted Spark experts; own otherwise")
}

pub fn glmf_draft_linear(explicit: Option<&str>, configured: Option<&str>, external_fp8: bool) -> Resolved<String> {
    selected(explicit.map(str::to_owned), configured.filter(|v| !v.is_empty()).map(str::to_owned),
        if external_fp8 { "w8a8" } else { "w8a16" }.into(), "w8a8 with an external FP8 drafter; w8a16 otherwise")
}

pub fn glmf_speculator(explicit: Option<&str>, configured: Option<&str>) -> Resolved<String> {
    selected(explicit.filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        configured.filter(|_| explicit.is_none()).filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        "dflash2".into(), "DFlash2; the default drafter must be available")
}

/// Occupancy is a supplied live measurement, never sampled by this resolver.
pub fn coordinator_weight_budget(explicit_gib: Option<f64>, configured_gib: Option<f64>,
    capacity_bytes: u64) -> Resolved<f64> {
    selected(explicit_gib, configured_gib, capacity_bytes as f64 / (1u64 << 30) as f64,
        "device capacity after explicit cap and occupancy adjustment")
}

#[derive(Debug,thiserror::Error)]
#[error("unsupported Kimi SPECULATOR={0}; use off, dflash2 or dspark")]
pub struct KimiSpeculatorError(String);

pub const KIMI_K31_DSPARK: &str = "Inferact/Kimi-K3.1-Pro-DSpark";
pub const KIMI_K31_DSPARK_REVISION: &str = "c197a09072916cf63172df2d9d4baef0b77ed83a";

pub const KIMI_K31_MC: &str = "wrldsuksgo2mars/Kimi-K3.1-Pro-EXL3-K2-MC";

#[derive(Debug, Clone, Copy, Default)]
pub struct KimiDraftQualification<'a> {
    pub content_qualified: bool,
    pub core_view: &'a str,
    pub rtx: usize,
    pub target_kv: &'a str,
    pub draft_kv: &'a str,
}

impl KimiDraftQualification<'_> {
    pub fn qualified(self) -> bool {
        self.content_qualified
    }
}

pub fn kimi_speculator(explicit:Option<&str>,configured:Option<&str>, qualification: KimiDraftQualification<'_>)
    -> Result<Resolved<String>,KimiSpeculatorError> {
    let resolved=selected(explicit.filter(|v|!v.is_empty() && *v!="auto").map(str::to_owned),
        configured.filter(|_|explicit.is_none()).filter(|v|!v.is_empty() && *v!="auto").map(str::to_owned),
        if qualification.qualified() { "dspark" } else { "off" }.into(),
        "DSpark for paired K3.1 target/drafter content in every view and layout; off otherwise");
    if !matches!(resolved.value.as_str(),"off"|"dflash2"|"dspark") {
        return Err(KimiSpeculatorError(resolved.value));
    }
    Ok(resolved)
}

pub fn kimi_draft_fp8(explicit: Option<bool>, dflash: bool) -> Resolved<bool> {
    selected(explicit, None, dflash, "FP8 for explicit Kimi DFlash2")
}

pub fn mimo_speculator(explicit: Option<&str>, configured: Option<&str>, model_id: &str) -> Resolved<String> {
    selected(explicit.filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        configured.filter(|_| explicit.is_none()).filter(|v| !v.is_empty() && *v != "auto").map(str::to_owned),
        if matches!(model_id, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD" | "XiaomiMiMo/MiMo-V2.6-Pro-MOPD") { "dflash2" } else { "off" }.into(), "DFlash2 for official MiMo V2.6 Flash/Pro MOPD model ids; off otherwise")
}

pub fn mimo_mtp_depth(explicit: Option<usize>, configured: Option<usize>, mtp: bool) -> Resolved<usize> {
    selected(explicit, configured, if mtp { 1 } else { 0 }, "one stage for explicit MiMo MTP; zero otherwise")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kimi_defaults_follow_content_pairing_not_view_layout_or_kv() {
        let qualified = KimiDraftQualification { content_qualified: true, core_view: "original", rtx: 2,
            target_kv: "fp8", draft_kv: "bf16" };
        assert_eq!(kimi_speculator(None, None, qualified).unwrap().value, "dspark");
        assert_eq!(kimi_speculator(Some("auto"), Some("off"), qualified).unwrap().value, "dspark");
        assert_eq!(kimi_speculator(Some("off"), None, qualified).unwrap().value, "off");
        assert_eq!(kimi_speculator(None, Some("off"), qualified).unwrap().value, "off");
        for view in ["original", "k6-compact", "k8-compact", "fp8-compact"] { for rtx in [1, 2, 3] {
            for target_kv in ["bf16", "fp8"] { for draft_kv in ["bf16", "fp8"] {
                assert_eq!(kimi_speculator(None, None, KimiDraftQualification {
                    core_view: view, rtx, target_kv, draft_kv, ..qualified }).unwrap().value, "dspark");
            } }
        } }
        for other in [KimiDraftQualification { content_qualified: false, ..qualified }] {
            assert_eq!(kimi_speculator(None, None, other).unwrap().value, "off");
            assert_eq!(kimi_speculator(Some("dspark"), None, other).unwrap().value, "dspark");
        }
    }

    #[test]
    fn defaults_and_precedence() {
        assert_eq!(glmf_index_cache(None, None, 1).value, "compact");
        assert_eq!(glmf_index_cache(None, None, 2).value, "keys");
        assert_eq!(glmf_index_cache(Some("keys"), Some("compact"), 1).provenance, "explicit");
        for tp in 1..=2 { for auto in [false, true] { for sparks in [false, true] {
            assert_eq!(glmf_replay_records(None, None, tp, auto, sparks).value == "shared", tp == 1 && auto && sparks);
        } } }
        assert_eq!(glmf_draft_linear(None, None, true).value, "w8a8");
        assert_eq!(glmf_draft_linear(None, Some("wide"), true).provenance, "config");
        assert_eq!(glmf_speculator(None, None).value, "dflash2");
        assert_eq!(glmf_speculator(None, Some("off")).value, "off");
        assert_eq!(coordinator_weight_budget(None, None, 32 << 30).value, 32.0);
        assert_eq!(coordinator_weight_budget(Some(20.0), None, 32 << 30).value, 20.0);
        assert_eq!(glmf_replay_records(Some("auto"), Some("shared"), 2, true, true).value, "own");
        assert_eq!(mimo_speculator(None, None, "community/lookalike").value, "off");
        assert_eq!(mimo_speculator(None, None, "XiaomiMiMo/MiMo-V2.6-Flash-MOPD").value, "dflash2");
        assert_eq!(mimo_speculator(Some("mtp"), Some("off"), "XiaomiMiMo/MiMo-V2.6-Pro-MOPD").value, "mtp");
        assert_eq!(mimo_mtp_depth(None, None, true).value, 1);
        assert_eq!(mimo_mtp_depth(Some(3), Some(2), true).value, 3);
        assert_eq!(kimi_speculator(None,None, KimiDraftQualification::default()).unwrap().value,"off");
        assert_eq!(kimi_speculator(None,Some("dspark"), KimiDraftQualification::default()).unwrap().value,"dspark");
        assert_eq!(kimi_speculator(Some("off"),Some("dspark"), KimiDraftQualification::default()).unwrap().value,"off");
        assert!(kimi_speculator(Some("mtp"),None, KimiDraftQualification::default()).is_err());
        assert!(kimi_draft_fp8(None, true).value);
        assert!(!kimi_draft_fp8(Some(false), true).value);
    }
}
