//! Attention prefill precision: the operand precision of the attention
//! kernel's own QK and PV products at prefill (`ATTENTION_PREFILL_PRECISION`).
//!
//! It is not a weight format (`CORE_VIEW`), a projection activation format or
//! the KV cache storage format. Decode attention has no key. `bf16` is BF16
//! Q, V and P with exact keys: an FP8 key enters the QK product as its exact
//! BF16 widening and its group scale applies to the FP32 partial. A family
//! states the modes its prefill programs run; `auto` takes the first, or the
//! family's measured choice for an FP8 core view.
use super::PlanError;

pub const OPTION: &str = "--attention-prefill-precision";

/// What one prefill attention program computes with. Both products
/// accumulate in FP32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionPrecision {
    /// BF16 Q, V and P; keys exact (see the module note).
    Bf16,
    /// BF16 Q and K; E4M3 P and V. Not a requestable value: the mode the
    /// DeepSeek prefill programs run until their BF16 PV export lands.
    Bf16QkFp8Pv,
    /// E4M3 Q, K, V and P (GLM's native kernel: two-term E4M3 Q and P).
    Fp8,
}

impl AttentionPrecision {
    pub fn name(self) -> &'static str {
        match self { Self::Bf16 => "bf16", Self::Bf16QkFp8Pv => "bf16-qk/fp8-pv", Self::Fp8 => "fp8" }
    }
}

/// The launch key's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Requested {
    #[default]
    Auto,
    Mode(AttentionPrecision),
}

impl std::str::FromStr for Requested {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "bf16" => Ok(Self::Mode(AttentionPrecision::Bf16)),
            "fp8" => Ok(Self::Mode(AttentionPrecision::Fp8)),
            other => Err(format!("{other:?}: expected auto, bf16 or fp8")),
        }
    }
}

/// A family's exported prefill attention modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPrecisionSupport {
    /// Modes with an exported program; the first is `auto`.
    pub modes: &'static [AttentionPrecision],
    /// `auto` for an FP8 core view (`CORE_VIEW=fp8-compact`), set only where
    /// FP8 prefill attention measured a speed win on that view.
    pub fp8_core_auto: Option<AttentionPrecision>,
    /// The measurement or audit behind the entry.
    pub source: &'static str,
}

/// Families whose attention kernels compute BF16 QK and PV and export nothing else.
pub const BF16_ONLY: PrefillPrecisionSupport = PrefillPrecisionSupport {
    modes: &[AttentionPrecision::Bf16], fp8_core_auto: None,
    source: "its attention kernels compute BF16 QK and PV; no FP8 attention program is exported (#2146)",
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Resolved {
    pub mode: AttentionPrecision,
    pub reason: String,
}

/// Resolve the prefill attention mode before layout or any CUDA allocation.
pub fn resolve(family: &dyn super::Family, requested: Requested, fp8_core: bool) -> Result<Resolved, PlanError> {
    let support = family.attention_prefill_precision();
    let default = *support.modes.first().expect("a family exports a prefill attention mode");
    match requested {
        Requested::Auto => Ok(match support.fp8_core_auto.filter(|_| fp8_core) {
            Some(mode) => Resolved { mode, reason: format!("auto: FP8 core view; {}", support.source) },
            None => Resolved { mode: default, reason: format!("auto: {}", support.source) },
        }),
        Requested::Mode(mode) if support.modes.contains(&mode) =>
            Ok(Resolved { mode, reason: "requested".into() }),
        Requested::Mode(mode) => Err(PlanError::UnsupportedValue {
            option: OPTION, value: mode.name().into(), family: family.id().into(),
            reason: format!("no {} prefill attention program is exported (available: {}); {}", mode.name(),
                support.modes.iter().map(|mode| mode.name()).collect::<Vec<_>>().join(", "), support.source),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use AttentionPrecision::{Bf16, Bf16QkFp8Pv, Fp8};

    fn family(id: &str) -> &'static dyn crate::plan::Family {
        crate::plan::spark_executions::descriptor(id).unwrap()
    }

    /// Every family resolves `auto` to the mode its programs run today, with
    /// or without an FP8 core view: adding the key changes no launch.
    #[test]
    fn auto_is_every_family_s_current_mode() {
        for (id, mode) in [("deepseek_v41", Bf16QkFp8Pv), ("deepseek_v4", Bf16QkFp8Pv), ("glm5", Fp8),
            ("glm5_flash", Fp8), ("mistral_large4", Fp8), ("kimi_k3", Bf16), ("mimo_v2", Bf16), ("qwen4", Bf16),
            ("qwen3_5", Bf16), ("step5", Bf16), ("muse_spark", Bf16), ("minimax_m3", Bf16), ("inkling2", Bf16)] {
            for fp8_core in [false, true] {
                let resolved = resolve(family(id), Requested::Auto, fp8_core).unwrap();
                assert_eq!(resolved.mode, mode, "{id}");
                assert!(resolved.reason.starts_with("auto: "), "{id}: {}", resolved.reason);
            }
        }
        assert_eq!(crate::plan::family::registry().len(), 15, "a new family states its prefill attention modes here");
    }

    #[test]
    fn a_mode_without_an_exported_program_is_refused_by_name() {
        for descriptor in crate::plan::family::registry() {
            let support = descriptor.attention_prefill_precision();
            assert!(!support.modes.is_empty() && !support.source.is_empty(), "{}", descriptor.id());
            assert!(support.fp8_core_auto.is_none_or(|mode| support.modes.contains(&mode)), "{}", descriptor.id());
            for mode in [Bf16, Fp8] {
                let result = resolve(*descriptor, Requested::Mode(mode), false);
                if support.modes.contains(&mode) {
                    assert_eq!(result.unwrap(), Resolved { mode, reason: "requested".into() });
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(error.contains(&format!("unsupported {OPTION}={} for {}", mode.name(), descriptor.id()))
                        && error.contains("prefill attention program"), "{error}");
                }
            }
        }
    }

    #[test]
    fn an_fp8_core_view_changes_auto_only_where_the_family_measured_it() {
        struct Measured;
        impl crate::plan::Family for Measured {
            fn id(&self) -> &'static str { "measured" }
            fn attention_prefill_precision(&self) -> PrefillPrecisionSupport {
                PrefillPrecisionSupport { modes: &[Bf16, Fp8], fp8_core_auto: Some(Fp8), source: "test" }
            }
            fn detect(&self, _: &crate::plan::Checkpoint) -> bool { false }
            fn runtime(&self) -> crate::plan::RuntimeStatus { crate::plan::RuntimeStatus::Planned }
            fn open(&self, _: &crate::plan::Checkpoint) -> Result<Box<dyn crate::plan::FamilyModel>, crate::plan::ConfigError> {
                unreachable!()
            }
            fn classify(&self, _: &crate::plan::ModelSpec, _: &str) -> Option<crate::plan::TensorRole> { None }
            fn component_hint(&self, _: crate::plan::Component) -> Option<crate::plan::Hint> { None }
        }
        assert_eq!(resolve(&Measured, Requested::Auto, false).unwrap().mode, Bf16);
        assert_eq!(resolve(&Measured, Requested::Auto, true).unwrap().mode, Fp8);
        assert_eq!(resolve(&Measured, Requested::Mode(Bf16), true).unwrap().mode, Bf16);
        assert_eq!(resolve(&Measured, Requested::Mode(Fp8), false).unwrap().mode, Fp8);
    }

    #[test]
    fn the_key_takes_auto_bf16_or_fp8() {
        assert_eq!("auto".parse(), Ok(Requested::Auto));
        assert_eq!("bf16".parse(), Ok(Requested::Mode(Bf16)));
        assert_eq!("fp8".parse(), Ok(Requested::Mode(Fp8)));
        for value in ["", "BF16", "bf16-qk", "bf16-qk/fp8-pv"] {
            assert!(value.parse::<Requested>().is_err(), "{value}");
        }
    }
}
