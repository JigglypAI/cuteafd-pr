//! Attention operand precision, independent of weights and KV storage.

/// Operand precision of both attention QK and PV products (FP32 accumulation).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionPrecision {
    #[default]
    Bf16,
    Fp8,
}

impl AttentionPrecision {
    pub const fn name(self) -> &'static str {
        match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" }
    }
}

impl std::fmt::Display for AttentionPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.name()) }
}

/// Accept only the shared AOT sparse MLA ABI, not context partials or other
/// programs which also use BF16/FP8 in their names.
pub fn sparse_mla_precision_stem(name: &str) -> Option<(&str, AttentionPrecision)> {
    let (stem, precision) = name.strip_suffix("_bf16").map_or((name, AttentionPrecision::Fp8),
        |stem| (stem, AttentionPrecision::Bf16));
    let (family, suffix) = stem.split_once("_sparse_mla_")?;
    let (kind, rows) = suffix.rsplit_once("_m")?;
    if rows.is_empty() || !rows.bytes().all(|x| x.is_ascii_digit()) { return None; }
    let supported = match family {
        "glm" | "glm2" | "glmf" | "glmf2" => matches!(kind, "decode" | "prefill"),
        "dsv4f" | "dsv4f2" | "dsv4p" | "dsv4p2" => matches!(kind,
            "decode_win" | "decode_c4" | "decode_c128" | "prefill_win" | "prefill_c4" | "prefill_c128"),
        _ => false,
    };
    supported.then_some((stem, precision))
}

/// Present the selected entry under its logical name for scratch and code
/// admission. Older planning-only inventories retain their estimates; serving
/// requires the exact selected symbol before allocating its owners.
pub fn select_sparse_mla_manifest(manifest: &mut serde_json::Value, precision: AttentionPrecision) {
    let Some(entries) = manifest["programs"].as_array_mut() else { return; };
    let alternatives: std::collections::HashSet<_> = entries.iter().filter_map(|p| {
        let (stem, mode) = sparse_mla_precision_stem(p["name"].as_str()?)?;
        (mode == AttentionPrecision::Bf16).then(|| stem.to_owned())
    }).collect();
    entries.retain_mut(|entry| {
        let Some((stem, mode)) = entry["name"].as_str().and_then(sparse_mla_precision_stem) else { return true; };
        match (precision, mode) {
            (AttentionPrecision::Bf16, AttentionPrecision::Bf16) => { entry["name"] = stem.to_owned().into(); true },
            (AttentionPrecision::Bf16, AttentionPrecision::Fp8) => !alternatives.contains(stem),
            (AttentionPrecision::Fp8, _) => mode == precision,
        }
    });
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?}: expected bf16 or fp8")]
pub struct AttentionPrecisionError(String);

impl std::str::FromStr for AttentionPrecision {
    type Err = AttentionPrecisionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            other => Err(AttentionPrecisionError(other.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_selects_code_and_scratch_under_the_same_logical_name() {
        let base = "glm_sparse_mla_decode_m64";
        let original = serde_json::json!({"programs": [
            {"name": base, "scratch": 12},
            {"name": "glm_sparse_mla_decode_m64_bf16", "scratch": 34},
            {"name": "glm_sparse_mla_partial_h0_s1_m64", "scratch": 56},
        ]});
        for (mode, bytes) in [(AttentionPrecision::Bf16, 34), (AttentionPrecision::Fp8, 12)] {
            let mut manifest = original.clone();
            select_sparse_mla_manifest(&mut manifest, mode);
            let programs = manifest["programs"].as_array().unwrap();
            assert_eq!(programs.len(), 2);
            assert_eq!(programs.iter().find(|p| p["name"] == base).unwrap()["scratch"], bytes);
            assert!(programs.iter().any(|p| p["scratch"] == 56));
        }
    }

    #[test]
    fn shared_default_never_depends_on_checkpoint_format() {
        assert_eq!(AttentionPrecision::default(), AttentionPrecision::Bf16);
        for mode in [AttentionPrecision::Bf16, AttentionPrecision::Fp8] {
            assert_eq!(mode.to_string().parse(), Ok(mode));
            assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{mode}\""));
        }
        for removed in ["auto", "", "BF16", "bf16-qk/fp8-pv"] {
            assert!(removed.parse::<AttentionPrecision>().is_err(), "{removed}");
        }
    }
}
