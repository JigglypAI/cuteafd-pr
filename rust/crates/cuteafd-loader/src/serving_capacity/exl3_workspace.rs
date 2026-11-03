//! Retained EXL3 capacity arenas, read from the matching exported manifests.
use super::{product, sum, CacheGeometryError};
use serde_json::Value;
use std::collections::BTreeMap;

/// The routed-expert epilogue a family's runtime executes after the EXL3 core.
///
/// The weighted top-k sum is its own native object; it reads the executor's
/// `output` owner. Families whose experts carry a per-expert output norm reply
/// unweighted per-expert partials instead, launch the core only and never read
/// that owner. Planner censuses and worker allocations must declare the same
/// epilogue, or admission and allocation disagree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exl3Epilogue {
    /// Core then the weighted top-k sum into the token output.
    #[default]
    Sum,
    /// Core only; the coordinator combines per-expert partial replies.
    PartialReplies,
}

/// The weighted token sum's allocation owner.
const OUTPUT_OWNER: &str = "output";

impl Exl3Epilogue {
    /// The epilogue the checkpoint's routed experts declare.
    pub fn for_catalog(catalog: &crate::OfficialV41Catalog) -> Self {
        if catalog.exl3().is_some() && catalog.per_expert_output_norm() {
            Self::PartialReplies
        } else {
            Self::Sum
        }
    }
    /// Whether `owner` is allocated for this epilogue.
    pub fn allocates(self, owner: &str) -> bool {
        self == Self::Sum || owner != OUTPUT_OWNER
    }
}

pub fn exl3_workspace_bytes(manifests: &[Value], fp8_k32: bool, epilogue: Exl3Epilogue)
    -> Result<u64, CacheGeometryError> {
    let invalid = |name: &str, what: &str| CacheGeometryError::ResidentTensor {
        name: format!("EXL3 workspace/{name}"), what: what.into(),
    };
    if manifests.is_empty() { return Err(invalid("manifests", "no capacity specializations")); }
    let mut shared = BTreeMap::<String, (u64, String)>::new();
    let mut private = Vec::new();
    for manifest in manifests {
        let buffers = manifest["buffers"].as_object().ok_or_else(|| invalid("buffers", "missing allocation map"))?;
        private.push(manifest["trellis_lut"]["bytes"].as_u64()
            .ok_or_else(|| invalid("trellis_lut", "missing asset extent"))?.max(16));
        for (name, spec) in buffers {
            let allocation = spec["allocation"].as_str().ok_or_else(|| invalid(name, "missing allocation owner"))?;
            if name != allocation || !epilogue.allocates(name) { continue; }
            let bytes = spec["bytes"].as_u64().ok_or_else(|| invalid(name, "missing byte extent"))?.max(16);
            let zero = spec["zero_on_create"].as_bool().ok_or_else(|| invalid(name, "missing initialization policy"))?;
            if zero { private.push(bytes); continue; }
            let dtype = spec["dtype"].as_str().ok_or_else(|| invalid(name, "missing dtype"))?;
            let entry = shared.entry(name.clone()).or_insert((0, dtype.into()));
            if entry.1 != dtype { return Err(invalid(name, "shared capacity arena has conflicting dtypes")); }
            entry.0 = entry.0.max(bytes);
        }
        match manifest["input_format"].as_str() {
            Some("e4m3_k32") => (),
            None | Some("bf16") if fp8_k32 => {
                let rows = manifest["capacity"].as_u64().ok_or_else(|| invalid("capacity", "missing row extent"))?;
                let hidden = manifest["hidden"].as_u64().ok_or_else(|| invalid("hidden", "missing hidden extent"))?;
                private.push(product("EXL3 wire decode", &[rows, hidden, 2])?);
            }
            None | Some("bf16") => (),
            _ => return Err(invalid("input_format", "unknown wire representation")),
        }
    }
    private.extend(shared.into_values().map(|(bytes, _)| bytes));
    sum("EXL3 capacity arenas", &private)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn aliases_share_scratch_but_capacity_state_and_wire_decode_stay_private() {
        let manifest = |rows, bytes| json!({"capacity": rows, "hidden": 8,
            "trellis_lut": {"bytes": 32}, "buffers": {
                "scratch": {"bytes": bytes, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
                "alias": {"bytes": 99999, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
                "state": {"bytes": 0, "allocation": "state", "dtype": "i32", "zero_on_create": true}}});
        let manifests = [manifest(1, 64), manifest(16, 1024)];
        let sum = |manifests, wire| exl3_workspace_bytes(manifests, wire, Exl3Epilogue::Sum);
        assert_eq!(sum(&manifests, true).unwrap(), 1024 + 2 * (32 + 16) + 17 * 8 * 2);
        assert_eq!(sum(&manifests, false).unwrap(), 1024 + 2 * (32 + 16));
        // A partial-reply epilogue launches the core only: the weighted sum's
        // output owner is never allocated, and every core owner is unchanged.
        let mut partial = manifests.clone();
        partial[0]["buffers"]["output"] = json!({"bytes": 64, "allocation": "output",
            "dtype": "bf16", "zero_on_create": false});
        partial[1]["buffers"]["output"] = json!({"bytes": 1024, "allocation": "output",
            "dtype": "bf16", "zero_on_create": false});
        assert_eq!(exl3_workspace_bytes(&partial, true, Exl3Epilogue::PartialReplies).unwrap(),
            1024 + 2 * (32 + 16) + 17 * 8 * 2);
        assert_eq!(exl3_workspace_bytes(&partial, true, Exl3Epilogue::Sum).unwrap(),
            1024 + 2 * (32 + 16) + 17 * 8 * 2 + 1024);
        let mut conflict = manifests.clone();
        conflict[1]["buffers"]["scratch"]["dtype"] = json!("f32");
        assert!(exl3_workspace_bytes(&conflict, true, Exl3Epilogue::Sum).is_err());
    }
}
