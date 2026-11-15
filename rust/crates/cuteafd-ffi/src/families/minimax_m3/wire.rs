//! Checked launch contract for the exact NVFP4 expert-input wire quantizer
//! (`SparkIngress::Nvfp4`): the Spark W4A4 package's own calibrated row
//! quantizer, run once per token on the coordinator.
use super::{bytes, invalid, validate_buffers, Result};
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;

pub const NAME: &str = "minimax_nvfp4_wire_rows";
const HIDDEN: usize = 6144;
const POINTERS: [&str; 2] = ["src", "dst"];

/// Bytes of one wire row (`cuteafd_core::nvfp4_wire_row_bytes(6144)`).
pub const ROW_BYTES: usize = cuteafd_core::nvfp4_wire_row_bytes(HIDDEN);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireLayout { pub rows: u32 }
impl WireLayout {
    pub fn from_spec(spec: &ProgramSpec) -> Result<Self> {
        let g = &spec.geometry;
        if spec.name != NAME || g["hidden"] != HIDDEN || g["mode"] != "calibrated"
            || g["scale_convention"] != "divisor" || g["row_bytes"] != ROW_BYTES
            || g["uniform_expert_input_scale_required"] != true
            || spec.capacity_rows == 0 || spec.capacity_rows > 4096 || g["max_rows"] != spec.capacity_rows
            || spec.scalar_kinds != b"if" || !spec.scratch.is_empty()
            || spec.pointers.iter().map(String::as_str).ne(POINTERS) {
            return Err(invalid("NVFP4 wire quantizer geometry, capacity or ABI"));
        }
        Ok(Self { rows: spec.capacity_rows })
    }
}

pub fn validate_entry(spec: &ProgramSpec, entry: &serde_json::Value) -> Result<WireLayout> {
    let layout = WireLayout::from_spec(spec)?;
    let expected = serde_json::json!([
        {"name": "src", "dtype": "bfloat16", "shape": format!("[rows,{HIDDEN}]"), "role": "in"},
        {"name": "dst", "dtype": "uint8", "shape": format!("[rows,{ROW_BYTES}]"), "role": "out"}]);
    let pointers = entry["pointers"].as_array().ok_or_else(|| invalid("wire pointer metadata"))?;
    if entry["op"] != "nvfp4_wire_rows" || pointers.len() != 2
        || pointers.iter().zip(expected.as_array().unwrap()).any(|(p, e)| ["name", "dtype", "shape", "role"]
            .iter().any(|key| p[*key] != e[*key]))
        || entry["scalars"] != serde_json::json!([{"name":"rows","type":"int32"},{"name":"input_scale","type":"float32"}]) {
        return Err(invalid("NVFP4 wire exported operand or operation contract"));
    }
    Ok(layout)
}

pub struct WireQuantizer<'a> { program: Program<'a>, pub layout: WireLayout }
impl<'a> WireQuantizer<'a> {
    pub fn bind(programs: &'a Programs<'a>) -> Result<Self> {
        let spec = programs.spec(NAME).map_err(|e| invalid(e.to_string()))?;
        let layout = WireLayout::from_spec(spec)?;
        let program = programs.program(NAME, &POINTERS).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { program, layout })
    }
    /// Quantizes `rows` BF16 `[rows, 6144]` rows into `[rows, ROW_BYTES]` wire
    /// rows with the layer's one shared raw checkpoint `input_global_scale`.
    ///
    /// # Safety
    /// Storage and stream must remain live through queued work and graph use.
    pub unsafe fn quantize(&self, rows: usize, input: Buffer, output: Buffer, input_scale: f32,
        stream: *mut c_void) -> Result<()> {
        if rows == 0 || rows > self.layout.rows as usize || !input_scale.is_finite() || input_scale <= 0.0 {
            return Err(invalid("NVFP4 wire live rows or input scale"));
        }
        validate_buffers(&[(input, bytes(rows, HIDDEN * 2)?), (output, bytes(rows, ROW_BYTES)?)], &[16; 2], &[1])
            .map_err(invalid)?;
        // SAFETY: host extents/devices/aliases checked; caller retains storage.
        unsafe { self.program.launch(&[input.ptr, output.ptr],
            &[Scalar::I32(rows as i32), Scalar::F32(input_scale)], stream) }.map_err(|e| invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> ProgramSpec {
        ProgramSpec { name: NAME.into(), pointers: POINTERS.map(String::from).to_vec(), scalar_kinds: b"if".to_vec(),
            scratch: Default::default(), capacity_rows: 4096,
            geometry: serde_json::json!({"hidden": 6144, "max_rows": 4096, "mode": "calibrated",
                "scale_convention": "divisor", "row_bytes": 3472, "uniform_expert_input_scale_required": true}) }
    }
    #[test]
    fn accepts_exact_quantizer_and_rejects_other_contracts() {
        assert_eq!(ROW_BYTES, 3472);
        assert_eq!(WireLayout::from_spec(&spec()).unwrap(), WireLayout { rows: 4096 });
        for edit in [
            |s: &mut ProgramSpec| s.geometry["mode"] = "dynamic".into(),
            |s: &mut ProgramSpec| s.geometry["scale_convention"] = "multiplier".into(),
            |s: &mut ProgramSpec| s.geometry["row_bytes"] = 12288.into(),
            |s: &mut ProgramSpec| s.scalar_kinds = b"i".to_vec(),
            |s: &mut ProgramSpec| s.pointers.swap(0, 1),
            |s: &mut ProgramSpec| s.capacity_rows = 2048,
        ] {
            let mut wrong = spec();
            edit(&mut wrong);
            assert!(WireLayout::from_spec(&wrong).is_err());
        }
        let entry = serde_json::json!({"op": "nvfp4_wire_rows",
            "pointers": [{"name":"src","dtype":"bfloat16","shape":"[rows,6144]","role":"in"},
                         {"name":"dst","dtype":"uint8","shape":"[rows,3472]","role":"out"}],
            "scalars": [{"name":"rows","type":"int32"},{"name":"input_scale","type":"float32"}]});
        validate_entry(&spec(), &entry).unwrap();
        let mut wrong = entry.clone();
        wrong["scalars"] = serde_json::json!([{"name":"rows","type":"int32"}]);
        assert!(validate_entry(&spec(), &wrong).is_err());
    }
}
