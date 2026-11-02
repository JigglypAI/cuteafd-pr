//! Checkpoint-native operand names and packing, shared with the dense loader.
use crate::families::qwen::HybridAttention;
use crate::plan::checkpoint::Checkpoint;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use super::NativeOperand;

/// Header-derived physical operands and the serialized read/packing heap peak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLayerInventory {
    pub allocations: Vec<NativeOperand>,
    pub loading_host_bytes: u64,
}

impl NativeLayerInventory {
    pub fn bytes(&self) -> Result<u64> {
        self.allocations.iter().try_fold(0u64, |sum, operand|
            sum.checked_add(operand.bytes).context("native layer size overflow"))
    }
}

/// Target and native MTP execute the same packed operands. Their header
/// validation, scale widening and upload peak must therefore be identical.
pub fn native_inventory(checkpoint: &Checkpoint, prefix: &str, attention: HybridAttention,
    fp8: bool, norm_dtype: DType) -> Result<NativeLayerInventory> {
    ensure!(matches!(norm_dtype, DType::Bf16 | DType::F32), "invalid native GDN norm dtype");
    let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .with_context(|| format!("checkpoint lacks {name}"));
    let mut allocations = Vec::new();
    let mut loading_host_bytes = 0;
    for operand in native_operands(prefix, attention, fp8) {
        let mut bytes = 0u64;
        let mut raw_bytes = 0u64;
        let mut largest_upload = 0u64;
        let mut columns = None;
        for name in &operand.tensors {
            let value = &tensor(name)?.meta;
            let expected = match operand.storage {
                Storage::Bf16 => value.dtype == DType::Bf16,
                Storage::NativeNorm => value.dtype == norm_dtype,
                Storage::F32 | Storage::Scales => matches!(value.dtype, DType::Bf16 | DType::F32),
                Storage::Fp8 => value.dtype == DType::F8E4M3,
            };
            ensure!(expected, "{name}: native operand dtype differs");
            if matches!(operand.storage, Storage::Fp8 | Storage::Scales) {
                if operand.storage == Storage::Fp8 {
                    ensure!(value.shape.len() == 2 && value.shape.iter().all(|n| *n > 0 && n % 128 == 0),
                        "{name}: expected whole 128x128 E4M3 blocks");
                    ensure!(columns.is_none_or(|n| n == value.shape[1]), "{name}: packed columns differ");
                    columns = Some(value.shape[1]);
                } else {
                    let weight = &tensor(name.strip_suffix("_scale_inv").context("invalid scale operand")?)?.meta;
                    ensure!(weight.shape.len() == 2 && value.shape == weight.shape.iter().map(|n| n / 128).collect::<Vec<_>>(),
                        "{name}: invalid native FP8 block scale grid");
                }
            }
            let widen = matches!(operand.storage, Storage::F32 | Storage::Scales) && value.dtype == DType::Bf16;
            raw_bytes = raw_bytes.checked_add(value.byte_length).context("dense host read overflow")?;
            largest_upload = largest_upload.max(value.byte_length);
            bytes = bytes.checked_add(value.byte_length.checked_mul(if widen { 2 } else { 1 })
                .context("dense operand size overflow")?).context("dense packed size overflow")?;
        }
        // Serialized rows() reads upload separately; packing/f32 conversion
        // retain raw payloads beside a Vec whose capacity can double.
        let packed = matches!(operand.storage, Storage::Fp8 | Storage::Scales | Storage::F32);
        let host = raw_bytes.checked_add(if packed { bytes.checked_mul(2).context("dense host pack overflow")? } else { 0 })
            .context("dense loading host overflow")?;
        loading_host_bytes = loading_host_bytes.max(host);
        let upload_bytes = if packed { bytes } else { largest_upload };
        allocations.push(NativeOperand { id: operand.name.into(), bytes: bytes.max(256), upload_bytes });
    }
    Ok(NativeLayerInventory { allocations, loading_host_bytes })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Bf16,
    F32,
    NativeNorm,
    Fp8,
    Scales,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Operand {
    pub name: &'static str,
    pub tensors: Vec<String>,
    pub storage: Storage,
}

pub fn layer_operands(prefix: &str, attention: HybridAttention) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut add = |name, tensors: &[&str], storage| {
        out.push(Operand {
            name,
            tensors: tensors.iter().map(|t| format!("{prefix}.{t}")).collect(),
            storage,
        })
    };
    for (operand, tensor) in [
        ("attn.norm", "input_layernorm.weight"),
        ("ffn.norm", "post_attention_layernorm.weight"),
        ("ffn.w_down", "mlp.down_proj.weight"),
    ] {
        add(operand, &[tensor], Storage::Bf16);
    }
    add(
        "ffn.w_gate_up",
        &["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        Storage::Bf16,
    );
    match attention {
        HybridAttention::Gdn => {
            add(
                "w_in",
                &[
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.in_proj_a.weight",
                ],
                Storage::Bf16,
            );
            for (operand, tensor) in [
                ("conv_w", "conv1d.weight"),
                ("a_log", "A_log"),
                ("dt_bias", "dt_bias"),
            ] {
                add(operand, &[&format!("linear_attn.{tensor}")], Storage::F32);
            }
            add("norm_w", &["linear_attn.norm.weight"], Storage::NativeNorm);
            add("w_out", &["linear_attn.out_proj.weight"], Storage::Bf16);
        }
        HybridAttention::Full => {
            add(
                "w_in",
                &[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                Storage::Bf16,
            );
            for (operand, tensor) in [
                ("q_norm", "q_norm.weight"),
                ("k_norm", "k_norm.weight"),
                ("w_o", "o_proj.weight"),
            ] {
                add(operand, &[&format!("self_attn.{tensor}")], Storage::Bf16);
            }
        }
    }
    out
}

pub fn native_operands(prefix: &str, attention: HybridAttention, fp8: bool) -> Vec<Operand> {
    let specs = layer_operands(prefix, attention);
    if !fp8 {
        return specs;
    }
    let mut out = Vec::new();
    for mut spec in specs {
        let projection = matches!(
            spec.name,
            "w_in" | "w_out" | "w_o" | "ffn.w_gate_up" | "ffn.w_down"
        );
        if !projection {
            out.push(spec);
            continue;
        }
        if attention == HybridAttention::Gdn && spec.name == "w_in" {
            let gates = spec.tensors.split_off(2);
            out.push(Operand {
                name: "w_ba",
                tensors: gates,
                storage: Storage::Bf16,
            });
        }
        let (weight, scales) = match spec.name {
            "w_in" if attention == HybridAttention::Gdn => ("w_qkvz_fp8", "w_qkvz_scale"),
            "w_in" => ("w_in_fp8", "w_in_scale"),
            "w_out" => ("w_out_fp8", "w_out_scale"),
            "w_o" => ("w_o_fp8", "w_o_scale"),
            "ffn.w_gate_up" => ("ffn.w_gate_up_fp8", "ffn.w_gate_up_scale"),
            "ffn.w_down" => ("ffn.w_down_fp8", "ffn.w_down_scale"),
            _ => unreachable!(),
        };
        out.push(Operand {
            name: scales,
            tensors: spec
                .tensors
                .iter()
                .map(|name| format!("{name}_scale_inv"))
                .collect(),
            storage: Storage::Scales,
        });
        out.push(Operand {
            name: weight,
            tensors: spec.tensors,
            storage: Storage::Fp8,
        });
    }
    out
}
