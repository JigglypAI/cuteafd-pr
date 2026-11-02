//! Checkpoint-native operand names and packing, shared with the dense loader.
use crate::families::qwen::HybridAttention;

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
