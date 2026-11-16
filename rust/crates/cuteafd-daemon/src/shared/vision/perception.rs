//! Header-only PE packing. Runtime dispatch stays gated until encoder oracles pass.
use super::*;
use cuteafd_ffi::vision::PerceptionSpec;
use cuteafd_loader::media::perception_vision::{PerceptionGeometry, PerceptionKind};

impl TowerSpec {
    pub fn perception(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let g = PerceptionGeometry::from_config(&cfg).map_err(VisionError::Unsupported)?;
        if !(1..=4096).contains(&max_tokens) {
            return Err(VisionError::Unsupported("PE image capacity must be 1..=4096".into()));
        }
        let tensors = match g.kind {
            PerceptionKind::Inkling2 => tower_catalog(snapshot,"model.visual.")?,
            PerceptionKind::Step5 => {
                let mut tensors = tower_catalog(snapshot,"vision_model.")?;
                tensors.extend(tower_catalog(snapshot,"vit_large_projector.")?);
                tensors
            }
        };
        let bytes = g.resident_weights(tensors.values().map(|(_,t)| t)).map_err(VisionError::Unsupported)?;
        let mut reads = Vec::new();
        let mut offsets = BTreeMap::new();
        let mut cursor = 0usize;
        for (name,_) in g.tensor_shapes() {
            let (path,metadata) = &tensors[&name];
            cursor = cursor.div_ceil(256)*256;
            offsets.insert(name,cursor as u64);
            reads.push(TensorRead { path:path.clone(),metadata:metadata.clone(),destination:cursor,vector:false });
            cursor += metadata.byte_length as usize;
        }
        if cursor.div_ceil(256)*256 != bytes as usize {
            return Err(VisionError::Unsupported("PE planner/packing arena mismatch".into()));
        }
        let inkling = g.kind == PerceptionKind::Inkling2;
        let prefix = if inkling { "model.visual.backbone" } else { "vision_model" };
        let get = |name: &str| offsets[name];
        let mut pe = PerceptionSpec { activation:u32::from(!inkling), position_grid:g.position_grid as u32,
            class_token:u32::from(inkling), output_kind:u32::from(!inkling),
            ln_pre:get(&format!("{prefix}.ln_pre.weight")), ln_pre_bias:get(&format!("{prefix}.ln_pre.bias")),
            class_embedding:NO_VISION_OFFSET,class_position:NO_VISION_OFFSET,
            downsample1:NO_VISION_OFFSET,downsample1_bias:NO_VISION_OFFSET,
            downsample2:NO_VISION_OFFSET,downsample2_bias:NO_VISION_OFFSET,
            scale1:vec![NO_VISION_OFFSET;g.depth],scale2:vec![NO_VISION_OFFSET;g.depth] };
        let mut native = VisionSpec { abi_version:3, reserved:if inkling {6} else {7},
            max_tokens:max_tokens as u32, output_width:g.output as u32, weight_bytes:bytes,
            hidden:g.hidden as u32,heads:g.heads as u32,kv_heads:g.heads as u32,
            head_dim:(g.hidden/g.heads) as u32,intermediate:g.intermediate as u32,
            patch_size:g.patch as u32,merger_width:(4*g.hidden) as u32,norm_eps:1e-5,
            inv_freq:NO_VISION_OFFSET,patch_bias:NO_VISION_OFFSET,
            merger_norm:NO_VISION_OFFSET,merger_norm_bias:NO_VISION_OFFSET,
            merger_fc1:NO_VISION_OFFSET,merger_fc2:NO_VISION_OFFSET,
            merger_fc1_bias:NO_VISION_OFFSET,merger_fc2_bias:NO_VISION_OFFSET,
            merger_extra:[NO_VISION_OFFSET;8],..Default::default() }.with_depth(g.depth);
        if inkling {
            native.patch = get("model.visual.backbone.embeddings.patch_embedding.weight");
            native.pos_embed = get("model.visual.backbone.embeddings.position_embedding");
            pe.class_embedding = get("model.visual.backbone.embeddings.class_embedding");
            pe.class_position = get("model.visual.backbone.embeddings.cls_position_embedding");
            native.merger_fc1 = get("model.visual.proj_in.weight");
            native.merger_fc2 = get("model.visual.proj_out.weight");
        } else {
            native.patch = get("vision_model.conv1.weight");
            native.pos_embed = get("vision_model.positional_embedding");
            pe.downsample1 = get("vision_model.vit_downsampler1.weight");
            pe.downsample1_bias = get("vision_model.vit_downsampler1.bias");
            pe.downsample2 = get("vision_model.vit_downsampler2.weight");
            pe.downsample2_bias = get("vision_model.vit_downsampler2.bias");
            native.merger_fc2 = get("vit_large_projector.weight");
        }
        for layer in 0..g.depth {
            let stem = if inkling { format!("{prefix}.layers.{layer}") }
                else { format!("{prefix}.transformer.resblocks.{layer}") };
            let field = |suffix: &str| get(&format!("{stem}.{suffix}"));
            let attention = if inkling {"self_attn"} else {"attn"};
            let (qkv,qkv_bias) = if inkling {
                let q = field("self_attn.q_proj.weight");
                let bias = field("self_attn.q_proj.bias");
                for (part,name) in [(1,"k"),(2,"v")] {
                    if field(&format!("self_attn.{name}_proj.weight")) != q+(part*g.hidden*g.hidden*2) as u64
                        || field(&format!("self_attn.{name}_proj.bias")) != bias+(part*g.hidden*2) as u64 {
                        return Err(VisionError::Unsupported("PE QKV concatenation extent mismatch".into()));
                    }
                }
                (q,bias)
            } else { (field("attn.in_proj_weight"),field("attn.in_proj_bias")) };
            native.blocks[layer] = VisionBlock {qkv,qkv_bias,
                proj:field(&format!("{attention}.out_proj.weight")),proj_bias:field(&format!("{attention}.out_proj.bias")),
                gate_up:field("mlp.c_fc.weight"),gate_up_bias:field("mlp.c_fc.bias"),
                down:field("mlp.c_proj.weight"),down_bias:field("mlp.c_proj.bias"),
                norm1:field("ln_1.weight"),norm2:field("ln_2.weight"),key0_bias:NO_VISION_OFFSET,window:0,column_order:0};
            native.norm1_bias[layer] = field("ln_1.bias");
            native.norm2_bias[layer] = field("ln_2.bias");
            native.q_norm[layer] = NO_VISION_OFFSET;
            native.k_norm[layer] = NO_VISION_OFFSET;
            pe.scale1[layer] = field("ls_1.gamma");
            pe.scale2[layer] = field("ls_2.gamma");
        }
        native.perception = Some(pe);
        Ok(Self {native,reads})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_pe_execution_is_not_advertised_before_oracles() {
        let source = include_str!("../vision.rs");
        let dispatch = source.split("pub fn from_snapshot").nth(1).unwrap().split("pub fn row_markers").next().unwrap();
        assert!(!dispatch.contains("inkling2"));
        assert!(!dispatch.contains("step5_vl"));
    }
}
