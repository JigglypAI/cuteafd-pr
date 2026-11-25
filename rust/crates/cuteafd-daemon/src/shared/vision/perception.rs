//! Header-only PE packing. Runtime dispatch stays gated until encoder oracles pass.
use super::*;
use cuteafd_ffi::vision::PerceptionSpec;
use cuteafd_loader::media::perception_vision::{PerceptionGeometry, PerceptionKind};

impl TowerSpec {
    pub fn perception(snapshot: &Path, max_tokens: usize) -> Result<Self> {
        let cfg: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let g = PerceptionGeometry::from_config(&cfg).map_err(VisionError::Unsupported)?;
        if !(1..=4096).contains(&max_tokens) {
            return Err(VisionError::Unsupported(
                "PE image capacity must be 1..=4096".into(),
            ));
        }
        let tensors = match g.kind {
            PerceptionKind::Inkling2 => tower_catalog(snapshot, "model.visual.")?,
            PerceptionKind::Step5 => {
                let mut tensors = tower_catalog(snapshot, "vision_model.")?;
                tensors.extend(tower_catalog(snapshot, "vit_large_projector.")?);
                tensors
            }
        };
        let bytes = g
            .resident_weights(tensors.values().map(|(_, t)| t))
            .map_err(VisionError::Unsupported)?;
        let mut reads = Vec::new();
        let mut offsets = BTreeMap::new();
        let mut cursor = 0usize;
        for (name, _) in g.tensor_shapes() {
            let (path, metadata) = &tensors[&name];
            cursor = cursor.div_ceil(256) * 256;
            offsets.insert(name, cursor as u64);
            reads.push(TensorRead {
                path: path.clone(),
                metadata: metadata.clone(),
                destination: cursor,
                transform: ReadTransform::Direct,
            });
            cursor += metadata.byte_length as usize;
        }
        if cursor.div_ceil(256) * 256 + g.inv_freq().len() * 4 != bytes as usize {
            return Err(VisionError::Unsupported(
                "PE planner/packing arena mismatch".into(),
            ));
        }
        let inkling = g.kind == PerceptionKind::Inkling2;
        let prefix = if inkling {
            "model.visual.backbone"
        } else {
            "vision_model"
        };
        let get = |name: &str| offsets[name];
        let mut pe = PerceptionSpec {
            activation: u32::from(!inkling),
            position_grid: g.position_grid as u32,
            class_token: u32::from(inkling),
            output_kind: u32::from(!inkling),
            ln_pre: get(&format!("{prefix}.ln_pre.weight")),
            ln_pre_bias: get(&format!("{prefix}.ln_pre.bias")),
            class_embedding: NO_VISION_OFFSET,
            class_position: NO_VISION_OFFSET,
            downsample1: NO_VISION_OFFSET,
            downsample1_bias: NO_VISION_OFFSET,
            downsample2: NO_VISION_OFFSET,
            downsample2_bias: NO_VISION_OFFSET,
            scale1: vec![NO_VISION_OFFSET; g.depth],
            scale2: vec![NO_VISION_OFFSET; g.depth],
        };
        let mut native = VisionSpec {
            abi_version: 3,
            reserved: if inkling { 6 } else { 7 },
            max_tokens: max_tokens as u32,
            output_width: g.output as u32,
            weight_bytes: bytes,
            hidden: g.hidden as u32,
            heads: g.heads as u32,
            kv_heads: g.heads as u32,
            head_dim: (g.hidden / g.heads) as u32,
            intermediate: g.intermediate as u32,
            patch_size: g.patch as u32,
            merger_width: (4 * g.hidden) as u32,
            norm_eps: 1e-5,
            inv_freq: (cursor.div_ceil(256) * 256) as u64,
            patch_bias: NO_VISION_OFFSET,
            merger_norm: NO_VISION_OFFSET,
            merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1: NO_VISION_OFFSET,
            merger_fc2: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET,
            merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra: [NO_VISION_OFFSET; 8],
            ..Default::default()
        }
        .with_depth(g.depth);
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
            let stem = if inkling {
                format!("{prefix}.layers.{layer}")
            } else {
                format!("{prefix}.transformer.resblocks.{layer}")
            };
            let field = |suffix: &str| get(&format!("{stem}.{suffix}"));
            let attention = if inkling { "self_attn" } else { "attn" };
            let (qkv, qkv_bias) = if inkling {
                let q = field("self_attn.q_proj.weight");
                let bias = field("self_attn.q_proj.bias");
                for (part, name) in [(1, "k"), (2, "v")] {
                    if field(&format!("self_attn.{name}_proj.weight"))
                        != q + (part * g.hidden * g.hidden * 2) as u64
                        || field(&format!("self_attn.{name}_proj.bias"))
                            != bias + (part * g.hidden * 2) as u64
                    {
                        return Err(VisionError::Unsupported(
                            "PE QKV concatenation extent mismatch".into(),
                        ));
                    }
                }
                (q, bias)
            } else {
                (field("attn.in_proj_weight"), field("attn.in_proj_bias"))
            };
            native.blocks[layer] = VisionBlock {
                qkv,
                qkv_bias,
                proj: field(&format!("{attention}.out_proj.weight")),
                proj_bias: field(&format!("{attention}.out_proj.bias")),
                gate_up: field("mlp.c_fc.weight"),
                gate_up_bias: field("mlp.c_fc.bias"),
                down: field("mlp.c_proj.weight"),
                down_bias: field("mlp.c_proj.bias"),
                norm1: field("ln_1.weight"),
                norm2: field("ln_2.weight"),
                key0_bias: NO_VISION_OFFSET,
                window: 0,
                column_order: 0,
            };
            native.norm1_bias[layer] = field("ln_1.bias");
            native.norm2_bias[layer] = field("ln_2.bias");
            native.q_norm[layer] = NO_VISION_OFFSET;
            native.k_norm[layer] = NO_VISION_OFFSET;
            pe.scale1[layer] = field("ls_1.gamma");
            pe.scale2[layer] = field("ls_2.gamma");
        }
        native.perception = Some(pe);
        let constant = ConstantRegion::fp32(
            "rotary.inv_freq",
            native.inv_freq,
            g.inv_freq(),
            "PerceptionKind::inv_freq vendor-exact per-family recipe",
            if inkling {
                "vllm@d664cd1e96052857aea2aa844eeba571c24758b4/vision.py sha256=a3ed563be68b63ee47c4103116bb4f77015eaaea31b040d9301b82af85349d16"
            } else {
                "vllm@d664cd1e96052857aea2aa844eeba571c24758b4/step_vl.py sha256=348a7e4d5260d68232d91fe13a3b1a9a6badca4799457f1b7c8a1a8a112adcb2"
            },
        );
        Self::from_parts_with_constants(native, reads, vec![constant])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sparse_snapshot(inkling: bool) -> tempfile::TempDir {
        use std::io::Write;
        let directory = tempfile::tempdir().unwrap();
        let cfg = if inkling {
            serde_json::json!({"model_type":"inkling2","text_config":{"hidden_size":4608},"vision_config":{
            "hidden_size":1024,"num_hidden_layers":23,"num_attention_heads":16,"native_patch_size":20,
            "temporal_patch_size":2,"adapter_merge_size":2,"projection_hidden_dim":4096,"posemb_grid_size":32,
            "mlp_ratio":4.0,"projector_bias":false,"layer_norm_eps":1e-5}})
        } else {
            serde_json::json!({"model_type":"step5_vl","text_config":{"hidden_size":4096},
            "image_token_len":169,"image_token_id":128001,"vision_config":{"model_type":"perception_encoder",
            "width":1536,"layers":47,"heads":16,"image_size":728,"patch_size":14,"pool_type":"none",
            "use_cls_token":false,"use_ln_post":false,"hidden_act":"quick_gelu"}})
        };
        let geometry = PerceptionGeometry::from_config(&cfg).unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&cfg).unwrap(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in geometry.tensor_shapes() {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                name,
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[end,end+bytes]}),
            );
            end += bytes;
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(directory.path().join("model.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + end).unwrap();
        directory
    }
    #[test]
    fn header_only_builders_pack_every_native_tensor_and_scale() {
        for inkling in [true, false] {
            let directory = sparse_snapshot(inkling);
            let tower = TowerSpec::perception(directory.path(), 169).unwrap();
            super::super::constants::assert_original_table(&tower);
            let pe = tower.native.perception.as_ref().unwrap();
            assert_eq!(tower.reads.len(), if inkling { 422 } else { 667 });
            assert_eq!(tower.native.depth, if inkling { 23 } else { 47 });
            assert_eq!(pe.class_token, u32::from(inkling));
            assert_eq!(pe.activation, u32::from(!inkling));
            assert_eq!(pe.scale1.len(), tower.native.depth as usize);
            assert_eq!(pe.scale2.len(), tower.native.depth as usize);
            assert!(pe
                .scale1
                .iter()
                .all(|offset| offset % 256 == 0 && *offset < tower.native.weight_bytes));
            assert!(tower
                .reads
                .iter()
                .all(|read| read.transform==ReadTransform::Direct && read.destination % 256 == 0));
            assert_eq!(
                tower.constants[0].bytes.len(),
                if inkling { 64 } else { 96 }
            );
            assert_eq!(
                tower.native.inv_freq + tower.constants[0].bytes.len() as u64,
                tower.native.weight_bytes
            );
            let contribution =
                push::Contribution::prepare(0, Some((&tower, 0x100000)), None, None, None).unwrap();
            let records = contribution.records();
            assert_eq!(records.operands.len(), tower.reads.len());
            assert!(records.steps.is_empty());
            assert_eq!(
                records
                    .operands
                    .iter()
                    .map(|operand| operand.bytes as u64)
                    .sum::<u64>()
                    + tower.constants[0].bytes.len() as u64,
                tower.native.weight_bytes
            );
            assert!(TowerSpec::perception(directory.path(), 0).is_err());
            assert!(TowerSpec::perception(directory.path(), 4097).is_err());
        }
    }
    #[test]
    #[ignore = "requires VISION_LIBRARY built with PE AOT; no CUDA initialization"]
    fn native_ledger_matches_perception_scratch_contract() {
        let library = std::path::PathBuf::from(std::env::var("VISION_LIBRARY").unwrap());
        for inkling in [true, false] {
            let directory = sparse_snapshot(inkling);
            let cfg =
                cuteafd_loader::read_json_file(&directory.path().join("config.json")).unwrap();
            let geometry = PerceptionGeometry::from_config(&cfg).unwrap();
            let capacities: &[usize] = if inkling {
                &[1, 20, 169, 4096]
            } else {
                &[81, 168, 169, 4096]
            };
            for &capacity in capacities {
                let tower = TowerSpec::perception(directory.path(), capacity).unwrap();
                let ledger =
                    cuteafd_ffi::vision::NativeVision::required(&library, &tower.native).unwrap();
                assert_eq!(ledger.weights, tower.native.weight_bytes);
                assert_eq!(ledger.blas_workspace, 0);
                assert_eq!(
                    ledger.scratch,
                    geometry.scratch_bytes(capacity as u64).unwrap()
                );
            }
        }
    }
    #[test]
    fn grouped_pe_requests_expand_features_and_native_markers_once() -> anyhow::Result<()> {
        use cuteafd_loader::media::{DecodedPixels, EncoderId, ImageFamily, ProcessorConfig, RgbImage};
        fn request(groups: Vec<cuteafd_loader::media::PreparedImageGroup>) -> cuteafd_api::openai::NativeRequest {
            let (events, _) = tokio::sync::mpsc::unbounded_channel();
            cuteafd_api::openai::NativeRequest { prompt: String::new(), literal_spans: vec![], constraint: None,
                images: vec![], media: groups.iter().flat_map(|g| g.images.iter().cloned()).collect(), image_groups: groups,
                audio: vec![], max_tokens: 1, sampling: Default::default(), stop_token_ids: vec![], events,
                probe: None, usage: None, prefix_cache_source: Default::default(), finish_detail: Default::default() }
        }
        for (inkling, family, size, hidden, features, prompt_rows) in [
            (true, ImageFamily::Inkling2, [197,157], 4608, 20, 20),
            (false, ImageFamily::Step5, [896,812], 4096, 493, 504),
        ] {
            let root = sparse_snapshot(inkling);
            cuteafd_loader::plan::testing::write_kimi_tokenizer(root.path());
            let mut config: serde_json::Value = cuteafd_loader::read_json_file(&root.path().join("tokenizer_config.json"))?;
            for (id, token) in [(258,"<|unused_200054|>"),(259,"<im_patch>"),(260,"<patch_start>"),
                (261,"<patch_end>"),(262,"<patch_newline>"),(263,"<im_start>"),(264,"<im_end>")] {
                config["added_tokens_decoder"][id.to_string()] = serde_json::json!({"content":token});
            }
            std::fs::write(root.path().join("tokenizer_config.json"), serde_json::to_vec(&config)?)?;
            let processor = ProcessorConfig::from_snapshot(root.path(), family)?;
            let group = processor.prepare_group(DecodedPixels::Rgb(RgbImage { width:size[0], height:size[1],
                data: vec![71; size[0] as usize*size[1] as usize*3] }), EncoderId([3;32]))?;
            let binding = Binding::new(root.path(), family, true, 512)?;
            let placeholder = if inkling {258} else {259};
            let (prompt, media, jobs) = binding.prepare(request(vec![group.clone()]), vec![1,placeholder,2], 512, hidden, 1024)?;
            assert_eq!(prompt.tokens.len(), prompt_rows+2);
            assert_eq!(media.spans().iter().map(|s| s.len).sum::<usize>(), features);
            assert_eq!(jobs.iter().map(|j| j.tokens).sum::<usize>(), features);
            assert_eq!(jobs.len(), if inkling {1} else {5});
            assert_eq!(prompt.job.media.len(), jobs.len());
            if !inkling {
                assert_eq!(media.spans().iter().map(|s|s.len).collect::<Vec<_>>(), [81,81,81,81,169]);
                assert_eq!(prompt.tokens[1],260);
                assert_eq!(prompt.tokens[prompt.tokens.len()-2],264);
            }
            let off = Binding::new(root.path(), family, false, 512)?;
            assert!(off.prepare(request(vec![group.clone()]), vec![1,placeholder,2], 512, hidden, 1024).is_err());
            let text = vec![4,5,6];
            let (plain, empty, jobs) = off.prepare(request(vec![]), text.clone(), 512, hidden, 1024)?;
            assert_eq!(plain.tokens,text);
            assert!(empty.spans().is_empty() && jobs.is_empty());
            assert!(binding.prepare(request(vec![group.clone()]), vec![1,placeholder,2], 512, hidden, prompt_rows).is_err());
            assert!(binding.prepare(request(vec![group.clone()]), vec![1,2], 512, hidden, 1024).is_err());
            let mut malformed = group;
            malformed.source[0] += 1;
            assert!(binding.prepare(request(vec![malformed]), vec![1,placeholder,2], 512, hidden, 1024).is_err());
        }
        Ok(())
    }

    #[test]
    fn qualified_pe_towers_have_explicit_shared_dispatch() {
        let source=include_str!("../vision.rs");
        assert!(source.contains("Some(\"inkling2\"|\"inkling2_loop\"|\"step5_vl\") => Self::perception"));
        for (inkling,family) in [(true,cuteafd_loader::media::ImageFamily::Inkling2),(false,cuteafd_loader::media::ImageFamily::Step5)] {
            let directory=sparse_snapshot(inkling);
            let tower=TowerSpec::from_snapshot(directory.path(),169).unwrap();
            assert_eq!(tower.image_family(),family);
        }
        // Also verify Inkling 2 Mini (inkling2_loop, output_width=2304) dispatches through Self::perception.
        let mini_dir = tempfile::tempdir().unwrap();
        let mini_cfg = serde_json::json!({"model_type":"inkling2_loop","text_config":{"hidden_size":2304},"vision_config":{
            "hidden_size":1024,"num_hidden_layers":23,"num_attention_heads":16,"native_patch_size":20,
            "temporal_patch_size":2,"adapter_merge_size":2,"projection_hidden_dim":4096,"posemb_grid_size":32,
            "mlp_ratio":4.0,"projector_bias":false,"layer_norm_eps":1e-5}});
        let mini_g = PerceptionGeometry::from_config(&mini_cfg).unwrap();
        std::fs::write(mini_dir.path().join("config.json"), serde_json::to_vec(&mini_cfg).unwrap()).unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in mini_g.tensor_shapes() {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(name, serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[end,end+bytes]}));
            end += bytes;
        }
        let hbytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(mini_dir.path().join("model.safetensors")).unwrap();
        use std::io::Write;
        file.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&hbytes).unwrap();
        file.set_len(8 + hbytes.len() as u64 + end).unwrap();
        let mini_tower = TowerSpec::from_snapshot(mini_dir.path(), 169).unwrap();
        assert_eq!(mini_tower.image_family(), cuteafd_loader::media::ImageFamily::Inkling2);
        assert_eq!(mini_tower.native.output_width, 2304);
    }
}

/// Checkpoint-native PE prompt markers. Crop construction stays in the shared
/// processor; this adapter supplies only the family's tokenizer layout.
pub(crate) struct Binding {
    processor: cuteafd_loader::media::ProcessorConfig,
    layout: cuteafd_loader::media::ImagePromptLayout,
    enabled: bool,
}
impl Binding {
    pub fn new(snapshot: &Path, family: cuteafd_loader::media::ImageFamily, enabled: bool,
        vocabulary: u32) -> anyhow::Result<Self> {
        use cuteafd_loader::media::{ImageFamily, ImagePromptLayout, ProcessorConfig};
        let tokenizer = cuteafd_loader::cached_tokenizer(snapshot)?;
        let marker = |name: &str| -> anyhow::Result<u32> {
            let ids = tokenizer.encode_ids(name)?;
            anyhow::ensure!(ids.len() == 1 && ids[0] < vocabulary, "PE marker {name} is not one native token");
            Ok(ids[0])
        };
        let layout = match family {
            ImageFamily::Inkling2 => ImagePromptLayout::Single { placeholder: marker("<|unused_200054|>")? },
            ImageFamily::Step5 => ImagePromptLayout::LocalGlobal { placeholder: marker("<im_patch>")?,
                local_start: marker("<patch_start>")?, local_end: marker("<patch_end>")?,
                newline: marker("<patch_newline>")?, global_start: marker("<im_start>")?, global_end: marker("<im_end>")? },
            _ => anyhow::bail!("not a PE prompt family"),
        };
        Ok(Self { processor: ProcessorConfig::from_snapshot(snapshot, family)?, layout, enabled })
    }
    pub fn prepare(&self, mut job: cuteafd_api::openai::NativeRequest, tokens: Vec<u32>, vocabulary: u32,
        hidden: usize, context: usize) -> anyhow::Result<(crate::shared::qwen::scheduler::Prompt,
            cuteafd_engine::media::RequestMedia, Vec<cuteafd_engine::media::EncodeJob>)> {
        anyhow::ensure!(self.enabled || (job.media.is_empty() && job.image_groups.is_empty()), "vision is disabled");
        let mut groups = std::mem::take(&mut job.image_groups);
        anyhow::ensure!(job.media.is_empty() || !groups.is_empty(), "PE images require grouped preparation");
        for group in &mut groups { self.processor.assemble_group_prompt(group, &self.layout)?; }
        super::still::Binding::prepare(job, tokens, vocabulary, hidden, context, self.layout.placeholder(), groups)
    }
}
