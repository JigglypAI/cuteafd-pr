//! Speculators dSpark block drafters (`DSparkDraftModel`, e.g.
//! RedHatAI/GLM-5.3-Flash-speculator.dspark-preview): Qwen3 sliding-window
//! layers over taps of a target's hidden states, a Markov logit-bias head and
//! a confidence head. Not a standalone model: the plan describes it so a
//! target engine can attach it as its speculator (serve-glmf --draft).
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::Checkpoint;
use crate::plan::family::{ConfigError, Described, Family, FamilyModel, Hint, RuntimeStatus};
use crate::plan::spec::*;

pub struct Dspark;
pub static DSPARK: Dspark = Dspark;

impl Family for Dspark {
    fn id(&self) -> &'static str {
        "dspark"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Serving
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.config.get("speculators_model_type").and_then(Value::as_str) == Some("dspark")
            || checkpoint.architectures().iter().any(|arch| arch == "DSparkDraftModel" || arch == "Qwen3DSparkModel")
    }

    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        self.spec(checkpoint).map(|spec| Box::new(Described(spec)) as Box<dyn FamilyModel>)
            .map_err(ConfigError::from_anyhow)
    }

    fn classify(&self, _spec: &ModelSpec, _name: &str) -> Option<TensorRole> {
        // Every tensor belongs to the drafter; it runs on the coordinator.
        Some(TensorRole::new(Component::Speculator))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        (component == Component::Speculator).then(|| Hint {
            what: "dSpark block drafter attached to a target engine".into(),
            how: "GLM 5.3 Flash drafts with it (serve-glmf --draft SNAP, SPECULATOR=dspark; \
                  cuteafd-daemon families/glm5_flash/dspark.rs, native/families/glm5_flash/cuda/glmf_dspark.cu). \
                  The kernels take 64-wide heads, a vanilla rank-256 Markov head, sample_from_anchor and a \
                  Markov confidence head; other targets need their taps wired into the drafter."
                .into(),
        })
    }
}

impl Dspark {
    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let c = crate::formats::block_dspark::BlockDsparkConfig::from_json(&checkpoint.config)?;
        let layer = LayerSpec { rope: None,
            attention: match c.window {
                Some(window)=>AttentionKind::SlidingGqa { heads:c.heads,kv_heads:c.kv_heads,head_dim:c.head_dim,window,sinks:false },
                None=>AttentionKind::Gqa { heads:c.heads,kv_heads:c.kv_heads,head_dim:c.head_dim },
            }, ffn:FfnKind::Dense { intermediate:c.intermediate } };
        let notes=vec![format!("block {} Markov rank {} target taps {:?}; native window {:?}",c.block,c.rank,c.taps,c.window)];
        Ok(ModelSpec {
            family: "dspark",
            architecture: checkpoint.architectures().first().cloned().unwrap_or_else(||"DSparkDraftModel".into()),
            hidden: c.hidden,
            vocab: c.vocab,
            layers: vec![layer; c.layers],
            moe: None,
            speculator: None,
            tables: Vec::new(),
            vision: false,
            notes,
        })
    }
}
