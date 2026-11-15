//! Inkling 2's checkpoint-native message channels and JSON tool calls.
pub mod parser;
pub mod prompt;

use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use super::glm5::template::ChatTemplate;
pub use parser::{InklingOutputParser, InklingParseError, InklingParserOptions};
pub use prompt::InklingPromptOptions;

pub const MODEL: &str = "<|message_model|>";
pub const THINK: &str = "<|content_thinking|>";
pub const TEXT: &str = "<|content_text|>";
pub const CALL: &str = "<|content_invoke_tool_json|>";
pub const END: &str = "<|end_message|>";
pub const EOS: &str = "<|content_model_end_sampling|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InklingTokenIds {
    pub eos: Vec<u32>,
    pub message_model: u32,
    pub thinking: u32,
    pub text: u32,
    pub invoke_tool_json: u32,
    pub end_message: u32,
}

#[derive(Debug)]
pub struct InklingEncoding {
    template: ChatTemplate,
    tokens: InklingTokenIds,
}

impl InklingEncoding {
    pub fn new(source: impl Into<String>, tokens: InklingTokenIds) -> Result<Self> {
        anyhow::ensure!(!tokens.eos.is_empty(), "Inkling EOS token list must not be empty");
        Ok(Self { template: ChatTemplate::new(source).context("compile Inkling chat template")?, tokens })
    }

    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(snapshot.join("tokenizer_config.json"))?)?;
        let id = |marker: &str| -> Result<u32> {
            let entries = config["added_tokens_decoder"].as_object().context("Inkling added_tokens_decoder missing")?;
            let mut matches = entries.iter().filter(|(_, value)| value["content"] == marker);
            let (key, _) = matches.next().with_context(|| format!("Inkling tokenizer has no {marker}"))?;
            anyhow::ensure!(matches.next().is_none(), "duplicate Inkling marker {marker}");
            key.parse().with_context(|| format!("invalid Inkling token ID for {marker}"))
        };
        Self::new(std::fs::read_to_string(snapshot.join("chat_template.jinja"))?, InklingTokenIds {
            eos: vec![id(EOS)?], message_model: id(MODEL)?, thinking: id(THINK)?, text: id(TEXT)?,
            invoke_tool_json: id(CALL)?, end_message: id(END)?,
        })
    }

    pub fn tokens(&self) -> &InklingTokenIds { &self.tokens }

    pub fn stop_token_ids(&self, _tools_declared: bool) -> Vec<u32> {
        // end_message closes a channel, not the assistant turn (including parallel calls).
        self.tokens.eos.clone()
    }

    pub fn render(&self, body: &Value, options: &InklingPromptOptions) -> Result<String, String> {
        self.template.render(&prompt::template_context(body, options)?)
            .map_err(|error| format!("Inkling chat template: {error:#}"))
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    pub const TEMPLATE: &str = include_str!("inkling2/fixtures/inkling2.jinja");
    pub fn encoding() -> super::InklingEncoding {
        super::InklingEncoding::new(TEMPLATE, super::InklingTokenIds { eos: vec![200006],
            message_model: 200001, thinking: 200008, text: 200004,
            invoke_tool_json: 200049, end_message: 200010 }).unwrap()
    }
}

#[cfg(test)]
mod tests;
