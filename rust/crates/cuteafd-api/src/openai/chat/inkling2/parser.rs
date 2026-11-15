//! Inkling's tokenizer response_template: repeated native channels, joined without trimming.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::held_prefix;
use super::{CALL, END, EOS, MODEL, TEXT, THINK};

#[derive(Debug, Clone, Default)]
pub struct InklingParserOptions {
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InklingParseError {
    #[error("unexpected Inkling output: {0}")]
    Unexpected(String),
    #[error("truncated Inkling {0}")]
    Truncated(&'static str),
    #[error("invalid Inkling tool call: {0}")]
    ToolCall(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode { Boundary, Header, Thinking, Text, Call, Done }

#[derive(Debug)]
pub struct InklingOutputParser {
    options: InklingParserOptions,
    mode: Mode,
    pending: String,
    recipient: String,
    call: String,
    calls: usize,
    stop: Option<GlmStop>,
    error: Option<InklingParseError>,
}

impl InklingOutputParser {
    pub fn new(mut options: InklingParserOptions) -> Self {
        options.stop_sequences.retain(|text| !text.is_empty());
        Self { options, mode: Mode::Boundary, pending: String::new(), recipient: String::new(),
            call: String::new(), calls: 0, stop: None, error: None }
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<OutputChunk>, InklingParseError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        if self.mode == Mode::Done { return Ok(Vec::new()); }
        self.pending.push_str(text);
        let result = self.drain(false);
        if let Err(error) = &result { self.error = Some(error.clone()); }
        result
    }

    /// EOS is often removed by the backend; a complete channel boundary is sufficient.
    pub fn finish(&mut self) -> Result<Vec<OutputChunk>, InklingParseError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        let result = self.drain(true).and_then(|outputs| {
            if self.mode != Mode::Done && (self.mode != Mode::Boundary || !self.pending.is_empty()) {
                return Err(InklingParseError::Truncated(match self.mode {
                    Mode::Call => "tool call", Mode::Thinking | Mode::Text => "content channel",
                    _ => "message header",
                }));
            }
            Ok(outputs)
        });
        if let Err(error) = &result { self.error = Some(error.clone()); }
        result
    }

    pub fn error(&self) -> Option<&InklingParseError> { self.error.as_ref() }
    pub fn tool_calls(&self) -> usize { self.calls }
    pub fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }

    fn drain(&mut self, final_chunk: bool) -> Result<Vec<OutputChunk>, InklingParseError> {
        let mut out = Vec::new();
        while self.mode != Mode::Done && !self.pending.is_empty() {
            let marker_at = self.pending.find("<|");
            if matches!(self.mode, Mode::Text | Mode::Thinking | Mode::Call) {
                let stop = if self.mode == Mode::Text {
                    self.options.stop_sequences.iter().filter_map(|s| self.pending.find(s).map(|i| (i, s.clone())))
                        .min_by_key(|(index, _)| *index)
                } else { None };
                if let Some((index, sequence)) = stop.filter(|(index, _)| marker_at.is_none_or(|m| *index < m)) {
                    let content = self.pending[..index].to_owned();
                    self.emit(content, &mut out);
                    self.stop = Some(GlmStop::Sequence(sequence));
                    self.mode = Mode::Done;
                    self.pending.clear();
                    break;
                }
                let safe = marker_at.unwrap_or_else(|| {
                    let held = if final_chunk { 0 } else {
                        let native = held_prefix(&self.pending, "<|");
                        self.options.stop_sequences.iter().filter(|_| self.mode == Mode::Text)
                            .map(|s| held_prefix(&self.pending, s)).max().unwrap_or(0).max(native)
                    };
                    self.pending.len() - held
                });
                if safe > 0 {
                    let content: String = self.pending.drain(..safe).collect();
                    self.emit(content, &mut out);
                    continue;
                }
                if marker_at.is_none() { break; }
            } else if self.mode == Mode::Header {
                let Some(index) = marker_at else { break };
                self.recipient.push_str(&self.pending[..index]);
                self.pending.drain(..index);
            } else if marker_at != Some(0) {
                if !final_chunk && "<|".starts_with(&self.pending) { break; }
                return Err(InklingParseError::Unexpected("expected a model/channel marker".into()));
            }
            let Some(end) = self.pending.find("|>") else { break };
            let marker = self.pending[..end + 2].to_owned();
            self.pending.drain(..end + 2);
            match (self.mode, marker.as_str()) {
                (Mode::Boundary, MODEL) => { self.mode = Mode::Header; self.recipient.clear(); }
                (Mode::Boundary, EOS) => {
                    self.mode = Mode::Done;
                    self.stop = Some(GlmStop::Marker(EOS.into()));
                    self.pending.clear();
                }
                (Mode::Boundary | Mode::Header, THINK) => { self.mode = Mode::Thinking; }
                (Mode::Boundary | Mode::Header, TEXT | "<|content_tool_error|>" | "<|content_invoke_tool_text|>") => {
                    self.mode = Mode::Text;
                }
                (Mode::Boundary | Mode::Header, CALL) => { self.mode = Mode::Call; self.call.clear(); }
                (Mode::Thinking | Mode::Text, END) => { self.mode = Mode::Boundary; self.recipient.clear(); }
                (Mode::Call, END) => {
                    out.push(self.complete_call()?);
                    self.mode = Mode::Boundary;
                    self.recipient.clear();
                }
                _ => return Err(InklingParseError::Unexpected(marker)),
            }
        }
        Ok(out)
    }

    fn emit(&mut self, content: String, out: &mut Vec<OutputChunk>) {
        if content.is_empty() { return; }
        match self.mode {
            Mode::Text => out.push(OutputChunk::Raw { content }),
            Mode::Thinking => out.push(OutputChunk::Reasoning { content }),
            Mode::Call => self.call.push_str(&content),
            _ => {}
        }
    }

    fn complete_call(&mut self) -> Result<OutputChunk, InklingParseError> {
        let invalid = |message: &str| InklingParseError::ToolCall(message.into());
        let value: Value = serde_json::from_str(&self.call).map_err(|error| invalid(&error.to_string()))?;
        let name = value["name"].as_str().filter(|name| !name.is_empty()).ok_or_else(|| invalid("name must be a nonempty string"))?;
        let args = value.get("args").filter(|value| value.is_object()).ok_or_else(|| invalid("args must be an object"))?;
        if !self.recipient.is_empty() && self.recipient != name { return Err(invalid("header and JSON names disagree")); }
        if !self.options.tools.as_ref().is_some_and(|tools| tools.iter().any(|tool| tool.name == name)) {
            return Err(invalid("undeclared tool"));
        }
        self.calls += 1;
        Ok(OutputChunk::ToolCall { tool_name: name.to_owned(), arguments: args.to_string() })
    }
}

impl TextParser for InklingOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { InklingOutputParser::push(self, text).unwrap_or_default() }
    fn finish(&mut self) -> Vec<OutputChunk> { InklingOutputParser::finish(self).unwrap_or_default() }
    fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    fn tool_calls(&self) -> usize { self.calls }
    fn error(&self) -> Option<String> { self.error.as_ref().map(ToString::to_string) }
}
