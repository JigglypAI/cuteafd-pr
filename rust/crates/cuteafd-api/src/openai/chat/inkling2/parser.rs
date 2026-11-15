//! Inkling's tokenizer response_template: repeated native channels, joined without trimming.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;
use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::{ArgStream, held_prefix};
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

#[derive(Debug, Default)]
struct Arguments {
    start: usize,
    scanned: usize,
    depth: usize,
    in_string: bool,
    escape: bool,
    closed: bool,
}

impl Arguments {
    fn advance(&mut self, call: &str) -> usize {
        if self.closed { return self.scanned; }
        for byte in call.as_bytes()[self.scanned..].iter().copied() {
            self.scanned += 1;
            if self.in_string {
                if self.escape { self.escape = false; }
                else if byte == b'\\' { self.escape = true; }
                else if byte == b'"' { self.in_string = false; }
            } else {
                match byte {
                    b'"' => self.in_string = true,
                    b'{' => self.depth += 1,
                    b'}' => {
                        self.depth = self.depth.saturating_sub(1);
                        if self.depth == 0 { self.closed = true; break; }
                    }
                    _ => {}
                }
            }
        }
        self.scanned
    }
}

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
    stream_args: ArgStream,
    arguments: Option<Arguments>,
    channels: usize,
}

impl InklingOutputParser {
    pub fn new(options: InklingParserOptions) -> Self { Self::with_stream_args(options, false) }

    pub fn with_stream_args(mut options: InklingParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|text| !text.is_empty());
        Self { options, mode: Mode::Boundary, pending: String::new(), recipient: String::new(),
            call: String::new(), calls: 0, stop: None, error: None,
            stream_args: ArgStream::new(stream_args), arguments: None, channels: 0 }
    }

    pub fn push(&mut self, text: &str) -> Result<Vec<OutputChunk>, InklingParseError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        if self.mode == Mode::Done { return Ok(Vec::new()); }
        self.pending.push_str(text);
        let result = self.drain(false);
        if let Err(error) = &result { self.error = Some(error.clone()); }
        result
    }

    /// Length/EOF may end ordinary text or reasoning mid-channel. Only an
    /// incomplete tool call is a protocol error; never publish it as valid JSON.
    pub fn finish(&mut self) -> Result<Vec<OutputChunk>, InklingParseError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        let result = self.drain(true).and_then(|mut outputs| {
            if self.mode == Mode::Call {
                if serde_json::from_str::<Value>(&self.call).is_err() {
                    return Err(InklingParseError::Truncated("tool call"));
                }
                self.complete_call(&mut outputs)?;
                self.mode = Mode::Boundary;
            }
            if self.channels == 0 {
                tracing::warn!("Inkling completion contained no recognized content channels");
            }
            if matches!(self.mode, Mode::Text | Mode::Thinking) {
                let tail = std::mem::take(&mut self.pending);
                self.emit(tail, &mut outputs);
            }
            self.pending.clear();
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
                // response_template ignores text outside recognized channels.
                let drop = marker_at.unwrap_or_else(|| self.pending.len() -
                    if final_chunk { 0 } else { held_prefix(&self.pending, "<|") });
                self.pending.drain(..drop);
                if drop == 0 { break; }
                continue;
            }
            let Some(end) = self.pending.find("|>") else { break };
            let marker = self.pending[..end + 2].to_owned();
            self.pending.drain(..end + 2);
            // Like the checkpoint response_template, native marker text inside a
            // JSON string is a known limit: decoded special markers are structural.
            match (self.mode, marker.as_str()) {
                (Mode::Boundary | Mode::Header, MODEL) => { self.mode = Mode::Header; self.recipient.clear(); }
                (Mode::Boundary | Mode::Header | Mode::Text | Mode::Thinking, EOS) => {
                    self.mode = Mode::Done;
                    self.stop = Some(GlmStop::Marker(EOS.into()));
                    self.pending.clear();
                }
                (Mode::Boundary | Mode::Header, THINK) => { self.mode = Mode::Thinking; self.channels += 1; }
                (Mode::Boundary | Mode::Header, TEXT | "<|content_tool_error|>" | "<|content_invoke_tool_text|>") => {
                    self.mode = Mode::Text; self.channels += 1;
                }
                (Mode::Boundary | Mode::Header, CALL) => {
                    self.mode = Mode::Call; self.call.clear(); self.arguments = None; self.channels += 1;
                }
                (Mode::Thinking | Mode::Text, END) => { self.mode = Mode::Boundary; self.recipient.clear(); }
                (Mode::Call, END) => {
                    self.complete_call(&mut out)?;
                    self.mode = Mode::Boundary;
                    self.recipient.clear();
                }
                (Mode::Boundary | Mode::Header, _) => { self.mode = Mode::Boundary; self.recipient.clear(); }
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
            Mode::Call => {
                self.call.push_str(&content);
                if !self.stream_args.enabled() { return; }
                if self.arguments.is_none() {
                    if let Some((name, start)) = arguments_prefix(&self.call) {
                        if self.valid_name(&name).is_err() { return; }
                        self.stream_args.begin(out);
                        self.stream_args.call(&name, out);
                        self.arguments = Some(Arguments { start, scanned: start, ..Arguments::default() });
                    }
                }
                if let Some(arguments) = self.arguments.as_mut() {
                    let start = arguments.scanned;
                    let end = arguments.advance(&self.call);
                    self.stream_args.delta(&self.call[start..end], out);
                }
            }
            _ => {}
        }
    }

    fn valid_name(&self, name: &str) -> Result<(), InklingParseError> {
        let invalid = |message: &str| InklingParseError::ToolCall(message.into());
        if name.is_empty() { return Err(invalid("name must be a nonempty string")); }
        if !self.recipient.is_empty() && self.recipient != name { return Err(invalid("header and JSON names disagree")); }
        if !self.options.tools.as_ref().is_some_and(|tools| tools.iter().any(|tool| tool.name == name)) {
            return Err(invalid("undeclared tool"));
        }
        Ok(())
    }

    fn complete_call(&mut self, out: &mut Vec<OutputChunk>) -> Result<(), InklingParseError> {
        let invalid = |message: &str| InklingParseError::ToolCall(message.into());
        let value: Value = serde_json::from_str(&self.call).map_err(|error| invalid(&error.to_string()))?;
        let name = value["name"].as_str().ok_or_else(|| invalid("name must be a string"))?;
        let args = value.get("args").filter(|value| value.is_object()).ok_or_else(|| invalid("args must be an object"))?;
        self.valid_name(name)?;
        if self.arguments.is_none() {
            // Noncanonical key order can put args before name; buffer that case.
            // Preserve raw argument spelling (whitespace and key order), so
            // buffered calls and concatenated streaming deltas are byte-identical.
            let arguments = if let Some((_, start)) = arguments_prefix(&self.call) {
                let mut scan = Arguments { start, scanned: start, ..Arguments::default() };
                let end = scan.advance(&self.call);
                self.call[scan.start..end].to_owned()
            } else { args.to_string() };
            out.push(OutputChunk::ToolCall { tool_name: name.to_owned(), arguments });
        }
        self.calls += 1;
        Ok(())
    }
}

/// Locate args after a complete name without parsing the unfinished argument
/// object. Once found, Arguments scans each arriving byte exactly once.
fn arguments_prefix(call: &str) -> Option<(String, usize)> {
    let mut parser = jiter::Jiter::new(call.as_bytes());
    let mut key = parser.next_object().ok()??.to_owned();
    let mut name = None;
    loop {
        if key == "name" { name = Some(parser.next_str().ok()?.to_owned()); }
        else if key == "args" {
            if parser.peek().ok()? != jiter::Peek::Object { return None; }
            return Some((name?, parser.current_index()));
        } else { parser.next_skip().ok()?; }
        key = parser.next_key().ok()??.to_owned();
    }
}

impl TextParser for InklingOutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> { InklingOutputParser::push(self, text).unwrap_or_default() }
    fn finish(&mut self) -> Vec<OutputChunk> { InklingOutputParser::finish(self).unwrap_or_default() }
    fn stop(&self) -> Option<&GlmStop> { self.stop.as_ref() }
    fn tool_calls(&self) -> usize { self.calls }
    fn error(&self) -> Option<String> { self.error.as_ref().map(ToString::to_string) }
}
