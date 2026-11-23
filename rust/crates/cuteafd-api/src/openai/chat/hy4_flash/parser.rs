//! Incremental Hy4-flash output parser: `<think:opensource>...</think:opensource>` reasoning,
//! visible content, and `<tool_calls:opensource><tool_call:opensource>name<arg_key:opensource>k</arg_key:opensource><arg_value:opensource>v</arg_value:opensource></tool_call:opensource></tool_calls:opensource>` calls.
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::Value;

use crate::openai::chat::glm5::{parser::GlmStop, TextParser};
use crate::openai::chat::stream_args::{held_prefix, json_string_contents, name_is_readable, ArgStream};

use super::{
    ARG_KEY_CLOSE, ARG_KEY_OPEN, ARG_VALUE_CLOSE, ARG_VALUE_OPEN, HY_END, HY_MIDDLE, HY_START,
    THINK_CLOSE, THINK_OPEN, TOOL_CALLS_CLOSE, TOOL_CALLS_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE, TOOL_RESPONSE_OPEN,
};

const STOP_MARKERS: [&str; 5] = [
    HY_END,
    HY_START,
    HY_MIDDLE,
    TOOL_RESPONSE_OPEN,
    TOOL_RESPONSE_CLOSE,
];

#[derive(Debug, Clone, Default)]
pub struct Hy4ParserOptions {
    /// True when the generation prompt already opened `<think:opensource>`.
    pub thinking: bool,
    pub tools: Option<Vec<ToolDefinition>>,
    pub stop_sequences: Vec<String>,
    pub id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    Content,
    CallsBlock,
    CallName,
    ArgKey,
    ArgValueStart,
    ArgValue,
    ArgKeyOrEnd,
    Done,
}

#[derive(Debug, Default)]
struct PendingCall {
    name: String,
    key: Option<String>,
    arguments: Vec<(String, String)>,
    raw_value: String,
    raw_text: String,
}

#[derive(Debug)]
pub struct Hy4OutputParser {
    options: Hy4ParserOptions,
    mode: Mode,
    pending: String,
    call: Option<PendingCall>,
    calls: usize,
    stop: Option<GlmStop>,
    stream_args: ArgStream,
}

impl Hy4OutputParser {
    pub fn new(options: Hy4ParserOptions) -> Self {
        Self::with_stream_args(options, false)
    }

    pub fn with_stream_args(mut options: Hy4ParserOptions, stream_args: bool) -> Self {
        options.stop_sequences.retain(|s| !s.is_empty());
        let mode = if options.thinking {
            Mode::Reasoning
        } else {
            Mode::Content
        };
        Self {
            options,
            mode,
            pending: String::new(),
            call: None,
            calls: 0,
            stop: None,
            stream_args: ArgStream::new(stream_args),
        }
    }

    fn tools_enabled(&self) -> bool {
        self.options.tools.is_some()
    }

    fn arg_is_string(&self, tool_name: &str, key: &str) -> bool {
        let Some(tools) = &self.options.tools else {
            return true;
        };
        let Some(tool) = tools.iter().find(|t| t.name == tool_name) else {
            return true;
        };
        let prop = &tool.parameters["properties"][key];
        match prop.get("type") {
            Some(Value::String(t)) => t == "string",
            _ => false,
        }
    }

    fn format_arg_value(&self, tool_name: &str, key: &str, raw: &str) -> String {
        if self.arg_is_string(tool_name, key) {
            format!("\"{}\"", json_string_contents(raw))
        } else {
            let trimmed = raw.trim();
            if serde_json::from_str::<Value>(trimmed).is_ok() {
                trimmed.to_owned()
            } else {
                format!("\"{}\"", json_string_contents(raw))
            }
        }
    }

    fn find_earliest<'a>(&self, markers: &[&'a str], include_stops: bool) -> Option<(usize, &'a str, bool)> {
        let mut best: Option<(usize, &'a str, bool)> = None;
        for &m in markers {
            if let Some(pos) = self.pending.find(m) {
                if best.is_none_or(|(b, _, _)| pos < b) {
                    best = Some((pos, m, false));
                }
            }
        }
        if include_stops {
            for &m in &STOP_MARKERS {
                if let Some(pos) = self.pending.find(m) {
                    if best.is_none_or(|(b, _, _)| pos < b) {
                        best = Some((pos, m, true));
                    }
                }
            }
            for seq in &self.options.stop_sequences {
                if let Some(pos) = self.pending.find(seq.as_str()) {
                    if best.is_none_or(|(b, _, _)| pos < b) {
                        // Safe because we only compare/inspect within this function.
                        best = Some((pos, "", true));
                    }
                }
            }
        }
        best
    }

    fn max_hold(&self, markers: &[&str], include_stops: bool) -> usize {
        let mut hold = markers
            .iter()
            .map(|&m| held_prefix(&self.pending, m))
            .max()
            .unwrap_or(0);
        if include_stops {
            hold = hold.max(
                STOP_MARKERS
                    .iter()
                    .map(|&m| held_prefix(&self.pending, m))
                    .max()
                    .unwrap_or(0),
            );
            hold = hold.max(
                self.options
                    .stop_sequences
                    .iter()
                    .map(|s| held_prefix(&self.pending, s.as_str()))
                    .max()
                    .unwrap_or(0),
            );
        }
        hold
    }

    fn emit_content(&self, text: String, out: &mut Vec<OutputChunk>) {
        if !text.is_empty() {
            out.push(OutputChunk::Raw { content: text });
        }
    }

    fn emit_reasoning(&self, text: String, out: &mut Vec<OutputChunk>) {
        if !text.is_empty() {
            out.push(OutputChunk::Reasoning { content: text });
        }
    }

    fn emit_completed_call(&mut self, call: PendingCall, out: &mut Vec<OutputChunk>) {
        let name = call.name.trim();
        if name.is_empty() || !name_is_readable(name, self.options.tools.as_deref()) {
            self.emit_content(call.raw_text, out);
            return;
        }
        let mut json_args = String::from("{");
        for (idx, (k, v)) in call.arguments.iter().enumerate() {
            if idx > 0 {
                json_args.push(',');
            }
            json_args.push('"');
            json_args.push_str(&json_string_contents(k));
            json_args.push_str("\":");
            json_args.push_str(v);
        }
        json_args.push('}');
        let _ = self.stream_args.enabled();
        self.calls += 1;
        out.push(OutputChunk::ToolCall {
            tool_name: name.to_owned(),
            arguments: json_args,
        });
    }

    fn step(&mut self, eof: bool, out: &mut Vec<OutputChunk>) -> bool {
        if self.pending.is_empty() || self.mode == Mode::Done {
            return false;
        }
        match self.mode {
            Mode::Reasoning => {
                let markers = [THINK_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let before = self.pending[..pos].to_owned();
                    self.emit_reasoning(before, out);
                    if is_stop {
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    self.pending.drain(..pos + matched.len());
                    self.mode = Mode::Content;
                    return true;
                }
                let hold = if eof { 0 } else { self.max_hold(&markers, true) };
                let emit_len = self.pending.len().saturating_sub(hold);
                if emit_len > 0 {
                    let chunk: String = self.pending.drain(..emit_len).collect();
                    self.emit_reasoning(chunk, out);
                }
                false
            }
            Mode::Content => {
                let markers: &[&str] = if self.tools_enabled() {
                    &[THINK_OPEN, TOOL_CALLS_OPEN, TOOL_CALL_OPEN]
                } else {
                    &[THINK_OPEN, TOOL_CALLS_OPEN, TOOL_CALL_OPEN]
                };
                // Check client stop sequences explicitly so we capture exact sequence text.
                let mut seq_hit: Option<(usize, String)> = None;
                for seq in &self.options.stop_sequences {
                    if let Some(pos) = self.pending.find(seq.as_str()) {
                        if seq_hit.as_ref().is_none_or(|(b, _)| pos < *b) {
                            seq_hit = Some((pos, seq.clone()));
                        }
                    }
                }
                let tag_hit = self.find_earliest(markers, true);
                let use_seq = match (&seq_hit, &tag_hit) {
                    (Some((sp, _)), Some((tp, _, _))) => sp <= tp,
                    (Some(_), None) => true,
                    _ => false,
                };
                if use_seq {
                    let (pos, seq) = seq_hit.unwrap();
                    let before = self.pending[..pos].to_owned();
                    self.emit_content(before, out);
                    self.stop = Some(GlmStop::Sequence(seq));
                    self.pending.clear();
                    self.mode = Mode::Done;
                    return false;
                }
                if let Some((pos, matched, is_stop)) = tag_hit {
                    let before = self.pending[..pos].to_owned();
                    self.emit_content(before, out);
                    if is_stop {
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    if matched == THINK_OPEN {
                        self.pending.drain(..pos + matched.len());
                        self.mode = Mode::Reasoning;
                        return true;
                    }
                    if !self.tools_enabled() {
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    self.pending.drain(..pos + matched.len());
                    if matched == TOOL_CALLS_OPEN {
                        self.mode = Mode::CallsBlock;
                    } else {
                        self.call = Some(PendingCall {
                            raw_text: TOOL_CALL_OPEN.to_owned(),
                            ..Default::default()
                        });
                        self.mode = Mode::CallName;
                    }
                    return true;
                }
                let hold = if eof { 0 } else { self.max_hold(markers, true) };
                let emit_len = self.pending.len().saturating_sub(hold);
                if emit_len > 0 {
                    let chunk: String = self.pending.drain(..emit_len).collect();
                    self.emit_content(chunk, out);
                }
                false
            }
            Mode::CallsBlock => {
                let markers = [TOOL_CALL_OPEN, TOOL_CALLS_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    if is_stop {
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    self.pending.drain(..pos + matched.len());
                    if matched == TOOL_CALL_OPEN {
                        self.call = Some(PendingCall {
                            raw_text: TOOL_CALL_OPEN.to_owned(),
                            ..Default::default()
                        });
                        self.mode = Mode::CallName;
                    } else {
                        self.mode = Mode::Content;
                    }
                    return true;
                }
                if eof {
                    self.pending.clear();
                    self.mode = Mode::Done;
                }
                false
            }
            Mode::CallName => {
                let markers = [ARG_KEY_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let piece = self.pending[..pos].to_owned();
                    self.pending.drain(..pos + matched.len());
                    if let Some(call) = self.call.as_mut() {
                        call.name.push_str(&piece);
                        call.raw_text.push_str(&piece);
                        call.raw_text.push_str(matched);
                    }
                    if is_stop {
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    if matched == ARG_KEY_OPEN {
                        self.mode = Mode::ArgKey;
                    } else if matched == TOOL_CALL_CLOSE {
                        let call = self.call.take().unwrap();
                        self.emit_completed_call(call, out);
                        self.mode = Mode::CallsBlock;
                    } else {
                        // Missing closing tag before next call: return incomplete call as content.
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.call = Some(PendingCall {
                            raw_text: TOOL_CALL_OPEN.to_owned(),
                            ..Default::default()
                        });
                        self.mode = Mode::CallName;
                    }
                    return true;
                }
                false
            }
            Mode::ArgKey => {
                let markers = [ARG_KEY_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let key = self.pending[..pos].trim().to_owned();
                    self.pending.drain(..pos + matched.len());
                    if is_stop {
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    if let Some(call) = self.call.as_mut() {
                        call.raw_text.push_str(&key);
                        call.raw_text.push_str(matched);
                        call.key = Some(key);
                    }
                    self.mode = Mode::ArgValueStart;
                    return true;
                }
                false
            }
            Mode::ArgValueStart => {
                let markers = [ARG_VALUE_OPEN, TOOL_CALL_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let piece = self.pending[..pos].to_owned();
                    self.pending.drain(..pos + matched.len());
                    if let Some(call) = self.call.as_mut() {
                        call.raw_text.push_str(&piece);
                        call.raw_text.push_str(matched);
                    }
                    if is_stop {
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    if matched == ARG_VALUE_OPEN {
                        self.mode = Mode::ArgValue;
                    } else {
                        let call = self.call.take().unwrap();
                        self.emit_content(call.raw_text, out);
                        self.mode = Mode::CallsBlock;
                    }
                    return true;
                }
                false
            }
            Mode::ArgValue => {
                let markers = [ARG_VALUE_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let val = self.pending[..pos].to_owned();
                    self.pending.drain(..pos + matched.len());
                    if is_stop {
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    let tool_name = self.call.as_ref().map(|c| c.name.trim().to_owned()).unwrap_or_default();
                    let key = self.call.as_mut().and_then(|c| c.key.take()).unwrap_or_default();
                    let formatted = self.format_arg_value(&tool_name, &key, &val);
                    if let Some(call) = self.call.as_mut() {
                        call.raw_text.push_str(&val);
                        call.raw_text.push_str(matched);
                        if !key.is_empty() && !call.arguments.iter().any(|(k, _)| k == &key) {
                            call.arguments.push((key, formatted));
                        }
                        call.raw_value.clear();
                    }
                    self.mode = Mode::ArgKeyOrEnd;
                    return true;
                }
                false
            }
            Mode::ArgKeyOrEnd => {
                let markers = [ARG_KEY_OPEN, TOOL_CALL_CLOSE];
                if let Some((pos, matched, is_stop)) = self.find_earliest(&markers, true) {
                    let piece = self.pending[..pos].to_owned();
                    self.pending.drain(..pos + matched.len());
                    if let Some(call) = self.call.as_mut() {
                        call.raw_text.push_str(&piece);
                        call.raw_text.push_str(matched);
                    }
                    if is_stop {
                        if let Some(call) = self.call.take() {
                            self.emit_content(call.raw_text, out);
                        }
                        self.stop = Some(GlmStop::Marker(matched.to_owned()));
                        self.pending.clear();
                        self.mode = Mode::Done;
                        return false;
                    }
                    if matched == ARG_KEY_OPEN {
                        self.mode = Mode::ArgKey;
                    } else {
                        let call = self.call.take().unwrap();
                        self.emit_completed_call(call, out);
                        self.mode = Mode::CallsBlock;
                    }
                    return true;
                }
                false
            }
            Mode::Done => false,
        }
    }
}

impl TextParser for Hy4OutputParser {
    fn push(&mut self, text: &str) -> Vec<OutputChunk> {
        if self.mode == Mode::Done {
            return Vec::new();
        }
        self.pending.push_str(text);
        let mut out = Vec::new();
        while self.step(false, &mut out) {}
        out
    }

    fn finish(&mut self) -> Vec<OutputChunk> {
        let mut out = Vec::new();
        if self.mode != Mode::Done {
            while self.step(true, &mut out) {}
            if let Some(mut call) = self.call.take() {
                call.raw_text.push_str(&self.pending);
                self.emit_content(call.raw_text, &mut out);
            }
            self.pending.clear();
            self.mode = Mode::Done;
        }
        out
    }

    fn stop(&self) -> Option<&GlmStop> {
        self.stop.as_ref()
    }

    fn tool_calls(&self) -> usize {
        self.calls
    }
}
