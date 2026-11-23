use super::{fixtures, Hy4OutputParser, Hy4ParserOptions, Hy4PromptOptions};
use crate::openai::chat::glm5::TextParser;
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::json;

#[test]
fn hy4_template_renders_thinking_and_no_think_modes() {
    let enc = fixtures::encoding();
    let body = json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "high"
    });
    let prompt = enc.render(&body, &Hy4PromptOptions::default()).unwrap();
    assert!(prompt.contains("<｜hy_start:opensource｜>system<｜hy_middle:opensource｜>"));
    assert!(prompt.contains("reasoning_effort:high"));
    assert!(prompt.ends_with("<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>"));

    let off = json!({
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "no_think"
    });
    let off_prompt = enc.render(&off, &Hy4PromptOptions::default()).unwrap();
    assert!(off_prompt.contains("reasoning_effort:no_think"));
    assert!(off_prompt.ends_with("<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>"));
}

#[test]
fn hy4_template_renders_tools_and_multi_turn_tool_calls() {
    let enc = fixtures::encoding();
    let body = json!({
        "tools": [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}
            }
        }],
        "messages": [
            {"role": "user", "content": "Check main.rs"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"main.rs\",\"limit\":50}"}
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"}
        ]
    });
    let opts = Hy4PromptOptions { tool_names: vec!["read_file".into()] };
    let prompt = enc.render(&body, &opts).unwrap();
    assert!(prompt.contains("<tools>"));
    assert!(prompt.contains("<tool_calls:opensource><tool_call:opensource>read_file"));
    assert!(prompt.contains("<arg_key:opensource>path</arg_key:opensource><arg_value:opensource>main.rs</arg_value:opensource>"));
    assert!(prompt.contains("<tool_response:opensource>fn main() {}</tool_response:opensource>"));
}

#[test]
fn hy4_parser_splits_reasoning_content_and_parallel_tool_calls_across_arbitrary_chunks() {
    let tool = ToolDefinition {
        name: "read_file".into(),
        description: Some("Read".into()),
        parameters: json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
            }
        }),
        strict: None,
    };
    let text = "Need to inspect.</think:opensource>Checking now.<tool_calls:opensource>\
<tool_call:opensource>read_file<arg_key:opensource>path</arg_key:opensource><arg_value:opensource>a.rs</arg_value:opensource>\
<arg_key:opensource>limit</arg_key:opensource><arg_value:opensource>42</arg_value:opensource></tool_call:opensource>\
<tool_call:opensource>read_file<arg_key:opensource>path</arg_key:opensource><arg_value:opensource>b.rs</arg_value:opensource></tool_call:opensource>\
</tool_calls:opensource><｜hy_end:opensource｜>";

    for step in [1usize, 3, 7, 1000] {
        let mut p = Hy4OutputParser::new(Hy4ParserOptions {
            thinking: true,
            tools: Some(vec![tool.clone()]),
            stop_sequences: Vec::new(),
            id: "req1".into(),
        });
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for win in chars.chunks(step) {
            let s: String = win.iter().collect();
            chunks.extend(p.push(&s));
        }
        chunks.extend(p.finish());
        assert_eq!(p.tool_calls(), 2, "step={step}");
        let reasoning: String = chunks.iter().filter_map(|c| match c { OutputChunk::Reasoning { content } => Some(content.as_str()), _ => None }).collect();
        let content: String = chunks.iter().filter_map(|c| match c { OutputChunk::Raw { content } => Some(content.as_str()), _ => None }).collect();
        assert_eq!(reasoning, "Need to inspect.");
        assert_eq!(content, "Checking now.");
        let args: Vec<serde_json::Value> = chunks.iter().filter_map(|c| match c {
            OutputChunk::ToolCall { arguments, .. } => Some(serde_json::from_str(arguments).unwrap()),
            _ => None,
        }).collect();
        assert_eq!(args, vec![json!({"path": "a.rs", "limit": 42}), json!({"path": "b.rs"})]);
    }
}
