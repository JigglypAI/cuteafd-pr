use super::{fixtures::encoding, *};
use deepseek_recipe::stream::OutputChunk;
use deepseek_recipe_core::tools::ToolDefinition;
use serde_json::{json, Value};

fn tool() -> ToolDefinition {
    ToolDefinition { name: "read_file".into(), description: None, strict: None,
        parameters: json!({"type":"object", "properties":{"path":{"type":"string"}}}) }
}
fn options() -> InklingParserOptions {
    InklingParserOptions { tools: Some(vec![tool()]), stop_sequences: Vec::new() }
}
fn parsed(parts: &[&str], options: InklingParserOptions) -> (String, String, Vec<(String, Value)>) {
    let mut parser = InklingOutputParser::new(options);
    let mut chunks = Vec::new();
    for part in parts { chunks.extend(parser.push(part).unwrap()); }
    chunks.extend(parser.finish().unwrap());
    let (mut text, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    for chunk in chunks {
        match chunk {
            OutputChunk::Raw { content } => text.push_str(&content),
            OutputChunk::Reasoning { content } => reasoning.push_str(&content),
            OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, serde_json::from_str(&arguments).unwrap())),
            _ => panic!("unexpected chunk"),
        }
    }
    assert_eq!(parser.tool_calls(), calls.len());
    (text, reasoning, calls)
}

#[test]
fn checkpoint_template_matches_official_tokenizer_goldens() {
    let goldens: Value = serde_json::from_str(include_str!("fixtures/goldens.json")).unwrap();
    assert_eq!(goldens["cases"].as_array().unwrap().len(), 21);
    for case in goldens["cases"].as_array().unwrap() {
        let body = &case["body"];
        let options = InklingPromptOptions { tool_names: body["tools"].as_array().into_iter().flatten()
            .map(|tool| tool["function"]["name"].as_str().unwrap().to_owned()).collect() };
        assert_eq!(encoding().render(body, &options).unwrap(), case["prompt"].as_str().unwrap(), "{}", case["name"]);
    }
}

#[test]
fn off_switches_match_native_none_effort_and_invalid_requests_do_not_panic() {
    let plain = json!({"messages":[{"role":"user", "content":"Hi"}], "reasoning_effort":"none"});
    let expected = encoding().render(&plain, &InklingPromptOptions::default()).unwrap();
    for extra in [json!({"thinking":{"type":"disabled"}}), json!({"enable_thinking":false}),
        json!({"chat_template_kwargs":{"enable_thinking":false}})] {
        let mut body = json!({"messages":[{"role":"user", "content":"Hi"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert_eq!(encoding().render(&body, &InklingPromptOptions::default()).unwrap(), expected);
    }
    for body in [json!({"messages":[]}), json!({"messages":[42]}),
        json!({"messages":[{"role":"bad", "content":"x"}]}),
        json!({"messages":[{"role":"assistant", "tool_calls":[42]}]}),
        json!({"messages":[{"role":"assistant", "tool_calls":[{"function":{"name":"x", "arguments":"[]"}}]}]})] {
        assert!(encoding().render(&body, &InklingPromptOptions::default()).is_err(), "{body}");
    }
    for effort in [json!(true), json!(-0.1), json!(1.0), json!("bogus")] {
        let body = json!({"messages":[{"role":"user", "content":"Hi"}], "reasoning_effort":effort});
        assert!(encoding().render(&body, &InklingPromptOptions::default()).is_err());
    }
}

#[test]
fn native_snapshot_headers_and_stops_do_not_depend_on_live_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("chat_template.jinja"), fixtures::TEMPLATE).unwrap();
    let mut config: Value = serde_json::from_str(include_str!("fixtures/tokenizer_config.json")).unwrap();
    assert_eq!(config["added_tokens_decoder"]["200006"]["content"], EOS);
    assert_eq!(config["added_tokens_decoder"]["200010"]["content"], END);
    let path = directory.path().join("tokenizer_config.json");
    std::fs::write(&path, config.to_string()).unwrap();
    let loaded = InklingEncoding::from_snapshot(directory.path()).unwrap();
    assert_eq!(loaded.tokens(), encoding().tokens());
    for tools in [false, true] { assert_eq!(loaded.stop_token_ids(tools), vec![200006]); }
    config["added_tokens_decoder"]["200010"]["content"] = json!("wrong");
    std::fs::write(path, config.to_string()).unwrap();
    assert!(InklingEncoding::from_snapshot(directory.path()).is_err());
}

#[test]
fn every_stream_split_preserves_reasoning_text_and_parallel_json_calls() {
    let wire = concat!("<|message_model|><|content_thinking|>Plan\n<|end_message|>",
        "<|message_model|><|content_text|>Opening README.\n<|end_message|>",
        "<|message_model|>read_file<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{\"path\":\"README.md\"}}<|end_message|>",
        "<|message_model|>read_file<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{\"path\":\"src/lib.rs\",\"nested\":{\"a\":[1,true]}}}<|end_message|>",
        "<|content_model_end_sampling|>");
    let expected = parsed(&[wire], options());
    assert_eq!(expected.0, "Opening README.\n");
    assert_eq!(expected.1, "Plan\n");
    assert_eq!(expected.2.len(), 2);
    for index in 0..=wire.len() {
        assert_eq!(parsed(&[&wire[..index], &wire[index..]], options()), expected, "split {index}");
    }
    let bytes: Vec<_> = (0..wire.len()).map(|i| &wire[i..i + 1]).collect();
    assert_eq!(parsed(&bytes, options()), expected);
}

#[test]
fn official_completion_spans_round_trip_through_parser_at_every_character_split() {
    let goldens: Value = serde_json::from_str(include_str!("fixtures/goldens.json")).unwrap();
    let cases = goldens["cases"].as_array().unwrap();
    let prefix = cases.iter().find(|case| case["name"] == "official_tools_high").unwrap()["prompt"].as_str().unwrap();
    for case in cases.iter().filter(|case| case["name"].as_str().unwrap().starts_with("official_completion_")) {
        let span = case["prompt"].as_str().unwrap().strip_prefix(prefix).unwrap();
        assert!(span.ends_with(EOS), "{} must stop on ID 200006", case["name"]);
        let expected = parsed(&[span], options());
        for (index, _) in span.char_indices().chain(std::iter::once((span.len(), '\0'))) {
            assert_eq!(parsed(&[&span[..index], &span[index..]], options()), expected, "{} at {index}", case["name"]);
        }
    }
}

#[test]
fn channels_stream_before_close_and_stops_only_match_visible_text() {
    let mut parser = InklingOutputParser::new(options());
    assert_eq!(parser.push("<|message_model|><|content_text|>Hello").unwrap(),
        vec![OutputChunk::Raw { content: "Hello".into() }]);
    parser.push("<|end_message|>").unwrap();
    assert!(parser.finish().unwrap().is_empty());
    let wire = "<|content_thinking|>STOP<|end_message|><|content_text|>before STOP after<|end_message|>";
    let mut opts = options(); opts.stop_sequences = vec!["STOP".into()];
    let expected = parsed(&[wire], opts.clone());
    assert_eq!(expected.0, "before "); assert_eq!(expected.1, "STOP");
    for index in 0..=wire.len() { assert_eq!(parsed(&[&wire[..index], &wire[index..]], opts.clone()), expected); }
}

#[test]
fn malformed_and_truncated_native_output_return_typed_errors_without_calls() {
    for wire in ["plain text", "<|message_user|>", "<|message_model|><|content_unknown|>",
        "<|content_invoke_tool_json|>not JSON<|end_message|>",
        "<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":[]}<|end_message|>",
        "<|content_invoke_tool_json|>{\"name\":\"unknown\",\"args\":{}}<|end_message|>",
        "<|message_model|>wrong<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{}}<|end_message|>"] {
        let mut parser = InklingOutputParser::new(options());
        assert!(parser.push(wire).is_err(), "{wire}");
        assert!(parser.error().is_some()); assert_eq!(parser.tool_calls(), 0);
    }
    for wire in ["<|message_mod", "<|message_model|>read_file", "<|content_text|>cut",
        "<|content_thinking|>cut", "<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{}"] {
        let mut parser = InklingOutputParser::new(options());
        parser.push(wire).unwrap();
        assert!(matches!(parser.finish(), Err(InklingParseError::Truncated(_))), "{wire}");
        assert_eq!(parser.tool_calls(), 0);
    }
}

#[test]
fn constraints_use_only_inkling_channels_and_json_arguments() {
    use crate::openai::{constraints::inkling_response_constraint, tools::{ToolConstraints, ToolSyntax}};
    use deepseek_recipe_core::tools::ToolChoice;
    let constraints = ToolConstraints::new(&[tool()], ToolChoice::Auto, true, true, true, ToolSyntax::InklingJson).unwrap().unwrap();
    let wire = inkling_response_constraint(None, Some(&constraints)).unwrap().unwrap().0;
    assert!(wire.contains("<|content_invoke_tool_json|>"));
    assert!(wire.contains("<|content_thinking|>"));
    assert!(wire.contains("ds41_json_schema"));
    assert!(!wire.contains("<tool_call>"));
    assert!(inkling_response_constraint(None, None).unwrap().is_none());
}
