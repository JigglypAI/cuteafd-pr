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
    for wire in ["<|content_invoke_tool_json|>not JSON<|end_message|>",
        "<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":[]}<|end_message|>",
        "<|content_invoke_tool_json|>{\"name\":\"unknown\",\"args\":{}}<|end_message|>",
        "<|message_model|>wrong<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{}}<|end_message|>"] {
        let mut parser = InklingOutputParser::new(options());
        assert!(parser.push(wire).is_err(), "{wire}");
        assert!(parser.error().is_some()); assert_eq!(parser.tool_calls(), 0);
    }
    for wire in ["<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{}"] {
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

#[test]
fn ordinary_channel_cuts_flush_and_boundary_noise_is_ignored_at_every_split() {
    for (wire, text, reasoning) in [
        ("<|message_model|><|content_text|>The hist", "The hist", ""),
        ("<|message_model|><|content_thinking|>The hist", "", "The hist"),
        ("\n<|message_model|><|content_text|>Hi<|end_message|>\n<|message_model|><|content_text|>there<|end_message|>", "Hithere", ""),
        ("noise<|unknown|>\n<|content_text|>Hi<|content_model_end_sampling|>", "Hi", ""),
        ("<|content_thinking|>Plan<|content_model_end_sampling|>", "", "Plan"),
        ("<|content_text|>partial<|end_mes", "partial<|end_mes", ""),
    ] {
        for index in 0..=wire.len() {
            let result = parsed(&[&wire[..index], &wire[index..]], options());
            assert_eq!((result.0.as_str(), result.1.as_str()), (text, reasoning), "{wire} at {index}");
        }
    }
}

#[test]
fn tool_arguments_stream_before_close_and_match_buffered_bytes_at_every_split() {
    let wire = concat!("<|message_model|>read_file<|content_invoke_tool_json|>",
        "{\"name\":\"read_file\",\"args\": {\"path\": \"README.md\", \"nested\": {\"z\": [1, true], \"quote\": \"}\\\"\"}}}",
        "<|end_message|><|content_model_end_sampling|>");
    fn collect(parts: &[&str], streaming: bool) -> Vec<(String, String)> {
        let mut parser = InklingOutputParser::with_stream_args(options(), streaming);
        let mut calls: Vec<(String, String)> = Vec::new();
        let mut outputs = Vec::new();
        for part in parts { outputs.extend(parser.push(part).unwrap()); }
        outputs.extend(parser.finish().unwrap());
        for output in outputs {
            match output {
                OutputChunk::ToolCall { tool_name, arguments } => calls.push((tool_name, arguments)),
                OutputChunk::ToolArgumentsDelta { content } => calls.last_mut().unwrap().1.push_str(&content),
                OutputChunk::ToolCallBegin => {}
                _ => panic!("unexpected chunk"),
            }
        }
        assert_eq!(parser.tool_calls(), calls.len());
        calls
    }
    let expected = collect(&[wire], false);
    assert_eq!(expected.len(), 1);
    assert!(expected[0].1.starts_with(r#"{"path": "README.md", "nested": "#));
    let parallel = format!("{}{}", wire.strip_suffix(EOS).unwrap(), wire);
    assert_eq!(collect(&[&parallel], true), collect(&[&parallel], false));
    let reordered = "<|content_invoke_tool_json|>{\"args\":{\"path\":\"README.md\"},\"name\":\"read_file\"}<|end_message|>";
    assert_eq!(collect(&[reordered], true), collect(&[reordered], false));
    for index in 0..=wire.len() {
        assert_eq!(collect(&[&wire[..index], &wire[index..]], true), expected, "split {index}");
    }
    let parts: Vec<_> = (0..wire.len()).map(|index| &wire[index..index + 1]).collect();
    assert_eq!(collect(&parts, true), expected);
    let mut parser = InklingOutputParser::with_stream_args(options(), true);
    let prefix = wire.split("README.md").next().unwrap();
    let outputs = parser.push(prefix).unwrap();
    assert!(outputs.iter().any(|output| matches!(output, OutputChunk::ToolArgumentsDelta { content } if content.contains("path"))));
    assert!(matches!(parser.finish(), Err(InklingParseError::Truncated("tool call"))));
}

#[test]
fn strict_parameterless_tool_schema_still_requires_object_arguments() {
    use crate::openai::tools::{ToolConstraints, ToolSyntax};
    use deepseek_recipe_core::tools::ToolChoice;
    let mut empty = tool(); empty.parameters = json!({}); empty.strict = Some(true);
    let constraints = ToolConstraints::new(&[empty], ToolChoice::Auto, true, true, true, ToolSyntax::InklingJson).unwrap().unwrap();
    let schema = &constraints.format.as_ref().unwrap()["tags"][0]["content"]["json_schema"];
    let validator = jsonschema::JSONSchema::compile(schema).unwrap();
    assert!(validator.is_valid(&json!({"name":"read_file", "args":{}})));
    assert!(!validator.is_valid(&json!({"name":"read_file", "args":[]})));
    assert!(!validator.is_valid(&json!({"name":"read_file", "args":"bad"})));
}

#[test]
fn complete_call_cut_before_end_message_is_retained_but_incomplete_json_is_not() {
    let wire = "<|message_model|>read_file<|content_invoke_tool_json|>{\"name\":\"read_file\",\"args\":{\"path\": \"README.md\"}}";
    for streaming in [false, true] {
        let mut parser = InklingOutputParser::with_stream_args(options(), streaming);
        let mut output = parser.push(wire).unwrap();
        output.extend(parser.finish().unwrap());
        assert_eq!(parser.tool_calls(), 1);
        assert!(output.iter().any(|chunk| matches!(chunk, OutputChunk::ToolCall { tool_name, .. } if tool_name == "read_file")));
    }
    let result = parsed(&[wire], options());
    assert_eq!(result.2, vec![("read_file".into(), json!({"path":"README.md"}))]);
    let eos = format!("{wire}{EOS}");
    for streaming in [false, true] {
        for index in 0..=eos.len() {
            let mut parser = InklingOutputParser::with_stream_args(options(), streaming);
            parser.push(&eos[..index]).unwrap();
            parser.push(&eos[index..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(parser.tool_calls(), 1, "EOS split {index}");
            assert_eq!(parser.stop(), Some(&crate::openai::chat::glm5::parser::GlmStop::Marker(EOS.into())));
        }
    }
    assert_eq!(parsed(&[&eos], options()), result);
    let mut parser = InklingOutputParser::new(options());
    parser.push(wire.strip_suffix('}').unwrap()).unwrap();
    assert!(matches!(parser.finish(), Err(InklingParseError::Truncated("tool call"))));
}

#[test]
fn markerless_output_stays_empty_and_warns_instead_of_silently_disappearing() {
    use std::sync::{Arc, Mutex};
    #[derive(Clone)]
    struct Log(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Log {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let sink = Log(bytes.clone());
    let subscriber = tracing_subscriber::fmt().with_ansi(false).without_time()
        .with_max_level(tracing::Level::WARN).with_writer(move || sink.clone()).finish();
    tracing::subscriber::with_default(subscriber, || {
        assert_eq!(parsed(&["no", " channels\n"], options()), (String::new(), String::new(), Vec::new()));
    });
    let log = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    assert!(log.contains("Inkling completion contained no recognized content channels"), "{log}");
}


#[test]
fn ordinary_openai_image_parts_keep_the_native_inkling_placeholder() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("tokenizer_config.json"), include_str!("fixtures/tokenizer_config.json")).unwrap();
    std::fs::write(root.path().join("chat_template.jinja"), include_str!("fixtures/inkling2.jinja")).unwrap();
    let encoding = super::InklingEncoding::from_snapshot(root.path()).unwrap();
    for kind in ["image", "image_url", "input_image"] {
        let body = serde_json::json!({"messages":[{"role":"user","content":[
            {"type":kind,"image_url":{"url":"data:image/png;base64,AA=="}},
            {"type":"text","text":"describe"}]}]});
        let prompt = encoding.render(&body, &super::InklingPromptOptions::default()).unwrap();
        assert_eq!(prompt.matches("<|unused_200054|>").count(),1,"{kind}: {prompt}");
        assert!(prompt.contains("<|content_image|>"));
    }
}

#[test]
fn inkling2_mini_generation_and_dictsort_tool_calls_render() {
    let tpl = include_str!("fixtures/inkling2.jinja").replace(
        r#"{{- (args | tojson(sort_keys=true, separators=(",", ":"))) -}}
            {{- "}<|end_message|>" -}}"#,
        r#"{{- "{" -}}
            {%- for key, value in args | dictsort(case_sensitive=true) -%}
                {{- (key | tojson) ~ ":" ~ (value | tojson(separators=(",", ":"))) -}}
                {%- if not loop.last -%},{%- endif -%}
            {%- endfor -%}
            {{- "}}<|end_message|>" -}}"#,
    );
    let encoding = super::InklingEncoding::new(
        tpl,
        super::InklingTokenIds {
            eos: vec![200006],
            message_model: 200001,
            thinking: 200008,
            text: 200004,
            invoke_tool_json: 200049,
            end_message: 200010,
        },
    )
    .unwrap();
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{\"z\":1,\"A\":2,\"a\":3}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}
        ]
    });
    let prompt = encoding.render(&body, &super::InklingPromptOptions::default()).unwrap();
    assert!(
        prompt.contains(r#"{"name":"search","args":{"A":2,"a":3,"z":1}}"#),
        "case-sensitive dictsort must place 'A' before 'a' before 'z': {prompt}"
    );
}
