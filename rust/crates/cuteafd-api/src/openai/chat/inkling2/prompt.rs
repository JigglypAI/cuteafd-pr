//! Normalize OpenAI arguments before executing the checkpoint's unmodified template.
use serde_json::{json, Value};
use crate::openai::chat::glm5::prompt::resolve_thinking;

#[derive(Debug, Clone, Default)]
pub struct InklingPromptOptions {
    pub tool_names: Vec<String>,
}

/// The native template accepts named effort levels or a number in [0, 0.99].
pub fn effort(body: &Value) -> Result<Value, String> {
    let value = body.get("reasoning_effort").filter(|v| !v.is_null())
        .or_else(|| body.get("chat_template_kwargs").and_then(|v| v.get("reasoning_effort")).filter(|v| !v.is_null()))
        .cloned().unwrap_or(json!("high"));
    let value = match value {
        Value::String(text) => {
            let text = text.trim();
            if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&text) {
                return Err(format!("unknown Inkling reasoning_effort: {text}"));
            }
            Value::String(text.to_owned())
        }
        Value::Number(ref number) if number.as_f64().is_some_and(|v| (0.0..=0.99).contains(&v)) => value,
        _ => return Err("Inkling reasoning_effort must be a named level or a number in [0, 0.99]".into()),
    };
    // Reuse the common explicit-switch precedence without imposing its string-only effort API.
    let mut switches = body.clone();
    if let Some(object) = switches.as_object_mut() { object.remove("reasoning_effort"); }
    if let Some(object) = switches.get_mut("chat_template_kwargs").and_then(Value::as_object_mut) {
        object.remove("reasoning_effort");
    }
    if !resolve_thinking(&switches)? { return Ok(json!("none")); }
    Ok(value)
}

pub fn template_context(body: &Value, options: &InklingPromptOptions) -> Result<Value, String> {
    let effort = effort(body)?;
    let raw = body.get("messages").and_then(Value::as_array).ok_or("messages must be an array")?;
    if raw.is_empty() { return Err("messages must not be empty".into()); }
    let mut messages = raw.clone();
    for (index, message) in messages.iter_mut().enumerate() {
        if !message.is_object() { return Err(format!("messages[{index}] must be an object")); }
        if let Some(calls) = message.get_mut("tool_calls").filter(|v| !v.is_null()) {
            for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                let function = call.get_mut("function").and_then(Value::as_object_mut)
                    .ok_or("tool call function must be an object")?;
                let args = function.entry("arguments").or_insert(json!({}));
                if let Some(text) = args.as_str() {
                    *args = serde_json::from_str(text).map_err(|error| format!("invalid Inkling tool-call arguments: {error}"))?;
                }
                if !args.is_object() { return Err("Inkling tool-call arguments must be an object".into()); }
            }
        }
    }
    let tools: Vec<_> = body["tools"].as_array().into_iter().flatten().filter(|tool| {
        tool["function"]["name"].as_str().is_some_and(|name| options.tool_names.iter().any(|want| want == name))
    }).cloned().collect();
    Ok(json!({"messages": messages, "tools": tools, "reasoning_effort": effort, "add_generation_prompt": true}))
}
