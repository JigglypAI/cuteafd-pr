//! Supported M3.1 image policy is the vendor default Pillow rectangle only.
use serde_json::Value;

/// Reject pixel-budget overrides before download/decoding or engine admission.
/// Do not recurse into arbitrary text/tool JSON where max_pixels is user data.
pub(crate) fn validate_policy(body: &Value) -> Result<(), String> {
    let forbidden = |value: &Value| value.get("max_pixels").is_some();
    if forbidden(body)
        || body.get("images_kwargs").is_some_and(forbidden)
        || body.get("mm_processor_kwargs").is_some_and(forbidden)
        || body
            .pointer("/extra_body/mm_processor_kwargs")
            .is_some_and(forbidden)
    {
        return Err("MiniMax image max_pixels override is unsupported; use the checkpoint's default image policy".into());
    }
    for message in body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for part in message
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if part.get("type").and_then(Value::as_str) == Some("image_url")
                && (forbidden(part) || part.get("image_url").is_some_and(forbidden))
            {
                return Err("MiniMax image max_pixels override is unsupported; use the checkpoint's default image policy".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn image_override_locations_are_refused_even_if_null() {
        for body in [
            json!({"max_pixels":1}),
            json!({"images_kwargs":{"max_pixels":1}}),
            json!({"mm_processor_kwargs":{"max_pixels":null}}),
            json!({"extra_body":{"mm_processor_kwargs":{"max_pixels":1}}}),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","max_pixels":1,"image_url":{"url":"x"}}]}]}),
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x","max_pixels":1}}]}]}),
        ] {
            assert!(validate_policy(&body).is_err());
        }
    }
    #[test]
    fn ordinary_text_and_tool_payloads_are_not_treated_as_processor_options() {
        assert!(validate_policy(&json!({"messages":[{"role":"user","content":"max_pixels=100"},
            {"role":"tool","content":"{\"max_pixels\":100}"}],"tools":[{"function":{"parameters":{"properties":{"max_pixels":{"type":"number"}}}}}]})).is_ok());
    }
}
