//! Explicit per-panel configuration shared by the CLI, smoke and server.
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;

pub type PanelConfigs = BTreeMap<String, Value>;

pub fn parse(text: &str) -> Result<(String, Value)> {
    let (panel, body) = text.split_once('=').context("panel-config needs PANEL=JSON")?;
    let value: Value = serde_json::from_str(body).context("panel-config JSON")?;
    validate(&BTreeMap::from([(panel.to_string(), value.clone())]))?;
    Ok((panel.to_string(), value))
}

pub fn validate(configs: &PanelConfigs) -> Result<()> {
    for (panel, value) in configs {
        match panel.as_str() {
            "decode_content" => { crate::panels::speed::DecodeOptions::parse(value)?; }
            _ => bail!("panel {panel} does not accept panel-config"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn panel_config_validates_names_fields_and_sampling() {
        for text in ["decode_content={\"contents\":[\"code\"],\"temperature\":1.0,\"top_p\":0.95,\"seed\":42}",
            "decode_content={}"] {
            assert!(super::parse(text).is_ok(), "{text}");
        }
        for text in ["decode_content", "other={}", "decode_content=null", "decode_content={\"typo\":1}",
            "decode_content={\"temperature\":-1}", "decode_content={\"top_p\":0}",
            "decode_content={\"contents\":[]}", "decode_content={\"contents\":[\"bad\"]}",
            "decode_content={\"contents\":[\"code\",\"code\"]}"] {
            assert!(super::parse(text).is_err(), "{text}");
        }
    }
}
