//! Server-side rendering of panels, reports and the share card as
//! self-contained SVG (and PNG through resvg). The dashboard shows these same
//! SVGs, so a chart looks the same in the browser, in an export and on GitHub.
pub mod svg;
pub mod theme;
pub mod bodies;
pub mod report;
pub mod card;
pub mod charts;
pub mod fabric;
pub mod layout;
pub mod panels;
pub mod png;

pub use theme::Theme;

/// Thousands separators: 2415.3 -> "2,415".
pub fn grouped(value: f64) -> String {
    let rounded = value.round() as i64;
    let digits = rounded.abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if rounded < 0 { format!("-{out}") } else { out }
}

/// A rate with sensible precision: 187.3, 41.06, 2,415.
pub fn rate(value: f64) -> String {
    if !value.is_finite() || value <= 0.0 {
        "—".into()
    } else if value >= 1000.0 {
        grouped(value)
    } else if value >= 100.0 {
        format!("{value:.0}")
    } else if value >= 10.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.2}")
    }
}

pub fn seconds(value: f64) -> String {
    if !value.is_finite() || value < 0.0 {
        "—".into()
    } else if value < 1.0 {
        format!("{:.0} ms", value * 1e3)
    } else if value < 100.0 {
        format!("{value:.2} s")
    } else if value < 3600.0 {
        format!("{}m {:02}s", (value / 60.0) as u64, (value % 60.0) as u64)
    } else {
        format!("{}h {:02}m", (value / 3600.0) as u64, (value % 3600.0 / 60.0) as u64)
    }
}

/// `2026-10-02` from an RFC 3339 time.
pub fn date(rfc3339: &str) -> &str {
    rfc3339.get(..10).unwrap_or(rfc3339)
}

#[cfg(test)]
mod tests {
    #[test]
    fn card_prefixes_mc_quality_and_preserves_badge_width_limit() {
        let mut report = crate::sample::full_report();
        let before = super::card::card_svg(&report);
        assert!(!before.contains("Core BF16"));
        for dtype in ["BF16", "K8", "K6", "FP8"] {
            report.baseline.as_mut().unwrap().quality.core_dtype = Some(dtype.into());
            let svg = super::card::card_svg(&report);
            assert!(svg.contains(&format!("Core {dtype} · KL")), "{dtype}");
            let badge = svg.split("<rect ").find(|rect| rect.contains("height=\"34\""))
                .expect("quality chip");
            let width: f64 = badge.split("width=\"").nth(1).unwrap()
                .split('"').next().unwrap().parse().unwrap();
            assert!(width <= 520.0);
        }
    }

    #[test]
    fn mc_card_without_baseline_labels_core_without_claiming_quality() {
        let mut report = crate::sample::full_report();
        report.baseline = None;
        let old = super::card::card_svg(&report);
        assert!(old.contains("quality not verified") && !old.contains("Core "));
        report.server.configuration.settings.push(crate::report::Setting {
            name: "core-original-dtype".into(), value: Some("BF16".into()),
            ..Default::default()
        });
        for (view, dtype) in [("original", "BF16"), ("k8-compact", "K8"),
            ("k6-compact", "K6"), ("fp8-compact", "FP8")] {
            report.server.configuration.settings.retain(|s| s.name != "core-view");
            report.server.configuration.settings.push(crate::report::Setting {
                name: "core-view".into(), value: Some(view.into()), ..Default::default()
            });
            let svg = super::card::card_svg(&report);
            assert!(svg.contains(&format!("Core {dtype} · quality not verified")));
        }
    }

    #[test]
    fn drafter_round_trips_in_server_info_and_is_optional_in_old_reports() {
        use crate::report::{DrafterKind, DrafterLayout, DrafterPlacement, DrafterRank, Report};
        let mut report = crate::sample::full_report();
        let legacy = serde_json::to_value(&report).unwrap();
        assert!(legacy["server"].get("drafter").is_none());
        let old: Report = serde_json::from_value(legacy).unwrap();
        assert!(!super::card::card_svg(&old).contains("Drafter:"));
        assert!(!super::report::panel_svg(&old, "configuration").contains("DRAFTER"));
        report.server.drafter = Some(DrafterLayout { kind: DrafterKind::Dflash2,
            placement: Some(DrafterPlacement::Split), ranks: vec![DrafterRank::CudaRank(1), DrafterRank::CudaRank(0)],
            lead: Some(DrafterRank::CudaRank(1)), arena_bytes: vec![1_200_000_000, 1_100_000_000], gpu_uuids: Default::default() });
        report.server.drafter.as_mut().unwrap().bind_visible_gpus("GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002");
        let wire = serde_json::to_value(&report).unwrap();
        assert_eq!(wire["server"]["drafter"], serde_json::json!({"kind":"dflash2", "placement":"split",
            "ranks":[{"cuda_rank":1},{"cuda_rank":0}], "lead":{"cuda_rank":1}, "arena_bytes":[1_200_000_000,1_100_000_000],
            "gpu_uuids":{"0":"GPU-00000000-0000-0000-0000-000000000001", "1":"GPU-00000000-0000-0000-0000-000000000002"}}));
        let restored: Report = serde_json::from_value(wire).unwrap();
        let line = "DFlash2 split over GPU rank 1 + GPU rank 0 (lead GPU rank 1), 1.20 / 1.10 GB";
        let card = super::card::card_svg(&restored);
        assert!(card.contains(&format!("Drafter: {line}")));
        let config = super::report::panel_svg(&restored, "configuration");
        assert!(config.contains(line), "{config}");
        assert!(super::png::png(&card, 1.0).is_ok());
    }

    #[test]
    fn card_marks_adaptive_and_fixed_drafts() {
        let mut report = crate::sample::full_report();
        for (mode, label) in [("adaptive", "adaptive draft"), ("fixed 4", "fixed draft 4")] {
            report.server.configuration.draft = mode.into();
            let svg = super::card::card_svg(&report);
            assert!(svg.contains(label));
            assert!(svg.contains("text-anchor=\"end\""));
            let png = super::png::png(&svg, 1.0).unwrap();
            assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
            if let Some(directory) = std::env::var_os("CUTEAFD_CARD_TEST_OUTPUT") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                let name = mode.replace(' ', "-");
                std::fs::write(directory.join(format!("{name}.svg")), svg).unwrap();
                std::fs::write(directory.join(format!("{name}.png")), png).unwrap();
            }
        }
    }

    #[test]
    fn card_without_speculator_never_claims_adaptive_or_fixed_drafts() {
        let mut report = crate::sample::full_report();
        report.server.configuration.speculator = None;
        for mode in ["adaptive", "fixed 4"] {
            report.server.configuration.draft = mode.into();
            let svg = super::card::card_svg(&report);
            assert!(svg.contains("no speculation"));
            assert!(!svg.contains("adaptive draft") && !svg.contains("fixed draft"));
        }
        report.server.configuration = serde_json::from_str("{}").unwrap();
        assert!(super::card::card_svg(&report).contains("no speculation"));
    }

    #[test]
    fn wrap_breaks_a_word_longer_than_the_line() {
        let lines = super::bodies::wrap("dev · https://github.com/tpurtell/cuteafd @ 9db3e0b7", 12.0, 120.0);
        assert!(lines.iter().all(|l| l.chars().count() <= 16), "{lines:?}");
        assert_eq!(lines.concat().replace(' ', ""), "dev·https://github.com/tpurtell/cuteafd@9db3e0b7");
    }

    #[test]
    fn always_thinking_is_recorded_and_visible_on_card_and_configuration() {
        let mut report = crate::sample::full_report();
        report.server.configuration.supports_thinking_off = Some(false);
        let label = report.server.configuration.thinking_label();
        report.baseline.as_mut().unwrap().card.thinking = Some(label.into());
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(encoded["server"]["configuration"]["supports_thinking_off"], false);
        assert_eq!(encoded["baseline"]["card"]["thinking"], label);
        let card = super::card::card_svg(&report);
        assert!(card.contains(label), "{card}");
        assert!(!card.contains("thinking off"));
        let configuration = super::report::panel_svg(&report, "configuration");
        assert!(configuration.contains(label), "{configuration}");
    }

    #[test]
    fn layer_timing_is_visible_on_cards_with_and_without_profiling() {
        for value in ["off", "on (perturbing)"] {
            let mut report = crate::sample::full_report();
            report.server.configuration.settings.push(crate::report::Setting {
                name: "per-layer-timing".into(), value: Some(value.into()), default: None, source: "resolved".into(),
            });
            let card = super::card::card_svg(&report);
            assert!(card.contains(&format!("per-layer-timing={value}")), "{card}");
        }
    }

    #[test]
    fn number_formats() {
        assert_eq!(super::grouped(2415.3), "2,415");
        assert_eq!(super::grouped(1234567.0), "1,234,567");
        assert_eq!(super::rate(187.34), "187");
        assert_eq!(super::rate(41.06), "41.1");
        assert_eq!(super::rate(2415.3), "2,415");
        assert_eq!(super::seconds(0.4), "400 ms");
        assert_eq!(super::seconds(125.0), "2m 05s");
    }
    #[test]
    fn health_footer_keeps_rates_and_pause_in_card_and_report() {
        let mut report = crate::sample::full_report();
        let h = cuteafd_transport::fabric_health::Health {
            verdict: cuteafd_transport::fabric_health::Verdict::Ok,
            retransmit_events: Some(600), retransmit_pct: Some(0.025), retransmit_judged: true,
            retransmit_warn_pct: 0.5, reasons: vec!["ECN and CNP observed".into()],
            ..Default::default()
        };
        report.fabric = vec![crate::fabric::Phase { name: "decode".into(), timed: true,
            ports: vec![crate::fabric::Port { host: "coordinator".into(), device: "nic".into(), port: 1,
                state: Some("4: ACTIVE".into()), seconds: 1.0, sent_bytes: Some(1_000_000_000), received_bytes: Some(0),
                engine_sent_bytes: Some(900_000_000), engine_received_bytes: Some(0),
                health: Some(h.clone()), health_since_start: Some(h), mtu: Some(9000), ..Default::default() }], ..Default::default() }];
        let card = super::card::card_svg(&report);
        assert!(!card.contains("Fabric: peak retransmits"));
        assert!(!card.contains("results remain valid"));
        let report = super::report::report_svg(&report);
        // The report draws the port as a small multiple with its loss verdict.
        assert!(report.contains("Fabric during this run") && report.contains("retransmits 0.025%"));
    }

}
