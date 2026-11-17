//! `cuteafd bench`: start a run on a server, follow its progress, write the
//! exports. The measurement happens inside the server (the same runner as the
//! dashboard); this side only asks and downloads.
use crate::report::{Report, RunStatus};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    pub url: String,
    pub profile: Option<String>,
    pub fabric: Option<bool>,
    pub panels: Option<Vec<String>>,
    pub passes: Vec<(String, u32)>,
    /// `Some(false)` asks the server to run explicit panels without its
    /// baseline (servers that predate the field run it anyway).
    pub baseline: Option<bool>,
    /// `svg` (report, card, panels), `png` (report and card), `card` (card.png only), `json`.
    pub export: Vec<String>,
    /// Output directory; `None`: `benchmarks/<family>/<date>-<profile>-<hardware>/` under `root`.
    pub out: Option<PathBuf>,
    /// Appended to the default directory name (a smoke card's name), so runs of
    /// one checkpoint on the same hardware and day do not overwrite each other.
    pub label: Option<String>,
    pub root: PathBuf,
    pub api_key: Option<String>,
    pub quiet: bool,
    /// Cancel the run and fail when it takes longer (a server too slow to finish).
    pub deadline: Option<Duration>,
    /// Cases for the `spec_lossless_set` panel (default: the server's shipped set).
    pub lossless_set: Option<crate::panels::lossless::CaseSet>,
    /// Opt-in row dumps for selected `spec_lossless_set` cases.
    pub lossless_dump: Option<crate::panels::lossless::RowDump>,
    /// `spec_lossless_set` cases in flight at once per arm (`None`: 1).
    pub lossless_concurrency: Option<usize>,
    pub context_lengths: Option<Vec<u64>>,
    pub context_levels: Option<Vec<usize>>,
    pub context_suffix: Option<u64>,
    pub panel_config: crate::panel_config::PanelConfigs,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(120)).build()
}

fn authorize(request: ureq::Request, key: &Option<String>) -> ureq::Request {
    match key {
        Some(key) => request.set("authorization", &format!("Bearer {key}")),
        None => request,
    }
}

/// `benchmarks/<family>/<date>-<profile>-<hardware>`.
pub fn default_dir(root: &Path, report: &Report) -> PathBuf {
    let family = report.server.family.clone().unwrap_or_else(|| "unknown".into());
    root.join("benchmarks").join(family).join(format!("{}-{}-{}-{}", crate::render::date(&report.created),
        report.profile, model_slug(&report.server.checkpoint()), report.server.hardware.slug()))
}

/// [`default_dir`] with `-<label>` appended (slugged) when a label is given.
pub fn labeled_dir(root: &Path, report: &Report, label: Option<&str>) -> PathBuf {
    let dir = default_dir(root, report);
    match label.map(model_slug).filter(|l| !l.is_empty()) {
        Some(label) => {
            let name = format!("{}-{label}", dir.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
            dir.with_file_name(name)
        }
        None => dir,
    }
}

/// The checkpoint's last path segment, lowercase, runs of other characters as one `-`
/// (two quants of one family on the same hardware and day get separate directories).
fn model_slug(model: &str) -> String {
    let name = model.trim_end_matches('/').rsplit('/').next().unwrap_or(model).to_lowercase();
    let mut slug = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_end_matches('-').to_string()
}

/// Writes the exports of `report` into `dir`.
pub fn write_exports(report: &Report, dir: &Path, kinds: &[String]) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut written = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let want = |kind: &str| kinds.iter().any(|k| k == kind);
    if want("json") {
        files.push("report.json".into());
    }
    if want("svg") {
        files.extend(["report.svg".to_string(), "card.svg".to_string()]);
        files.extend(crate::render::report::shown(report).iter().map(|id| format!("panel-{id}.svg")));
    }
    if want("png") {
        files.extend(["report.png".to_string(), "card.png".to_string()]);
    } else if want("card") {
        files.push("card.png".to_string());
    }
    for file in files {
        let bytes = crate::http::export(report, &file).with_context(|| format!("exporting {file}"))?
            .with_context(|| format!("no export {file}"))?.1;
        let path = dir.join(&file);
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

fn check_context_suffix(report: &mut Report, requested: u64) {
    if requested == 0 { return; }
    let evidence = report.panels.iter().position(|p| p.id == "context_load");
    let error = match evidence {
        Some(index) if !report.panels[index].passes.is_empty() => report.panels[index].passes.iter()
            .find_map(|pass| crate::panels::context_load::validate_suffix(pass, requested).err()),
        _ => Some("server does not support context suffix: no measured context_load output".to_string()),
    };
    if let Some(error) = error {
        if let Some(index) = evidence {
            report.panels[index].status = crate::report::PanelStatus::Failed;
            report.panels[index].error = Some(error.clone());
        }
        report.status = RunStatus::Failed;
        report.error = Some(error);
    }
}

/// An older server may ignore unknown fields: require evidence of the configured run.
fn check_panel_config(report: &mut Report, configs: &crate::panel_config::PanelConfigs) {
    for (panel, config) in configs {
        if panel == "spec_lossless_set" {
            let warm = crate::panels::lossless::Options::parse(config).map(|o| o.warm).unwrap_or(false);
            let supported = !warm || report.panels.iter().find(|p| p.id == *panel).is_some_and(|p|
                !p.passes.is_empty() && p.passes.iter().all(|pass| pass["warm"] == true));
            if !supported {
                report.status = RunStatus::Failed;
                report.error = Some("server did not record requested warm spec_lossless_set".into());
            }
            continue;
        }
        if panel == "concurrency" {
            let sustained = crate::sustained::Options::parse(config).map(|o| o.sustained).unwrap_or(false);
            // Without the in-server panel the config is for the external sweep,
            // which validates its own output (it fails when no sustained point ran).
            let Some(evidence) = report.panels.iter().find(|p| &p.id == panel) else { continue };
            let points: Vec<&Value> = evidence.passes.iter()
                .filter_map(|pass| pass["points"].as_array()).flatten().collect();
            let supported = if sustained {
                // Every level the panel measured needs its sustained point and window.
                let levels: std::collections::BTreeSet<u64> = points.iter()
                    .filter(|p| p["phase"] != "sustained").filter_map(|p| p["c"].as_u64()).collect();
                !levels.is_empty() && levels.iter().all(|level| points.iter().any(|p|
                    p["phase"] == "sustained" && p["c"].as_u64() == Some(*level) && p["sustained"].is_object()))
            } else {
                !points.iter().any(|p| p["phase"] == "sustained")
            };
            if !supported {
                report.status = RunStatus::Failed;
                report.error = Some("server did not record requested concurrency panel-config settings".into());
            }
            continue;
        }
        let options = match panel.as_str() {
            "decode_content" => crate::panels::speed::DecodeOptions::parse(config).ok(),
            _ => None,
        };
        let supported = options.is_some_and(|options| report.panels.iter().find(|p| &p.id == panel).is_some_and(|p|
            !p.passes.is_empty() && p.passes.iter().all(|pass| {
                let rows = pass["rows"].as_array();
                pass.get("sampling") == Some(&options.sampling()) && rows.is_some_and(|rows|
                    !rows.is_empty() && options.contents.as_ref().is_none_or(|contents|
                        rows.len() == contents.len() && contents.iter().all(|content|
                            rows.iter().any(|r| r["content"].as_str() == Some(content.as_str())))))
            })));
        if !supported {
            report.status = RunStatus::Failed;
            report.error = Some(format!("server did not record requested {panel} panel-config settings"));
        }
    }
}

/// Starts a run on `options.url`, follows it to the end and writes the exports.
pub fn run(options: &RunOptions) -> Result<(Report, PathBuf)> {
    crate::panel_config::validate(&options.panel_config)?;
    let base = options.url.trim_end_matches('/').to_string();
    let agent = agent();
    let key = options.api_key.as_deref().map(cuteafd_api::openai::auth::ApiKey::new).transpose().map_err(anyhow::Error::msg)?;
    let client = crate::client::Client::new(&base, None, Default::default()).with_api_key(key);
    let prelude = client.warm_prelude(crate::sweep::TOKENS)?;
    let mut body = json!({});
    if let Some(profile) = &options.profile {
        body["profile"] = json!(profile);
    }
    if let Some(panels) = &options.panels {
        body["panels"] = json!(panels);
    }
    body["passes"] = Value::Object(options.passes.iter().map(|(k, v)| (k.clone(), json!(v))).collect());
    if let Some(fabric) = options.fabric { body["fabric"] = json!(fabric); }
    if let Some(baseline) = options.baseline {
        body["baseline"] = json!(baseline);
    }
    if let Some(set) = &options.lossless_set {
        body["lossless_set"] = json!(crate::panels::lossless::SetRequest::of(set));
    }
    if let Some(dump) = &options.lossless_dump { body["lossless_dump"] = json!(dump); }
    if let Some(n) = options.lossless_concurrency { body["lossless_concurrency"] = json!(n); }
    if let Some(lengths) = &options.context_lengths { body["context_lengths"] = json!(lengths); }
    if let Some(levels) = &options.context_levels { body["context_levels"] = json!(levels); }
    if let Some(suffix) = options.context_suffix { body["context_suffix"] = json!(suffix); }
    if !options.panel_config.is_empty() { body["panel_config"] = json!(options.panel_config); }
    let response = authorize(agent.post(&format!("{base}/v1/bench/runs")), &options.api_key)
        .send_json(body);
    let id = match response {
        Ok(response) => response.into_json::<Value>()?["id"].as_str().context("run id")?.to_string(),
        Err(ureq::Error::Status(code, response)) => {
            bail!("starting the run: HTTP {code}: {}", response.into_string().unwrap_or_default())
        }
        Err(error) => return Err(error).context("starting the run"),
    };
    if !options.quiet {
        eprintln!("run {id} on {base}");
    }
    let followed = follow(&base, &id, options.quiet, options.deadline, &options.api_key);
    if let Err(error) = &followed {
        if error.downcast_ref::<Overdue>().is_some() {
            let _ = authorize(agent.post(&format!("{base}/v1/bench/runs/{id}/cancel")), &options.api_key).call();
            return Err(followed.unwrap_err()).with_context(|| format!("run {id} on {base}"));
        }
    }
    let served = followed.as_ref().ok().and_then(|_| authorize(agent.get(&format!("{base}/v1/bench/runs/{id}")), &options.api_key).call().ok())
        .map(|response| response.into_json::<Report>().context("the finished report"));
    let report = match served {
        Some(report) => report?,
        // The server went away (it exited after or during the run): its history is
        // on this host when the server runs here (run.sh mounts the store).
        None => match crate::store::Store::open_read(&crate::store::default_dir()).ok().and_then(|s| s.load(&id).ok().flatten())
            .filter(|r| !matches!(r.status, RunStatus::Running | RunStatus::Queued)) {
            Some(report) => {
                eprintln!("warning: the server at {base} stopped answering; run {id} read from the local store");
                report
            }
            None => return Err(followed.err().unwrap_or_else(|| anyhow::anyhow!("the server stopped answering")))
                .with_context(|| format!("run {id} on {base}")),
        },
    };
    let dir = options.out.clone().unwrap_or_else(|| labeled_dir(&options.root, &report, options.label.as_deref()));
    // The report this returns is read the way every surface reads it, so the
    // run's own gate check and a run recovered from the local store after the
    // server went away cannot carry a verdict the data no longer earns.
    let mut report = crate::quality::under_current_rule(&report);
    anyhow::ensure!(report.prelude.as_ref() == Some(&prelude), "timed report omitted or changed this server's warm prelude receipt");
    check_context_suffix(&mut report, options.context_suffix.unwrap_or(0));
    check_panel_config(&mut report, &options.panel_config);
    let written = write_exports(&report, &dir, &options.export)?;
    if !options.quiet {
        if let Some(card) = report.baseline.as_ref().map(|b| &b.card) {
            for (label, point) in [("8K prefill", &card.prefill), ("8K w/tail", &card.prefill_tail)] {
                if let Some(point) = point {
                    let shape = point.shape.as_ref().map(|s| format!("; {} x {} + {} rows", s.full_units, s.unit_rows, s.tail_rows)).unwrap_or_default();
                    eprintln!("{label}: {} prompt tokens, {:.1} tok/s{shape}", point.prompt_tokens, point.tok_s);
                }
            }
        }
        for path in &written {
            eprintln!("wrote {}", path.display());
        }
    }
    match report.status {
        RunStatus::Done => {}
        RunStatus::Cancelled => bail!("the run was cancelled"),
        _ => bail!("the run ended {:?}: {}", report.status, report.error.clone().unwrap_or_default()),
    }
    Ok((report, dir))
}

/// A followed run passed its deadline.
#[derive(Debug, thiserror::Error)]
#[error("the run did not finish within {0} s")]
struct Overdue(u64);

/// How long a server may refuse connections before a followed run counts as lost.
const UNREACHABLE: Duration = Duration::from_secs(60);

/// Prints progress lines from the event stream until run `id` finishes.
fn follow(base: &str, id: &str, quiet: bool, deadline: Option<Duration>, api_key: &Option<String>) -> Result<()> {
    let started = std::time::Instant::now();
    let overdue = || deadline.filter(|d| started.elapsed() > *d).map(|d| Overdue(d.as_secs()));
    let events = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60)).build();
    let mut last_line = String::new();
    let mut unreachable: Option<std::time::Instant> = None;
    loop {
        let response = match authorize(events.get(&format!("{base}/v1/bench/events")), api_key).call() {
            Ok(response) => {
                unreachable = None;
                response
            }
            Err(error) => {
                // A dropped stream: check whether the run ended meanwhile, else reconnect,
                // giving up once the server has refused connections for a while.
                if finished(base, id, api_key)? {
                    return Ok(());
                }
                if let Some(overdue) = overdue() {
                    return Err(overdue.into());
                }
                let since = *unreachable.get_or_insert_with(std::time::Instant::now);
                if since.elapsed() > UNREACHABLE {
                    bail!("the server has not answered for {} s: {error}", UNREACHABLE.as_secs());
                }
                if !quiet {
                    eprintln!("event stream: {error}; reconnecting");
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };
        for line in BufReader::new(response.into_reader()).lines() {
            if let Some(overdue) = overdue() {
                return Err(overdue.into());
            }
            let Ok(line) = line else { break };
            let Some(data) = line.strip_prefix("data: ") else { continue };
            let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
            match event["type"].as_str() {
                Some("progress") if event["run"] == id && !quiet => {
                    let live = &event["live"];
                    let mut text = format!("[{:>3.0}%] {} · {} · ETA {}", 100.0 * event["fraction"].as_f64().unwrap_or(0.0),
                        event["title"].as_str().unwrap_or(""), event["label"].as_str().unwrap_or(""),
                        crate::render::seconds(event["eta_s"].as_f64().unwrap_or(0.0)));
                    if let Some(tok_s) = live["tok_s"].as_f64().filter(|v| *v > 0.0) {
                        text.push_str(&format!(" · {} tok/s", crate::render::rate(tok_s)));
                    }
                    if text != last_line {
                        eprintln!("{text}");
                        last_line = text;
                    }
                }
                Some("report") if event["run"] == id => {
                    let status = event["report"]["status"].as_str().unwrap_or("");
                    if !matches!(status, "running" | "queued") {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        if finished(base, id, api_key)? {
            return Ok(());
        }
    }
}

fn finished(base: &str, id: &str, api_key: &Option<String>) -> Result<bool> {
    let report: Value = match authorize(agent().get(&format!("{base}/v1/bench/runs/{id}")), api_key).call() {
        Ok(response) => response.into_json()?,
        Err(_) => return Ok(false),
    };
    Ok(!matches!(report["status"].as_str(), Some("running" | "queued")))
}

/// Cancels the server's active run.
pub fn cancel(url: &str, api_key: &Option<String>) -> Result<()> {
    let base = url.trim_end_matches('/');
    let status: Value = authorize(agent().get(&format!("{base}/v1/bench/status")), api_key).call()?.into_json()?;
    let Some(id) = status["active"]["id"].as_str() else { bail!("no run is active") };
    authorize(agent().post(&format!("{base}/v1/bench/runs/{id}/cancel")), api_key).call()?;
    eprintln!("cancelled {id}");
    Ok(())
}

/// Writes `text` to `path` only when it changed.
pub fn write_if_changed(path: &Path, text: &str) -> Result<bool> {
    write_bytes_if_changed(path, text.as_bytes())
}

/// Writes `bytes` to `path` only when they changed, so re-rendering a file
/// that is already current leaves it untouched.
pub fn write_bytes_if_changed(path: &Path, bytes: &[u8]) -> Result<bool> {
    if std::fs::read(path).ok().as_deref() == Some(bytes) {
        return Ok(false);
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    #[test]
    fn warm_lossless_requires_server_evidence_not_ignored_configuration() {
        let configs = std::collections::BTreeMap::from([("spec_lossless_set".into(), serde_json::json!({"warm":true}))]);
        let mut report = crate::sample::report(false);
        report.panels.retain(|p| p.id != "spec_lossless_set");
        report.panels.push(crate::report::PanelResult { id:"spec_lossless_set".into(),
            passes:vec![serde_json::json!({"gate":"pass"})], ..Default::default() });
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Failed);
        report.status = crate::report::RunStatus::Done;
        report.panels.last_mut().unwrap().passes[0]["warm"] = serde_json::json!(true);
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Done);
    }

    #[test]
    fn older_server_cannot_silently_ignore_sampled_panel_config() {
        let configs = std::collections::BTreeMap::from([("decode_content".into(),
            serde_json::json!({"contents":["code"],"temperature":1.0,"top_p":0.95,"seed":42}))]);
        let mut report = crate::sample::report(false);
        report.panels.retain(|p| p.id != "decode_content");
        report.panels.push(crate::report::PanelResult { id:"decode_content".into(),
            passes:vec![serde_json::json!({"rows":[{"content":"code"}]})], ..Default::default() });
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Failed);
        report.status = crate::report::RunStatus::Done;
        report.panels.last_mut().unwrap().passes[0]["sampling"] =
            serde_json::json!({"temperature":1.0,"top_p":0.95,"seed":42});
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Done);
    }

    #[test]
    fn sustained_concurrency_requires_server_evidence_not_ignored_configuration() {
        let configs = std::collections::BTreeMap::from([("concurrency".into(),
            serde_json::json!({"sustained":true,"multiplier":4}))]);
        let mut report = crate::sample::report(false);
        report.panels.retain(|p| p.id != "concurrency");
        report.panels.push(crate::report::PanelResult { id:"concurrency".into(),
            passes:vec![serde_json::json!({"points":[{"c":1},{"c":4}]})], ..Default::default() });
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Failed);
        report.status = crate::report::RunStatus::Done;
        report.error = None;
        report.panels.last_mut().unwrap().passes[0]["points"] = serde_json::json!([
            {"c":1},{"c":4},
            {"c":1,"phase":"sustained","sustained":{"credited_tokens":10.0,"elapsed_s":1.0}},
            {"c":4,"phase":"sustained","sustained":{"credited_tokens":10.0,"elapsed_s":1.0}}]);
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Done);
    }

    #[test]
    fn external_only_concurrency_config_skips_the_server_check() {
        let configs = std::collections::BTreeMap::from([("concurrency".into(),
            serde_json::json!({"sustained":true,"multiplier":4}))]);
        let mut report = crate::sample::report(false);
        report.panels.retain(|p| p.id != "concurrency");
        super::check_panel_config(&mut report, &configs);
        assert_eq!(report.status, crate::report::RunStatus::Done);
    }

    #[test]
    fn old_server_ignoring_suffix_cannot_return_a_passing_report() {
        let mut report = crate::sample::report(false);
        report.status = crate::report::RunStatus::Done;
        report.panels.push(crate::report::PanelResult { id: "context_load".into(),
            status: crate::report::PanelStatus::Done,
            passes: vec![serde_json::json!({"points": [{"status": "ok", "c": 1,
                "requests": [{"timing": {"prompt_tokens": 8149, "cached_tokens": 8092}}]}]})],
            ..Default::default() });
        super::check_context_suffix(&mut report, 512);
        assert_eq!(report.status, crate::report::RunStatus::Failed);
        assert_eq!(report.panel("context_load").unwrap().status, crate::report::PanelStatus::Failed);
        assert!(report.error.as_deref().unwrap().contains("server does not support context suffix"));
    }
    #[test]
    fn a_label_keeps_two_cards_of_one_checkpoint_apart() {
        let report = crate::sample::report(false);
        let root = std::path::Path::new("/r");
        let base = super::default_dir(root, &report);
        let a = super::labeled_dir(root, &report, Some("glm53f-exl3-min"));
        let b = super::labeled_dir(root, &report, Some("glm53f-bf16kda-c8"));
        assert_ne!(a, b);
        assert_eq!(a.parent(), base.parent());
        assert!(a.file_name().unwrap().to_string_lossy().ends_with("-glm53f-exl3-min"));
        assert_eq!(super::labeled_dir(root, &report, None), base);
    }
}
