//! Warmed verify-only exact-shape timing: a benchmark probe that runs chosen
//! per-request verify row shapes through the serving decode path (captured
//! graphs, the family's own bucket padding) on cold scratch sequences, then
//! resets them. Nothing is committed, retained or emitted.
use serde::{Deserialize, Serialize};

/// Most shapes one probe times.
pub const MAX_SHAPES: usize = 32;
/// Most concurrent requests (verify sequences) of one shape.
pub const MAX_SEQUENCES: usize = 64;
/// Most rows one request of a shape verifies.
pub const MAX_ROWS: usize = 128;

/// One verify step's shape: each request's rows (the list's length is the
/// concurrency), optionally padded to a forced bucket at least the natural one.
/// Text form: comma-separated `ROWS` or `ROWSxCOUNT` groups, then `@BUCKET`,
/// e.g. `4x8`, `5,4x7`, `4x8@64`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct VerifyShape {
    pub rows: Vec<usize>,
    pub bucket: Option<usize>,
}

impl VerifyShape {
    pub fn real_rows(&self) -> usize { self.rows.iter().sum() }
}

impl TryFrom<String> for VerifyShape {
    type Error = String;
    fn try_from(text: String) -> Result<Self, String> { text.parse() }
}

impl From<VerifyShape> for String {
    fn from(shape: VerifyShape) -> String { shape.to_string() }
}

impl std::str::FromStr for VerifyShape {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, String> {
        let (groups, bucket) = match text.trim().split_once('@') {
            Some((groups, bucket)) => (groups, Some(bucket.trim().parse::<usize>()
                .map_err(|_| format!("verify shape {text:?}: bucket {bucket:?} is not a number"))?)),
            None => (text.trim(), None),
        };
        let number = |part: &str| part.trim().parse::<usize>()
            .map_err(|_| format!("verify shape {text:?}: {part:?} is not a number"));
        let mut rows = Vec::new();
        for group in groups.split(',') {
            let (value, count) = match group.split_once('x') {
                Some((value, count)) => (number(value)?, number(count)?),
                None => (number(group)?, 1),
            };
            if !(1..=MAX_ROWS).contains(&value) || count == 0 {
                return Err(format!("verify shape {text:?}: rows per request are 1..={MAX_ROWS}, counts at least 1"));
            }
            rows.extend(std::iter::repeat_n(value, count));
            if rows.len() > MAX_SEQUENCES {
                return Err(format!("verify shape {text:?}: at most {MAX_SEQUENCES} requests"));
            }
        }
        let shape = VerifyShape { rows, bucket };
        if bucket.is_some_and(|bucket| bucket < shape.real_rows()) {
            return Err(format!("verify shape {text:?}: bucket below its {} real rows", shape.real_rows()));
        }
        Ok(shape)
    }
}

impl std::fmt::Display for VerifyShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut groups = Vec::new();
        for &rows in &self.rows {
            match groups.last_mut() {
                Some((value, count)) if *value == rows => *count += 1,
                _ => groups.push((rows, 1usize)),
            }
        }
        let text = groups.iter().map(|&(value, count)| if count == 1 { value.to_string() } else { format!("{value}x{count}") })
            .collect::<Vec<_>>().join(",");
        match self.bucket {
            Some(bucket) => write!(f, "{text}@{bucket}"),
            None => f.write_str(&text),
        }
    }
}

fn default_context() -> usize { 1024 }
fn default_warmups() -> usize { 3 }
fn default_repeats() -> usize { 20 }

/// What a verify-timing probe runs: every shape at `context` tokens of
/// prefilled history per request, `warmups` untimed then `repeats` timed steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyTiming {
    pub shapes: Vec<VerifyShape>,
    #[serde(default = "default_context")]
    pub context: usize,
    #[serde(default = "default_warmups")]
    pub warmups: usize,
    /// Zero runs the warm-up alone (a pre-window pass that captures the graphs).
    #[serde(default = "default_repeats")]
    pub repeats: usize,
}

impl VerifyTiming {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!((1..=MAX_SHAPES).contains(&self.shapes.len()), "verify timing takes 1..={MAX_SHAPES} shapes");
        anyhow::ensure!((16..=131_072).contains(&self.context), "verify timing context is 16..=131072 tokens");
        anyhow::ensure!(self.warmups <= 50 && self.repeats <= 500, "verify timing takes at most 50 warm-ups and 500 repeats");
        anyhow::ensure!(self.warmups + self.repeats > 0, "verify timing runs at least one step per shape");
        Ok(())
    }

    /// The scratch sequences the probe admits: the widest shape's.
    pub fn sequences(&self) -> usize { self.shapes.iter().map(|s| s.rows.len()).max().unwrap_or(0) }

    /// Rows past the context each sequence may write: the most any shape gives it.
    pub fn max_rows(&self) -> usize { self.shapes.iter().flat_map(|s| s.rows.iter().copied()).max().unwrap_or(0) }

    /// Prompt tokens the probe needs: sequence `i` takes `context` tokens from offset
    /// `i`, then the following tokens as its verify rows.
    pub fn prompt_tokens(&self) -> usize { self.context + self.sequences() + self.max_rows() }
}

/// Graph cache activity over a sample's timed steps (every device, every cycle).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphDelta {
    /// Process-wide CUDA graph captures begun.
    pub captures: u64,
    pub misses: u64,
    pub eager_runs: u64,
}

/// One shape's timing.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VerifySample {
    pub shape: String,
    /// R: the step's real rows.
    pub real_rows: usize,
    /// The bucket serving pads R to.
    pub natural_bucket: usize,
    /// B: the rows the step ran.
    pub bucket: usize,
    /// Per request: its history length before the verify rows.
    pub contexts: Vec<usize>,
    /// Graph replay was ready (startup prewarm done, graphs enabled).
    pub replay_ready: bool,
    /// Ready, and the timed steps captured nothing and ran nothing eagerly.
    pub graphed: bool,
    pub graph: GraphDelta,
    pub warmups: usize,
    /// Synchronized verify-step milliseconds, one per timed repeat.
    pub verify_ms: Vec<f64>,
    /// Synchronized token-selection milliseconds after each verify.
    pub select_ms: Vec<f64>,
}

impl VerifySample {
    pub fn verify_median_ms(&self) -> Option<f64> { median(&self.verify_ms) }
    pub fn select_median_ms(&self) -> Option<f64> { median(&self.select_ms) }
}

pub fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => None,
        n if n % 2 == 1 => Some(sorted[n / 2]),
        n => Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VerifyTimingRecord {
    pub family: String,
    pub timing_scope: String,
    pub samples: Vec<VerifySample>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_parse_round_trip_and_refuse_bad_text() {
        for (text, rows, bucket) in [("4x8", vec![4; 8], None), ("5,4x7", [vec![5], vec![4; 7]].concat(), None),
            ("4x8@64", vec![4; 8], Some(64)), ("8", vec![8], None)] {
            let shape: VerifyShape = text.parse().unwrap();
            assert_eq!((shape.rows.clone(), shape.bucket), (rows, bucket), "{text}");
            assert_eq!(shape.to_string(), text);
            let json = serde_json::to_value(&shape).unwrap();
            assert_eq!(json, serde_json::json!(text));
            assert_eq!(serde_json::from_value::<VerifyShape>(json).unwrap(), shape);
        }
        assert_eq!("5,4x7".parse::<VerifyShape>().unwrap().real_rows(), 33);
        for bad in ["", "0x8", "4x0", "4x8@31", "a", "4x65", "129", "4x8@b", "4,,4"] {
            assert!(bad.parse::<VerifyShape>().is_err(), "{bad}");
        }
    }

    #[test]
    fn options_default_validate_and_size_the_prompt() {
        let timing: VerifyTiming = serde_json::from_value(serde_json::json!({"shapes": ["4x8", "5,4x7", "8x8"]})).unwrap();
        assert_eq!((timing.context, timing.warmups, timing.repeats), (1024, 3, 20));
        timing.validate().unwrap();
        assert_eq!((timing.sequences(), timing.max_rows(), timing.prompt_tokens()), (8, 8, 1040));
        for bad in [serde_json::json!({"shapes": []}), serde_json::json!({"shapes": ["4x8"], "context": 8}),
            serde_json::json!({"shapes": ["4x8"], "warmups": 0, "repeats": 0})] {
            let timing: VerifyTiming = serde_json::from_value(bad.clone()).unwrap();
            assert!(timing.validate().is_err(), "{bad}");
        }
        assert!(serde_json::from_value::<VerifyTiming>(serde_json::json!({"shapes": ["4x8"], "typo": 1})).is_err());
    }

    #[test]
    fn medians_handle_even_and_empty_samples() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
    }
}
