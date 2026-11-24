//! Trace-owned Hy4-flash decode row geometries shared by admission and startup.
use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Hy4GraphLayout {
    #[serde(rename = "exl3-k4-mc-1rtx-0spark")]
    Exl3Min,
    #[serde(rename = "exl3-k4-mc-2rtx-0spark")]
    Exl3Max,
    #[serde(rename = "exl3-k4-mc-1rtx-spark")]
    Exl3MinSpark,
    #[serde(rename = "exl3-k4-mc-2rtx-spark")]
    Exl3MaxSpark,
}

impl Hy4GraphLayout {
    pub fn resolve(ranks: usize, sparks: usize) -> Option<Self> {
        match (ranks, sparks) {
            (1, 0) => Some(Self::Exl3Min),
            (2, 0) => Some(Self::Exl3Max),
            (1, s) if s > 0 => Some(Self::Exl3MinSpark),
            (2, s) if s > 0 => Some(Self::Exl3MaxSpark),
            _ => None,
        }
    }

    pub fn ranks(self) -> usize {
        match self {
            Self::Exl3Min | Self::Exl3MinSpark => 1,
            Self::Exl3Max | Self::Exl3MaxSpark => 2,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Hy4GraphSection {
    pub layer: usize,
    pub section: String,
    pub capture_bytes: u64,
    pub replay_hits: u64,
}

#[derive(Debug, Deserialize)]
pub struct Hy4GraphRow {
    pub rows: usize,
    pub pin: bool,
    pub replay_hits: u64,
    pub capture_bytes: u64,
    pub sections: Vec<Hy4GraphSection>,
}

#[derive(Debug, Deserialize)]
pub struct Hy4GraphRank {
    pub rank: usize,
    pub reserve_base_bytes: u64,
    pub reserve_extra_bytes: u64,
    pub entries: Vec<Hy4GraphRow>,
}

#[derive(Debug, Deserialize)]
pub struct Hy4GraphTrace {
    pub schema_version: u32,
    pub family: String,
    pub layout: Hy4GraphLayout,
    pub ranks: Vec<Hy4GraphRank>,
}

#[derive(Deserialize)]
struct List {
    layouts: Vec<Hy4GraphTrace>,
}

pub fn prewarm(layout: Option<Hy4GraphLayout>) -> Option<&'static Hy4GraphTrace> {
    static LIST: OnceLock<List> = OnceLock::new();
    let layout = layout?;
    LIST.get_or_init(|| {
        let list: List = serde_json::from_str(include_str!("hy4_prewarm.json"))
            .expect("Hy4-flash trace inventory");
        for trace in &list.layouts {
            assert_eq!(trace.schema_version, 1);
            assert_eq!(trace.family, "hy4_flash");
            assert_eq!(trace.ranks.len(), trace.layout.ranks());
            for (rank, data) in trace.ranks.iter().enumerate() {
                assert_eq!(data.rank, rank);
                assert_eq!(data.reserve_extra_bytes, 2 << 20);
                assert!(!data.entries.is_empty());
                for row in &data.entries {
                    assert!(row.rows > 0 && row.replay_hits > 0);
                    assert_eq!(
                        row.capture_bytes,
                        row.sections.iter().map(|s| s.capture_bytes).sum::<u64>()
                    );
                }
                assert_eq!(
                    data.reserve_base_bytes,
                    data.entries
                        .iter()
                        .map(|e| e.capture_bytes)
                        .sum::<u64>()
                        .div_ceil(2 << 20)
                        * (2 << 20)
                );
                assert_eq!(
                    data.entries
                        .iter()
                        .map(|e| (e.rows, e.pin))
                        .collect::<Vec<_>>(),
                    trace.ranks[0]
                        .entries
                        .iter()
                        .map(|e| (e.rows, e.pin))
                        .collect::<Vec<_>>()
                );
            }
        }
        list
    })
    .layouts
    .iter()
    .find(|trace| trace.layout == layout)
}

pub fn reserve(layout: Option<Hy4GraphLayout>, rank: usize) -> (u64, u64) {
    prewarm(layout)
        .and_then(|trace| trace.ranks.get(rank))
        .map_or((0, 0), |data| (data.reserve_base_bytes, data.reserve_extra_bytes))
}

/// Two donor-off, no-prewarm C1 passes (short and 32K). W1 uses the larger
/// of charged and settled physical bytes. Step growth is the trace-priced
/// draft + verify bundle; capture growth is the largest single capture in
/// those C1 traces, not the sum across the bundle.
pub fn draft_working_set(layout: Option<Hy4GraphLayout>, rank: usize)
    -> Option<crate::placement::graph_reserve::WorkingSet> {
    let (w1, step_growth, capture_growth) = match (layout, rank) {
        (Some(Hy4GraphLayout::Exl3Min), 0) => (86, 22, 4),
        (Some(Hy4GraphLayout::Exl3Max), 0) => (120, 24, 2),
        (Some(Hy4GraphLayout::Exl3Max), 1) => (118, 22, 2),
        _ => return None,
    };
    Some(crate::placement::graph_reserve::WorkingSet {
        w1: w1 << 20, step_growth: step_growth << 20, capture_growth: capture_growth << 20,
    })
}

pub fn reserve_policy(layout: Option<Hy4GraphLayout>, rank: usize, dfly2: bool)
    -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    if let Some(measured) = dfly2.then(|| draft_working_set(layout, rank)).flatten() {
        return crate::placement::graph_reserve::current_working_set(rank, measured);
    }
    let (base, tail) = reserve(layout, rank);
    crate::placement::graph_reserve::current(rank, false, base, tail)
}

/// Select complete rows against all devices; never borrow an untraced layout.
pub fn selected_rows(layout: Option<Hy4GraphLayout>, max_rows: usize) -> anyhow::Result<Vec<(usize, bool)>> {
    let Some(trace) = prewarm(layout) else {
        return Ok(Vec::new());
    };
    let candidates = trace.ranks[0]
        .entries
        .iter()
        .enumerate()
        .filter_map(|(i, row)| (row.rows <= max_rows).then_some(i))
        .collect::<Vec<_>>();
    let bases = trace
        .ranks
        .iter()
        .map(|rank| {
            crate::placement::graph_reserve::current(
                rank.rank,
                false,
                rank.reserve_base_bytes,
                rank.reserve_extra_bytes,
            )
            .map(|p| p.base_bytes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pins = candidates
        .iter()
        .map(|&i| trace.ranks[0].entries[i].pin)
        .collect::<Vec<_>>();
    let selected = crate::placement::graph_reserve::prewarm_selection(&pins, &bases, |selected, bases| {
        trace.ranks.iter().zip(bases).all(|(rank, &base)| {
            let bytes = selected
                .iter()
                .map(|&i| rank.entries[candidates[i]].capture_bytes)
                .sum::<u64>();
            bytes.div_ceil(2 << 20) * (2 << 20) + rank.reserve_extra_bytes <= base
        })
    });
    Ok(selected
        .into_iter()
        .map(|i| {
            let row = &trace.ranks[0].entries[candidates[i]];
            (row.rows, row.pin)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_resolution_distinguishes_rank_and_spark_shapes() {
        assert_eq!(Hy4GraphLayout::resolve(1, 0), Some(Hy4GraphLayout::Exl3Min));
        assert_eq!(Hy4GraphLayout::resolve(2, 0), Some(Hy4GraphLayout::Exl3Max));
        assert_eq!(Hy4GraphLayout::resolve(1, 4), Some(Hy4GraphLayout::Exl3MinSpark));
        assert_eq!(Hy4GraphLayout::resolve(2, 6), Some(Hy4GraphLayout::Exl3MaxSpark));
        assert_eq!(Hy4GraphLayout::resolve(0, 0), None);
        assert_eq!(Hy4GraphLayout::resolve(3, 0), None);
        assert_eq!(Hy4GraphLayout::Exl3Min.ranks(), 1);
        assert_eq!(Hy4GraphLayout::Exl3Max.ranks(), 2);
        assert_eq!(Hy4GraphLayout::Exl3MinSpark.ranks(), 1);
        assert_eq!(Hy4GraphLayout::Exl3MaxSpark.ranks(), 2);
    }

    #[test]
    fn draft_working_sets_are_rank_local_and_do_not_cover_spark_layouts() {
        let min = Some(Hy4GraphLayout::Exl3Min);
        let max = Some(Hy4GraphLayout::Exl3Max);
        for (layout, rank, w1, step, capture, soft, hard) in [
            (min, 0, 86, 22, 4, 182, 190),
            (max, 0, 120, 24, 2, 216, 220),
            (max, 1, 118, 22, 2, 214, 218),
        ] {
            let measured = draft_working_set(layout, rank).unwrap();
            assert_eq!((measured.w1, measured.step_growth, measured.capture_growth), (w1 << 20, step << 20, capture << 20));
            let policy = reserve_policy(layout, rank, true).unwrap();
            assert_eq!((policy.base_bytes, policy.bytes), (soft << 20, hard << 20));
            assert_eq!(policy.growth_headroom, Some((2 * capture) << 20));
        }
        assert_eq!(draft_working_set(min, 1), None);
        assert_eq!(draft_working_set(Some(Hy4GraphLayout::Exl3MinSpark), 0), None);
        assert!(reserve_policy(min, 0, false).unwrap().growth_headroom.is_none());
        assert_eq!(reserve_policy(Some(Hy4GraphLayout::Exl3MinSpark), 0, true).unwrap().bytes, 0);
    }

    #[test]
    fn untraced_layout_has_no_pins_no_charge_and_no_borrowed_rows() {
        assert!(prewarm(None).is_none());
        assert_eq!(reserve(None, 0), (0, 0));
        assert!(selected_rows(None, 64).unwrap().is_empty());
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3Max), 0), (0, 0));
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3MinSpark), 0), (0, 0));
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3MaxSpark), 0), (0, 0));
    }

    #[test]
    fn traced_exl3_min_layout_validates_and_selects_pinned_and_replayed_rows() {
        let trace = prewarm(Some(Hy4GraphLayout::Exl3Min)).expect("exl3-k4-mc-1rtx-0spark traced");
        assert_eq!(trace.ranks.len(), 1);
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3Min), 0), (85_983_232, 2 << 20));
        assert_eq!(reserve(Some(Hy4GraphLayout::Exl3Min), 1), (0, 0));
        let selected = selected_rows(Some(Hy4GraphLayout::Exl3Min), 64).unwrap();
        assert_eq!(
            selected,
            vec![
                (1, true),
                (4, true),
                (2, true),
                (8, true),
                (16, true),
                (3, false),
                (7, false),
                (14, false),
                (15, false),
            ]
        );
    }
}
