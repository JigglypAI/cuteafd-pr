//! The one KV pool policy (`cuteafd_core::memory_policy`): the layout is
//! sized for `KV_TARGET_TOKENS` (2M tokens on cards above 32 GiB, 1M at 32 GiB
//! or less by default); the pool then grows into what
//! the fixed layout leaves, up to `KV_MAX_TOKENS`; below `KV_MIN_TOKENS` the
//! launch is refused.
use cuteafd_core::memory_policy::{MemoryPolicy, SMALL_CARD_BYTES};

/// Agentic context floor: the default KV min.
pub use cuteafd_core::memory_policy::AGENTIC_FLOOR_TOKENS;

/// Physical card identity and an independent operator memory ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatorCard {
    pub physical_bytes: u64,
    pub logical_bytes: u64,
}

impl CoordinatorCard {
    /// Offline caps are not card identities. Absent physical facts, assume
    /// the architecture's default PRO class, and report that assumption.
    pub fn planned(logical_bytes: u64, physical_sms: Option<u32>) -> Self {
        let class = physical_sms.and_then(|sms| super::inventory::ARCH_CONTEXTS.iter()
            .find(|class| class.arch == "sm_120" && class.sms == sms));
        let physical_bytes = class.map_or(super::inventory::PRO_TOTAL_BYTES, |class|
            if class.total_bytes == 0 { SMALL_CARD_BYTES } else { class.total_bytes });
        Self { physical_bytes, logical_bytes }
    }


}

/// Geometry and operator inputs, before the shared placement preparation resolves policy.
#[derive(Debug, Clone, Copy)]
pub struct PoolFacts<'a> {
    /// Logical per-card ceilings; a capped PRO intentionally simulates a small card.
    pub card_bytes: &'a [u64],
    pub context: super::planner::Context,
    pub requested: Option<u64>,
    pub target: Option<u64>,
    pub unit_rows: u64,
    /// Exclusive of private tails; an encoding limit, never a memory reservation.
    pub encoding_limit: Option<u64>,
}

impl<'a> PoolFacts<'a> {
    pub fn new(card_bytes: &'a [u64], context: impl Into<super::planner::Context>,
        requested: Option<u64>, unit_rows: u64) -> Self {
        Self { card_bytes, context: context.into(), requested, target: None,
            unit_rows, encoding_limit: None }
    }
}

/// The only policy resolution entry point for plan and runtime preparation.
pub fn prepare_pool(facts: PoolFacts<'_>) -> Result<PoolPolicy, cuteafd_core::memory_policy::PolicyError> {
    let mut policy = MemoryPolicy::current();
    policy.kv_target = facts.target.or(policy.kv_target);
    let smallest = facts.card_bytes.iter().copied().min().unwrap_or(u64::MAX);
    let bounds = policy.kv(smallest, facts.context.maximum)?;
    let raw_requested = PoolPolicy::normalize_request(facts.requested);
    let requested = raw_requested.map(|tokens| {
        // A raw pin below context must retain its typed refusal, not round into validity.
        if tokens < facts.context.required() { return Ok(tokens); }
        let rows = facts.unit_rows.max(1);
        tokens.div_ceil(rows).checked_mul(rows).ok_or(
            cuteafd_core::memory_policy::PolicyError::PoolAlignment { requested: tokens, unit_rows: rows })
    }).transpose()?;
    Ok(PoolPolicy {
        context: facts.context, compiled_context: facts.context.maximum, raw_requested, requested,
        target: bounds.target.max(facts.context.required()),
        floor: requested.unwrap_or(bounds.min).max(facts.context.required()),
        ceiling: facts.encoding_limit.unwrap_or(u64::MAX),
        max: requested.unwrap_or(bounds.max), unit_rows: facts.unit_rows,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoolPolicy {
    pub(super) context: super::planner::Context,
    /// Immutable workspace extent, independent of the pool fit floor.
    pub(super) compiled_context: u64,
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub(super) requested: Option<u64>,
    pub(super) raw_requested: Option<u64>,
    /// The layout (onboarding, buffers) is sized for this pool.
    pub(super) target: u64,
    /// Admission refuses below this (`KV_MIN_TOKENS` and any pinned context).
    pub(super) floor: u64,
    /// The largest pool the build can index; a fixed onboard fills up to it.
    pub(super) ceiling: u64,
    /// After an automatic layout is fixed for the target, the pool grows into
    /// what is left up to this (`KV_MAX_TOKENS`, default the target).
    pub(super) max: u64,
    pub(super) unit_rows: u64,
}

impl PoolPolicy {
    pub fn context(&self) -> super::planner::Context { self.context }
    pub fn compiled_context(&self) -> u64 { self.compiled_context }
    pub fn requested(&self) -> Option<u64> { self.requested }
    pub(crate) fn raw_requested(&self) -> Option<u64> { self.raw_requested }
    pub fn target(&self) -> u64 { self.target }
    pub fn floor(&self) -> u64 { self.floor }
    pub fn ceiling(&self) -> u64 { self.ceiling }
    pub fn max(&self) -> u64 { self.max }
    pub fn unit_rows(&self) -> u64 { self.unit_rows }

    /// Both CLI zero and an omitted pool request select automatic admission.
    pub fn normalize_request(requested: Option<u64>) -> Option<u64> {
        requested.filter(|&tokens| tokens > 0)
    }

    /// Finite index/curve domain, not an allocation or a layout reservation.
    /// Layout selection still uses `target`; only the selected pool's owners
    /// are charged when it grows into the memory left after onboarding.
    pub fn maximum_tokens(&self) -> u64 {
        self.requested.unwrap_or(self.max.max(self.target)).min(self.ceiling)
    }

    /// Diagnostic capacity reports use the layout target or strict request,
    /// never the planner's potentially much larger index domain.
    pub fn capacity_policy(&self, concurrency: u32) -> cuteafd_core::serving_capacity::CapacityPolicy {
        cuteafd_core::serving_capacity::CapacityPolicy {
            concurrency, target_pool_tokens: self.target,
            max_context_tokens: Some(self.compiled_context),
            pool_tokens: Some(self.requested.unwrap_or(self.target)), ..Default::default()
        }
    }

    /// `card_bytes` are the coordinator GPUs' totals; `context` the compiled
    /// serving context; `requested` 0 or `None` means automatic. KV target,
    /// min and max come from the installed memory policy.
    pub fn bounds(&self) -> cuteafd_core::memory_policy::KvBounds {
        cuteafd_core::memory_policy::KvBounds { target: self.target, min: self.floor, max: self.max }
    }

    /// Pin a selected pool for re-admission without inventing an operator request.
    pub(super) fn pin_selected(&mut self, tokens: u64) {
        self.target = tokens;
        self.floor = tokens;
        self.ceiling = tokens;
        self.max = tokens;
    }

    pub fn validate_explicit(&self) -> Result<(), super::PlacementError> {
        if let Some(requested) = self.requested.filter(|&n| n > 0 && n < self.context.required()) {
            return Err(super::PlacementError::RequestedBelowContext {
                requested, floor: self.context.required(),
            });
        }
        Ok(())
    }

    pub(crate) fn wanted_units(&self) -> u64 {
        self.requested.unwrap_or(self.target).div_ceil(self.unit_rows.max(1))
    }

    pub(crate) fn ceiling_units(&self) -> u64 {
        self.ceiling / self.unit_rows.max(1)
    }

    /// Automatic growth bound: KV max within the index limit.
    pub(crate) fn max_units(&self) -> u64 {
        self.max.min(self.ceiling) / self.unit_rows.max(1)
    }
}

#[cfg(test)]
fn legacy_fixture_pool(cards: &[u64], context: u64, requested: Option<u64>, unit_rows: u64) -> PoolPolicy {
    prepare_pool(PoolFacts::new(cards, context, requested, unit_rows)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_uses_one_shared_target_precedence_and_small_card_default() {
        MemoryPolicy { kv_target: Some(3 << 20), kv_max: Some(100_000_000), ..Default::default() }.scoped(|| {
            let mut facts = PoolFacts::new(&[32 << 30], super::super::planner::Context::automatic(262_144), None, 64);
            assert_eq!(prepare_pool(facts).unwrap().target(), 3 << 20);
            facts.target = Some(4 << 20);
            let pool = prepare_pool(facts).unwrap();
            assert_eq!(pool.target(), 4 << 20);
            assert_eq!(pool.maximum_tokens(), 100_000_000);
            facts.requested = Some(524_288);
            let explicit = prepare_pool(facts).unwrap();
            assert_eq!(explicit.maximum_tokens(), 524_288);
            assert_eq!(explicit.capacity_policy(16).pool_tokens, Some(524_288));
        });
        MemoryPolicy::default().scoped(|| {
            for cards in [&[32 << 30][..], &[96 << 30, 32 << 30][..]] {
                assert_eq!(prepare_pool(PoolFacts::new(cards, 262_144, None, 64)).unwrap().target(), 1 << 20);
            }
            assert_eq!(prepare_pool(PoolFacts::new(&[96 << 30], 262_144, None, 64)).unwrap().target(), 2 << 20);
        });
    }

    #[test]
    fn maximum_extent_includes_growth_without_changing_the_layout_target() {
        let defaults = legacy_fixture_pool(&[96 << 30], 262_144, None, 256);
        assert_eq!(defaults.maximum_tokens(), defaults.target);
        let growing = MemoryPolicy { kv_max: Some(100_000_000), ..Default::default() }.scoped(||
            legacy_fixture_pool(&[96 << 30], 262_144, None, 256));
        assert_eq!(growing.maximum_tokens(), 100_000_000);
        assert_eq!((growing.target, growing.floor, growing.compiled_context),
            (defaults.target, defaults.floor, defaults.compiled_context));
        assert_eq!(growing.wanted_units(), defaults.wanted_units());
        assert_eq!(growing.max_units(), 100_000_000 / 256);
    }

    #[test]
    fn explicit_pool_extent_is_strict_and_zero_is_automatic() {
        MemoryPolicy { kv_target: Some(1 << 20), kv_max: Some(100_000_000),
            ..Default::default() }.scoped(|| {
            for requested in [Some(524_288), Some(4 << 20)] {
                let pool = legacy_fixture_pool(&[96 << 30], 262_144, requested, 256);
                assert_eq!(pool.maximum_tokens(), requested.unwrap());
                assert_eq!(pool.max, requested.unwrap());
                assert_eq!(pool.wanted_units(), requested.unwrap() / 256);
            }
            assert_eq!(legacy_fixture_pool(&[96 << 30], 262_144, Some(0), 256),
                legacy_fixture_pool(&[96 << 30], 262_144, None, 256));
        });
    }

    #[test]
    fn explicit_pool_rounds_once_and_preserves_raw_context_refusals() {
        let pool = prepare_pool(PoolFacts::new(&[96 << 30], 4096, Some(4097), 256)).unwrap();
        assert_eq!(pool.requested(), Some(4352));
        assert_eq!(pool.raw_requested(), Some(4097));
        assert_eq!(pool.maximum_tokens(), 4352);
        assert_eq!(pool.capacity_policy(16).pool_tokens, Some(4352));
        let short = prepare_pool(PoolFacts::new(&[96 << 30], 4096, Some(4095), 256)).unwrap();
        assert_eq!(short.requested(), Some(4095));
        assert!(short.validate_explicit().is_err());
        assert!(matches!(prepare_pool(PoolFacts::new(&[96 << 30], 4096, Some(u64::MAX), 256)),
            Err(cuteafd_core::memory_policy::PolicyError::PoolAlignment { .. })));
    }

    #[test]
    fn capped_pro_uses_the_same_small_card_pool_as_a_logical_simulation() {
        let planned = CoordinatorCard::planned(32 << 30, Some(188));
        let sampled = CoordinatorCard { physical_bytes: super::super::inventory::PRO_TOTAL_BYTES,
            logical_bytes: 32 << 30 };
        assert_eq!(planned, sampled);
        assert_eq!(prepare_pool(PoolFacts::new(&[planned.logical_bytes], 262_144, None, 64)).unwrap().maximum_tokens(), 1 << 20);
        assert_eq!(prepare_pool(PoolFacts::new(&[planned.logical_bytes], 262_144, Some(0), 64)).unwrap().maximum_tokens(),
            prepare_pool(PoolFacts::new(&[sampled.logical_bytes], 262_144, None, 64)).unwrap().maximum_tokens());
        assert_eq!(CoordinatorCard::planned(32 << 30, None), planned);
        assert_eq!(prepare_pool(PoolFacts::new(&[28 << 30], 262_144, None, 64)).unwrap().maximum_tokens(), 1 << 20);
    }

    #[test]
    fn targets_follow_the_smallest_card_and_the_context() {
        let gib = 1u64 << 30;
        assert_eq!(legacy_fixture_pool(&[96 * gib], 131_072, None, 256).target, 2 << 20);
        assert_eq!(legacy_fixture_pool(&[96 * gib, 32 * gib], 131_072, None, 256).target, 1 << 20);
        assert_eq!(legacy_fixture_pool(&[32 * gib], 1 << 21, None, 256).target, 1 << 21);
        let explicit = legacy_fixture_pool(&[96 * gib], 131_072, Some(32_768), 256);
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 131_072));
        assert_eq!(explicit.validate_explicit(), Err(super::super::PlacementError::RequestedBelowContext {
            requested: 32_768, floor: 131_072,
        }));
        assert!(legacy_fixture_pool(&[96 * gib], 32_768, Some(32_768), 256)
            .validate_explicit().is_ok());
        assert_eq!(legacy_fixture_pool(&[96 * gib], 131_072, Some(0), 256).floor, 131_072);
        assert_eq!(legacy_fixture_pool(&[96 * gib], 131_072, Some(0), 256).requested, None);

    }
}
