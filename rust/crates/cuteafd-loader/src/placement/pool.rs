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

    pub fn maximum_pool(cards: &[Self], context: u64, requested: Option<u64>, page_rows: u64) -> u64 {
        let physical = cards.iter().map(|card| card.physical_bytes).collect::<Vec<_>>();
        PoolPolicy::resolve(&physical, context, requested, page_rows).maximum_tokens()
    }
}

/// Geometry and operator inputs, before the shared placement preparation resolves policy.
#[derive(Debug, Clone, Copy)]
pub struct PoolFacts<'a> {
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
pub fn prepare_pool(facts: PoolFacts<'_>) -> PoolPolicy {
    let policy = MemoryPolicy::current();
    let requested = PoolPolicy::normalize_request(facts.requested);
    let smallest = facts.card_bytes.iter().copied().min().unwrap_or(u64::MAX);
    let default = if smallest <= SMALL_CARD_BYTES { cuteafd_core::memory_policy::SMALL_CARD_KV_TARGET_TOKENS }
        else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS };
    let target = facts.target.or(policy.kv_target).unwrap_or(default.max(facts.context.maximum))
        .max(facts.context.required());
    let minimum = policy.kv_min.unwrap_or(facts.context.maximum.min(AGENTIC_FLOOR_TOKENS).min(target));
    PoolPolicy {
        context: facts.context, compiled_context: facts.context.maximum, requested, target,
        floor: requested.unwrap_or(minimum).max(facts.context.required()),
        ceiling: facts.encoding_limit.unwrap_or(u64::MAX),
        max: requested.unwrap_or_else(|| policy.kv_max.unwrap_or(target).max(target)),
        unit_rows: facts.unit_rows,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoolPolicy {
    pub context: super::planner::Context,
    /// Immutable workspace extent, independent of the pool fit floor.
    pub compiled_context: u64,
    /// An explicit pool is strict: admission fails rather than shrink it.
    pub requested: Option<u64>,
    /// The layout (onboarding, buffers) is sized for this pool.
    pub target: u64,
    /// Admission refuses below this (`KV_MIN_TOKENS` and any pinned context).
    pub floor: u64,
    /// The largest pool the build can index; a fixed onboard fills up to it.
    pub ceiling: u64,
    /// After an automatic layout is fixed for the target, the pool grows into
    /// what is left up to this (`KV_MAX_TOKENS`, default the target).
    pub max: u64,
    pub unit_rows: u64,
}

impl PoolPolicy {
    pub fn context(&self) -> super::planner::Context { self.context }
    pub fn compiled_context(&self) -> u64 { self.compiled_context }
    pub fn requested(&self) -> Option<u64> { self.requested }
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
    pub fn resolve(card_bytes: &[u64], context: u64, requested: Option<u64>, unit_rows: u64) -> Self {
        let requested = Self::normalize_request(requested);
        let smallest = card_bytes.iter().copied().min().unwrap_or(u64::MAX);
        let policy = MemoryPolicy::current();
        let target_default = if smallest <= SMALL_CARD_BYTES { cuteafd_core::memory_policy::SMALL_CARD_KV_TARGET_TOKENS }
            else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS };
        let target = policy.kv_target.unwrap_or(target_default.max(context));
        let minimum = policy.kv_min.unwrap_or(context.min(AGENTIC_FLOOR_TOKENS).min(target));
        let floor = match requested {
            Some(n) => n.max(context),
            None => context.max(minimum),
        };
        let max = requested.unwrap_or_else(|| policy.kv_max.unwrap_or(target).max(target));
        Self { context: context.into(), compiled_context: context, requested, target, floor, ceiling: u64::MAX, max, unit_rows }
    }

    /// Apply the shared context requirement without changing workspace extents.
    pub fn with_context(mut self, context: super::planner::Context) -> Self {
        self.context = context;
        self.compiled_context = context.maximum;
        self.target = MemoryPolicy::current().kv_target.unwrap_or(self.target).max(context.required());
        let minimum = MemoryPolicy::current().kv_min
            .unwrap_or(context.maximum.min(AGENTIC_FLOOR_TOKENS).min(self.target));
        self.floor = self.requested.unwrap_or(minimum).max(context.required());
        self
    }

    /// Pin a selected pool for re-admission without inventing an operator request.
    pub fn pin_selected(&mut self, tokens: u64) {
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
mod tests {
    use super::*;

    #[test]
    fn maximum_extent_includes_growth_without_changing_the_layout_target() {
        let defaults = PoolPolicy::resolve(&[96 << 30], 262_144, None, 256);
        assert_eq!(defaults.maximum_tokens(), defaults.target);
        let growing = MemoryPolicy { kv_max: Some(100_000_000), ..Default::default() }.scoped(||
            PoolPolicy::resolve(&[96 << 30], 262_144, None, 256));
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
                let pool = PoolPolicy::resolve(&[96 << 30], 262_144, requested, 256);
                assert_eq!(pool.maximum_tokens(), requested.unwrap());
                assert_eq!(pool.max, requested.unwrap());
                assert_eq!(pool.wanted_units(), requested.unwrap() / 256);
            }
            assert_eq!(PoolPolicy::resolve(&[96 << 30], 262_144, Some(0), 256),
                PoolPolicy::resolve(&[96 << 30], 262_144, None, 256));
        });
    }

    #[test]
    fn capped_pro_has_the_same_planned_and_sampled_pool_extent() {
        let planned = CoordinatorCard::planned(32 << 30, Some(188));
        let sampled = CoordinatorCard { physical_bytes: super::super::inventory::PRO_TOTAL_BYTES,
            logical_bytes: 32 << 30 };
        assert_eq!(planned, sampled);
        assert_eq!(CoordinatorCard::maximum_pool(&[planned], 262_144, None, 64), 2 << 20);
        assert_eq!(CoordinatorCard::maximum_pool(&[planned], 262_144, Some(0), 64),
            CoordinatorCard::maximum_pool(&[sampled], 262_144, None, 64));
        assert_eq!(CoordinatorCard::planned(32 << 30, None), planned);
        assert_eq!(CoordinatorCard::maximum_pool(&[CoordinatorCard::planned(28 << 30, Some(170))],
            262_144, None, 64), 1 << 20);
    }

    #[test]
    fn targets_follow_the_smallest_card_and_the_context() {
        let gib = 1u64 << 30;
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, None, 256).target, 2 << 20);
        assert_eq!(PoolPolicy::resolve(&[96 * gib, 32 * gib], 131_072, None, 256).target, 1 << 20);
        assert_eq!(PoolPolicy::resolve(&[32 * gib], 1 << 21, None, 256).target, 1 << 21);
        let explicit = PoolPolicy::resolve(&[96 * gib], 131_072, Some(32_768), 256);
        assert_eq!((explicit.requested, explicit.floor), (Some(32_768), 131_072));
        assert_eq!(explicit.validate_explicit(), Err(super::super::PlacementError::RequestedBelowContext {
            requested: 32_768, floor: 131_072,
        }));
        assert!(PoolPolicy::resolve(&[96 * gib], 32_768, Some(32_768), 256)
            .validate_explicit().is_ok());
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256).floor, 131_072);
        assert_eq!(PoolPolicy::resolve(&[96 * gib], 131_072, Some(0), 256).requested, None);

    }
}
