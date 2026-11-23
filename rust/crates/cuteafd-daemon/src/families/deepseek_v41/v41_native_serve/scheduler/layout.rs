//! Layout selection stays outside lane execution and uses static dispatch.
// A dropped transport result (a failed reset or re-bootstrap) is a bug here.
#![deny(unused_must_use)]
use super::*;
use crate::families::deepseek_v41::v41_experts::dspark::{DsparkChain, DistributedDsparkChain};
use crate::families::deepseek_v41::v41_target_pass::DistributedTargetPass;
use super::super::prefill_target::PrefillTarget;

pub(in crate::families::deepseek_v41::v41_native_serve) trait ServingTarget<'w, 'a>: PrefillTarget<'a> {
    type Chain: DraftChain<'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()>;
    fn reset_connections(transport: &mut Self::Transport) -> Result<()>;
    /// One decode round over the lanes' `members`. `second` and
    /// `second_transport` are the second lane's owners, absent when one decode
    /// lane resolved (every member is then on lane 0).
    #[allow(clippy::too_many_arguments)]
    fn decode_round(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: Option<&mut Self>, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: Option<&mut Self::Transport>,
        active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()>;
}

/// The second lane's pass and transport together, or neither (one resolved
/// lane, whose members are then all on lane 0).
fn second_lane<'p, P, T>(second: Option<&'p mut P>, transport: Option<&'p mut T>, members: &[Vec<usize>; 2])
    -> Result<Option<(&'p mut P, &'p mut T)>> {
    match (second, transport) {
        (Some(second), Some(transport)) => Ok(Some((second, transport))),
        (None, None) => {
            ensure!(members[1].is_empty(), "lane 1 has members but one decode lane resolved");
            Ok(None)
        }
        _ => anyhow::bail!("V4.1 second decode lane is missing its pass or transport"),
    }
}

impl<'w, 'a: 'w> ServingTarget<'w, 'a> for TargetPass<'_, 'a> {
    type Chain = DsparkChain<'w, 'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()> {
        transport.begin_request(); Ok(())
    }
    fn reset_connections(transport: &mut Self::Transport) -> Result<()> {
        transport.reset_connections()
    }
    fn decode_round(lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: Option<&mut Self>, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: Option<&mut Self::Transport>,
        active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()> {
        let second = second_lane(second, second_transport, members)?;
        match second {
            Some((second, second_transport)) if members.iter().all(|lane| !lane.is_empty()) =>
                independent::run(runtime, first, Some(second), requests, first_transport,
                    Some(second_transport), active, draft, prefixes, receive, wake),
            // One busy lane (always, with one resolved lane): the single-RTX
            // C1 path without shared-bank/async delivery.
            Some((second, second_transport)) if members[0].is_empty() => single_lane_round(lib, runtime, 1, second,
                requests, second_transport, active, &members[1], draft, prefixes.turn_bank_enabled()),
            _ => single_lane_round(lib, runtime, 0, first, requests, first_transport, active, &members[0], draft,
                prefixes.turn_bank_enabled()),
        }
    }
}

impl<'w, 'a: 'w> ServingTarget<'w, 'a> for DistributedTargetPass<'_, 'a> {
    type Chain = DistributedDsparkChain<'w, 'a>;
    fn begin_request(transport: &mut Self::Transport) -> Result<()> {
        transport.begin_request(); Ok(())
    }
    fn reset_connections(transport: &mut Self::Transport) -> Result<()> {
        let device = transport.device;
        device.run(|| transport.reset_connections())
    }
    fn decode_round(_lib: &'a NativeLibrary, runtime: &tokio::runtime::Runtime,
        first: &mut Self, second: Option<&mut Self>, requests: &mut Requests<'a>,
        first_transport: &mut Self::Transport, second_transport: Option<&mut Self::Transport>,
        active: &mut [Option<Active<'a>>], members: &[Vec<usize>; 2],
        draft: Option<&mut DraftRuntime<'w, 'a, Self::Chain>>,
        prefixes: &mut PrefixCache<'a>, receive: &mpsc::Receiver<NativeRequest>, wake: admission::Wake<'_>) -> Result<()> {
        // A single active lane still drives its distributed draft via polling:
        // with two lanes the empty peer returns at once without joining
        // per-token work; with one resolved lane, lane 0 runs alone
        // (`LaneSet::run_one`) on the same lane loop.
        if second.is_none() { ensure!(members[1].is_empty(), "lane 1 has members but one decode lane resolved"); }
        independent::run(runtime, first, second, requests, first_transport,
            second_transport, active, draft, prefixes, receive, wake)
    }
}

#[cfg(test)]
mod tests {
    /// A reset that cannot restart its Spark exchange (a failed re-bootstrap)
    /// fails the call: serve then fails the request instead of sending the
    /// next wave into a stopped exchange. Both layouts forward the transport's
    /// result, and the module denies dropped results.
    #[test]
    fn serving_resets_forward_the_transports_result() {
        let source = include_str!("layout.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("#![deny(unused_must_use)]"));
        let bodies: Vec<_> = source.split("fn reset_connections(transport: &mut Self::Transport) -> Result<()> {")
            .skip(1).map(|rest| rest.split("\n    }").next().unwrap()).collect();
        assert_eq!(bodies.len(), 2);
        assert!(bodies[0].trim() == "transport.reset_connections()", "{}", bodies[0]);
        assert!(bodies[1].contains("device.run(|| transport.reset_connections())"), "{}", bodies[1]);
        assert!(bodies.iter().all(|body| !body.contains("Ok(())")), "{bodies:?}");
    }
}
