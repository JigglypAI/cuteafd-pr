"""Native family hooks must retain shared engine ownership boundaries."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT/'rust/crates/cuteafd-daemon/src/families/inkling2'


def test_native_golden_uses_shared_executor_and_native_expert_dispatch():
    prefill = (FAMILY/'prefill.rs').read_text()
    assert "impl<'a, O: UnitBuffers<'a>> PrefillHooks for Hooks" in prefill
    assert 'run_prefill_units(' in prefill
    assert 'FfnRoute::Local' in prefill
    layer = (FAMILY/'layers.rs').read_text()
    assert 'experts: &dyn Dispatcher' in layer
    assert 'DraftProjections::load_with_output' in layer
    attention = layer.split('pub(crate) fn attention(')[1].split('/// Complete layer')[0]
    assert 'cuda_stream_synchronize' not in attention and '.drain(' not in attention
    golden = (FAMILY/'golden.rs').read_text()
    assert 'super::prefill::layer(' in golden
    assert 'Mxfp8::load(' in golden
    assert 'rms<=0.02 && cosine>=0.999' in golden
    assert '(nll-reference_nll).abs() <= 0.03' in golden
    assert 'python' not in golden.lower()


def test_persistent_layer_state_is_borrowed_and_metadata_uses_retained_uploads():
    owners = (FAMILY/'owners.rs').read_text()
    cache = (FAMILY/'cache.rs').read_text()
    assert "impl<'a> UnitBuffers<'a> for LayerUnit" in owners
    assert 'self.cache.field(self.sequence, self.layer, name)' in owners
    assert 'self.cache.global(self.layer)' in owners
    assert 'cache.require_reader(&self.stream)' in owners
    assert 'RetainedUploads' in owners and 'uploads.upload(' in owners
    assert 'copy_h2d(' not in owners
    assert 'reader.raw == stream.raw' in cache
    assert 'self.copies.copy_d2d(' in cache
    assert 'fatal_drain(self.copies.drain()' in cache
    layer = (FAMILY/'layers.rs').read_text()
    assert "owner: &impl UnitBuffers<'_>" in layer
    assert 'Allocation::new' not in layer


def test_arenas_are_single_backings_and_head_is_shared():
    weights = (FAMILY/'weights.rs').read_text()
    experts = (FAMILY/'experts.rs').read_text()
    assert 'DeviceWeightArena::new' in weights and 'DenseLoad::mount' in weights
    assert 'DeviceWeightArena::new' in experts and 'DenseLoad::mount' in experts
    assert 'dequant' not in experts.lower()
    assert 'transpose_relative' in weights
    assert 'padding_zero_exact = true' in weights
    head = (FAMILY/'head.rs').read_text()
    assert 'target_head::launch(' in head
    assert 'cfg.unpadded_vocab_size as u32' in head
    assert 'cuda_logits_softcap_f32_async' in head
    assert 'drain_or_abort' in head
    assert 'drain_or_abort' in (FAMILY/'owners.rs').read_text()


def test_owner_reconciliation_is_per_id_and_stream_slabs_are_allowances():
    admission = (FAMILY/'admission.rs').read_text()
    assert 'devices.remap_plan_weight_arenas(plan)' in admission
    assert 'placement::planner::reconcile(&projected' in admission
    assert 'ComponentKind::Allowance(Allowance' in admission
    assert 'driver.stream-slabs: one ordered cold golden stream' in admission
    golden = (FAMILY/'golden.rs').read_text()
    assert 'admission::reconcile_native(&plan, devices, &receipts)' in golden
    assert 'measured == admitted' in golden
    assert 'measured <= admitted' not in golden


def test_reference_full_depth_metadata_and_no_stage_hooks_in_arithmetic():
    import ast
    path=ROOT/'scripts/diagnostics/inkling2-prefix-reference.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='is_partial')
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    assert scope['is_partial'](0,48)
    assert scope['is_partial'](35,48)
    assert not scope['is_partial'](47,48)
    assert 'partial=is_partial(args.last_layer,cfg.num_hidden_layers)' in path.read_text()
    assert 'owner.tap(' not in (FAMILY/'layers.rs').read_text()
    assert 'diagnostics' not in (FAMILY/'owners.rs').read_text()
