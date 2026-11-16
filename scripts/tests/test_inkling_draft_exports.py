"""Recurrent draft export inventory without importing CUDA or the fork."""
import ast
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def catalog(monkeypatch):
    module = types.ModuleType("b12x.integration.cuteafd.inkling_drafter")
    for name in ("dflare_mix", "mask_init", "recurrent_gate", "head_fold", "logit_finish", "residual_norm",
                 "qk_norm_rope", "silu_mul", "bias_add", "kv_store"):
        setattr(module, f"compile_inkling_{name}_aot", lambda _name=name, **kwargs: (_name, kwargs))
    package = types.ModuleType("b12x.integration.cuteafd")
    package.inkling_drafter = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    path = ROOT / "python/tools/aot/export_b12x_dsv4_aot.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("inkling_draft_programs", "inkling_draft_projection_shapes")]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_recurrent_exports_cover_all_ten_primitives_and_rank_variants(monkeypatch):
    exports = catalog(monkeypatch)["inkling_draft_programs"]()
    assert len(exports) == 18
    by_name = {name: (op, params, compile_) for name, op, params, compile_ in exports}
    assert len(by_name) == len(exports)
    for q, k in ((48, 8), (24, 4), (0, 8), (0, 4)):
        op, params, compile_ = by_name[f"qk_norm_rope_q{q}_k{k}"]
        assert compile_() == ("qk_norm_rope", {"q_heads": q, "kv_heads": k})
        assert params == {"max_rows": 4096}
    for heads in (8, 4):
        for fmt in ("fp8", "bf16"):
            assert by_name[f"kv_store_k{heads}_{fmt}"][2]() == ("kv_store", {"kv_heads": heads, "kv_format": fmt})
    for name in ("mask_init", "recurrent_gate"):
        assert by_name[name][1] == {"max_rows": 64}
    for name in ("logit_finish", "bias_add_w768_f32", "bias_add_w201024_bf16"):
        assert by_name[name][1] == {"max_rows": 448}


def test_recurrent_projection_shapes_include_native_head_and_tp2_columns(monkeypatch):
    shapes = catalog(monkeypatch)["inkling_draft_projection_shapes"]
    whole, split = shapes(1), shapes(2)
    for expected in ((4608, 6144), (4608, 1024), (6144, 4608), (4608, 18432), (18432, 4608),
                     (9728, 768), (256, 201024), (4608, 201024)):
        assert expected in whole
    for expected in ((4608, 3072), (4608, 512), (3072, 4608), (4608, 9216), (9216, 4608)):
        assert expected in split
