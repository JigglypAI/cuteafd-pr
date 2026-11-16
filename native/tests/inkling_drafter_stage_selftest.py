"""Teacher-forced bisection of real-weight recurrent drafter stages.

Diagnostic only, never a serving/build driver. Run through golden-run --command.
Every reference receives the native stage's input, so errors do not accumulate
in this table. Pinned vendor test expressions are executed without top-level
imports or device-specific skips; the supplied vendor file must match its SHA.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from families.inkling2 import drafter as eager

VENDOR_SHA = "0f5c89ea5db33c4240f9faa2d70051393f36d88dc8b32ff8d73df50c074b80cc"


class Vendor:
    def __init__(self, path):
        source = Path(path).read_bytes()
        if hashlib.sha256(source).hexdigest() != VENDOR_SHA:
            raise ValueError("pinned vendor test source SHA differs")
        self.path, self.tree = str(path), ast.parse(source)

    def nodes(self, function, first, last):
        function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        return [n for n in ast.walk(function) if isinstance(n, ast.Assign) and first <= n.lineno <= last]

    def execute(self, nodes, scope):
        scope["torch"] = torch
        exec(compile(ast.Module(body=nodes, type_ignores=[]), self.path, "exec"), scope)
        return scope

    def mix(self, features, alpha, weight):
        return self.execute(self.nodes("test_mix_rmsnorm_rounds_once", 43, 45),
                            dict(features=features, alpha=alpha, weight=weight))["ref"]

    def gate(self, raw, state):
        nodes = self.nodes("test_markov_head_fold_and_softcapped_logits", 82, 83)
        scope = self.execute(nodes, dict(raw=raw, state=state))
        function = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "test_markov_head_fold_and_softcapped_logits")
        result = []
        for node in function.body:
            if isinstance(node, ast.Assert) and node.lineno in (85, 86):
                result.append(eval(compile(ast.Expression(node.test.args[1]), self.path, "eval"), scope))
        return tuple(result)

    def finish(self, base, bias):
        return self.execute(self.nodes("test_markov_head_fold_and_softcapped_logits", 76, 76),
                            dict(base=base, bias=bias, cap=30.))["ref"]

    def attention(self, q, ck, cv, dk, dv, window):
        keys, values = torch.cat((ck, dk))[None], torch.cat((cv, dv))[None]
        scope = dict(q=q, key_cache=keys, value_cache=values, i=0, n=len(ck), window=window or 0,
                     is_local=window is not None, num_pages=1, page=keys.shape[1], block=len(q),
                     head_dim=q.shape[-1], group=q.shape[1]//ck.shape[1])
        return self.execute(self.nodes("test_block_attention_anchored_window", 250, 256), scope)["ref"].bfloat16()


class Info(ct.Structure):
    _fields_ = [("name", ct.c_char*64), ("pointers", ct.c_uint32),
                ("scalars", ct.c_uint32), ("kinds", ct.c_char*16)]


class Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        self.lib.cuteafd_dsv4_program_count.restype = ct.c_uint32
        self.lib.cuteafd_dsv4_program_info.argtypes = [ct.c_uint32, ct.POINTER(Info)]
        self.lib.cuteafd_dsv4_program_load.argtypes = [ct.c_uint32]
        self.lib.cuteafd_dsv4_program_launch.argtypes = [ct.c_uint32, ct.POINTER(ct.c_void_p),
                                                       ct.POINTER(ct.c_uint64), ct.c_void_p]
        self.ids = {}
        for index in range(self.lib.cuteafd_dsv4_program_count()):
            info = Info()
            assert self.lib.cuteafd_dsv4_program_info(index, ct.byref(info)) == 0
            self.ids[info.name.decode()] = index
        self.loaded = set()
        self.stream = torch.cuda.current_stream().cuda_stream

    def run(self, name, operands, scalars):
        index = self.ids[name]
        if index not in self.loaded:
            assert self.lib.cuteafd_dsv4_program_load(index) == 0
            self.loaded.add(index)
        pointers = (ct.c_void_p*len(operands))(*(t.data_ptr() for t in operands))
        slots = [(struct.unpack("<I", struct.pack("<f", v))[0] if isinstance(v, float) else v)
                 for v in scalars]
        args = (ct.c_uint64*len(slots))(*slots)
        status = self.lib.cuteafd_dsv4_program_launch(index, pointers, args, self.stream)
        if status:
            raise RuntimeError(f"{name}: native status {status}")

    def linear(self, x, weight, f32=False):
        k, n = x.shape[-1], weight.shape[0]
        output = torch.empty((len(x), n), device=x.device, dtype=torch.float32 if f32 else torch.bfloat16)
        self.run(f"draft_bf16_k{k}_n{n}"+("_f32" if f32 else ""), [x, weight, output], [len(x)])
        return output

    def norm(self, x, weight):
        out = torch.empty_like(x)
        fn = self.lib.cuteafd_cuda_ds4_rmsnorm_bf16_rne_async
        fn.argtypes = [ct.c_void_p]*3+[ct.c_int32, ct.c_int32, ct.c_float, ct.c_void_p]
        assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), len(x), x.shape[-1], 1e-6, self.stream) == 0
        return out

    def residual(self, residual, delta, gamma):
        new, norm = torch.empty_like(residual), torch.empty_like(residual)
        self.run("inkling_draft_residual_norm", [residual, delta, gamma, new, norm], [len(residual), 1e-6])
        return new, norm

    def rotary(self, qkv, qnorm, knorm, positions, inverse, heads):
        rows = len(qkv)
        q = torch.empty((rows, heads, 128), device=qkv.device, dtype=torch.bfloat16)
        k, v = [torch.empty((rows, 8, 128), device=qkv.device, dtype=torch.bfloat16) for _ in range(2)]
        self.run(f"inkling_draft_qk_norm_rope_q{heads}_k8", [qkv, qnorm, knorm, positions, inverse, q, k, v], [rows, 1e-6])
        return q, k, v

    def attention(self, q, dk, dv, ck, cv):
        # Use same native block attention as runtime, BF16 retained planes.
        size = self.lib.cuteafd_block_dspark_attention_workspace
        size.argtypes = [ct.c_int32]*6
        size.restype = ct.c_uint64
        context = len(ck)
        scratch = torch.empty(size(1, 48, 8, 8, context+8, 128), device=q.device, dtype=torch.uint8)
        output = torch.empty((8, 6144), device=q.device, dtype=torch.bfloat16)
        tables = torch.tensor([0, context, context], device=q.device, dtype=torch.int32)
        fn = self.lib.cuteafd_block_dspark_attention
        fn.argtypes = [ct.c_void_p]*10+[ct.c_int32]*8+[ct.c_float, ct.c_void_p]
        status = fn(q.data_ptr(), dk.data_ptr(), dv.data_ptr(), ck.data_ptr(), cv.data_ptr(),
                    tables.data_ptr(), tables.data_ptr()+4, tables.data_ptr()+8,
                    output.data_ptr(), scratch.data_ptr(), 1, 8, 48, 8, context, context+8,
                    0, 128, 128**-.5, self.stream)
        assert status == 0, status
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--vendor-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-reference", action="store_true",
                        help="diagnostic: use pinned vendor FP32 attention after measuring native stage")
    parser.add_argument("--teacher-force-markov", action="store_true",
                        help="diagnostic: keep reference previous token to remove recurrent cascade")
    args = parser.parse_args()
    vendor = Vendor(args.vendor_test)
    native = Native(os.environ["CUTEAFD_NATIVE_LIB"])
    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(str(args.snapshot/"mtp.safetensors"), framework="pt", device="cuda:0") as file:
        w = {name: file.get_tensor(name) for name in file.keys()}
    index = json.loads((args.snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    def target(name):
        with safe_open(str(args.snapshot/index[name]), framework="pt", device="cuda:0") as file:
            return file.get_tensor(name)
    embedding, embed_norm, lm_head = (target(f"model.llm.{suffix}") for suffix in ("embed.weight", "embed_norm.weight", "unembed.weight"))
    fixture = load_file(str(args.fixtures/"rows-3.safetensors"), device="cuda:0")
    features = fixture["features"]
    rows = []
    def check(name, actual, reference):
        actual, reference = actual.contiguous(), reference.contiguous()
        diff = actual.float()-reference.float()
        relative = float(diff.double().norm()/reference.double().norm().clamp_min(1e-20))
        bit_equal = float((actual.view(torch.int16) == reference.bfloat16().view(torch.int16)).float().mean()) if actual.dtype == torch.bfloat16 else float((actual == reference).float().mean())
        entry = dict(stage=name, relative_l2=relative, bit_equal=bit_equal, max_abs=float(diff.abs().max()))
        rows.append(entry)
        print(json.dumps(entry), flush=True)
        return actual
    alpha = w["fuse_weight"].float().softmax(-1).bfloat16()
    inverse = 1./(1e6**(torch.arange(0, 128, 2, device="cuda:0").float()/128))
    context_positions = torch.arange(3, device="cuda:0", dtype=torch.int64)
    positions = torch.arange(3, 11, device="cuda:0", dtype=torch.int64)
    contexts = []
    for layer in range(5):
        p = f"layers.{layer}"
        mixed = torch.empty((3, 4608), device="cuda:0", dtype=torch.bfloat16)
        native.run("inkling_draft_dflare_mix", [features, alpha[layer], w[f"{p}.dflare_norm.weight"], mixed], [3, 1e-6])
        check(f"{layer}.context.mix", mixed, vendor.mix(features, alpha[layer], w[f"{p}.dflare_norm.weight"]))
        kv = []
        for name in ("k_proj_ctx", "v_proj_ctx"):
            result = native.linear(mixed, w[f"{p}.self_attn.{name}.weight"])
            check(f"{layer}.context.{name}", result, F.linear(mixed, w[f"{p}.self_attn.{name}.weight"]))
            kv.append(result)
        qkv = torch.cat(kv, -1)
        _, key, value = native.rotary(qkv, w[f"{p}.self_attn.q_norm.weight"], w[f"{p}.self_attn.k_norm_ctx.weight"], context_positions, inverse, 0)
        refkey = eager.rope(eager.norm(kv[0].view(3, 8, 128), w[f"{p}.self_attn.k_norm_ctx.weight"], 1e-6), context_positions, 1e6)
        check(f"{layer}.context.norm_rope", key, refkey)
        contexts.append((key, value))
    anchor = F.embedding(torch.tensor([101], device="cuda:0"), embedding)
    normalized_anchor = check("anchor.norm", native.norm(anchor, embed_norm), eager.norm(anchor, embed_norm, 1e-6))
    residual = torch.empty((1, 8, 4608), device="cuda:0", dtype=torch.bfloat16)
    native.run("inkling_draft_mask_init", [normalized_anchor, w["mask_embed"], residual], [1])
    residual = residual[0]
    normalized = check("initial.norm", native.norm(residual, w["layers.0.input_layernorm.weight"]), eager.norm(residual, w["layers.0.input_layernorm.weight"], 1e-6))
    layer_hidden = []
    for layer in range(5):
        p, a = f"layers.{layer}", f"layers.{layer}.self_attn"
        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            result = native.linear(normalized, w[f"{a}.{name}.weight"])
            check(f"{layer}.{name}", result, F.linear(normalized, w[f"{a}.{name}.weight"]))
            projections.append(result)
        q, k, v = native.rotary(torch.cat(projections, -1), w[f"{a}.q_norm.weight"], w[f"{a}.k_norm.weight"], positions, inverse, 48)
        check(f"{layer}.query.norm_rope", q, eager.rope(eager.norm(projections[0].view(8,48,128), w[f"{a}.q_norm.weight"], 1e-6), positions, 1e6))
        check(f"{layer}.key.norm_rope", k, eager.rope(eager.norm(projections[1].view(8,8,128), w[f"{a}.k_norm.weight"], 1e-6), positions, 1e6))
        ck, cv = contexts[layer]
        attended = native.attention(q, k, v, ck, cv)
        reference_attention = vendor.attention(q, ck, cv, k, v, None if layer == 2 else 1024).flatten(1)
        check(f"{layer}.attention", attended, reference_attention)
        if args.attention_reference:
            attended = reference_attention
        delta = check(f"{layer}.o_proj", native.linear(attended, w[f"{a}.o_proj.weight"]), F.linear(attended, w[f"{a}.o_proj.weight"]))
        updated, normed = native.residual(residual, delta, w[f"{p}.post_attention_layernorm.weight"])
        refnorm, refupdated = eager.add_norm(delta, residual, w[f"{p}.post_attention_layernorm.weight"], 1e-6)
        check(f"{layer}.attention.residual", updated, refupdated)
        normalized = check(f"{layer}.attention.norm", normed, refnorm)
        residual = updated
        gate, up = [native.linear(normalized, w[f"{p}.mlp.{name}.weight"]) for name in ("gate_proj", "up_proj")]
        check(f"{layer}.gate_proj", gate, F.linear(normalized, w[f"{p}.mlp.gate_proj.weight"]))
        check(f"{layer}.up_proj", up, F.linear(normalized, w[f"{p}.mlp.up_proj.weight"]))
        activation = torch.empty_like(gate)
        native.run("inkling_draft_silu_mul_i18432", [torch.cat((gate, up), -1), activation], [8])
        check(f"{layer}.activation", activation, (F.silu(gate.float())*up.float()).bfloat16())
        delta = check(f"{layer}.down_proj", native.linear(activation, w[f"{p}.mlp.down_proj.weight"]), F.linear(activation, w[f"{p}.mlp.down_proj.weight"]))
        gamma = w["norm.weight"] if layer == 4 else w[f"layers.{layer+1}.input_layernorm.weight"]
        updated, normalized = native.residual(residual, delta, gamma)
        refnorm, refupdated = eager.add_norm(delta, residual, gamma, 1e-6)
        check(f"{layer}.ffn.residual", updated, refupdated)
        check(f"{layer}.ffn.norm", normalized, refnorm)
        residual = updated
        layer_hidden.append(normalized)
    check("final.accumulated", normalized, fixture["hidden"])
    fold, fold_bias = eager.fold_head(lm_head, w["markov_head.hidden_out.weight"], w["markov_head.hidden_out.bias"])
    state = torch.zeros((1,256), device="cuda:0", dtype=torch.bfloat16)
    previous = torch.tensor([101], device="cuda:0")
    f_rnn = vendor.mix(features[-1:], alpha[-1], w["markov_head.dflare_norm.weight"])
    per_position = []
    for step in range(7):
        hidden = normalized[step+1:step+2]
        inputs = torch.cat((state, F.embedding(previous, w["markov_head.markov_w1.weight"]), hidden, f_rnn), -1)
        projection = native.linear(inputs, w["markov_head.rnn_proj.weight"], f32=True)
        check(f"head.{step+1}.projection", projection, F.linear(inputs.float(), w["markov_head.rnn_proj.weight"].float()))
        raw = torch.empty_like(projection)
        native.run("inkling_draft_bias_add_w768_f32", [projection, w["markov_head.rnn_proj.bias"], raw], [1])
        new, latent = torch.empty_like(state), torch.empty_like(state)
        native.run("inkling_draft_recurrent_gate", [raw, state, new, latent], [1])
        rn, rl = vendor.gate(raw, state)
        check(f"head.{step+1}.state", new, rn)
        check(f"head.{step+1}.latent", latent, rl)
        base = native.linear(hidden, lm_head)
        correction_f32 = native.linear(latent, fold, f32=True)
        correction = torch.empty_like(base)
        native.run("inkling_draft_bias_add_w201024_bf16", [correction_f32, fold_bias, correction], [1])
        logits = torch.empty((1,200058), device="cuda:0", dtype=torch.bfloat16)
        native.run("inkling_draft_logit_finish", [base, correction, logits], [1])
        check(f"head.{step+1}.finish", logits, vendor.finish(base, correction)[:, :200058])
        ref = fixture["logits"][step].float()
        top = ref.topk(2).values
        difference = (logits[0].float()-ref).abs()
        previous = logits.float().argmax(-1)
        entry = dict(position=step+1, actual=int(previous), reference=int(fixture["tokens"][step]),
                     top1_margin=float(top[0]-top[1]), max_logit_error=float(difference.max()),
                     relative_l2=float(difference.double().norm()/ref.double().norm().clamp_min(1e-20)))
        per_position.append(entry)
        print(json.dumps(entry), flush=True)
        state = new
        if args.teacher_force_markov:
            previous = fixture["tokens"][step:step+1].long()
    args.output.write_text(json.dumps(dict(stages=rows, positions=per_position, vendor_source_sha256=VENDOR_SHA,
        reference_kind="vendor-authored test expressions for mix/gate/finish/attention; eager norm and torch linear diagnostics"), indent=2)+"\n")


if __name__ == "__main__":
    main()
