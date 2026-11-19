"""Bounded Hy4 parity and projection pilot on the pinned source."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from transformers import PreTrainedTokenizerFast

from inkling2 import numerical
from kimi_k3.uniform_k2.storage import atomic_json
from families.hy4_flash import golden, model as reference, sources
from . import forward, recipe
from .source import HeaderSource

TRAIN = (
    "Explain how rain becomes snow and why cold clouds contain liquid droplets.",
    "用中文简述快速排序与归并排序的空间和时间复杂度。",
    "Write a short Python function to merge two sorted lists.",
)
HELDOUT = (
    "Why do some network requests need an idempotency key?",
    "Summarize the difference between an array and a linked list.",
    "A train travels 90 km in 45 minutes. Calculate its average speed.",
    "用三句话说明月相变化的原因。",
)
BARS = {4: 1e-2, 6: 1e-3, 8: 1e-4}


def source_code_identity():
    paths = (Path(__file__), Path(forward.__file__), Path(reference.__file__), Path(recipe.__file__))
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class Pilot:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("Hy4 pilot requires maxq CUDA")
        self.args = args
        self.device = "cuda:0"
        self.started = time.monotonic()
        self.reader = forward.Reader(args.source, self.device)
        self.model = forward.Model(self.reader)
        self.source = HeaderSource(args.source)
        self.oracle = golden.Family(args.source, self.device)
        self.oracle.initialize_official()
        self.report = dict(status="running", schema="cuteafd-hy4-short-pilot-v1",
            source_revision=recipe.BASE_REVISION, reference_revision=sources.TRANSFORMERS_REVISION,
            code_identity=source_code_identity(), checks=[], completed_layers=0,
            latest_layer=None, encodes=0, core_policy=dict(include_head=True,
            include_kv_b=True, k6_drafter_bits=6))
        self.checkpoint()

    def checkpoint(self):
        self.report["seconds"] = time.monotonic() - self.started
        atomic_json(self.args.output, self.report)

    def record(self, kind, **values):
        self.report["checks"].append(dict(kind=kind, **values))
        self.checkpoint()
        print(json.dumps(dict(kind=kind, **values), allow_nan=False), flush=True)

    def heldout_error(self, weight, reconstructed, heldout):
        actual = forward.native_projection("", heldout, reconstructed)
        expected = forward.native_projection("", heldout, weight)
        error = float((actual - expected).square().sum() / expected.square().sum().clamp_min(1e-30))
        if not math.isfinite(error):
            raise ValueError("nonfinite pilot heldout NMSE")
        return error

    def encode(self, name, weight, training, heldout, widths=(8, 6), rows=None):
        matrix = (weight if rows is None else weight[:rows]).contiguous()
        accumulator = numerical.covariance_accumulator(matrix.shape[1], self.device)
        accumulator.add(training)
        raw = dict(H=accumulator.snapshot(), count=accumulator.count, finalized=False)
        result = {}
        for bits in widths:
            started = time.monotonic()
            packed, metrics, evidence = numerical.quantize_projection(matrix, raw,
                bits=bits, device=self.device)
            # Reconstruction consumes the packed EXL3 tensor representation.
            reconstruction = numerical.reconstruct_projection(packed, list(matrix.shape),
                                                               device=self.device, dtype=torch.bfloat16).float()
            error = self.heldout_error(matrix, reconstruction, heldout)
            bpw = 8 * sum(value.numel() * value.element_size() for value in packed.values()) / matrix.numel()
            torch.cuda.synchronize()
            self.report["encodes"] += 1
            self.record("projection", module=name, bits=bits, shape=list(matrix.shape),
                        full_shape=list(weight.shape), heldout_nmse=error, bar=BARS[bits],
                        effective_bits_per_weight=bpw, passed=error <= BARS[bits],
                        seconds=time.monotonic() - started, hessian_rows=accumulator.count)
            result[bits] = reconstruction, error
        return result

    def projection_witness(self, capture, names):
        for name in names:
            values, weight = capture[name]
            n = len(TRAIN)
            result = self.encode(name, weight, torch.cat(values[:n]), torch.cat(values[n:]), rows=128)
            if any(err > BARS[bits] for bits, (_, err) in result.items()):
                raise ValueError(f"target projection exceeds heldout bar: {name}")

    def expert_witness(self, vectors, layer):
        # Use rows of actual routed activations in the native expert order.
        first = vectors[0][1].reshape(-1)
        hits = first.bincount(minlength=self.reader.config.n_routed_experts)
        candidates = hits.argsort(descending=True).tolist()
        for expert in candidates:
            selected = [latent[torch.where(ids == expert)[0]] for latent, ids in vectors]
            train = torch.cat(selected[:len(TRAIN)])
            heldout = torch.cat(selected[len(TRAIN):])
            if len(train) >= 4 and len(heldout) >= 4:
                break
        else:
            raise ValueError("pilot did not find a commonly routed expert across splits")
        gate, up, down = (w.to(self.device, dtype=torch.float32)
                          for w in self.reader.expert_weights(layer.index, expert))
        name = f"{layer.prefix}mlp.experts.{expert}."
        low = {}
        for role, matrix in (("gate_proj", gate), ("up_proj", up)):
            low[role], error = self.encode(name + role, matrix, train, heldout, (4,), rows=128)[4]
            if error > BARS[4]:
                raise ValueError(f"{name + role} K4 misses the heldout bar")

        def activation(inputs):
            g = forward.native_projection("", inputs, gate)
            u = forward.native_projection("", inputs, up)
            g[:, :128] = forward.native_projection("", inputs, low["gate_proj"])
            u[:, :128] = forward.native_projection("", inputs, low["up_proj"])
            limit = self.reader.config.swiglu_limit
            return torch.nn.functional.silu(g.clamp_max(limit)) * u.clamp(-limit, limit)

        _, error = self.encode(name + "down_proj", down, activation(train), activation(heldout),
                               (4,), rows=128)[4]
        if error > BARS[4]:
            raise ValueError(f"{name}down_proj K4 misses the heldout bar")

    def parity(self, layer, tokens, states, adapted):
        weights = dict(layer.weights)
        for key in reference.EXPERTS:
            if key in weights:
                weights[key] = weights[key].to(self.device)
        module = {"weights": weights, "official": None}
        errors = []
        for index in (0, len(TRAIN)):
            expected = self.oracle.official_forward(module, layer.index,
                states[index].unsqueeze(0), tokens[index])[0]
            errors.append(golden.rel_l2(adapted[index], expected))
        del module
        self.record("official_layer_parity", layer=layer.index, relative_l2=errors,
                    passed=max(errors) <= golden.REL_LIMIT)
        if max(errors) > golden.REL_LIMIT:
            raise ValueError(f"Hy4 official parity failed at layer {layer.index}")

    def long_context(self, model_layer):
        row = json.loads((self.args.corpus / "long-context-smoke.json").read_text())
        ids = torch.tensor(row["input_ids"], dtype=torch.long, device=self.device)
        if len(ids) <= 2048 or row["hessian_eligible"] is not False:
            raise ValueError("invalid active-DSA witness")
        state = self.model.initial(ids)
        _, _, inputs = self.model.attention_start(state, ids, model_layer)
        actual, tap = self.model.attention(inputs, model_layer, capture_tap=True)
        expected = reference.attention(inputs, model_layer.weights, self.model.config, model_layer.index)
        error = golden.rel_l2(actual, expected)
        actual_layer, _ = self.model.forward_layer(state, ids, model_layer)
        expected_layer = self.oracle.official_forward(
            {"weights": model_layer.weights, "official": None}, model_layer.index, state.unsqueeze(0), ids)[0]
        official_error = golden.rel_l2(actual_layer, expected_layer)
        self.record("long_context_dsa", tokens=len(ids), selected_columns=tap.groups.shape[-1],
                    relative_l2=error, official_relative_l2=official_error,
                    passed=max(error, official_error) <= golden.REL_LIMIT and bool(tap.groups.numel()))
        if max(error, official_error) > golden.REL_LIMIT or not tap.groups.numel():
            raise ValueError("Hy4 active DSA staged attention witness failed")

    def drafter(self, tokens, states, taps):
        draft = forward.Drafter(self.model)
        embedding = self.reader.raw("model.embed_tokens.weight")
        head = self.reader.load("lm_head.weight")
        features = [self.model.head_features(state)[-1] for state in states]
        bonuses = [self.model.head(feature, weight=head).argmax() for feature in features]
        anchors = [draft.start(bonus, state, embedding) for bonus, state in zip(bonuses, features)]
        target_ends = list(map(len, tokens))
        original = anchors
        for stage in range(len(draft.layers)):
            original = [draft.layer(h, stage, t, end) for h, t, end in zip(original, taps, target_ends)]
        expected = [draft.proposals(draft.finish(h), b, embedding, head_weight=head).argmax(-1)
                    for h, b in zip(original[len(TRAIN):], bonuses[len(TRAIN):])]
        different = {8: list(anchors), 6: list(anchors)}
        missed = {8: False, 6: False}
        for stage, layer in enumerate(draft.layers):
            substitutions = {8: {}, 6: {}}
            # Collect the stage inputs along K8 causal drafter states. Scope the
            # smoke to a representative 128-output block of each matrix.
            sequences = (("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.linear_gate"),
                         ("self_attn.q_b_proj", "self_attn.kv_b_proj"), ("self_attn.o_proj",),
                         ("mlp.gate_proj", "mlp.up_proj"), ("mlp.down_proj",))
            for group in sequences:
                capture = {}
                names = {layer.prefix + leaf for leaf in group}

                def observe(name, inputs, weight):
                    if name in names:
                        values, _ = capture.setdefault(name, ([], weight))
                        values.append(inputs.reshape(-1, inputs.shape[-1]))
                    return self.project(substitutions[8])(name, inputs, weight)

                for h, t, end in zip(different[8], taps, target_ends):
                    draft.layer(h, stage, t, end, observe)
                if set(capture) != names:
                    raise ValueError("missing DFly2 staged projection input")
                for name, (values, weight) in capture.items():
                    result = self.encode(name, weight, torch.cat(values[:len(TRAIN)]),
                                         torch.cat(values[len(TRAIN):]), rows=128)
                    for bits, (value, error) in result.items():
                        substitutions[bits][name] = value
                        missed[bits] |= error > BARS[bits]
            for bits in substitutions:
                different[bits] = [draft.layer(h, stage, t, end, self.project(substitutions[bits]))
                    for h, t, end in zip(different[bits], taps, target_ends)]
        ratings = {}
        for bits, arm in different.items():
            observed = [draft.proposals(draft.finish(h), b, embedding, head_weight=head).argmax(-1)
                        for h, b in zip(arm[len(TRAIN):], bonuses[len(TRAIN):])]
            ratings[bits] = sum(int((x == y).sum()) for x, y in zip(expected, observed)) / sum(map(len, expected))
        self.record("dfly2_top1", scope="128-output slices in all stages", agreements=ratings,
                    compared_tokens=sum(map(len, expected)), passed=ratings[8] >= .98 and not missed[8])
        if missed[8] or ratings[8] < .98:
            raise ValueError("DFly2 representative K8 misses the pilot bar")
        if missed[6] or ratings[6] < .98:
            self.report["core_policy"]["k6_drafter_bits"] = 8
        # Pilot head error is measured on heldout FP32 decoder features.
        fs = [self.model.head_features(state) for state in states]
        result = self.encode("lm_head", head, torch.cat(fs[:len(TRAIN)]),
                             torch.cat(fs[len(TRAIN):]), rows=128)
        if any(error > BARS[bits] for bits, (_, error) in result.items()):
            self.report["core_policy"]["include_head"] = False

    @staticmethod
    def project(substitutions):
        def apply(name, inputs, weight):
            matrix = substitutions.get(name)
            output = forward.native_projection(name, inputs, weight)
            if matrix is not None:
                output[..., :len(matrix)] = forward.native_projection(name, inputs, matrix)
            return output
        return apply

    @torch.inference_mode()
    def run(self):
        reference.configure_f32()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(self.args.source, local_files_only=True)
        from .prepare_corpus import CHAT
        tokens = [torch.tensor(tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    **CHAT), dtype=torch.long, device=self.device) for prompt in TRAIN + HELDOUT]
        self.report["prompt_tokens"] = list(map(len, tokens))
        states = [self.model.initial(ids) for ids in tokens]
        taps = [{} for _ in states]
        witnesses = {
            0: ("self_attn.q_a_proj", "mlp.down_proj"),
            1: ("mlp.routed_expert_down_proj", "mlp.routed_expert_up_proj"),
            20: ("self_attn.linear_gate",),
            21: ("self_attn.kv_b_proj",),
            41: ("self_attn.o_proj",),
        }
        kv_layers, idx_layers = forward.draft_targets(self.reader.config)
        tap_layers = set(kv_layers) | set(idx_layers)
        for index in range(self.reader.config.num_hidden_layers):
            layer = self.model.load_layer(index)
            self.report["latest_layer"] = index
            self.checkpoint()
            selected = {layer.prefix + role for role in witnesses.get(index, ())}
            capture = {}
            routes = []

            def observe(name, inputs, weight):
                if name in selected:
                    values, _ = capture.setdefault(name, ([], weight))
                    values.append(inputs)
                return forward.native_projection(name, inputs, weight)

            adapted = []
            for ordinal, (state, ids) in enumerate(zip(states, tokens)):
                residual, post, values = self.model.attention_start(state, ids, layer)
                output, tap = self.model.attention(values, layer, observe, capture_tap=index in tap_layers)
                hidden, ffn_post, inputs = self.model.mlp_start(residual, post, output, layer)
                routed = None
                if self.reader.config.mlp_layer_types[index] == "sparse":
                    latent, experts, gamma = self.model.moe_route(inputs, layer, observe)
                    if index == 1:
                        routes.append((latent, experts))
                    routed = torch.zeros_like(latent)
                    for expert in experts.unique().tolist():
                        rows, coefficients = self.model.expert_rows(experts, gamma, expert)
                        routed[rows] += self.model.expert(expert, latent[rows], layer) * coefficients[:, None]
                adapted.append(self.model.mlp_finish(hidden, ffn_post, inputs, layer, routed, observe))
                if index in tap_layers:
                    taps[ordinal][index] = tap
            if index in witnesses:
                self.parity(layer, tokens, states, adapted)
                self.projection_witness(capture, sorted(selected))
            if index == 0:
                self.long_context(layer)
            if index == 1:
                self.expert_witness(routes, layer)
            states = adapted
            del layer
            self.reader.close()
            self.report["completed_layers"] = index + 1
            self.checkpoint()
        self.drafter(tokens, states, taps)
        self.report["status"] = "passed"
        self.report["inventory"] = self.source.audit
        self.checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("pilot receipt exists; use a separate attempt path")
    pilot = Pilot(args)
    try:
        pilot.run()
    except Exception as error:
        pilot.report.update(status="failed", error=f"{type(error).__name__}: {error}")
        pilot.checkpoint()
        raise


if __name__ == "__main__":
    main()
