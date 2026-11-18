"""One K8-causal Hy4 quantization pass, resumed at committed stages."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from inkling2 import numerical
from kimi_k3.uniform_k2.storage import atomic_json, fsync_directory, verify_receipt
from . import recipe
from .export import DirectMCWriter, verify_original_discovery
from .forward import AttentionTap, Drafter, Layer, Model, Reader, native_projection, recalled_context, reference
from .prepare_corpus import load_token_batches
from .source import HeaderSource
from .storage import Routes, Stages


REPO = Path(__file__).resolve().parents[2]
CODE = (
    "quantization/hy4_flash/run.py", "quantization/hy4_flash/storage.py",
    "quantization/hy4_flash/recipe.py", "quantization/hy4_flash/forward.py",
    "quantization/hy4_flash/source.py", "quantization/hy4_flash/prepare_corpus.py",
    "quantization/hy4_flash/pilot.py", "quantization/hy4_flash/export.py",
    "quantization/inkling2/numerical.py",
    "quantization/inkling2/export.py", "quantization/kimi_k3/core_k6/storage.py",
    "quantization/kimi_k3/uniform_k2/storage.py", "quantization/kimi_k3/prepare_corpus.py",
    "quantization/glm5/core_k6/recipe.py", "quantization/mc_package.py",
    "scripts/lib/mc_manifest.py",
    "python/reference/families/hy4_flash", "python/reference/shape_invariant.py",
)
SEGMENT_BYTES = 4 * (1 << 30)
EXPERT_GROUP = 32
EXPERT_ROWS = 2048
PILOT_POLICY = {"include_head", "include_kv_b", "k6_drafter_bits"}
LEGACY_EXPERT_SPOOL_COMMIT = "3c5ee4bb0cecedacdf87006bf336f96719ffc3d2"
LEGACY_EXPERT_SPOOL_FILES = ("quantization/hy4_flash/run.py", "quantization/hy4_flash/storage.py")
RESIDENT_EXPERTS_COMMIT = "495873eaa903490d11bdb2df80392520204dc83a"
LEGACY_GPTQMODEL_COMMIT = "6b7f7c48d6be9a3737958e1dac3b94b8ce251695"
BATCHED_HESSIAN_COMMIT = "d606eaccc7fe026b8c83e97a8a8693f6e364e21b"


def read_json(path):
    return json.loads(path.read_text())


def load_pilot(path):
    pilot = read_json(path)
    policy = pilot.get("core_policy", {})
    paths = (Path(__file__).with_name("pilot.py"), Path(__file__).with_name("forward.py"),
             Path(reference.__file__), Path(recipe.__file__))
    tested = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if (pilot.get("status") != "passed"
            or pilot.get("completed_layers") != recipe.LAYER_COUNT or pilot.get("source_revision") != recipe.BASE_REVISION
            or pilot.get("reference_revision") != recipe.REFERENCE_REVISION
            or pilot.get("code_identity") != tested or not pilot.get("checks")
            or set(policy) != PILOT_POLICY
            or type(policy["include_head"]) is not bool
            or type(policy["include_kv_b"]) is not bool
            or type(policy["k6_drafter_bits"]) is not int
            or policy["k6_drafter_bits"] not in (6, 8)):
        raise ValueError("Hy4 production requires a passing pinned pilot and explicit compact-view policy")
    return pilot


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def identity(source, corpus, pilot):
    if os.environ.get("INKLING_GPTQMODEL_ROOT"):
        raise ValueError("Hy4 production requires the committed GPTQModel pin")
    if git("status", "--porcelain", "--", *CODE):
        raise ValueError("Hy4 producer files must be committed before production")
    submodule = REPO / "third_party/gptqmodel"
    fork = git("rev-parse", "HEAD", cwd=submodule)
    if (git("status", "--porcelain", cwd=submodule)
            or git("ls-tree", "HEAD", "third_party/gptqmodel").split()[2] != fork):
        raise ValueError("Hy4 GPTQModel differs from the committed fork pin")
    paths = git("ls-files", "--", *CODE).splitlines()
    return dict(schema="cuteafd-hy4-causal-mc-v1", source_repo=recipe.BASE_REPO,
                source_revision=recipe.BASE_REVISION, reference_revision=recipe.REFERENCE_REVISION,
                source_fingerprints=source.source_fingerprints,
                corpus_manifest_sha256=hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
                code=dict(fork=fork, files={path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
                                           for path in paths}),
                pilot=pilot, seed=787, sigma=.025, expert_bits=4, core_bits=[8, 6],
                upstream_core_bits=8, calibration="ieee-fp32-reference",
                covariance=numerical.ACCUMULATION_CONTRACT,
                expert_rows=EXPERT_ROWS, expert_group=EXPERT_GROUP,
                replay_weights="serialized-exl3-reconstructed-bf16-promoted-fp32",
                drafter_anchor="last-committed-target-state-k8-head-greedy-bonus")


def resident_code_at(commit):
    return {path: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=REPO)).hexdigest()
        for path in LEGACY_EXPERT_SPOOL_FILES}


def checkpoint_identity(root, candidate, *, resume_legacy):
    """Retain the sealed numerical identity across reviewed backend transitions."""
    identity_path = root / "identity.json"
    recovery_path = root / "resident-expert-recovery.json"
    batched_path = root / "batched-hessian-recovery.json"
    if not identity_path.exists():
        if resume_legacy or recovery_path.exists() or batched_path.exists():
            raise ValueError("Hy4 resident recovery requires an existing numerical identity")
        atomic_json(identity_path, candidate)
        return candidate
    frozen = read_json(identity_path)
    if frozen == candidate and not recovery_path.exists() and not batched_path.exists():
        return frozen
    if not resume_legacy:
        raise ValueError("Hy4 legacy expert recovery requires --resume-legacy-expert-spool")

    frozen_fork, candidate_fork = frozen["code"]["fork"], candidate["code"]["fork"]
    batched = frozen_fork != candidate_fork
    if batched and (frozen_fork, candidate_fork) != (LEGACY_GPTQMODEL_COMMIT, BATCHED_HESSIAN_COMMIT):
        raise ValueError("Hy4 recovery only accepts the reviewed batched Hessian fork change")
    files = candidate["code"]["files"] | resident_code_at(LEGACY_EXPERT_SPOOL_COMMIT)
    legacy = candidate | {"code": candidate["code"] | {"files": files, "fork": frozen_fork}}
    if frozen != legacy:
        raise ValueError("Hy4 numerical or unrelated code identity differs from the sealed run")

    expected = dict(schema="hy4-resident-experts-v1", legacy_commit=LEGACY_EXPERT_SPOOL_COMMIT,
                    code={path: candidate["code"]["files"][path] for path in LEGACY_EXPERT_SPOOL_FILES},
                    frozen_identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
    if batched:
        previous = expected | {"code": resident_code_at(RESIDENT_EXPERTS_COMMIT)}
        if not recovery_path.is_file() or read_json(recovery_path) != previous:
            raise ValueError("Hy4 batched Hessian recovery needs the sealed resident backend receipt")
        recovery_path = batched_path
        expected = expected | dict(schema="hy4-batched-hessian-v1", fork_commit=candidate_fork,
                                   prior_backend_commit=RESIDENT_EXPERTS_COMMIT)
    elif batched_path.exists():
        raise ValueError("Hy4 batched Hessian recovery already owns this run")

    if recovery_path.exists():
        if read_json(recovery_path) != expected:
            raise ValueError("Hy4 resident recovery code differs from the committed backend")
    else:
        atomic_json(recovery_path, expected)
    return frozen


class Progress:
    def __init__(self, root, total):
        self.root, self.total = root, total
        self.base = dict(target_layers=recipe.LAYER_COUNT, drafter_layers=recipe.MTP_LAYER_COUNT,
                         total_encode_operations=total, completed_target_layers=0,
                         completed_drafter_layers=0, committed_encodes=0)

    def event(self, phase, *, snapshot=True, **fields):
        row = self.base | fields | dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat())
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if snapshot:
            atomic_json(self.root / "status.json", dict(status="running", **row))
        print(json.dumps(row), flush=True)


def raw(acc):
    if acc.count == 0:
        raise ValueError("Hy4 projection has no natural calibration rows")
    return dict(H=acc.snapshot(), count=acc.count, finalized=False)


def reconstruction(values, name, bits, shape, device):
    packed = {suffix: values[f"k{bits}.{name}.{suffix}"] for suffix in recipe.SUFFIXES}
    return numerical.reconstruct_projection(packed, list(shape), device=str(device),
                                            dtype=torch.bfloat16).float()


def encode(name, weight, covariance, bits, device, metrics):
    values = {}
    for width in bits:
        packed, measured, evidence = numerical.quantize_projection(
            weight, covariance, bits=width, device=str(device))
        values.update({f"k{width}.{name}.{key}": value for key, value in packed.items()})
        metrics.append(evidence | measured | dict(module=name, bits=width))
    return values


def weights_project(overrides):
    def project(name, inputs, weight):
        return native_projection(name, inputs, overrides.get(name, weight))
    return project


class Runner:
    def __init__(self, args):
        self.args, self.root = args, args.root.resolve()
        self.output = args.output.resolve()
        self.device = torch.device(args.device)
        self.root.mkdir(parents=True, exist_ok=True)
        self.header = HeaderSource(args.source)
        self.pilot = load_pilot(args.pilot)
        self.policy = self.pilot["core_policy"]
        self.identity = checkpoint_identity(
            self.root, identity(self.header, args.corpus, self.pilot),
            resume_legacy=args.resume_legacy_expert_spool)
        self.core = recipe.reviewed_core_modules(
            include_head=self.policy["include_head"], include_kv_b=self.policy["include_kv_b"])
        self.total = len(self.core) * 2 + recipe.EXPERT_MODULE_COUNT
        self.progress = Progress(self.root, self.total)
        self.stages = Stages(self.root / "stages", self.identity)
        self.writer = DirectMCWriter(self.output, self.header.snapshot, self.identity,
                                     self.header.as_inventory())
        self.records = [r["input_ids"].reshape(-1).cpu()
                        for r in load_token_batches(args.corpus, split="calibration")]
        if not self.records or any(not len(tokens) for tokens in self.records):
            raise ValueError("Hy4 needs nonempty causal calibration prompts")
        self.reader = Reader(self.header.snapshot, self.device)
        self.model = Model(self.reader)
        self.embedding = self.reader.raw(reference.OUTER_NAMES[0])
        self.frontier_path = self.root / "frontier.json"
        self.frontier = self.load_frontier()
        self.progress.base.update(completed_target_layers=self.frontier["target"],
                                  completed_drafter_layers=self.frontier["drafter"],
                                  committed_encodes=self.frontier["encodes"])

    def load_frontier(self):
        if not self.frontier_path.exists():
            return dict(stage="initial", target=0, drafter=0, head=False, encodes=0)
        frontier = read_json(self.frontier_path)
        if frontier.pop("identity") != self.identity:
            raise ValueError("Hy4 frontier numerical identity differs")
        target, draft, head = (frontier[key] for key in ("target", "drafter", "head"))
        if (type(target) is not int or not 0 <= target <= recipe.LAYER_COUNT
                or type(draft) is not int or not 0 <= draft <= recipe.MTP_LAYER_COUNT
                or type(head) is not bool or (draft and not head)
                or (head and target != recipe.LAYER_COUNT)):
            raise ValueError("invalid Hy4 rolling frontier")
        expected = ["initial"] if target == 0 else [f"target-{target - 1:03d}-next"]
        if head:
            expected = [f"draft-{draft - 1:03d}-next"] if draft else ["draft-initial"]
        if frontier["stage"] not in expected or not self.stages.load(frontier["stage"]):
            raise ValueError("committed Hy4 successor stage is missing")
        return frontier

    def advance(self, *, completed, old_stages, **fields):
        self.frontier.update(fields)
        self.frontier["encodes"] += completed
        atomic_json(self.frontier_path, dict(self.frontier, identity=self.identity))
        self.progress.base.update(completed_target_layers=self.frontier["target"],
                                  completed_drafter_layers=self.frontier["drafter"],
                                  committed_encodes=self.frontier["encodes"])
        self.stages.retire(*old_stages)

    def rows(self, stage, *keys):
        return self.stages.records(stage, device=str(self.device), keys=keys or None)

    def save(self, name, batches):
        def checked():
            for index, values in batches():
                if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
                       for value in values.values()):
                    raise ValueError("nonfinite Hy4 calibration stage")
                yield index, values
        self.stages.write_batches(name, checked(), segment_bytes=SEGMENT_BYTES, segment_batches=128)

    def core_fit(self, stage, layer, groups, samples, overrides):
        """Fit one shared covariance per input boundary and replay serialized K8."""
        groups = [tuple(name for name in group if name in self.core) for group in groups]
        names = [name for group in groups for name in group]
        if not names:
            return None
        metrics = []
        def fit():
            accumulators = [numerical.covariance_accumulator(
                self.core[group[0]]["in_features"], device=self.device) if group else None
                for group in groups]
            for sample in samples():
                for acc, inputs in zip(accumulators, sample, strict=True):
                    if acc is not None:
                        acc.add(inputs)
            ordinal = 0
            for group, acc in zip(groups, accumulators, strict=True):
                if acc is None:
                    continue
                covariance = raw(acc)
                for name in group:
                    self.progress.event("core_projection", stage=stage, projection=name)
                    weight = layer.weights[name.removeprefix(layer.prefix) + ".weight"]
                    yield ordinal, encode(name, weight, covariance, (8, 6), self.device, metrics)
                    ordinal += 1
        manifest = self.stages.write_batches(stage, fit(), {"names": names, "metrics": metrics})
        if manifest["metadata"]["names"] != names:
            raise ValueError("Hy4 compact scratch inventory differs")
        for index, name in enumerate(names):
            geometry = self.core[name]
            overrides[name] = reconstruction(self.stages.read_batch(stage, index), name, 8,
                (geometry["out_features"], geometry["in_features"]), self.device)
        return stage

    @staticmethod
    def first_names(prefix):
        return (prefix + leaf for leaf in ("q_a_proj", "kv_a_proj_with_mqa", "linear_gate"))

    def compact_entries(self, stages, *, draft=False):
        entries = {6: {}, 8: {}}
        metrics = []
        for stage in filter(None, stages):
            manifest = self.stages.load(stage)
            metrics.extend(manifest["metadata"]["metrics"])
            for index, name in enumerate(manifest["metadata"]["names"]):
                for bits in (8, 6):
                    if draft and bits == 6 and self.policy["k6_drafter_bits"] == 8:
                        continue
                    for suffix in recipe.SUFFIXES:
                        key = f"{name}.{suffix}"
                        entries[bits][key] = self.stages.reference(stage, index, f"k{bits}.{key}")
        return entries, metrics

    def save_metrics(self, label, metrics):
        directory = self.root / "metrics"
        directory.mkdir(exist_ok=True)
        path = directory / f"{label}.json"
        if path.exists():
            if read_json(path) != metrics:
                raise ValueError(f"committed Hy4 metrics changed for {label}")
        else:
            atomic_json(path, metrics)

    def source_group(self, prefix, *, exclude=()):
        original = {recipe.source_tensor(name)[0] for name in self.core if name.startswith(prefix)}
        native = {name for name in self.header.specs if name.startswith(prefix)}
        native -= original | set(exclude) | recipe.ENGRAM_TABLES.keys()
        return self.header.source_entries(native), self.header.source_entries(original)

    def target(self, index):
        label = f"target-{index:03d}"
        tick = time.monotonic()
        incoming = self.frontier["stage"]
        successor = label + "-next"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, successor, target=index + 1)
            return
        self.progress.base.update(current_layer=index, expert_projections=0,
                                  total_expert_projections=recipe.EXPERTS_PER_LAYER * 3 if index else 0)
        self.progress.event("target_layer")
        layer = self.model.load_layer(index)
        p, a = layer.prefix, layer.prefix + "self_attn."
        overrides = {}
        project = weights_project(overrides)
        capture = index in recipe.DFLY_TARGET_LAYERS
        moe = index > 0
        mlp = p + ("mlp.shared_experts." if moe else "mlp.")
        a_start = label + "-attention-input"
        mlp_stage = label + "-mlp"
        route_stage = label + "-routes"
        fit_stages = []

        def attention_input():
            for number, values in self.rows(incoming):
                tokens = self.records[number].to(self.device)
                residual, post, inputs = self.model.attention_start(values["state"], tokens, layer)
                yield number, dict(residual=residual, post=post, inputs=inputs)

        self.save(a_start, attention_input)
        fit_stages.append(self.core_fit(label + "-attn-first", layer, [self.first_names(a)],
            lambda: ((values["inputs"],) for _, values in self.rows(a_start, "inputs")), overrides))

        def attention_first(*keys):
            for number, values in self.rows(a_start, *keys):
                first = self.model.attention_first(values["inputs"], layer, project)
                yield number, values, first

        fit_stages.append(self.core_fit(label + "-attn-expand", layer,
            [(a + "q_b_proj",), (a + "kv_b_proj",)],
            lambda: ((first.normalized_q, first.kv) for _, _, first in attention_first("inputs")), overrides))

        def attention_output(*keys):
            for number, values, first in attention_first(*keys):
                output, tap = self.model.attention_output_inputs(
                    values["inputs"], first, layer, project, capture_tap=capture)
                yield number, values, output, tap

        fit_stages.append(self.core_fit(label + "-attn-out", layer, [(a + "o_proj",)],
            lambda: ((o,) for _, _, o, _ in attention_output("inputs")), overrides))

        def attention_finish():
            for number, values, output, tap in attention_output():
                attended = self.model.attention_finish(output, layer, project)
                state, post, inputs = self.model.mlp_start(values["residual"], values["post"],
                                                           attended, layer)
                result = dict(state=state, post=post, inputs=inputs)
                if tap is not None:
                    result.update(kv=tap.kv, rope_key=tap.rope_key, groups=tap.groups)
                yield number, result

        self.save(mlp_stage, attention_finish)
        if capture:
            tap_stage = f"tap-{index:03d}"
            self.save(tap_stage, lambda: self.stages.records(mlp_stage, keys=("kv", "rope_key", "groups")))

        entry = [mlp + "gate_proj", mlp + "up_proj"]
        if moe:
            entry.append(p + "mlp.routed_expert_down_proj")
        fit_stages.append(self.core_fit(label + "-mlp-first", layer, [entry],
            lambda: ((values["inputs"],) for _, values in self.rows(mlp_stage, "inputs")), overrides))

        def activated(inputs):
            return F.silu(layer.project((mlp + "gate_proj").removeprefix(p), inputs, project)) * (
                layer.project((mlp + "up_proj").removeprefix(p), inputs, project))

        fit_stages.append(self.core_fit(label + "-mlp-down", layer, [(mlp + "down_proj",)],
            lambda: ((activated(values["inputs"]),) for _, values in self.rows(mlp_stage, "inputs")), overrides))
        packed, extra_metrics, expert_stages = {}, [], []
        combined = None
        if moe:
            def routes():
                for number, values in self.rows(mlp_stage, "inputs"):
                    latent, ids, gamma = self.model.moe_route(values["inputs"], layer, project)
                    yield number, dict(latent=latent, ids=ids, gamma=gamma)
            self.save(route_stage, routes)
            spool = Routes(self.stages.records(route_stage), recipe.EXPERTS_PER_LAYER,
                           sum(len(tokens) for tokens in self.records), device=self.device)
            packed, extra_metrics, expert_stages, combined = self.experts(label, layer, spool)
            del spool
            fit_stages.append(self.core_fit(label + "-latent-up", layer,
                [(p + "mlp.routed_expert_up_proj",)],
                lambda: ((rows,) for rows in combined.split(EXPERT_ROWS)), overrides))

        def finish():
            routed = iter(combined.split([len(tokens) for tokens in self.records])) if moe else None
            for number, values in self.rows(mlp_stage, "state", "post", "inputs"):
                result = self.model.mlp_finish(values["state"], values["post"], values["inputs"],
                    layer, next(routed) if moe else None, project)
                yield number, dict(state=result)

        self.save(successor, finish)
        views, metrics = self.compact_entries(fit_stages)
        metrics += extra_metrics
        self.save_metrics(label, metrics)
        fused = {p + leaf for leaf in reference.EXPERTS} if moe else ()
        native, original = self.source_group(p, exclude=fused)
        self.writer.write_group(label, shared=native | packed, original=original,
                                k8=views[8], k6=views[6])
        self.advance(completed=len(metrics), target=index + 1, stage=successor,
            old_stages=[incoming, a_start, mlp_stage, route_stage, *filter(None, fit_stages), *expert_stages])
        self.progress.event("target_layer_complete", current_layer=index,
                            layer_seconds=time.monotonic() - tick)
        self.reader.close()

    def experts(self, label, layer, spool):
        packed, errors, stages = {}, [], []
        combined = torch.zeros_like(spool.latent)
        for start in range(0, recipe.EXPERTS_PER_LAYER, EXPERT_GROUP):
            stage = f"{label}-expert-{start:03d}"
            metrics = []
            tick = time.monotonic()
            manifest = self.stages.load(stage)
            if manifest is None:
                def encode_group():
                    for expert in range(start, min(start + EXPERT_GROUP, recipe.EXPERTS_PER_LAYER)):
                        yield expert - start, self.expert(layer, spool, expert, metrics, combined)

                manifest = self.stages.write_batches(stage, encode_group(), {"metrics": metrics},
                                                      segment_bytes=SEGMENT_BYTES, segment_batches=EXPERT_GROUP)
            else:
                for entry in manifest["entries"]:
                    self.replay_expert(layer, spool, start + entry["index"], combined, stage, entry["index"])
            for entry in manifest["entries"]:
                ordinal = entry["index"]
                for name in self.expert_names(layer, start + ordinal):
                    for suffix in recipe.SUFFIXES:
                        tensor = f"{name}.{suffix}"
                        packed[tensor] = self.stages.reference(stage, ordinal, f"k4.{tensor}")
            errors.extend(manifest["metadata"]["metrics"])
            stages.append(stage)
            self.progress.base["expert_projections"] = len(errors)
            phase = "expert_group_complete" if metrics else "expert_group_replayed"
            self.progress.event(phase, current_layer=layer.index,
                                expert_group_seconds=time.monotonic() - tick,
                                new_expert_projections=len(metrics))
        return packed, errors, stages, combined

    @staticmethod
    def expert_names(layer, expert):
        prefix = f"{layer.prefix}mlp.experts.{expert}."
        return tuple(prefix + leaf for leaf in ("gate_proj", "up_proj", "down_proj"))

    def expert_activations(self, spool, rows, names, gate, up):
        limit = self.model.config.swiglu_limit
        for selection in rows.split(EXPERT_ROWS):
            chunk = spool.latent.index_select(0, selection)
            g = native_projection(names[0], chunk, gate).clamp_max(limit)
            u = native_projection(names[1], chunk, up).clamp(-limit, limit)
            yield F.silu(g) * u

    def accumulate_expert(self, combined, rows, gamma, name, down, activations):
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        for selection, coefficients, chunk in zip(
                rows.split(EXPERT_ROWS), gamma.split(EXPERT_ROWS), activations, strict=True):
            weighted = native_projection(name, chunk, down) * coefficients[:, None]
            finite.logical_and_(torch.isfinite(weighted).all())
            # Every expert visits a token at most once. Add in ascending expert
            # order, as the stock reference and the legacy CPU reduction do.
            combined.index_copy_(0, selection, combined.index_select(0, selection) + weighted)
        if not bool(finite):
            raise ValueError(f"nonfinite Hy4 expert output: {name}")

    def replay_expert(self, layer, spool, expert, combined, stage, ordinal):
        rows, gamma = spool.expert(expert)
        rows, gamma = rows.to(self.device), gamma.to(self.device)
        names = self.expert_names(layer, expert)
        keys = tuple(f"k4.{name}.{suffix}" for name in names for suffix in recipe.SUFFIXES)
        # Select only packed weights, including when a frozen group still
        # contains the legacy unsummed output. Replaying needs no Hessian.
        values = self.stages.fields(stage, ordinal, keys)
        latent = spool.latent.shape[-1]
        intermediate = self.model.config.moe_intermediate_size
        gate, up, down = (reconstruction(values, name, 4, shape, self.device)
                          for name, shape in zip(names, ((intermediate, latent), (intermediate, latent),
                                                        (latent, intermediate)), strict=True))
        activated = self.expert_activations(spool, rows, names, gate, up)
        self.accumulate_expert(combined, rows, gamma, names[2], down, activated)

    def expert(self, layer, spool, expert, metrics, combined):
        rows, gamma = spool.expert(expert)
        if not len(rows):
            raise ValueError(f"Hy4 layer {layer.index} expert {expert} has no natural calibration rows")
        rows, gamma = rows.to(self.device), gamma.to(self.device)
        names = self.expert_names(layer, expert)
        weights = self.reader.expert_weights(layer.index, expert)
        acc = numerical.covariance_accumulator(weights[0].shape[-1], device=self.device)
        for part in rows.split(EXPERT_ROWS):
            acc.add(spool.latent.index_select(0, part))
        covariance = raw(acc)
        values = {}
        for name, weight in zip(names[:2], weights[:2], strict=True):
            values.update(encode(name, weight.to(self.device), covariance, (4,), self.device, metrics))
        gate, up = (reconstruction(values, name, 4, weight.shape, self.device)
                    for name, weight in zip(names[:2], weights[:2], strict=True))
        acc = numerical.covariance_accumulator(weights[2].shape[-1], device=self.device)
        for chunk in self.expert_activations(spool, rows, names, gate, up):
            acc.add(chunk)
        values.update(encode(names[2], weights[2].to(self.device), raw(acc), (4,), self.device, metrics))
        down = reconstruction(values, names[2], 4, weights[2].shape, self.device)
        activated = self.expert_activations(spool, rows, names, gate, up)
        self.accumulate_expert(combined, rows, gamma, names[2], down, activated)
        return values

    def recover_complete_group(self, label, successor, **frontier):
        """A sealed output and successor authorize adopting a missing frontier."""
        if self.stages.load(successor) is None:
            raise ValueError("Hy4 sealed output has no committed successor; diagnose before reuse")
        metrics = read_json(self.root / "metrics" / f"{label}.json")
        # If a crash preceded this barrier, all former stages are still
        # available. Retire only after committing the adoption.
        discard = [path.name for path in self.stages.root.iterdir()
                   if path.name.startswith(label + "-") and path.name != successor]
        discard.append(self.frontier["stage"])
        self.advance(completed=len(metrics), old_stages=discard, stage=successor, **frontier)
        self.progress.event("group_frontier_recovered", group=label)

    def head_and_draft_anchors(self):
        label = "head"
        incoming = self.frontier["stage"]
        starter = "draft-initial"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, starter, head=True)
            return
        weights = {name: self.reader.load(name) for name in reference.OUTER_NAMES[1:5]}
        weights["lm_head.weight"] = self.reader.load("lm_head.weight")
        layer = Layer(recipe.LAYER_COUNT, "", weights)
        overrides = {}
        self.progress.event("head_calibration")
        fit = self.core_fit(label + "-fit", layer, [("lm_head",)],
            lambda: ((self.model.head_features(row["state"], weights),)
                     for _, row in self.rows(incoming)), overrides)
        aux = self.reader.load_draft_aux()
        drafter = Drafter(self.model,
            layers=[self.reader.load_draft_layer(index) for index in range(recipe.MTP_LAYER_COUNT)],
            aux=aux)

        def anchors():
            for number, row in self.rows(incoming):
                features = self.model.head_features(row["state"][-1:], weights)
                bonus = self.model.head(features, project=weights_project(overrides),
                                        weight=weights["lm_head.weight"]).argmax()
                state = drafter.start(bonus, features[0], self.embedding)
                yield number, dict(state=state)

        self.save(starter, anchors)
        views, metrics = self.compact_entries([fit])
        self.save_metrics(label, metrics)
        native, original = self.source_group("lm_head.")
        self.writer.write_group(label, shared=native, original=original, k8=views[8], k6=views[6])
        self.advance(completed=len(metrics), head=True, stage=starter,
                     old_stages=[incoming, *filter(None, [fit])])
        self.progress.event("head_complete")
        self.reader.close()

    def draft_taps(self, drafter, index, prompt):
        taps = {}
        for source in {drafter.kv_layers[index], drafter.index_layers[index]}:
            values = self.stages.read_batch(f"tap-{source:03d}", prompt, device=str(self.device))
            taps[source] = AttentionTap(values["kv"], values["rope_key"], values["groups"])
        return taps

    def draft(self, drafter, index):
        label = f"draft-{index:03d}"
        incoming = self.frontier["stage"]
        successor = label + "-next"
        if self.writer.group_complete(label):
            self.recover_complete_group(label, successor, drafter=index + 1)
            return
        self.progress.event("drafter_layer", current_drafter_layer=index)
        tick = time.monotonic()
        layer = drafter.layers[index]
        p = layer.prefix
        a, mlp = p + "self_attn.", p + "mlp."
        overrides = {}
        project = weights_project(overrides)

        def inputs():
            for number, values in self.rows(incoming):
                state = values["state"]
                yield number, state, drafter.attention_inputs(state, index)

        fit_stages = [self.core_fit(label + "-attn-first", layer, [self.first_names(a)],
            lambda: ((boundary,) for _, _, boundary in inputs()), overrides)]

        def first():
            for number, state, boundary in inputs():
                yield number, state, self.model.attention_first(boundary, layer, project)

        def expansions():
            ratio = drafter.config.compressed_indexer_compress_ratio
            for number, _, projected in first():
                kv = projected.kv
                if self.policy["include_kv_b"]:
                    taps = self.draft_taps(drafter, index, number)
                    end = len(self.records[number])
                    ids = recalled_context(taps[drafter.index_layers[index]], end, ratio)
                    kv = torch.cat((taps[drafter.kv_layers[index]].kv[ids], kv))
                yield projected.normalized_q, kv

        fit_stages.append(self.core_fit(label + "-attn-expand", layer,
            [(a + "q_b_proj",), (a + "kv_b_proj",)], expansions, overrides))

        def attention():
            for number, state, projected in first():
                taps = self.draft_taps(drafter, index, number)
                end = len(self.records[number])
                boundary = drafter.attention_output_inputs(projected, index, taps, end, project)
                yield number, state, boundary

        fit_stages.append(self.core_fit(label + "-attn-out", layer, [(a + "o_proj",)],
            lambda: ((out,) for _, _, out in attention()), overrides))

        mlp_stage = label + "-mlp"
        def after_attention():
            for number, state, out in attention():
                result = self.model.attention_finish(out, layer, project)
                residual, normalized = drafter.ffn_inputs(state, result, index)
                yield number, dict(residual=residual, inputs=normalized)

        self.save(mlp_stage, after_attention)
        fit_stages.append(self.core_fit(label + "-mlp-first", layer,
            [(mlp + "gate_proj", mlp + "up_proj")],
            lambda: ((values["inputs"],) for _, values in self.rows(mlp_stage, "inputs")), overrides))

        def down_inputs():
            for _, v in self.rows(mlp_stage, "inputs"):
                gate = layer.project("mlp.gate_proj", v["inputs"], project)
                up = layer.project("mlp.up_proj", v["inputs"], project)
                yield (F.silu(gate) * up,)

        fit_stages.append(self.core_fit(label + "-mlp-down", layer, [(mlp + "down_proj",)],
                                       down_inputs, overrides))

        def finish():
            for number, v in self.rows(mlp_stage):
                result = self.model.ffn_parts(v["inputs"], layer, "mlp.", project)
                yield number, dict(state=v["residual"] + result)

        self.save(successor, finish)
        views, metrics = self.compact_entries(fit_stages, draft=True)
        self.save_metrics(label, metrics)
        native, original = self.source_group(p)
        self.writer.write_group(label, shared=native, original=original, k8=views[8], k6=views[6])
        self.advance(completed=len(metrics), drafter=index + 1, stage=successor,
                     old_stages=[incoming, mlp_stage, *filter(None, fit_stages)])
        self.progress.event("drafter_layer_complete", current_drafter_layer=index,
                            layer_seconds=time.monotonic() - tick)

    def globals(self):
        if self.writer.group_complete("globals"):
            return
        native = {name for name in self.header.specs
                  if not name.startswith(("model.layers.", "lm_head."))
                  and not any(name.startswith(f"model.dfly_layers.{index}.")
                              for index in range(recipe.MTP_LAYER_COUNT))}
        self.writer.write_group("globals", shared=self.header.source_entries(native))

    def write_error_csv(self):
        rows = [row for path in sorted((self.root / "metrics").glob("*.json"))
                for row in read_json(path)]
        expected = {(module, bits) for module in self.core for bits in (8, 6)}
        expected.update((module, 4) for module in recipe.reviewed_expert_modules())
        if len(rows) != len(expected) or {(row["module"], row["bits"]) for row in rows} != expected:
            raise ValueError("Hy4 calibration metrics do not cover every expected encode")
        names = ["module", "bits"] + sorted(set().union(*(row.keys() for row in rows)) - {"module", "bits"})
        path = self.root / "per-module-errors.csv"
        temporary = path.with_suffix(".csv.partial")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, names)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(self.root)
        return path

    def export_engram(self):
        self.progress.event("engram_export")
        receipt = read_json(self.args.engram_receipt)
        source = self.args.engram_dir.resolve()
        if source.stat().st_dev != self.output.stat().st_dev:
            raise ValueError("Hy4 Engram and MC output need the same filesystem for no-copy adoption")
        tables = receipt["tables"]
        expected = set(tables.values()) | {part["file"] for layer in receipt["manifest"]["layers"]
                                           for part in layer["parts"]}
        if tables.keys() != recipe.ENGRAM_TABLES.keys() or receipt["files"].keys() != expected:
            raise ValueError("Hy4 Engram receipt has an unexpected file inventory")
        for name, proof in receipt["files"].items():
            if Path(name).name != name or not name.startswith("engram-") or not name.endswith(".safetensors"):
                raise ValueError("invalid Hy4 Engram file receipt")
            before = source / name
            after = self.output / name
            if before.is_symlink() or after.is_symlink():
                raise ValueError("Hy4 Engram adoption requires regular files")
            verify_receipt(before, proof)
            if not after.exists():
                os.link(before, after)
            verify_receipt(after, proof)
        fsync_directory(self.output)
        return receipt

    def run(self):
        reference.configure_f32()
        self.globals()
        if not self.frontier["target"]:
            self.save("initial", lambda: ((index, {"state": self.model.initial(tokens.to(self.device),
                                               self.embedding)})
                                         for index, tokens in enumerate(self.records)))
        for layer in range(self.frontier["target"], recipe.LAYER_COUNT):
            self.target(layer)
        if not self.frontier["head"]:
            self.head_and_draft_anchors()
        if self.frontier["drafter"] < recipe.MTP_LAYER_COUNT:
            drafter = Drafter(self.model)
            for index in range(self.frontier["drafter"], recipe.MTP_LAYER_COUNT):
                self.draft(drafter, index)
            del drafter
        self.reader.close()
        if self.frontier["encodes"] != self.total:
            raise ValueError("Hy4 completed encode count differs from the pilot-selected recipe")
        error_csv = self.write_error_csv()
        tables = self.export_engram()
        self.progress.event("package_audit")
        result = self.writer.finalize(core_selection=self.core, engram=tables, document=self.header.document,
                                      error_csv=error_csv, card_path=self.root / "README.md",
                                      k6_drafter_bits=self.policy["k6_drafter_bits"])
        state = read_json(self.output / "package-state.json")
        discovery = verify_original_discovery(self.output, state["direct_expected_mappings"]["original"])
        complete = dict(status="passed", identity=self.identity, output=str(self.output),
                        revision=recipe.BASE_REVISION, repository=recipe.TARGET_REPO,
                        completed_target_layers=recipe.LAYER_COUNT,
                        completed_drafter_layers=recipe.MTP_LAYER_COUNT,
                        completed_encodes=self.total, error_csv=str(error_csv),
                        task_intermediates=[str(self.stages.root)],
                        package_validation=str(self.output / "package-validation.json"),
                        package_status=result["status"], original_discovery=discovery)
        recovery = self.root / "resident-expert-recovery.json"
        if recovery.is_file():
            complete["resident_expert_recovery"] = read_json(recovery)
        batched = self.root / "batched-hessian-recovery.json"
        if batched.is_file():
            complete["batched_hessian_recovery"] = read_json(batched)
        atomic_json(self.root / "numerical-complete.json", complete)
        atomic_json(self.root / "status.json", dict(self.progress.base, status="passed",
                    phase="numerical_complete", updated_at=datetime.now(timezone.utc).isoformat()))
        return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "pilot", "root", "output", "engram-dir", "engram-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--resume-legacy-expert-spool", action="store_true",
                        help="resume the sealed legacy production run with resident expert replay")
    args = parser.parse_args()
    complete = args.root / "numerical-complete.json"
    if complete.exists():
        receipt = read_json(complete)
        package = args.output.resolve() / "package-validation.json"
        if (receipt["status"] != "passed" or receipt["output"] != str(args.output.resolve())
                or receipt["identity"] != read_json(args.root / "identity.json")
                or receipt["package_validation"] != str(package)
                or read_json(package)["status"] != "validated-headers-and-byte-exactness"):
            raise ValueError("Hy4 completion receipt disagrees with requested output")
        print(json.dumps(receipt), flush=True)
        return
    runner = None
    try:
        with torch.inference_mode():
            runner = Runner(args)
            result = runner.run()
        print(json.dumps(result), flush=True)
    except Exception as error:
        path = args.root / "status.json"
        if path.exists():
            status = read_json(path)
        else:
            status = {}
        if args.root.is_dir():
            atomic_json(path, status | dict(status="blocked",
                error=f"{type(error).__name__}: {error}",
                updated_at=datetime.now(timezone.utc).isoformat()))
        raise
    finally:
        if runner is not None:
            runner.reader.close()


if __name__ == "__main__":
    main()
