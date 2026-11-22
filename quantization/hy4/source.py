"""Complete header and byte census for the Tencent Hy4 BF16-source revision."""
from collections import Counter
import hashlib

from hy4_family_source import PinnedHeaderSource
from kimi_k3.uniform_k2.storage import canonical

BASE_REPO = "tencent/Hy4"
BASE_REVISION = "c10c0de5b6f850703fb0a90e65d1c32b558b9167"
BASE_PAYLOAD_BYTES = 1_548_929_595_126
SHARD_COUNT = 245
SOURCE_CONFIG_SHA256 = "dab00b0f4c9265f650eabccc1033bd6aa29080cf1459856a9f2a06ae074d7714"
SOURCE_INDEX_SHA256 = "8433cb5cfa21cf7dda97139c4f3aba58682063ef95df7b4105674284872041d6"
SOURCE_SPECS_SHA256 = "5f0ceb39314a09ecd136fea91a17e1b253b6b0eee65e90fa30421b4eb35d122b"


def group_for(name):
    if name.startswith("vit."):
        return "vision"
    if ".mlp.experts." in name:
        return "routed_experts"
    if name.startswith("model.dfly_layers."):
        return "dfly2"
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return "embedding_and_head"
    return "target_core"


def audit_source(source):
    specs_digest = hashlib.sha256(canonical(source.specs).encode()).hexdigest()
    if (source.tensor_count != 2710 or source.payload_bytes != BASE_PAYLOAD_BYTES
            or specs_digest != SOURCE_SPECS_SHA256):
        raise ValueError("Hy4 source byte total or reviewed tensor inventory differs")
    groups, tensors = Counter(), Counter()
    for name, size in source.sizes.items():
        group = group_for(name)
        groups[group] += size
        tensors[group] += 1
    return dict(source_tensors=source.tensor_count, source_bytes=source.payload_bytes,
                source_dtypes=dict(Counter(dtype for dtype, _ in source.specs.values())),
                source_group_bytes=dict(groups), source_group_tensors=dict(tensors),
                config_sha256=SOURCE_CONFIG_SHA256, index_sha256=SOURCE_INDEX_SHA256,
                specs_sha256=specs_digest, source_revision=BASE_REVISION)


class HeaderSource(PinnedHeaderSource):
    def __init__(self, snapshot):
        super().__init__(snapshot, repo=BASE_REPO, revision=BASE_REVISION,
                         config_sha256=SOURCE_CONFIG_SHA256, index_sha256=SOURCE_INDEX_SHA256,
                         shard_count=SHARD_COUNT, payload_bytes=BASE_PAYLOAD_BYTES,
                         audit_source=audit_source)
