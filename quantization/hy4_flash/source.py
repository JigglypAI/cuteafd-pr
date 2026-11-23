"""Header inventory for the pinned Hy4-flash checkpoint."""
from hy4_family_source import PinnedHeaderSource
from . import recipe


class HeaderSource(PinnedHeaderSource):
    def __init__(self, snapshot):
        super().__init__(snapshot, repo=recipe.BASE_REPO, revision=recipe.BASE_REVISION,
                         config_sha256=recipe.SOURCE_CONFIG_SHA256,
                         index_sha256=recipe.SOURCE_INDEX_SHA256, shard_count=recipe.SHARD_COUNT,
                         payload_bytes=recipe.BASE_PAYLOAD_BYTES, audit_source=recipe.audit_source)
