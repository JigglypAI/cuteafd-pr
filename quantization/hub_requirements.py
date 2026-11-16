"""One qualified Hub SDK for receipt-backed, resumable Xet publication."""
import importlib.metadata

HUB_VERSION = "1.33.0"


def require_qualified_hub():
    from huggingface_hub.utils._runtime import is_xet_available

    if importlib.metadata.version("huggingface_hub") != HUB_VERSION or not is_xet_available():
        raise ValueError(f"publication requires qualified huggingface_hub {HUB_VERSION} with Xet enabled")
