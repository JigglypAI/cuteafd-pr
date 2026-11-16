#!/usr/bin/env python3
"""Publish a completed immutable run with the qualified streaming Hub API."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hub_requirements import require_qualified_hub


def install_upload_compat():
    require_qualified_hub()
    from huggingface_hub import HfApi

    if hasattr(HfApi, "upload_large_folder"):
        return
    # Hub 2 removed the old name. With Xet, upload_folder now provides streamed,
    # resumable, adaptive multi-commit uploads itself. Its worker/report settings
    # are managed by that pipeline rather than the retired large-folder API.
    def upload_large_folder(self, *, repo_id, repo_type, folder_path,
                            ignore_patterns, num_workers=4, print_report=True,
                            print_report_every=1800):
        return self.upload_folder(repo_id=repo_id, repo_type=repo_type,
                                  folder_path=folder_path, ignore_patterns=ignore_patterns)

    HfApi.upload_large_folder = upload_large_folder


def main():
    if "--publish-only" not in sys.argv[1:]:
        raise SystemExit("This entry point requires --publish-only")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    install_upload_compat()
    # Keep the original producer and its plan hashes unchanged on recovery;
    # its completion-marker checks and remote checksum audit still run.
    runpy.run_path(str(Path(__file__).with_name("run.py")), run_name="__main__")


if __name__ == "__main__":
    main()
