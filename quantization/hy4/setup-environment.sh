#!/usr/bin/env bash
# Prepare the full Hy4 reference and production environment on maxq.
set -euo pipefail

task_root="$HOME/Developer/quantization-runs/hy4-k4-mc"
build_root="$HOME/.cache/cuteafd/builds/hy4-k4-mc"
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
reference="$build_root/transformers-reference"
baseline=2f1bc5e321b9a736200a590d6e7743ccdeb60772
rope_fix=bd5b6b6d979cd22a7d237d9e84b88d212b08a545
python="$task_root/environment/bin/python"

if [[ -e "$task_root/environment" ]]; then
    echo "The Hy4 task environment already exists at $task_root/environment" >&2
    exit 1
fi
"$repo_root/scripts/build/assert-build-filesystem.py" "$build_root"
mkdir -p "$build_root" "$task_root"
git init --quiet "$reference"
git -C "$reference" fetch --depth=1 https://github.com/huggingface/transformers.git "$baseline" "$rope_fix"
git -C "$reference" checkout --detach "$baseline"
git -C "$reference" diff "$baseline" "$rope_fix" --stat -- src/transformers/models/hy_v4 > "$task_root/reference-pr-diff.txt"

uv venv --python 3.12.7 "$task_root/environment"
uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130'
CC=/usr/bin/gcc CXX=/usr/bin/g++ uv pip install --python "$python" "$reference" \
    'huggingface-hub==1.33.0' 'safetensors==0.8.0' 'tokenizers==0.23.3' 'numpy==2.2.6' \
    'xxhash==4.1.0' 'hf-xet==1.8.0' 'ninja==1.13.2' 'pypcre==0.6.6' 'logbar==0.4.16' \
    'jinja2==3.1.6' 'psutil==7.2.2' 'packaging==26.3' 'pytest==9.2.0' 'tokenicer==0.0.21'
uv pip check --python "$python"
