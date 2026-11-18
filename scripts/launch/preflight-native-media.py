#!/usr/bin/env python3
"""Check the serving library's exported AOT symbols without loading CUDA."""
import argparse
import mmap
from pathlib import Path
import struct

# Check real generated entrypoints, not the always-present vision C ABI stub.
REQUIRED = {
    'vision': ('vision attention AOT', 'VISION_ATTENTION_AOT', 'VISION', (
        '_mlir_cuteafd_vision_attention_d64_cuda_init',
        '_mlir_cuteafd_vision_attention_d72_cuda_init',
    )),
    'mimo-audio': ('MiMo audio AOT', 'AUDIO_AOT', 'AUDIO', (
        'cuteafd_audio_create', '_mlir_cuteafd_audio_layer_norm_1024_1_1_cuda_init',
    )),
    'pcm-audio': ('PCM audio AOT', 'AUDIO_AOT', 'AUDIO', (
        'cuteafd_pcm_create', '_mlir_cuteafd_audio_pcm_causal_conv7_h256_cuda_init',
    )),
}


def exports(path):
    # All shipping images are little-endian ELF64 Linux. Use SHT_DYNSYM and
    # its linked string table so neither binutils nor driver libraries are needed.
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as elf:
        if elf[:7] != b'\x7fELF\x02\x01\x01':
            raise ValueError('not a little-endian ELF64 library')
        offset, = struct.unpack_from('<Q', elf, 40)
        width, count = struct.unpack_from('<HH', elf, 58)
        if width != 64 or not count:
            raise ValueError('missing ELF section table')
        sections = [struct.unpack_from('<IIQQQQIIQQ', elf, offset + width * i) for i in range(count)]
        for _, kind, _, _, start, size, strings, _, _, entry in sections:
            if kind == 11:  # SHT_DYNSYM
                if entry != 24:
                    raise ValueError('invalid ELF symbol size')
                names = sections[strings][4]
                defined = set()
                for pos in range(start, start + size, entry):
                    name, info, visibility, section = struct.unpack_from('<IBBH', elf, pos)
                    if section and info >> 4 in (1, 2) and visibility & 3 in (0, 3):
                        end = elf.find(b'\0', names + name)
                        defined.add(elf[names + name:end].decode('ascii', errors='replace'))
                return defined
        raise ValueError('missing ELF dynamic symbols')


def check(path, scope, capabilities):
    # The daemon may select any installed host-ISA sibling. Verify each carries
    # the same capability; this also covers a forced HOST_ISA override.
    for lib in [path, *sorted(path.parent.glob(f'*/{path.name}')),
                *sorted(path.parent.glob(f'lib-variants/*/{path.name}'))]:
        available = exports(lib)
        for capability in capabilities:
            name, flag, option, symbols = REQUIRED[capability]
            missing = sorted(set(symbols) - available)
            if missing:
                raise ValueError(f'{lib}: this build lacks {name}; rebuild with '
                                 f'CUTEAFD_{scope}_{flag}=ON or set {option}=off '
                                 f'(missing {", ".join(missing)})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    parser.add_argument('scope', choices=('WIP', 'RELEASE'))
    parser.add_argument('capabilities', nargs='+', choices=REQUIRED)
    args = parser.parse_args()
    try:
        check(args.library, args.scope, args.capabilities)
    except (OSError, ValueError, IndexError, struct.error) as error:
        parser.exit(2, f'native media preflight: {error}\n')


if __name__ == '__main__':
    main()
