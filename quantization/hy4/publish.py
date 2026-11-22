"""Publish the sealed full Hy4 K4 MC snapshot using the verified Hub lifecycle."""
from hy4_flash.publish import run_cli, validate_mc
from inkling2.publish import PublicationTarget
from .source import BASE_REPO, BASE_REVISION

TARGET_REPO = 'wrldsuksgo2mars/Hy4-EXL3-K4-MC'
FAMILY = 'hy4'


def _validate_package(output, state, validation):
    manifest = validate_mc(output, state, validation, repo=BASE_REPO, revision=BASE_REVISION)
    if 'engram_fp8' in manifest:
        raise ValueError('unexpected Engram alternate in full Hy4 package')


TARGET = PublicationTarget(TARGET_REPO, FAMILY, 'Publish calibrated Hy4 K4 MC checkpoint', _validate_package)


def main():
    run_cli(TARGET, BASE_REVISION)


if __name__ == '__main__':
    main()
