#!/usr/bin/env python3
"""Own a build process group, including descendants that start new groups.

Linux subreaping keeps escaped grandchildren observable when their parents
exit. --channel ties a remote or Docker exec to its client's open stdin;
--pipe supplies that lifetime channel without forwarding terminal input.
"""
import argparse
import contextlib
import ctypes
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def children():
    # The supervisor is single-threaded. Only its direct children are signalled
    # individually; their own orphaned children are reparented here, not to PID 1.
    return [int(pid) for pid in Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()]


def supervise(command, *, channel=False, pipe=False, timeout=None, grace=5):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'cannot supervise build descendants')
    stopped = None

    def stop(signum, _frame=None):
        nonlocal stopped
        if stopped is None:
            stopped = signum

    # SIGINT/QUIT can be inherited as ignored from a background Bash command.
    previous = {sig: signal.signal(sig, stop) for sig in SIGNALS}
    selector = selectors.DefaultSelector()
    process = None
    try:
        if channel:
            selector.register(sys.stdin, selectors.EVENT_READ)
        if stopped is not None:
            return 128 + stopped
        # Build and instance flock descriptors intentionally survive through the
        # transport process until it finishes draining its remote command.
        process = subprocess.Popen(command, stdin=subprocess.PIPE if pipe else subprocess.DEVNULL if channel else None,
                                   start_new_session=True, close_fds=False)
        deadline = None if timeout is None else time.monotonic() + timeout
        expired = False
        while stopped is None and process.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                expired = True
                stop(signal.SIGTERM)
                break
            if channel:
                if selector.select(0.1) and not os.read(sys.stdin.fileno(), 4096):
                    stop(signal.SIGTERM)
            else:
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
        if stopped is not None:
            print(f'build process terminated (exit {128 + stopped})', file=sys.stderr)
        if process.stdin is not None:
            process.stdin.close()
        if stopped is not None or children():
            # Signal the whole original group even if its leader has exited.
            # Nested supervisors handle their own remote cleanup; use a bounded
            # grace period before killing and reaping any remaining descendants.
            kill_at = time.monotonic() + grace
            sent = set()
            if not pipe:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            while True:
                process.poll()
                pids = children()
                if not pids:
                    break
                force = time.monotonic() >= kill_at
                for pid in pids:
                    if force or (not pipe and pid not in sent):
                        signum = signal.SIGKILL if force else signal.SIGTERM
                        # Most adopted children own a fresh timeout/session group.
                        # Signal that group immediately, so a nested shepherd
                        # can run its own cleanup while we still have time.
                        with contextlib.suppress(ProcessLookupError):
                            if os.getpgid(pid) == pid:
                                os.killpg(pid, signum)
                            else:
                                os.kill(pid, signum)
                        sent.add(pid)
                    if pid != process.pid:
                        with contextlib.suppress(ChildProcessError):
                            os.waitpid(pid, os.WNOHANG)
                time.sleep(0.02)
        code = process.wait()
        if expired:
            return 124
        if stopped is not None:
            return 128 + stopped
        return code if code >= 0 else 128 - code
    finally:
        selector.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', action='store_true')
    parser.add_argument('--pipe', action='store_true')
    parser.add_argument('--timeout', type=float)
    parser.add_argument('--grace', type=float, default=5)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    return supervise(command, channel=args.channel, pipe=args.pipe, timeout=args.timeout,
                     grace=args.grace)


if __name__ == '__main__':
    sys.exit(main())
