"""Run model-written programs in a sandboxed subprocess and grade them against GSM8K answers.

Each program runs in a fresh `python -I` process, in an empty temporary directory, with a minimal
environment, a wall-clock timeout (the whole process group is killed), and CPU-time, memory and
file-size limits. This contains accidents from our own small model (infinite loops, huge
allocations, stray file writes); it is not a hardened boundary against adversarial code.
"""

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# The only variables the child inherits. A module-loaded Python may need LD_LIBRARY_PATH to start.
_ENV_VARS = ("PATH", "LD_LIBRARY_PATH")

# Runs inside the child process. Prints from the program are discarded; the only thing written to
# the real stdout is one JSON object describing the return value of simple_math_problem().
_RUNNER = r"""
import json, math, numbers, os, resource, sys
for name, value in (("RLIMIT_CPU", %(cpu_s)d), ("RLIMIT_AS", %(mem_bytes)d), ("RLIMIT_FSIZE", 0)):
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
    except (AttributeError, ValueError, OSError):
        pass  # e.g. RLIMIT_AS cannot be set on macOS
source = sys.stdin.buffer.read().decode("utf-8")
out = sys.stdout
sys.stdout = sys.stderr = open(os.devnull, "w")
try:
    namespace = {"__name__": "__generated__"}
    exec(compile(source, "<generated>", "exec"), namespace)
    value = namespace["simple_math_problem"]()
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        result = {"status": "non_numeric", "detail": type(value).__name__}
    elif not math.isfinite(float(value)):
        result = {"status": "non_numeric", "detail": repr(float(value))}
    else:
        result = {"status": "ok", "value": float(value)}
except BaseException as e:
    result = {"status": "error", "detail": (type(e).__name__ + ": " + str(e))[:300]}
out.write(json.dumps(result))
"""


@dataclass
class ExecResult:
    status: str  # ok | error | non_numeric | timeout | crash
    value: float | None = None
    detail: str = ""


def run_program(source, timeout=5.0, memory_mb=1024):
    """Execute `source` and return what its simple_math_problem() returned."""
    runner = _RUNNER % {"cpu_s": math.ceil(timeout) + 1, "mem_bytes": memory_mb << 20}
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", runner],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            env={k: os.environ[k] for k in _ENV_VARS if k in os.environ},
            start_new_session=True,  # own process group, so a timeout kills any children too
        )
        try:
            out, _ = proc.communicate(source.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return ExecResult("timeout")
    try:
        return ExecResult(**json.loads(out))
    except (ValueError, TypeError):
        return ExecResult("crash", detail=f"exit code {proc.returncode}")


def self_test():
    """Fail loudly if the sandbox itself is broken, instead of silently grading everything as wrong."""
    result = run_program("def simple_math_problem() -> int:\n    return 2 + 2\n")
    if result.status != "ok" or result.value != 4:
        raise RuntimeError(f"sandbox self-test failed: {result}")


def run_programs(sources, timeout=5.0, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda s: run_program(s, timeout), sources))


def parse_gold(answer):
    """GSM8K answers end with '#### <number>', possibly with thousands separators."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


def is_correct(result, gold):
    """Numeric match with a small tolerance (40 and 40.0 both match 40). Anything but 'ok' is wrong."""
    return result.status == "ok" and math.isclose(result.value, gold, rel_tol=1e-9, abs_tol=1e-4)
