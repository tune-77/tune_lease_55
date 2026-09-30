"""runtime_hooks/tunelease_lgbm_thread_guard.py: lightgbm 読込時にスレッド上限1が掛かること。"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PROBE = """
import ctypes, sys
sys.path.insert(0, {hooks!r})
import tunelease_lgbm_thread_guard
assert "lightgbm" not in sys.modules
import lightgbm.basic as b
n = ctypes.c_int()
b._LIB.LGBM_GetMaxThreads(ctypes.byref(n))
print(n.value)
"""


def test_guard_limits_lightgbm_threads_on_import():
    pytest.importorskip("lightgbm")
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(hooks=str(ROOT / "runtime_hooks"))],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "1"
