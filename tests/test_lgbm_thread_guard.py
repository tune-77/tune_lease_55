"""runtime_hooks/tunelease_lgbm_thread_guard.py: lightgbm.basic 読込時にスレッド上限1が掛かること。

conftest.py が lightgbm を MagicMock に差し替えるため、別プロセスで偽の
lightgbm パッケージを読ませてフックの発火を確かめる（CI に lightgbm は無い）。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_FAKE_BASIC = """
class _Lib:
    max_threads = None
    def LGBM_SetMaxThreads(self, n):
        _Lib.max_threads = n.value
_LIB = _Lib()
"""

_PROBE = """
import sys
sys.path.insert(0, {fake!r})
sys.path.insert(0, {hooks!r})
import tunelease_lgbm_thread_guard
assert "lightgbm" not in sys.modules
import lightgbm.basic as b
print(b._LIB.max_threads)
"""


def test_guard_limits_lightgbm_threads_on_import(tmp_path):
    pkg = tmp_path / "lightgbm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import basic\n")
    (pkg / "basic.py").write_text(_FAKE_BASIC)
    probe = _PROBE.format(fake=str(tmp_path), hooks=str(ROOT / "runtime_hooks"))
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "1"
