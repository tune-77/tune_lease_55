"""fetch_fincept_data.py の耐障害性テスト（REV-034a 復旧）。

akshare は任意依存（requirements 未収録）。未導入/壊れていても、マクロ更新ステップは
日次改善パイプラインを止めないよう **exit 0（健全スキップ）** で終わることを保証する。
以前は ImportError で exit 1 になり、パイプライン健全性モニタに「失敗」として検出されていた。
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_fincept_data.py"


def test_exits_zero_when_akshare_unavailable(tmp_path):
    # 実行環境へのakshareインストール有無に依存しないよう、ImportErrorを起こす
    # シャドーモジュールをPYTHONPATHの先頭に置いてスキップ経路を再現する。
    (tmp_path / "akshare.py").write_text(
        'raise ImportError("akshare intentionally unavailable in test")\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}, stderr={proc.stderr[:500]}"
    combined = proc.stdout + proc.stderr
    assert "スキップ" in combined


def test_import_guard_catches_broad_exception():
    # 未導入だけでなく壊れた依存(ImportError以外)も握れるよう、広い except になっていること。
    src = _SCRIPT.read_text(encoding="utf-8")
    assert "except Exception" in src
    # ImportError で exit 1 する旧実装に戻っていないこと。
    assert "sys.exit(1)" not in src
