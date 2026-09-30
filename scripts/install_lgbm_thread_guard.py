"""runtime_hooks/tunelease_lgbm_thread_guard.py を venv の起動処理に登録する（冪等）。

site-packages に .pth を置き、この venv の python で起動する全プロセス
（API・launchd ジョブ・日次パイプラインの各ステップ・手動スクリプト）で
LightGBM のスレッド上限 1 が効くようにする。venv を作り直したら再実行する。
日次パイプライン末尾でも毎回実行して自己修復する。
"""
import site
import sys
from pathlib import Path

PTH_NAME = "tunelease_lgbm_thread_guard.pth"
HOOK_DIR = Path(__file__).resolve().parent.parent / "runtime_hooks"


def main() -> int:
    site_packages = Path(site.getsitepackages()[0])
    content = f"{HOOK_DIR}\nimport tunelease_lgbm_thread_guard\n"
    pth = site_packages / PTH_NAME
    if pth.exists() and pth.read_text() == content:
        print(f"[lgbm-thread-guard] 登録済み: {pth}")
        return 0
    pth.write_text(content)
    print(f"[lgbm-thread-guard] 登録しました: {pth} ({sys.executable})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
