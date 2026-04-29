"""リース審査AIシステムの統一エントリポイント。"""

from __future__ import annotations

import runpy
from pathlib import Path


APP_FILE = Path(__file__).with_name("lease_logic_sumaho12.py")


def main() -> None:
    """Streamlit実行時にUIロジックをロードする。"""
    runpy.run_path(str(APP_FILE), run_name="__main__")


if __name__ == "__main__":
    main()
