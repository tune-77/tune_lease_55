"""
app_logger.py — アプリケーションログ管理
==========================================
エラー・警告・情報を logs/app.log に記録する。
Streamlit 画面には「ログを確認してください」とだけ表示し、
tracebackを画面に出さない。

使い方:
    from app_logger import log_error, log_info, log_warning

    try:
        ...
    except Exception as e:
        log_error(e, context="スコア計算中")
        st.error("❌ エラーが発生しました。管理者にお問い合わせください。（logs/app.log）")
"""

import logging
import traceback
import os
import sys
from pathlib import Path
from datetime import datetime

# ログ出力先
_PKG_DIR = Path(__file__).parent
_LOG_DIR = _PKG_DIR / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

# ログローテーション（10MB × 3世代）
_MAX_BYTES   = 10 * 1024 * 1024
_BACKUP_COUNT = 3


def _get_logger() -> logging.Logger:
    """アプリケーションロガーを返す（初回のみセットアップ）。"""
    logger = logging.getLogger("lease_logic")
    if logger.handlers:
        return logger  # 既にセットアップ済み

    logger.setLevel(logging.DEBUG)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ファイルハンドラ（ローテーション付き）
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
    except Exception:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")

    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_error(exc: Exception, context: str = "") -> None:
    """
    例外をログに記録する。

    Parameters
    ----------
    exc     : 捕捉した例外
    context : どの処理で発生したかの説明文
    """
    logger = _get_logger()
    tb = traceback.format_exc()
    loc = f"[{context}] " if context else ""
    logger.error(f"{loc}{type(exc).__name__}: {exc}\n{tb}")


def log_warning(msg: str, context: str = "") -> None:
    """警告メッセージをログに記録する。"""
    logger = _get_logger()
    loc = f"[{context}] " if context else ""
    logger.warning(f"{loc}{msg}")


def log_info(msg: str, context: str = "") -> None:
    """情報メッセージをログに記録する。"""
    logger = _get_logger()
    loc = f"[{context}] " if context else ""
    logger.info(f"{loc}{msg}")


def get_log_path() -> str:
    """ログファイルのパスを返す。"""
    return str(_LOG_FILE)


def read_recent_logs(n_lines: int = 50) -> list[str]:
    """
    ログファイルの末尾 n_lines 行を返す（Streamlit表示用）。
    """
    if not _LOG_FILE.exists():
        return ["（ログファイルはまだありません）"]
    try:
        with open(_LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n_lines:]]
    except Exception:
        return ["（ログの読み込みに失敗しました）"]
