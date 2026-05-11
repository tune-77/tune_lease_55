"""軽量な計測ユーティリティ。

提供:
- `timeit` デコレータ: 関数実行時間を logger.info で出力
- `log_dataframe_info` : pandas.DataFrame のサイズ・メモリ情報を logger.debug で出力
"""
from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


def timeit(fn: Callable):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.time() - t0
            try:
                logger.info("%s took %.3fs", fn.__name__, dt)
            except Exception:
                pass

    return wrapped


def log_dataframe_info(df, name: str = "df"):
    """DataFrame の行数・列数・メモリ使用量をデバッグ出力する。"""
    try:
        import pandas as pd
        if not hasattr(df, "shape"):
            logger.debug("%s is not a DataFrame", name)
            return
        rows, cols = df.shape
        mem = df.memory_usage(deep=True).sum()
        logger.debug("%s: rows=%d cols=%d mem=%.2fKB", name, rows, cols, mem / 1024.0)
    except Exception as e:
        logger.debug("log_dataframe_info error: %s", e)
