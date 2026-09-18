"""Shared helper for reporting a pipeline script's failure reason to stderr.

scripts/detect_pipeline_failures.py の parse_pipeline_log() は
"エラー:"/"ERROR:"/"警告:"/"WARNING:" 接頭辞の行だけを critical_errors/warnings
として拾う。各スクリプトが SystemExit(message) や print() を個別に書いていると
接頭辞の付け忘れで、原因がログに残っていても分類・検出されない（REV-395a/396a/397a
調査時に確認）。ここに一本化する。
"""
from __future__ import annotations

import sys


def report_pipeline_failure(message: str, *, level: str = "エラー") -> None:
    print(f"{level}: {message}", file=sys.stderr)
