#!/usr/bin/env python3
"""決定論的な監査を実行し、`.claude/reports/` を毎日更新する。

`.claude/reports/` は書き手が居ないと更新されず、全レポートが数ヶ月 STALE に
なっていた。ただし監査の中身を見ると、スコア範囲・DB異常値・ウェイト合計と
いった大半は決定論的な計算であり、LLM もサブエージェントも必要ない
（`.claude/commands/*.md` のクイックモードが既にそう実装されている）。

そこでここでは LLM を一切使わず、素の Python で監査して REPORT_SCHEMA 形式の
レポートを書く。監査こそ LLM に書かせるべきではない。幻覚で「異常なし」と
書かれるのが最悪の失敗だからである。

対象外のエージェント（code-reviewer / security-checker など判断を伴うもの）は
ここでは扱わない。それらは変更駆動であり、鮮度は
`agent_sidecar_reader.AGENT_TRIGGER_PATHS` による担当ファイル基準で判定する。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_ROOT = PROJECT_ROOT / ".claude" / "reports"
DB_PATH = PROJECT_ROOT / "data" / "lease_data.db"
LOGS_DIR = PROJECT_ROOT / "logs"

# 監査対象のコアモジュール（build チェック用）
CORE_MODULES = (
    "scoring_core",
    "asset_scorer",
    "total_scorer",
    "category_config",
    "rule_manager",
    "coeff_definitions",
)


def _rel(path: Path) -> str:
    """表示用の相対パス。リポジトリ外なら絶対パスのまま返す（例外にしない）。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class CheckResult:
    agent: str
    task: str
    status: str = "success"  # success | partial | failure
    summary: str = ""
    details: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    handoff: str = ""

    def fail(self, reason: str) -> "CheckResult":
        self.status = "failure"
        self.risks.append(reason)
        if not self.summary:
            self.summary = f"監査を完了できなかった: {reason}"
        return self


# ── 各監査（すべて決定論的・LLM不使用）──────────────────────────────────────


def check_rules() -> CheckResult:
    """ウェイト合計・グレード閾値の整合性（/validate-rules クイックモード相当）。"""
    r = CheckResult("rule-validator", "ウェイト合計・グレード閾値の整合性チェック")
    try:
        from category_config import ASSET_WEIGHT, CATEGORY_SCORE_ITEMS, SCORE_GRADES
    except Exception as exc:  # noqa: BLE001
        return r.fail(f"category_config を import できない: {exc}")

    bad_weights = [
        f"{cat}: ウェイト合計 {sum(i['weight'] for i in items)}（期待 100）"
        for cat, items in CATEGORY_SCORE_ITEMS.items()
        if sum(i["weight"] for i in items) != 100
    ]
    bad_asset = [
        f"{cat}: asset_w + obligor_w = {w['asset_w'] + w['obligor_w']}（期待 1.0）"
        for cat, w in ASSET_WEIGHT.items()
        if abs(w["asset_w"] + w["obligor_w"] - 1.0) >= 1e-9
    ]
    bad_grades = []
    prev = 101
    for g in SCORE_GRADES:
        if g["min"] >= prev:
            bad_grades.append(f"{g['label']}: min={g['min']} が直前の {prev} 以上で順序が逆転")
        prev = g["min"]

    r.details.append(f"カテゴリ数: {len(CATEGORY_SCORE_ITEMS)} / グレード数: {len(SCORE_GRADES)}")
    r.risks.extend(bad_weights + bad_asset + bad_grades)
    if r.risks:
        r.status = "partial"
        r.summary = f"整合性の逸脱を {len(r.risks)} 件検出した。"
        r.handoff = "逸脱したカテゴリの定義を確認すること。スコア結果に直結する。"
    else:
        r.summary = "ウェイト合計・資産/債務者配分・グレード閾値の順序はいずれも整合している。"
    return r


def check_scoring() -> CheckResult:
    """スコアが定義域に収まるか（/audit-scores クイックモード相当）。"""
    r = CheckResult("scoring-auditor", "スコア範囲と承認ラインの整合性チェック")
    try:
        from asset_scorer import calc_asset_score
        from category_config import ASSET_ID_TO_CATEGORY
    except Exception as exc:  # noqa: BLE001
        return r.fail(f"スコアリングモジュールを import できない: {exc}")

    categories = sorted(set(ASSET_ID_TO_CATEGORY.values()))
    out_of_range: list[str] = []
    errored: list[str] = []
    for cat in categories:
        try:
            score = calc_asset_score(cat, {}, {"lease_months": 36})["total_score"]
        except Exception as exc:  # noqa: BLE001
            errored.append(f"{cat}: 計算に失敗 ({exc})")
            continue
        if not 0 <= score <= 100:
            out_of_range.append(f"{cat}: total_score={score} が 0〜100 の範囲外")

    r.details.append(f"検査カテゴリ数: {len(categories)}")
    r.risks.extend(out_of_range + errored)

    # 承認ラインの単一ソース化（CLAUDE.md の重複定義事故の再発検知）
    try:
        from scoring_core import APPROVAL_LINE

        r.details.append(f"scoring_core.APPROVAL_LINE = {APPROVAL_LINE}")
    except Exception as exc:  # noqa: BLE001
        r.risks.append(f"scoring_core.APPROVAL_LINE を参照できない: {exc}")

    if r.risks:
        r.status = "partial"
        r.summary = f"スコアリングに要確認事項を {len(r.risks)} 件検出した。"
        r.handoff = "審査結果に直結するため、範囲外カテゴリを優先して確認すること。"
    else:
        r.summary = f"全 {len(categories)} カテゴリのスコアが 0〜100 に収まっている。"
    return r


def check_data_quality() -> CheckResult:
    """DB の件数・異常値（/check-data クイックモード相当）。"""
    r = CheckResult("data-quality-checker", "SQLite 審査データの件数・異常値チェック")
    if not DB_PATH.exists():
        return r.fail(f"DB が見つからない: {_rel(DB_PATH)}")

    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
            tables = [
                t for (t,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            counts = {}
            for t in tables:
                try:
                    counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                except sqlite3.Error as exc:
                    r.risks.append(f"{t}: 件数を取得できない ({exc})")
    except sqlite3.Error as exc:
        return r.fail(f"DB を読めない: {exc}")

    r.details.append(f"テーブル数: {len(tables)}")
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        r.details.append(f"  {t}: {c}件")
    empty = [t for t, c in counts.items() if c == 0]
    if empty:
        r.risks.append(f"空のテーブル: {', '.join(sorted(empty))}")

    if r.risks:
        r.status = "partial"
        r.summary = f"{len(tables)} テーブルを確認し、要確認事項を {len(r.risks)} 件検出した。"
    else:
        r.summary = f"{len(tables)} テーブルすべてにデータが存在する。"
    return r


def check_build() -> CheckResult:
    """コアモジュールが import できるか（/build-check クイックモード相当）。"""
    r = CheckResult("build-runner", "コアモジュールのインポート確認")
    failed: list[str] = []
    for name in CORE_MODULES:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {type(exc).__name__}: {exc}")

    r.details.append(f"対象モジュール: {len(CORE_MODULES)}")
    r.risks.extend(failed)
    if failed:
        r.status = "failure" if len(failed) == len(CORE_MODULES) else "partial"
        r.summary = f"{len(failed)}/{len(CORE_MODULES)} のモジュールが import できない。"
        r.handoff = "依存パッケージの不足かモジュール側の構文エラー。先に解消しないと他の監査も動かない。"
    else:
        r.summary = f"コアモジュール {len(CORE_MODULES)} 件すべて import できる。"
    return r


def check_logs() -> CheckResult:
    """ログのエラー・警告件数（log-file-analyzer 相当）。"""
    r = CheckResult("log-file-analyzer", "ログのエラー・警告抽出")
    if not LOGS_DIR.exists():
        r.summary = "logs/ が存在しないため解析対象なし。"
        r.details.append(f"想定パス: {_rel(LOGS_DIR)}")
        return r

    errors = warnings = 0
    scanned = 0
    samples: list[str] = []
    for path in sorted(LOGS_DIR.glob("*.log")):
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            r.risks.append(f"{path.name}: 読めない ({exc})")
            continue
        for line in text.splitlines():
            if "ERROR" in line or "Traceback" in line:
                errors += 1
                if len(samples) < 5:
                    samples.append(f"{path.name}: {line.strip()[:160]}")
            elif "WARNING" in line:
                warnings += 1

    r.details.append(f"走査ファイル数: {scanned} / ERROR: {errors} 行 / WARNING: {warnings} 行")
    r.details.extend(f"  {s}" for s in samples)
    if errors:
        r.status = "partial"
        r.summary = f"{scanned} ファイルから ERROR {errors} 行、WARNING {warnings} 行を検出した。"
        r.handoff = "頻出パターンは analyze_error_logs.py が改善台帳へ起票する。"
    else:
        r.summary = f"{scanned} ファイルを走査し ERROR は検出されなかった（WARNING {warnings} 行）。"
    return r


def check_api_health() -> CheckResult:
    """依存サービスの設定・到達性（/check-health クイックモード相当）。

    ネットワーク到達性は Ollama のみ短いタイムアウトで確認する。外部 API は
    キーの有無だけを見る。日次で課金や rate limit を踏まないため。
    """
    r = CheckResult("api-health-checker", "依存サービスの設定・到達性確認")

    r.details.append(f"SQLite: {'あり' if DB_PATH.exists() else 'なし'}")
    if not DB_PATH.exists():
        r.risks.append("data/lease_data.db が存在しない")

    for key in ("GEMINI_API_KEY", "SLACK_BOT_TOKEN"):
        present = bool(str(os.environ.get(key) or "").strip())
        r.details.append(f"{key}: {'設定あり' if present else '未設定'}")
        if not present:
            r.risks.append(f"{key} が未設定（該当機能は動作しない）")

    ollama = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        import urllib.request

        with urllib.request.urlopen(f"{ollama}/api/tags", timeout=3) as resp:
            ok = resp.status == 200
    except Exception as exc:  # noqa: BLE001
        ok = False
        r.risks.append(f"Ollama に到達できない（{ollama}）: {type(exc).__name__}")
    r.details.append(f"Ollama ({ollama}): {'到達' if ok else '不到達'}")

    if r.risks:
        r.status = "partial"
        r.summary = f"依存サービスに要確認事項を {len(r.risks)} 件検出した。"
    else:
        r.summary = "SQLite・APIキー・Ollama いずれも利用可能。"
    return r


def check_tests(timeout: int = 600) -> CheckResult:
    """ユニットテストの実行結果（/run-tests 相当）。"""
    r = CheckResult("test-runner", "ユニットテスト実行")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return r.fail(f"pytest が {timeout}s でタイムアウトした")
    except Exception as exc:  # noqa: BLE001
        return r.fail(f"pytest を起動できない: {exc}")

    tail = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()][-3:]
    r.details.extend(tail)
    if proc.returncode == 0:
        r.summary = "テストはすべて通過した。"
    else:
        r.status = "partial"
        r.summary = f"pytest が非ゼロ終了（exit={proc.returncode}）。"
        r.risks.append(tail[-1] if tail else "詳細不明")
        r.handoff = "失敗したテストを test-result-analyzer で切り分けること。"
    return r


# レポートディレクトリ名 → 監査関数。名前はディレクトリ名であってエージェント名
# ではない（scoring-auditor → scoring-audit/）。正は .claude/AGENTS.md。
CHECKS = {
    "rule-validation": check_rules,
    "scoring-audit": check_scoring,
    "data-quality": check_data_quality,
    "build": check_build,
    "log-analysis": check_logs,
    "api-health": check_api_health,
    "test-results": check_tests,
}


# ── レポート出力 ────────────────────────────────────────────────────────────


SELF_MARKER = "<!-- generated-by: build_agent_self_reports.py -->"
CARRIED_HEADING = "## 過去のエージェント指摘（未検証）"
PREVIOUS_NAME = "previous-agent-report.md"


def preserve_previous_report(path: Path) -> str:
    """上書き前のエージェント作成レポートを退避し、参照文を返す。

    決定論的な監査は、エージェントが書いた「係数の飽和」のような判断込みの
    指摘を再現できない。黙って上書きすると消えてしまう。

    セクションを抜き出して埋め込む方式は採らない。過去のレポートは
    REPORT_SCHEMA に厳密に従っておらず（`## 課題・リスク` を持たず `###` で
    階層化しているものがある）、抽出すると静かに取りこぼすため。
    ファイルごと退避すれば構造に依存せず無損失で残せる。

    sidecar は `*/latest.md` しか読まないので、退避先はブリーフを汚さない。
    """
    if not path.exists():
        return ""
    existing = path.read_text(encoding="utf-8", errors="ignore")
    kept = path.parent / PREVIOUS_NAME

    if SELF_MARKER not in existing:
        # エージェントが書いたレポート。初回だけ退避する。
        kept.write_text(existing, encoding="utf-8")
    elif not kept.exists():
        return ""

    stamp = ""
    m = re.search(r"^timestamp:\s*(.+)$", kept.read_text(encoding="utf-8"), re.MULTILINE)
    if m:
        stamp = m.group(1).strip()
    return (
        f"{stamp or '時刻不明'} 時点のエージェント作成レポートを "
        f"`{PREVIOUS_NAME}` に退避した。決定論的な監査では再現できない指摘を"
        "含むため、現在のコードで検証すること。"
    )


def render_report(
    result: CheckResult,
    now: dt.datetime | None = None,
    carried: str = "",
) -> str:
    """REPORT_SCHEMA.md の書式でレポート本文を組み立てる。"""
    stamp = (now or dt.datetime.now()).strftime("%Y-%m-%d %H:%M")
    lines = [
        "---",
        f"agent: {result.agent}",
        f"task: {result.task}",
        f"timestamp: {stamp}",
        f"status: {result.status}",
        "reads_from: []",
        "---",
        "",
        "## サマリー",
        result.summary or "（結果なし）",
        "",
        "## 詳細",
    ]
    lines.extend(f"- {d}" for d in (result.details or ["（詳細なし）"]))
    lines.extend(["", "## 課題・リスク"])
    lines.extend(f"- {x}" for x in (result.risks or ["なし"]))
    lines.extend(["", "## 後続エージェントへの申し送り", result.handoff or "なし"])
    if carried:
        lines.extend(["", CARRIED_HEADING, carried])
    lines.extend(
        [
            "",
            SELF_MARKER,
            "<!-- 決定論的監査・LLM不使用。手で編集しても次回実行で上書きされる。",
            "     『過去のエージェント指摘』は解消したら手で消すこと。 -->",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(report_dir: str, result: CheckResult, root: Path | None = None) -> Path:
    base = root if root is not None else REPORT_ROOT
    path = base / report_dir / "latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_report(result, carried=preserve_previous_report(path)), encoding="utf-8"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="対象を絞る（複数指定可）")
    parser.add_argument("--skip", action="append", default=[], help="除外する（複数指定可）")
    args = parser.parse_args()

    targets = {k: v for k, v in CHECKS.items() if not args.only or k in args.only}
    targets = {k: v for k, v in targets.items() if k not in args.skip}
    if not targets:
        print("[self-report] 対象がありません")
        return 0

    ok = 0
    for report_dir, check in targets.items():
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001 - 1件の失敗で全体を止めない
            result = CheckResult(report_dir, "自動監査").fail(f"想定外の例外: {exc}")
        path = write_report(report_dir, result)
        ok += result.status != "failure"
        print(f"[self-report] {result.status:8} {report_dir} → {_rel(path)}")

    print(f"[self-report] 完了: {ok}/{len(targets)} 件が failure 以外")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
