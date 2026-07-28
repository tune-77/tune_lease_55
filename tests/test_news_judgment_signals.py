from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "lease_news_digest.py"
_SPEC = importlib.util.spec_from_file_location("lease_news_digest", _MODULE_PATH)
assert _SPEC and _SPEC.loader
digest = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = digest
_SPEC.loader.exec_module(digest)


def test_write_news_judgment_signals_converts_actions_to_screening_candidates(tmp_path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    note = news_dir / "2026-07-28_業界リスクニュース_物流燃料費.md"
    note.write_text(
        """---
date: 2026-07-28
tags: ["物流/車両"]
region: 国内
source: Example News
importance: 中
---
# 物流会社で燃料費と人件費が上昇

## 3行要約
- 運送業で燃料費と人件費の負担が増えている。
- 車両更新投資は増えるが採算確認が必要。
- 稼働率と運賃改定が焦点になる。

## 活用メモ
審査では増車理由、ルート別採算、運転手確保を確認する。
""",
        encoding="utf-8",
    )

    output = tmp_path / "news_judgment_signals.jsonl"
    latest = tmp_path / "news_judgment_signals_latest.json"

    result = digest.write_news_judgment_signals(
        date_str="2026-07-28",
        vault=vault,
        path=output,
        latest_path=latest,
    )

    assert result["count"] == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    signal = rows[0]
    assert signal["id"].startswith("ns-")
    assert signal["source"] == "news_judgment_signals"
    assert signal["promotion_status"] == "news_signal"
    assert signal["actionability"] == "actionable"
    assert "物流・運輸" in signal["affected_industries"]
    assert "車両" in signal["affected_assets"]
    assert "稼働率" in signal["claim"]
    assert signal["do_not_use_when"]

    latest_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert latest_payload["policy"] == "news_signals_are_screening_context_not_promoted_judgment_assets"


def test_write_news_judgment_signals_replaces_same_day_rows_and_drops_parts_noise(tmp_path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    (news_dir / "2026-07-28_業界リスクニュース_部品販売.md").write_text(
        """---
date: 2026-07-28
tags: ["物流/車両"]
region: 国内
source: Example News
importance: 中
---
# いすゞ イスズ純正部品 EGRバルブ 新品未使用 - Universidad de Sevilla

## 3行要約
- 部品販売の記事。

## 活用メモ
審査では車両更新を確認する。
""",
        encoding="utf-8",
    )

    output = tmp_path / "news_judgment_signals.jsonl"
    output.write_text(
        json.dumps(
            {
                "id": "ns-stale",
                "source": "news_judgment_signals",
                "research_date": "2026-07-28",
                "actionability": "actionable",
                "claim": "古い同日シグナル",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = digest.write_news_judgment_signals(
        date_str="2026-07-28",
        vault=vault,
        path=output,
        latest_path=tmp_path / "latest.json",
    )

    assert result["count"] == 0
    assert output.read_text(encoding="utf-8") == ""


def test_build_news_judgment_signals_collapses_same_day_similar_viewpoints(tmp_path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    for name, title in [
        ("共同物流.md", "シーオス、共同物流の実践手法を公開"),
        ("物流チャット.md", "物流倉庫の現場チャットとトラック呼出しを提供開始"),
    ]:
        (news_dir / f"2026-07-28_業界リスクニュース_{name}").write_text(
            f"""---
date: 2026-07-28
tags: ["物流/車両"]
region: 国内
source: Example News
importance: 中
---
# {title}

## 3行要約
- 物流現場の効率化に関する記事。

## 活用メモ
審査では増車理由、ルート別採算、運転手確保を確認する。
""",
            encoding="utf-8",
        )

    signals = digest.build_news_judgment_signals(date_str="2026-07-28", vault=vault, limit=5)

    assert len(signals) == 1
    assert signals[0]["dedupe_summary"]["suppressed"] == 1
    assert signals[0]["duplicates_collapsed"]


def test_write_news_judgment_signals_suppresses_repeated_valid_prior_signal(tmp_path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    (news_dir / "2026-07-28_業界リスクニュース_物流.md").write_text(
        """---
date: 2026-07-28
tags: ["物流/車両"]
region: 国内
source: Example News
importance: 中
---
# 物流現場の効率化

## 3行要約
- 物流現場の効率化に関する記事。

## 活用メモ
審査では増車理由、ルート別採算、運転手確保を確認する。
""",
        encoding="utf-8",
    )
    prior = digest.build_news_judgment_signals(date_str="2026-07-28", vault=vault, limit=5)[0]
    prior["id"] = "ns-prior"
    prior["research_date"] = "2026-07-20"
    prior["valid_until"] = "2026-10-26"
    output = tmp_path / "news_judgment_signals.jsonl"
    output.write_text(json.dumps(prior, ensure_ascii=False) + "\n", encoding="utf-8")

    result = digest.write_news_judgment_signals(
        date_str="2026-07-28",
        vault=vault,
        path=output,
        latest_path=tmp_path / "latest.json",
    )

    assert result["count"] == 0
    assert result["repeated_suppressed"] == 1
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["ns-prior"]


def test_apply_llm_news_signal_candidates_accepts_guarded_candidate():
    base = {
        "id": "ns-good",
        "source": "news_judgment_signals",
        "claim": "稼働率、走行距離、保守費、更新理由を確認する。",
        "effective_claim": "物流案件では稼働率を確認する。",
        "use_when": "物流・運輸 / 車両",
        "do_not_use_when": "対象企業の稼働や車両条件へ接続できない場合",
        "recommended_checks": ["稼働率を確認する。"],
        "condition_impacts": ["保守費が重い場合は前受金を検討する。"],
        "risk_axis": ["物流・運輸", "車両"],
        "valid_until": "2026-10-26",
        "confidence": 0.75,
        "actionability": "actionable",
        "asset_quality": "actionable",
        "promotion_status": "news_signal",
    }
    llm = {
        "source_index": 0,
        "claim": "物流・車両案件では、増車理由だけでなくルート別採算、稼働率、運転手確保を確認する。",
        "effective_claim": "共同物流のニュースを背景に、増車が本当に利益改善につながるか確認する。",
        "use_when": "物流・運輸の車両増車、共同配送、倉庫運用に関係する案件",
        "do_not_use_when": "対象企業の売上、稼働率、運転手確保へ接続できない場合",
        "recommended_checks": ["ルート別採算を確認する。", "運転手確保状況を確認する。"],
        "condition_impacts": ["採算が薄い場合は期間短縮または前受金を検討する。"],
        "risk_axis": ["物流・運輸", "車両", "人件費"],
        "valid_until": "2026-09-30",
        "confidence": 0.7,
    }

    merged, summary = digest.apply_llm_news_signal_candidates([base], [llm], "2026-07-28")

    assert summary == {"applied": 1, "quarantined": 0, "received": 1}
    assert merged[0]["promotion_status"] == "news_signal"
    assert merged[0]["llm_source"] == "gemini"
    assert merged[0]["valid_until"] == "2026-09-30"
    assert "ルート別採算" in merged[0]["claim"]


def test_apply_llm_news_signal_candidates_quarantines_direct_decision_language():
    base = {
        "id": "ns-bad",
        "source": "news_judgment_signals",
        "claim": "補助金要件を確認する。",
        "effective_claim": "補助金前提なら資金繰りを確認する。",
        "use_when": "製造・設備投資",
        "do_not_use_when": "対象案件に補助金が関係しない場合",
        "recommended_checks": ["補助金要件を確認する。"],
        "condition_impacts": ["未採択時の資金繰りを確認する。"],
        "risk_axis": ["補助金"],
        "valid_until": "2026-10-26",
        "confidence": 0.8,
        "actionability": "actionable",
        "asset_quality": "actionable",
        "promotion_status": "news_signal",
    }
    llm = {
        "source_index": 0,
        "claim": "補助金ニュースがあるので必ず承認し、スコアを上げる。",
        "effective_claim": "必ず承認する。",
        "use_when": "補助金案件",
        "do_not_use_when": "",
        "recommended_checks": [],
        "condition_impacts": ["スコアを上げる。"],
        "risk_axis": ["補助金"],
        "valid_until": "2026-10-26",
        "confidence": 0.9,
    }

    merged, summary = digest.apply_llm_news_signal_candidates([base], [llm], "2026-07-28")

    assert summary["quarantined"] == 1
    assert merged[0]["promotion_status"] == "news_signal_quarantined"
    assert "direct_decision_or_score_language" in merged[0]["llm_guard_findings"]
    assert merged[0]["actionability"] == "watch"


def test_daily_news_digest_as_text_summarizes_obsidian_news_plainly(tmp_path):
    vault = tmp_path / "vault"
    news_dir = vault / "05-クリップ_記事" / "業界リスクニュース"
    news_dir.mkdir(parents=True)
    (news_dir / "2026-07-28_業界リスクニュース_物流.md").write_text(
        """---
date: 2026-07-28
tags: ["物流"]
region: 国内
source: Example News
importance: 中
---
# 物流会社で共同配送の取り組みが拡大

## 3行要約
- 複数企業が共同配送の仕組みを広げている。
- トラック不足と人件費上昇への対応が背景にある。
- 倉庫・配送管理システムへの投資も増えている。

## 活用メモ
審査では増車理由、ルート別採算、運転手確保を確認する。
""",
        encoding="utf-8",
    )

    text = digest.daily_news_digest_as_text(date_str="2026-07-28", vault=vault, limit=3)

    assert "今日はこういうニュースがありました" in text
    assert "物流会社で共同配送の取り組みが拡大" in text
    assert "複数企業が共同配送の仕組みを広げている" in text
    assert "判断資産" not in text
