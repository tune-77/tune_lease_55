"""紫苑記憶システム強化（ヘルスチェック・矛盾検知・評価候補・昇格・忘却）のテスト。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.check_shion_memory_health import check, load_index_summary, load_state, save_state  # noqa: E402
from scripts.detect_shion_memory_contradictions import find_contradictions  # noqa: E402
from scripts.build_shion_eval_candidates import collect_candidates as collect_eval_candidates  # noqa: E402
from scripts.build_shion_memory_promotion_queue import collect_candidates as collect_promotions  # noqa: E402
from scripts.apply_shion_memory_promotions import apply_promotions  # noqa: E402
from scripts.update_shion_memory_freshness import apply_freshness, load_feedback_signals  # noqa: E402


# ── ヘルスチェック ──────────────────────────────────────────────────────────


def _write_index(path: Path, n: int) -> None:
    records = [
        {"id": f"m{i}", "content": f"記憶{i}", "memory_type": "factual_memory", "status": "active"}
        for i in range(n)
    ]
    path.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")


class TestMemoryHealth:
    def test_healthy_when_counts_stable(self, tmp_path):
        index = tmp_path / "index.json"
        _write_index(index, 100)
        summary = load_index_summary(index)
        healthy, _ = check(summary, {"total": 95})
        assert healthy

    def test_alarm_on_sharp_drop(self, tmp_path):
        index = tmp_path / "index.json"
        _write_index(index, 60)
        summary = load_index_summary(index)
        healthy, message = check(summary, {"total": 100})
        assert not healthy
        assert "急減" in message

    def test_alarm_on_missing_or_empty_index(self, tmp_path):
        healthy, _ = check(load_index_summary(tmp_path / "none.json"), {})
        assert not healthy
        index = tmp_path / "index.json"
        _write_index(index, 0)
        healthy, _ = check(load_index_summary(index), {})
        assert not healthy

    def test_state_roundtrip(self, tmp_path):
        state_path = tmp_path / "state.json"
        save_state(state_path, {"total": 42, "by_type": {}, "by_status": {}})
        assert load_state(state_path)["total"] == 42


# ── 矛盾検知 ────────────────────────────────────────────────────────────────


class TestContradictions:
    def test_detects_conflicting_numbers(self):
        records = [
            {"id": "a", "status": "active", "content": "ブルドーザー・油圧ショベルの法定耐用年数は6年として審査する", "memory_type": "judgment_memory", "source_path": "x.md"},
            {"id": "b", "status": "active", "content": "ブルドーザー・油圧ショベルの法定耐用年数は8年として審査する", "memory_type": "judgment_memory", "source_path": "y.md"},
        ]
        found = find_contradictions(records)
        assert len(found) == 1
        assert found[0]["conflicting_values"]["年"] == {"a": ["6"], "b": ["8"]}

    def test_ignores_dissimilar_or_agreeing_pairs(self):
        records = [
            {"id": "a", "status": "active", "content": "ブルドーザーの法定耐用年数は6年として扱う", "memory_type": "judgment_memory"},
            {"id": "b", "status": "active", "content": "医療機器のリースでは薬機法の広告規制を必ず確認する", "memory_type": "judgment_memory"},
            {"id": "c", "status": "active", "content": "ブルドーザーの法定耐用年数は6年として扱う方針", "memory_type": "judgment_memory"},
        ]
        assert find_contradictions(records) == []

    def test_skips_non_active_records(self):
        records = [
            {"id": "a", "status": "active", "content": "建機の耐用年数は6年で審査する", "memory_type": "judgment_memory"},
            {"id": "b", "status": "revised", "content": "建機の耐用年数は8年で審査する", "memory_type": "judgment_memory"},
        ]
        assert find_contradictions(records) == []


# ── 評価セット候補 ──────────────────────────────────────────────────────────


class TestEvalCandidates:
    def test_negative_feedback_is_high_priority(self):
        usage = [{"question": "補助金の返還リスクは？", "route": "case_screening"}] * 3
        feedback = [{"query": "残価の見方を教えて", "rating": "bad"}]
        candidates = collect_eval_candidates(usage, feedback, set())
        assert candidates[0]["_priority"] == "high"
        assert candidates[0]["query"] == "残価の見方を教えて"
        assert any(c["query"] == "補助金の返還リスクは？" for c in candidates)

    def test_excludes_existing_eval_queries_and_rare_queries(self):
        usage = [
            {"question": "既存の質問です？", "route": "case_screening"},
            {"question": "既存の質問です？", "route": "case_screening"},
            {"question": "一度しか聞かれていない質問", "route": "policy_review"},
        ]
        candidates = collect_eval_candidates(usage, [], {"既存の質問です？"})
        assert candidates == []


# ── 昇格キュー・適用 ────────────────────────────────────────────────────────


class TestPromotionQueue:
    def test_teaching_pattern_detected_and_question_excluded(self):
        rows = [
            {"event_id": "e1", "ts": "2026-07-08T10:00:00", "user_message": "補助金案件は入金時期まで見ると覚えておいて"},
            {"event_id": "e2", "ts": "2026-07-08T11:00:00", "user_message": "残価って覚えておくべきですか？"},
        ]
        candidates = collect_promotions(rows, set())
        teaching = [c for c in candidates if c["kind"] == "teaching"]
        assert len(teaching) == 1
        assert "入金時期" in teaching[0]["proposed_content"]

    def test_recurring_topic_detected(self):
        rows = [
            {"event_id": f"e{i}", "ts": f"2026-07-0{i+1}T10:00:00", "user_message": f"残価リスクの件、{i}回目の相談です。リースの残価をどう見るか"}
            for i in range(3)
        ]
        candidates = collect_promotions(rows, set())
        topics = [c for c in candidates if c["kind"] == "recurring_topic"]
        assert any(c.get("topic") == "残価" for c in topics)

    def test_applied_ids_are_skipped(self):
        rows = [{"event_id": "e1", "ts": "t", "user_message": "補助金案件は入金時期まで見ると覚えておいて"}]
        first = collect_promotions(rows, set())
        applied = {c["candidate_id"] for c in first}
        assert collect_promotions(rows, applied) == []


class TestApplyPromotions:
    def _queue(self, tmp_path) -> Path:
        queue = tmp_path / "queue.json"
        queue.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"candidate_id": "promo_abc", "kind": "teaching", "proposed_content": "補助金案件は入金時期・未採択時の返済余力まで確認する"}
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return queue

    def test_apply_appends_bullet_and_records_log(self, tmp_path):
        queue = self._queue(tmp_path)
        target = tmp_path / "promoted.md"
        log = tmp_path / "applied.jsonl"
        rc = apply_promotions(queue, target, log, ids={"promo_abc"}, apply_all=False, dry_run=False)
        assert rc == 0
        text = target.read_text(encoding="utf-8")
        assert "- 補助金案件は入金時期・未採択時の返済余力まで確認する" in text
        assert "promo_abc" in log.read_text(encoding="utf-8")

    def test_second_apply_is_noop(self, tmp_path):
        queue = self._queue(tmp_path)
        target = tmp_path / "promoted.md"
        log = tmp_path / "applied.jsonl"
        apply_promotions(queue, target, log, ids=None, apply_all=True, dry_run=False)
        before = target.read_text(encoding="utf-8")
        apply_promotions(queue, target, log, ids=None, apply_all=True, dry_run=False)
        assert target.read_text(encoding="utf-8") == before


# ── フィードバック連動の忘却 ────────────────────────────────────────────────


class TestFeedbackFreshness:
    def _index(self, memory_type: str = "factual_memory") -> dict:
        return {
            "records": [
                {
                    "id": "m1",
                    "content": "テスト記憶",
                    "memory_type": memory_type,
                    "status": "active",
                    "source_path": "knowledge_base/note.md",
                    "created_at": "2026-01-01",
                }
            ]
        }

    def test_negative_feedback_demotes_immediately(self):
        index = self._index()
        summary = apply_freshness(index, {}, negative_files={"note.md"})
        assert summary["demoted_by_feedback"] == 1
        assert index["records"][0]["status"] == "stale"

    def test_positive_feedback_protects_from_time_demotion(self):
        index = self._index()
        summary = apply_freshness(index, {}, positive_files={"note.md"})
        assert summary["demoted_to_stale"] == 0
        assert index["records"][0]["status"] == "active"

    def test_value_memory_never_demoted_by_feedback(self):
        index = self._index(memory_type="value_memory")
        summary = apply_freshness(index, {}, negative_files={"note.md"})
        assert summary["demoted_by_feedback"] == 0

    def test_load_feedback_signals_conservative_on_conflict(self, tmp_path):
        log = tmp_path / "fb.jsonl"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"obsidian_ref": "a.md", "rating": "bad"}),
                    json.dumps({"obsidian_ref": "a.md", "rating": "good"}),
                    json.dumps({"obsidian_ref": "b.md", "rating": "wrong"}),
                ]
            ),
            encoding="utf-8",
        )
        negative, positive = load_feedback_signals(log)
        assert negative == {"b.md"}  # 高低両方付いた a.md は降格させない
        assert positive == {"a.md"}
