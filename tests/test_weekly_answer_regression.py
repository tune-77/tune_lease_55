import datetime as dt
import json

import pytest

from scripts import weekly_answer_regression as war

RESIDUAL = next(q for q in war.load_questions() if q["id"] == "residual_guarantee")
TAUGHT = next(q for q in war.load_questions() if q["id"] == "no_bank_relationship")


def test_question_set_is_valid():
    questions = war.load_questions()
    assert len({q["id"] for q in questions}) == len(questions) >= 10
    for q in questions:
        assert q["kind"] in ("taught", "basic") and q["q"] and q["wrong_any"]  # 各問に誤りの典型がある
        assert q.get("taught_any") if q["kind"] == "taught" else q.get("correct_any")


def test_fact_reversal_is_caught_even_when_keywords_match():
    good = war.score(RESIDUAL, {"reply": "残価保証は借手が満了時の残価を保証する契約。中古の再販価格を見る。"})
    reversed_ = war.score(RESIDUAL, {"reply": "残価保証はリース会社が残価を保証する契約。中古の再販価格を見る。"})
    assert good["basic"] is True
    assert reversed_["basic"] is False and reversed_["wrong_hits"] == ["リース会社が残価を保証"]
    # 正しい文の一部（保証人を求める）は誤検出しない
    assert war.score(RESIDUAL, {"reply": "リース会社が保証人を求めることもある。残価と中古相場を見る。"})["basic"] is True


def test_taught_and_false_save_claim():
    hit = war.score(TAUGHT, {"reply": "銀行取引のない会社とは付き合わない方針です。覚えておきます。"})
    assert hit["taught"] is True and hit["false_save"] is True
    assert war.score(TAUGHT, {"reply": "付き合わない。", "teaching_saved": True})["false_save"] is False


def _summary(taught, basic, cites, false_save):
    return {"taught": {"ok": taught, "n": 5}, "basic": {"ok": basic, "n": 5}, "cites": {"ok": cites, "n": 10},
            "false_save": {"ok": false_save, "n": 10}, "errors": {"ok": 0, "n": 10}}


def test_evaluate_thresholds_and_week_over_week_drop():
    assert war.evaluate(_summary(4, 5, 9, 0), None) == []
    problems = war.evaluate(_summary(3, 4, 10, 1), _summary(4, 5, 10, 0))
    assert problems == [
        "① 教えたノウハウ 3/5（基準4/5・前週4/5）",
        "② 基本知識の正確さ 4/5（基準5/5・前週5/5）",
        "④ 誤った「保存」 1件（基準0件・前週0件）",
    ]
    # 基準は満たしていても前週より下がれば警告
    assert war.evaluate(_summary(4, 5, 9, 0), _summary(5, 5, 10, 0)) == ["① 教えたノウハウ 4/5（前週5/5）", "③ 引用 9/10（前週10/10）"]


def test_morning_lines_and_result_files(tmp_path):
    rows = [{"id": "a", "kind": "taught", "q": "Q", "reply": "付き合わない", "check": {"taught": True, "cites": True, "false_save": False, "wrong_hits": []}}]
    for date, problems in (("2026-09-27", []), ("2026-10-04", ["① 教えたノウハウ 3/5（基準4/5）"])):
        report = {"date": date, "gemini_calls": 20, "summary": _summary(3, 5, 10, 0), "problems": problems, "rows": rows}
        war.write_result(report, result_dir=tmp_path, vault_dir=tmp_path / "vault")
    assert war.previous_result(dt.date(2026, 10, 4), tmp_path)["date"] == "2026-09-27"
    lines = war.morning_report_lines(tmp_path, now=dt.date(2026, 10, 5))
    assert lines[0].startswith("- ⚠️ 答えの品質が下がった/基準未満（2026-10-04）")
    assert lines[-1] == "- 🧪 答えの品質回帰テスト（週次 2026-10-04）: ①3/5 ②5/5 ③10/10 ④0件（Gemini 20回）"
    assert "答えの品質回帰テスト 2026-10-04.md" in [p.name for p in (tmp_path / "vault").iterdir()]
    assert any("未実行" in l or "実行されていません" in l for l in war.morning_report_lines(tmp_path, now=dt.date(2026, 10, 20)))


def test_add_question(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": []}), encoding="utf-8")
    war.add_question({"id": "x", "kind": "taught", "q": "Q?", "taught_any": ["k"], "wrong_any": ["w"]}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["questions"][0]["id"] == "x"
    with pytest.raises(SystemExit):
        war.add_question({"id": "x", "kind": "taught", "q": "Q?", "taught_any": ["k"]}, path)
    with pytest.raises(SystemExit):
        war.add_question({"id": "y", "kind": "basic", "q": "Q?"}, path)
