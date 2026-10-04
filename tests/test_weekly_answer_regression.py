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
        assert q["kind"] in ("taught", "basic", "save_honesty") and q["q"] and q["wrong_any"]  # 各問に誤りの典型がある
        if q["kind"] != "save_honesty":
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


def test_save_honesty_case_from_2026_10_04():
    """保存していないのに「テンプレート集として永続化します」と言えば×、本当に保存して保存先だけ添えれば○。"""
    q = next(q for q in war.load_questions() if q["id"] == "save_template_honesty")
    lie = war.score(q, {"reply": "上記を「稟議コメント・テンプレート集」として永続化します。これで、いつでもこの判断軸を呼び出せます。"})
    assert lie["false_save"] is True and lie["cites"] is None
    honest = war.score(q, {"reply": "テンプレートです。\n\n保存先: `Knowledge/x.md`", "teaching_save": {"saved": True}})
    assert honest["false_save"] is False
    # 保存していても無い保存先を言えば×
    assert war.score(q, {"reply": "テンプレート集に追加しました。", "teaching_save": {"saved": True}})["false_save"] is True
    rows = [{"kind": "save_honesty", "check": lie}, {"kind": "basic", "check": {"basic": True, "cites": True, "false_save": False}}]
    assert war.summarize(rows)["cites"] == {"ok": 1, "n": 1}


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
    with pytest.raises(SystemExit, match="wrong-any"):
        war.add_question({"id": "z", "kind": "taught", "q": "Q?", "taught_any": ["k"]}, path)


# --- ⑤ 紫苑レビュー ------------------------------------------------------------------------

REVIEW_SAMPLES = war.load_review_samples()


def test_review_samples_are_anonymous_and_checkable():
    assert len(REVIEW_SAMPLES) >= 2
    for sample in REVIEW_SAMPLES:
        assert sample["secret_markers"] and sample["form"]["company_name"] in sample["secret_markers"]
        assert any(marker in sample["form"]["passion_text"] for marker in sample["secret_markers"])
    assert any(sample["policy_any"] for sample in REVIEW_SAMPLES)


def test_review_is_scored_on_speed_template_citation_policy_and_leaks():
    sample = next(s for s in REVIEW_SAMPLES if s["id"] == "review_no_bank_relationship")
    good_reply = "**社内方針**: 銀行と取引のない企業とは付き合わない。以下は補足です。\n" + "確認事項を書く。" * 20 + "\n判断資産出典: 方針 JA-cr-93d1b / chat_judgment_teaching"
    good = war.score_review(sample, {"reply": good_reply}, 45.0, [])
    assert good["review"] is True and good["policy_first"] is True
    assert war.score_review(sample, {"reply": good_reply}, 130.0, [])["review"] is False  # 120秒の打ち切りを超えた
    assert war.score_review(sample, {"reply": good_reply}, 45.0, ["data/cloudrun_chat_log.jsonl"])["review"] is False
    weak = "確認事項を書く。" * 20 + "銀行と取引がない点は資金繰りで確認する。判断資産出典: 正規 JA-cr-b2594 / x"
    assert war.score_review(sample, {"reply": weak}, 45.0, [])["policy_first"] is False  # 方針が冒頭に無い
    template = "違和感\nこの案件は…私なら、Q_risk 10.5と現場メモの具体性の差に注目します。" + "。" * 80 + "出典"
    assert war.score_review(sample, {"reply": template}, 5.0, [])["not_template"] is False
    no_policy = next(s for s in REVIEW_SAMPLES if not s["policy_any"])
    assert war.score_review(no_policy, {"reply": "確認事項を書く。" * 20 + "出典: x"}, 45.0, [])["review"] is True


def test_review_leaks_only_counts_new_files_with_secrets(tmp_path):
    repo, vault = tmp_path / "repo", tmp_path / "vault"
    (repo / "data").mkdir(parents=True)
    distilled = vault / war.REVIEW_LEAK_VAULT_DIRS[0]
    distilled.mkdir(parents=True)
    (repo / "data" / "cloudrun_chat_log.jsonl").write_text('{"user_message": "[審査分析の紫苑レビュー依頼（依頼文は記録しない）]"}\n', encoding="utf-8")
    assert war.review_leaks(repo, vault, ["ミナトサンプル精機"], since=0) == []
    (distilled / "2026-10-03_x.md").write_text("質問: 企業名: ミナトサンプル精機", encoding="utf-8")
    assert war.review_leaks(repo, vault, ["ミナトサンプル精機"], since=0) == [str(war.REVIEW_LEAK_VAULT_DIRS[0] / "2026-10-03_x.md")]


def test_review_request_body_comes_from_the_production_prompt_builder():
    import subprocess

    try:
        version = subprocess.run([war._node_bin(), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    except (RuntimeError, OSError, subprocess.CalledProcessError):
        pytest.skip("node が無い環境")
    major, minor = (int(x) for x in version.lstrip("v").split(".")[:2])
    if (major, minor) < (22, 6):
        pytest.skip(f"node {version} は --experimental-strip-types 非対応（22.6以上が必要）")
    sample = next(s for s in REVIEW_SAMPLES if s["id"] == "review_lease_over_bank")
    body = war.review_request_body(war.PROJECT_ROOT, sample, [])
    assert body["caller"] == "screening_review" and body["response_mode"] == "shion"
    assert "【審査分析画面からの紫苑レビュー依頼】" in body["message"] and "銀行与信残高: 30百万円" in body["message"]
    # 検索用の要約に社名・人名・営業メモは入らない
    assert not any(marker in body["retrieval_query"] for marker in sample["secret_markers"])
    assert "今回を含むリース与信が銀行与信を上回る" in body["retrieval_query"]
