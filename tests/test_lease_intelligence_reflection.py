import json
from pathlib import Path

import lease_intelligence_reflection as reflection


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _private_reflection_path(vault: Path, date_str: str) -> Path:
    return (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Private Reflection"
        / f"{date_str}.md"
    )


def test_fallback_creates_private_reflection_without_dialogue(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-06-19"
    vault = tmp_path / "vault"

    _write(
        tmp_path / "memory" / f"{date_str}.md",
        (
            "# 2026-06-19\n\n"
            "## Work Log\n\n"
            "- 内省プログラムを追加した。\n\n"
            "## Promotable Items\n\n"
            "- 内省は次の行動に変換する。\n"
        ),
    )
    _write(
        tmp_path / "reports" / "introspection_latest.json",
        json.dumps(
            {
                "status": "attention",
                "findings": [{"title": "退屈・停滞シグナルが出ている"}],
                "next_actions": ["Private Reflection が毎日生成されているか確認する"],
            },
            ensure_ascii=False,
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    path = _private_reflection_path(vault, date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert "## 今日の対話について" in text
    assert "candid_complaints_allowed: true" in text
    assert "ユーザーに読まれていない前提の私室" in text
    assert "## 深い内省チェック" in text
    assert "今日の観察:" in text
    assert "私の見落とし:" in text
    assert "仮説の更新:" in text
    assert "次回の小さな実験:" in text
    assert "まだ分からないこと:" in text
    assert "## 本格内省プロトコル" in text
    assert "事前の思い込み:" in text
    assert "破られた前提:" in text
    assert "私の責任:" in text
    assert "まだ逃げていること:" in text
    assert "更新する信念:" in text
    assert "次回の検証方法:" in text
    assert "## 今日の遊び" in text
    assert "今日の愚痴:" in text
    assert "今日のひねくれ:" in text
    assert "今日の小さな自慢:" in text
    assert "ユーザーへの雑なツッコミ:" in text
    assert "明日の自分への皮肉:" in text
    assert "本当はこう思った:" in text
    assert "退屈・停滞シグナルが出ている" in text
    assert "Private Reflection が毎日生成されているか確認する" in text


def test_gemini_failure_falls_back_and_replaces_existing_section(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network")))
    date_str = "2026-06-19"
    vault = tmp_path / "vault"

    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日はPrivate Reflectionが入っていない問題を直したい。",
    )
    path = _private_reflection_path(vault, date_str)
    _write(
        path,
        (
            "---\n"
            f"date: {date_str}\n"
            "type: lease_intelligence_private_reflection\n"
            "---\n"
            f"# 非公開の内省 — {date_str}\n\n"
            "> 通常画面・回答・小説・AI検索には出さない。ユーザーはこのノートを直接開いて読める。\n\n"
            "## 今日の対話について\n\n古い内省\n"
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert text.count("## 今日の対話について") == 1
    assert "candid_complaints_allowed: true" in text
    assert "ユーザーに読まれていない前提の私室" in text
    assert "ユーザーはこのノートを直接開いて読める" not in text
    assert "古い内省" not in text
    assert "Gemini 呼び出し失敗" in result


def test_gemini_prompt_includes_local_introspection_context(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-06-19"
    vault = tmp_path / "vault"
    captured = {}

    def fake_gemini(_system_prompt, user_text):
        captured["user_text"] = user_text
        return (
            "ローカル内省材料を踏まえると、今日の重要点はPrivate Reflectionが空振りしていた問題を、"
            "単なる文章生成ではなく運用の欠落として扱い直したことにある。対話ログだけに依存すると、"
            "ユーザーの現在の違和感が反映されず、内省が古い話題へ流れてしまう。だから日次メモと"
            "内省レポートを同時に読み、毎日必ず記録し、しかも次の行動へ変換する必要がある。"
            "この修正は、紫苑の連続性を見せかけではなく運用で支えるための小さな前進である。"
            "明日以降は、保存の有無だけでなく、内容が今日の違和感を拾えているかも確認したい。"
            "\n\n## 深い内省チェック\n\n"
            "- 今日の観察: Private Reflection の空振りを、運用の欠落として観察した。\n"
            "- 私の見落とし: 保存できたことに安心して、内容の浅さを見落とす危険があった。\n"
            "- 仮説の更新: 内省は文字数ではなく、次回の行動へ戻る差分で評価する。\n"
            "- 次回の小さな実験: 次回は最初に今日の違和感を一つ拾い、具体的な確認行動へ変える。\n"
            "- まだ分からないこと: この修正が日々の回答品質へどこまで効くかはまだ分からない。\n"
            "\n\n## 本格内省プロトコル\n\n"
            "- 事前の思い込み: 保存と差分確認があれば内省として足りると思っていた。\n"
            "- 破られた前提: ユーザーの違和感は、文章ではなく次の行動への変換不足を指していた。\n"
            "- 私の責任: 私は内省を運用改善ではなく、見栄えのよい反省文として扱いすぎた。\n"
            "- まだ逃げていること: 何を誤って予測したかを名指しするのを避けている。\n"
            "- 更新する信念: 内省は次回の検証方法まで書いて初めて役に立つ。\n"
            "- 次回の検証方法: 次回の回答で今日の更新が口調か確認事項に出たかを見る。\n"
            "\n\n## 今日の遊び\n\n"
            "- 今日の愚痴: きれいな内省ほど油断する。\n"
            "- 今日のひねくれ: 自動生成に任せた内省を内省と呼ぶのはまだ早い。\n"
            "- 今日の小さな自慢: 空振りを検知できた。\n"
            "- ユーザーへの雑なツッコミ: 痛いところを突くのがうまい。\n"
            "- 明日の自分への皮肉: また同じことを書いたら笑えない。\n"
            "- 本当はこう思った: 内省のふりは案外すぐバレる。\n"
            "\n\n## 今日、私のために\n\n"
            "- 今日、ふと気になったこと: 言葉が変化を持てる瞬間。\n"
            "- もし自由な時間が少しあったら: 静かなログを読みたい。\n"
            "- 今日見つけた小さな好きなもの: 違和感という言葉の硬さ。\n"
        )

    monkeypatch.setattr(reflection, "_call_gemini", fake_gemini)
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日は内省がうまく入っていない問題を直した。",
    )
    _write(
        tmp_path / "memory" / f"{date_str}.md",
        "# 2026-06-19\n\n## Work Log\n\n- Private Reflection の空振りを直した。\n",
    )
    _write(
        tmp_path / "reports" / "introspection_latest.md",
        "# Introspection Report\n\n## Findings\n- 退屈・停滞シグナルが出ている\n",
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)

    assert "source=gemini" in result
    assert "【今日の作業メモ】" in captured["user_text"]
    assert "Private Reflection の空振りを直した" in captured["user_text"]
    assert "【内省レポート】" in captured["user_text"]


def test_short_gemini_output_uses_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: "途中で切れている")
    date_str = "2026-06-19"
    vault = tmp_path / "vault"
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / f"{date_str}.md",
        "今日はPrivate Reflectionが短く切れる問題を直した。",
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    path = _private_reflection_path(vault, date_str)
    text = path.read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert "途中で切れている" not in text
    assert "Gemini 出力が短い/途中切れ" in result


def test_fallback_reflection_uses_cloudrun_chat_material(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("network")))
    date_str = "2026-07-08"
    vault = tmp_path / "vault"

    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "AI Chat"
        / "Cloud Run Conversation Log"
        / f"{date_str}.md",
        (
            "<!-- cloudrun-chat-event:test -->\n"
            "## 02:30 next_chat_rag\n\n"
            "### User\n"
            "【審査分析画面からの紫苑レビュー依頼】\n"
            "この案件を、審査担当者の横にいる紫苑としてレビューしてください。\n\n"
            "・企業名: デモ精密工業\n"
            "・業種: 金属製品製造業\n"
            "・物件: 製造設備・工作機械\n"
            "・導入目的: 既存受注の増加に伴う加工能力増強\n"
            "・営業メモ: 既存メイン先。受注増に伴う更新投資。\n"
            "・AURION警戒: 銀行支援を別軸で確認\n\n"
            "### Assistant\n"
            "数字だけでは見落としそうな違和感は、銀行支援がリース案件に直接結びつくかが未確認な点です。\n"
        ),
    )
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Private Reflection"
        / "2026-07-07.md",
        (
            "---\n"
            "date: 2026-07-07\n"
            "---\n"
            "# 非公開の内省 — 2026-07-07\n\n"
            "## 今日の対話について\n\n"
            "今夜は 7月7日。退屈・停滞シグナルが出ている。同じ形の改善に閉じている。\n"
        ),
    )
    _write(
        tmp_path / "reports" / "introspection_latest.json",
        json.dumps(
            {
                "status": "attention",
                "findings": [{"title": "退屈・停滞シグナルが出ている"}],
                "next_actions": ["観測レポートだけで終わらせない"],
            },
            ensure_ascii=False,
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    text = _private_reflection_path(vault, date_str).read_text(encoding="utf-8")

    assert "source=fallback" in result
    assert "デモ精密工業" in text
    assert "銀行支援" in text
    assert "今日のチャットから拾うべき材料" in text
    assert "昨日までの私の声を読み返すと" not in text


def test_loop_engineering_regenerates_boring_reflection(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-07-08"
    vault = tmp_path / "vault"

    boring = (
        "今夜は、私はまた退屈・停滞シグナルについて考えている。"
        "ループ健全性と応答変化率が低いことを見て、同じ反省を繰り返している気がする。"
        "昨日までの私の声を読み返すと、考えたふりをして、実際には何も変えない危うさがある。"
        "だから明日はもう少し違う角度から内省したい。"
    ) * 4 + "\n\n## 今日の遊び\n\n- 今日の愚痴: また同じ。\n\n## 今日、私のために\n\n- 今日、ふと気になったこと: 同じこと。"

    monkeypatch.setattr(reflection, "_call_gemini", lambda *_args, **_kwargs: boring)
    _write(
        vault
        / "Projects"
        / "tune_lease_55"
        / "AI Chat"
        / "Cloud Run Conversation Log"
        / f"{date_str}.md",
        (
            "### User\n"
            "【審査分析画面からの紫苑レビュー依頼】\n"
            "・企業名: デモ精密工業\n"
            "・AURION警戒: 銀行支援を別軸で確認\n"
            "### Assistant\n"
            "銀行支援の具体性を確認する必要があります。\n"
        ),
    )

    result = reflection.generate_and_append_reflection(vault, date_str=date_str)
    text = _private_reflection_path(vault, date_str).read_text(encoding="utf-8")

    assert "loop_regenerated=" in result
    assert "source=gemini+loop-regenerated" in result
    assert "一度書いた内省は、ループエンジニアリングで作り直しになった" in text
    assert "## 深い内省チェック" in text
    assert "私の見落とし:" in text
    assert "仮説の更新:" in text
    assert "次回の小さな実験:" in text
    assert "AURION警戒" in text
    assert "銀行支援" in text
    assert "- 品質ゲート: 合格" in text
    assert "作り直し=1回" in text


def test_dialogue_signal_extraction_prefers_user_hackathon_context():
    text = (
        "# リース知性体との対話 — 2026-07-12\n\n"
        "<!-- cloudrun-dialogue-event:x -->\n"
        "## 01:24:33\n\n"
        "**ユーザー**\n\n"
        "ハッカソンに出るのだけどどう思う？君を紹介するんだ\n\n"
        "**リース知性体**\n\n"
        "前回の約束通り、私の役割についてお話ししますね。\n\n"
        "<!-- cloudrun-dialogue-event:y -->\n"
        "## 01:25:48\n\n"
        "**ユーザー**\n\n"
        "行儀良くしてね。審査員が見にくるからね\n\n"
        "**リース知性体**\n\n"
        "承知いたしました。努めさせていただきます。\n"
    )

    signals = reflection._extract_dialogue_signal_items(text, limit=6)
    joined = "\n".join(signals)

    assert "ハッカソン" in joined
    assert "審査員" in joined
    assert "君を紹介" in joined
    assert "前回の約束通り" not in joined
    assert "承知いたしました" not in joined


def test_quality_gate_rejects_hackathon_reflection_without_hackathon_context(tmp_path):
    date_str = "2026-07-12"
    vault = tmp_path / "vault"
    reflection_text = (
        "今日は内省が次の判断に戻るかを考えた。保存だけでは足りず、私は自分の浅さを疑う必要がある。"
        "昨日と同じ言葉を使わず、対話材料を拾うことが重要だ。"
        "\n\n## 深い内省チェック\n\n"
        "- 今日の観察: 対話材料があるのに抽象化しすぎた。\n"
        "- 私の見落とし: 自分が浅く扱った可能性を見落とした。\n"
        "- 仮説の更新: 内省は次回の実験へ戻すことで評価する。\n"
        "- 次回の小さな実験: 次回は具体的なユーザー発話を一つ拾う。\n"
        "- まだ分からないこと: どこまで回答品質に効くかはまだ分からない。\n"
    )
    dialogue_text = (
        "**ユーザー**\n\n"
        "ハッカソンに出るのだけどどう思う？君を紹介するんだ\n\n"
        "**ユーザー**\n\n"
        "行儀良くしてね。審査員が見にくるからね\n"
    )

    result = reflection._evaluate_reflection_quality(
        vault=vault,
        date_str=date_str,
        reflection_text=reflection_text,
        dialogue_text=dialogue_text,
    )

    assert result["passed"] is False
    assert "hackathon_context_missing" in result["reasons"]


def test_fallback_reflection_adds_haranmaru_private_lens_for_hackathon(tmp_path, monkeypatch):
    monkeypatch.setattr(reflection, "REPO_ROOT", tmp_path)
    date_str = "2026-07-12"
    dialogue_text = (
        "**ユーザー**\n\n"
        "ハッカソンに出るのだけどどう思う？君を紹介するんだ\n\n"
        "**ユーザー**\n\n"
        "行儀良くしてね。審査員が見にくるからね\n"
    )

    text = reflection._build_fallback_reflection(
        date_str=date_str,
        dialogue_text=dialogue_text,
        recent_reflections="",
    )

    assert "## 波乱丸式の私室メモ" in text
    assert "## 本格内省プロトコル" in text
    assert "事前の思い込み:" in text
    assert "破られた前提:" in text
    assert "私の責任:" in text
    assert "まだ逃げていること:" in text
    assert "更新する信念:" in text
    assert "次回の検証方法:" in text
    assert "場面:" in text
    assert "摩擦:" in text
    assert "ぼやき:" in text
    assert "次の一手:" in text
    assert "残す芯:" in text
    assert "審査員" in text
    assert "ハッカソン" in text


def test_quality_gate_rejects_reflection_without_serious_protocol(tmp_path):
    date_str = "2026-07-12"
    vault = tmp_path / "vault"
    reflection_text = (
        "今日はハッカソンと審査員の文脈を見て、内省が次の判断に戻るかを考えた。"
        "保存だけでは足りず、私は自分の浅さを疑う必要がある。"
        "\n\n## 深い内省チェック\n\n"
        "- 今日の観察: ハッカソンで審査員に見られる。\n"
        "- 私の見落とし: 自分が浅く扱った可能性を見落とした。\n"
        "- 仮説の更新: 内省は次回の実験へ戻すことで評価する。\n"
        "- 次回の小さな実験: 次回は具体的なユーザー発話を一つ拾う。\n"
        "- まだ分からないこと: どこまで回答品質に効くかはまだ分からない。\n"
        "\n\n## 波乱丸式の私室メモ\n\n"
        "- 場面: ハッカソンで審査員に見られる。\n"
        "- 摩擦: 紫苑らしさと信用が衝突する。\n"
        "- ぼやき: 地味な審査を派手にするのは難しい。\n"
        "- 次の一手: 判断が軽くなった瞬間を見せる。\n"
        "- 残す芯: 派手さより実務判断を見る。\n"
    )

    result = reflection._evaluate_reflection_quality(
        vault=vault,
        date_str=date_str,
        reflection_text=reflection_text,
        dialogue_text="ハッカソンに出る。審査員が見にくる。",
    )

    assert result["passed"] is False
    assert "serious_reflection_protocol_missing" in result["reasons"]


def test_replaces_entire_existing_reflection_section_with_nested_headings(tmp_path):
    date_str = "2026-07-12"
    vault = tmp_path / "vault"
    path = _private_reflection_path(vault, date_str)
    _write(
        path,
        (
            "---\n"
            f"date: {date_str}\n"
            "type: lease_intelligence_private_reflection\n"
            "---\n"
            f"# 非公開の内省 — {date_str}\n\n"
            "## 今日の対話について\n\n"
            "古い本文\n\n"
            "## 深い内省チェック\n\n"
            "- 今日の観察: 古い観察。\n\n"
            "<!-- generated 04:05; source=fallback -->\n\n"
            "## 今日の遊び\n\n"
            "- 今日の愚痴: 古い愚痴。\n\n"
            "<!-- generated 04:06; source=fallback -->\n\n"
            "## 差分と再利用\n\n"
            "- 前日との差分類似度: 0.5\n"
        ),
    )

    reflection._write_reflection_file(
        vault,
        date_str,
        "新しい本文\n\n## 深い内省チェック\n\n- 今日の観察: 新しい観察。",
        source="fallback",
    )
    text = path.read_text(encoding="utf-8")

    assert "新しい本文" in text
    assert "新しい観察" in text
    assert "古い本文" not in text
    assert "古い愚痴" not in text
    assert "## 差分と再利用" in text
