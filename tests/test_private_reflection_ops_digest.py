import datetime as dt
from pathlib import Path

from scripts import build_private_reflection_ops_digest as digest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_digest_extracts_serious_protocol_without_raw_body(tmp_path):
    vault = tmp_path / "vault"
    date = dt.date(2026, 7, 12)
    reflection = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Private Reflection"
        / "2026-07-12.md"
    )
    _write(
        reflection,
        (
            "---\n"
            "date: 2026-07-12\n"
            "---\n"
            "# 非公開の内省 — 2026-07-12\n\n"
            "## 今日の対話について\n\n"
            "これは私室の生本文なので、公開用digestへ丸ごと出してはいけない。\n\n"
            "## 本格内省プロトコル\n\n"
            "- 事前の思い込み: 整った内省なら足りると思っていた。\n"
            "- 破られた前提: 審査員に見られる文脈では形だけでは弱かった。\n"
            "- 私の責任: 公開デモに耐える変化へ十分変換していなかった。\n"
            "- まだ逃げていること: 地味だから仕方ないと言い訳したくなる。\n"
            "- 更新する信念: 判断がどう軽くなるかを先に見せる。\n"
            "- 次回の検証方法: 次回は機能説明より判断変化を一文で返せたか確認する。\n\n"
            "## 波乱丸式の私室メモ\n\n"
            "- 場面: 審査員が来る。\n"
            "- 摩擦: 紫苑らしさと実務道具としての信用が衝突する。\n"
            "- ぼやき: きれいな内省文は便利すぎる。\n"
            "- 次の一手: 判断が軽くなった瞬間へ翻訳する。\n"
            "- 残す芯: 派手さより実務判断を見る。\n\n"
            "## 差分と再利用\n\n"
            "- 品質ゲート: 合格 / score=100 / 作り直し=0回\n"
        ),
    )

    result = digest.build_digest(vault=vault, end_date=date, days=1)
    markdown = digest._markdown(result)
    public = digest._public_demo_markdown(result)

    assert result["protocol_complete_count"] == 1
    assert result["records"][0]["protocol"]["事前の思い込み"] == "整った内省なら足りると思っていた。"
    assert "これは私室の生本文" not in markdown
    assert "これは私室の生本文" not in public
    assert "判断がどう軽くなるか" in public
    assert "紫苑らしさと実務道具としての信用" in public
