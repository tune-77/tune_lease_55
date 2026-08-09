import json

from scripts import build_obsidian_retrieval_graph as graph


def test_build_index_extracts_nodes_edges_and_routes(tmp_path):
    vault = tmp_path / "vault"
    project = vault / "Projects" / "tune_lease_55" / "Research"
    project.mkdir(parents=True)
    (project / "residual-value.md").write_text(
        "---\ntags: [residual, lease]\n---\n# 残価リスク\n[[asset-market]] と中古売却を確認する。\n",
        encoding="utf-8",
    )
    (project / "asset-market.md").write_text("# 中古市場\n換金性のメモ。\n", encoding="utf-8")

    index = graph.build_index(vault)

    assert index["summary"]["notes"] == 2
    assert index["summary"]["edges"] == 1
    node = next(n for n in index["nodes"] if n["key"] == "residual-value")
    assert node["route"] == "judgment_memory"
    assert node["links"] == ["asset-market"]
    target = next(n for n in index["nodes"] if n["key"] == "asset-market")
    assert target["backlinks"] == ["residual-value"]


def test_route_query_scores_without_opening_note_bodies(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "billing-rules.md").write_text("# Billing\nrefund rate rules\n", encoding="utf-8")
    (vault / "client-acme.md").write_text("# Acme\ncontact list\n", encoding="utf-8")

    index = graph.build_index(vault)
    candidates = graph.route_query("refund rate を教えて", index, limit=1)

    assert len(candidates) == 1
    assert candidates[0]["path"] == "billing-rules.md"
    assert candidates[0]["route_score"] > 0


def test_route_query_penalizes_logs_and_news_for_judgment_questions(tmp_path):
    vault = tmp_path / "vault"
    good = vault / "Projects" / "tune_lease_55" / "Research" / "cash-flow.md"
    good.parent.mkdir(parents=True)
    good.write_text("# 返済余力と資金繰りの異常兆候\n担当者が確認する質問。", encoding="utf-8")
    news = vault / "05-クリップ_記事" / "業界リスクニュース" / "cash-news.md"
    news.parent.mkdir(parents=True)
    news.write_text("# 資金繰り支援ニュース\n審査上の確認事項。", encoding="utf-8")
    log = vault / "06-日記_作業ログ" / "改善ログ" / "rev.md"
    log.parent.mkdir(parents=True)
    log.write_text("# 条件付き承認の推奨アクション自動提示\n改善ログ。", encoding="utf-8")

    index = graph.build_index(vault)

    cash = graph.route_query("資金繰りが厳しい会社の確認事項", index, limit=2)
    assert cash[0]["path"] == "Projects/tune_lease_55/Research/cash-flow.md"
    assert all(not item["path"].startswith("05-クリップ_記事/") for item in cash)
    conditional = graph.route_query("条件付き承認の具体的な方法", index, limit=2)
    assert conditional == []


def test_markdown_contains_router_and_one_line_index(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "voice.md").write_text("# Voice\nwriting tone rules\n", encoding="utf-8")

    text = graph.markdown(graph.build_index(vault))

    assert "## Router" in text
    assert "## One Line Index" in text
    assert "`voice.md`" in text


def test_cli_dry_run_outputs_json(tmp_path, capsys):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "A.md").write_text("# A\nbody\n", encoding="utf-8")

    assert graph.main(["--vault", str(vault), "--dry-run"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["summary"]["notes"] == 1
