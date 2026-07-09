"""compare_embedding_models.py の純粋関数の単体テスト（ネットワーク・Vault不要）。"""
import pytest

from scripts.compare_embedding_models import (
    batched,
    cosine_similarity,
    evaluate_case,
    path_matches_any,
    summarize,
)


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)  # ゼロベクトルでも例外なし


def test_path_matches_any_is_partial_match():
    assert path_matches_any("リース知識/補助金・税制優遇とリース.md", ["補助金・税制優遇とリース.md"])
    assert not path_matches_any("Humor/つん子.md", ["リース知識/"])
    assert not path_matches_any("a.md", [""])  # 空パターンはマッチしない


def test_evaluate_case_ranks_and_forbidden():
    ranked = ["Daily/memo.md", "リース知識/補助金.md", "Humor/つん子.md"]
    metrics = evaluate_case(ranked, expected=["補助金"], forbidden=["Humor/"], top_k=3)
    assert metrics["first_hit_rank"] == 2
    assert not metrics["hit_at_1"]
    assert metrics["hit_at_3"]
    assert metrics["reciprocal_rank"] == pytest.approx(0.5)
    assert metrics["forbidden_in_top"]


def test_evaluate_case_miss():
    metrics = evaluate_case(["a.md", "b.md"], expected=["c.md"], forbidden=[], top_k=5)
    assert metrics["first_hit_rank"] == 0
    assert metrics["reciprocal_rank"] == 0.0
    assert not metrics["hit_at_5"]


def test_summarize_aggregates():
    summary = summarize([
        {"hit_at_1": True, "hit_at_3": True, "hit_at_5": True, "reciprocal_rank": 1.0, "forbidden_in_top": False},
        {"hit_at_1": False, "hit_at_3": True, "hit_at_5": True, "reciprocal_rank": 0.5, "forbidden_in_top": True},
    ])
    assert summary["cases"] == 2
    assert summary["hit_at_1"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.75)
    assert summary["forbidden_rate"] == pytest.approx(0.5)


def test_batched_splits_evenly():
    assert batched(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert batched([], 3) == []


class _StubEmbedder:
    """決定的なハッシュ埋め込み。モデルロード・API呼び出しなしで run() を通す。"""

    model = "stub"
    total_tokens = 0

    def __init__(self, model_name: str = ""):
        pass

    def embed(self, texts, kind="document"):
        return [[float((hash(t) >> shift) % 97) for shift in (0, 8, 16, 24)] for t in texts]

    def cost_usd(self):
        return 0.0


def test_run_end_to_end_with_stub_embedder(tmp_path, monkeypatch):
    import argparse

    import scripts.compare_embedding_models as cem

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "リース知識").mkdir()
    (vault / "リース知識" / "補助金・税制優遇とリース.md").write_text(
        "# 補助金\n\nものづくり補助金はリースでも対象になる場合がある。", encoding="utf-8"
    )
    (vault / "審査メモ.md").write_text("# 審査\n\n与信基準のメモ。", encoding="utf-8")

    monkeypatch.setattr(cem, "LocalEmbedder", _StubEmbedder)
    monkeypatch.setattr(cem, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cem, "REPORT_DIR", tmp_path / "reports")

    args = argparse.Namespace(vault=str(vault), models="local", top_k=5, max_chunks=0)
    assert cem.run(args) == 0

    reports = list((tmp_path / "reports").glob("embedding_model_comparison_*.md"))
    assert len(reports) == 1
    body = reports[0].read_text(encoding="utf-8")
    assert "| local |" in body or "| stub |" in body

    # キャッシュが効くこと（2回目はAPI/エンコードを呼ばずに完走する）
    assert cem.run(args) == 0


def test_run_requires_gemini_key_for_gemini_model(tmp_path, monkeypatch):
    import argparse

    import scripts.compare_embedding_models as cem

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# a\n\nメモ", encoding="utf-8")
    monkeypatch.setattr(cem, "load_api_key", lambda name: None)

    args = argparse.Namespace(vault=str(vault), models="gemini", top_k=5, max_chunks=0)
    assert cem.run(args) == 2


def test_estimate_tokens_scales_with_chars():
    from scripts.compare_embedding_models import estimate_tokens

    assert estimate_tokens(["あ" * 100]) == 120
    assert estimate_tokens([]) == 0
