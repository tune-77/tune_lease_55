"""compare_embedding_models.py の純粋関数の単体テスト（ネットワーク・Vault不要）。"""
import pytest

from scripts.compare_embedding_models import (
    CLOUDFLARE_BGE_M3_MODEL,
    CLOUDFLARE_QWEN3_MODEL,
    CloudflareEmbedder,
    batched,
    build_exported_corpus,
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


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_cloudflare_embedder_parses_result_and_hides_token():
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        texts = kwargs["json"]["text"]
        return _FakeResponse({"result": {"data": [[1.0, float(i)] for i, _ in enumerate(texts)]}})

    embedder = CloudflareEmbedder("account-1", "secret-token", post_fn=post)
    vectors = embedder.embed(["再リース", "残価リスク"])

    assert vectors == [[1.0, 0.0], [1.0, 1.0]]
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret-token"
    assert "@cf/pfnet/plamo-embedding-1b" in calls[0][0]
    assert embedder.total_tokens > 0


@pytest.mark.parametrize("model", [CLOUDFLARE_QWEN3_MODEL, CLOUDFLARE_BGE_M3_MODEL])
def test_cloudflare_embedder_supports_vectorize_compatible_models(model):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return _FakeResponse({"result": {"data": [[0.1, 0.2]]}})

    embedder = CloudflareEmbedder("account-1", "token", model=model, post_fn=post)
    assert embedder.embed(["日本語検索"]) == [[0.1, 0.2]]
    assert model in calls[0]
    assert embedder.cost_usd() > 0


def test_cloudflare_embedder_rejects_unknown_model():
    with pytest.raises(ValueError, match="未対応"):
        CloudflareEmbedder("account-1", "token", model="@cf/example/unknown")


def test_cloudflare_embedder_retries_only_retryable_status():
    responses = iter([
        _FakeResponse({}, status_code=429),
        _FakeResponse({"result": {"data": [[0.1, 0.2]]}}),
    ])
    waits = []
    embedder = CloudflareEmbedder(
        "account-1",
        "token",
        post_fn=lambda *_args, **_kwargs: next(responses),
        sleep_fn=waits.append,
    )
    assert embedder.embed(["補助金"]) == [[0.1, 0.2]]
    assert waits == [2]


def test_cloudflare_embedder_retries_connection_failure():
    calls = 0
    waits = []

    def post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary network failure")
        return _FakeResponse({"result": {"data": [[0.3, 0.4]]}})

    embedder = CloudflareEmbedder("account-1", "token", post_fn=post, sleep_fn=waits.append)
    assert embedder.embed(["再リース"]) == [[0.3, 0.4]]
    assert waits == [2]


def test_build_exported_corpus_uses_manifest_paths_safely(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "doc.txt").write_text("Title: 再リース\n\n満了後の確認事項", encoding="utf-8")
    (export / "manifest.json").write_text(
        '{"documents":['
        '{"output_path":"doc.txt","source_path":"Projects/tune_lease_55/Research/release.md"},'
        '{"output_path":"../outside.txt","source_path":"bad.md"}'
        ']}',
        encoding="utf-8",
    )

    corpus = build_exported_corpus(export)
    assert [item["rel_path"] for item in corpus] == ["Projects/tune_lease_55/Research/release.md"]
    assert "満了後" in corpus[0]["text"]


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
    assert "| local | 4 | 可 |" in body

    # キャッシュが効くこと（2回目はAPI/エンコードを呼ばずに完走する）
    assert cem.run(args) == 0


def test_run_excludes_cases_missing_expected_document_from_metrics(tmp_path, monkeypatch):
    import argparse
    import json

    import scripts.compare_embedding_models as cem

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "unrelated.md").write_text("# unrelated\n\n関係のない文書です。", encoding="utf-8")
    eval_set = tmp_path / "eval.json"
    eval_set.write_text(json.dumps([{
        "id": "missing",
        "query": "補助金",
        "expected_path_any": ["not-in-corpus.md"],
        "forbidden_path_any": [],
    }]), encoding="utf-8")

    monkeypatch.setattr(cem, "LocalEmbedder", _StubEmbedder)
    monkeypatch.setattr(cem, "EVAL_SET_PATH", eval_set)
    monkeypatch.setattr(cem, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cem, "REPORT_DIR", tmp_path / "reports")

    args = argparse.Namespace(vault=str(vault), models="local", top_k=5, max_chunks=0)
    assert cem.run(args) == 0
    body = next((tmp_path / "reports").glob("*.md")).read_text(encoding="utf-8")
    assert "| local | 4 | 可 | 0/1 (0%) |" in body
    assert "対象外（正解文書がサニタイズ済みコーパスに無い）" in body


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
