import pytest

from scripts.cloudflare_vectorize_shadow import (
    CloudflareReranker,
    DIMENSIONS,
    INDEX_NAME,
    VectorizeClient,
    build_vector_records,
    mutation_is_processed,
    stable_vector_id,
    stale_vector_ids,
    _bounded_reranker_inputs,
)


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_stable_vector_id_is_deterministic_and_short():
    first = stable_vector_id("Research/再リース.md")
    assert first == stable_vector_id("Research/再リース.md")
    assert first != stable_vector_id("Research/残価.md")
    assert len(first.encode("utf-8")) <= 64


def test_build_vector_records_keeps_only_sanitized_fields():
    corpus = [{"rel_path": "Research/再リース.md", "text": "再リースの注意点"}]
    records = build_vector_records(corpus, [[0.1] * DIMENSIONS])
    assert records[0]["metadata"] == {
        "path": "Research/再リース.md",
        "text": "再リースの注意点",
        "corpus": "lease-knowledge-sanitized-v1",
    }
    assert len(records[0]["values"]) == DIMENSIONS


def test_build_vector_records_gives_each_source_chunk_a_unique_id():
    corpus = [
        {"key": "Research/a.md#chunk-0", "rel_path": "Research/a.md", "text": "first"},
        {"key": "Research/a.md#chunk-1", "rel_path": "Research/a.md", "text": "second"},
    ]

    records = build_vector_records(corpus, [[0.1] * DIMENSIONS, [0.2] * DIMENSIONS])

    assert records[0]["id"] != records[1]["id"]
    assert {record["metadata"]["path"] for record in records} == {"Research/a.md"}


def test_build_vector_records_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="次元不一致"):
        build_vector_records([{"rel_path": "a.md", "text": "a"}], [[0.1, 0.2]])


def test_ensure_index_creates_missing_index_without_leaking_token():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET":
            return _FakeResponse(404, {"success": False, "errors": [{"message": "not found"}]})
        return _FakeResponse(200, {
            "success": True,
            "result": {"name": INDEX_NAME, "config": {"dimensions": DIMENSIONS, "metric": "cosine"}},
        })

    client = VectorizeClient("account", "secret-token", request_fn=request)
    index, created = client.ensure_index()
    assert created
    assert index["name"] == INDEX_NAME
    assert calls[1][2]["json"]["config"]["dimensions"] == DIMENSIONS
    assert calls[1][2]["headers"]["Authorization"] == "Bearer secret-token"


def test_ensure_index_refuses_incompatible_existing_index():
    def request(_method, _url, **_kwargs):
        return _FakeResponse(200, {
            "success": True,
            "result": {"config": {"dimensions": 768, "metric": "cosine"}},
        })

    client = VectorizeClient("account", "token", request_fn=request)
    with pytest.raises(RuntimeError, match="設定が不一致"):
        client.ensure_index()


def test_query_caps_top_k_and_requests_metadata():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _FakeResponse(200, {"success": True, "result": {"matches": [{"id": "one"}]}})

    client = VectorizeClient("account", "token", request_fn=request)
    assert client.query([0.1] * DIMENSIONS, top_k=99) == [{"id": "one"}]
    assert calls[0][2]["json"]["topK"] == 20
    assert calls[0][2]["json"]["returnMetadata"] == "all"


def test_upsert_uses_vectorize_vectors_multipart_field():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _FakeResponse(200, {"success": True, "result": {"mutationId": "mutation-1"}})

    client = VectorizeClient("account", "token", request_fn=request)
    records = build_vector_records(
        [{"rel_path": "Research/再リース.md", "text": "再リース"}],
        [[0.1] * DIMENSIONS],
    )
    assert client.upsert(records) == ["mutation-1"]
    assert set(calls[0][2]["files"]) == {"vectors"}


def test_list_vector_ids_follows_cursor_pages():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if len(calls) == 1:
            return _FakeResponse(200, {"result": {
                "vectors": [{"id": "old"}],
                "isTruncated": True,
                "nextCursor": "next-page",
            }})
        return _FakeResponse(200, {"result": {
            "vectors": [{"id": "current"}],
            "isTruncated": False,
        }})

    client = VectorizeClient("account", "token", request_fn=request)
    assert client.list_vector_ids() == ["old", "current"]
    assert calls[0][2]["params"] == {"count": 1000}
    assert calls[1][2]["params"] == {"count": 1000, "cursor": "next-page"}


def test_delete_by_ids_uses_json_endpoint():
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _FakeResponse(200, {"result": {"mutationId": "delete-1"}})

    client = VectorizeClient("account", "token", request_fn=request)
    assert client.delete_by_ids(["stale-a", "stale-b"]) == ["delete-1"]
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/delete_by_ids")
    assert calls[0][2]["json"] == {"ids": ["stale-a", "stale-b"]}


def test_stale_vector_ids_excludes_current_export_records():
    records = [{"id": "current"}, {"id": "new"}]
    assert stale_vector_ids(["current", "withdrawn", "withdrawn"], records) == ["withdrawn"]


def test_mutation_is_processed_requires_final_mutation_and_exact_count():
    assert mutation_is_processed(
        {"processedUpToMutation": "upsert-2", "vectorCount": 3},
        "upsert-2",
        3,
    )
    assert not mutation_is_processed(
        {"processedUpToMutation": "upsert-1", "vectorCount": 3},
        "upsert-2",
        3,
    )
    assert not mutation_is_processed(
        {"processedUpToMutation": "upsert-2", "vectorCount": 4},
        "upsert-2",
        3,
    )


def test_reranker_reorders_vectorize_candidates_and_keeps_original_rank():
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _FakeResponse(200, {
            "result": {"response": [
                {"id": 1, "score": 0.91},
                {"id": 0, "score": 0.22},
            ]}
        })

    matches = [
        {"id": "a", "score": 0.8, "metadata": {"path": "a.md", "text": "一般説明"}},
        {"id": "b", "score": 0.7, "metadata": {"path": "b.md", "text": "再リースの注意点"}},
    ]
    reranker = CloudflareReranker("account", "secret", post_fn=post)
    ranked = reranker.rerank("再リース", matches, top_k=2)

    assert [item["id"] for item in ranked] == ["b", "a"]
    assert ranked[0]["vector_rank"] == 2
    assert ranked[0]["rerank_score"] == pytest.approx(0.91)
    assert calls[0][1]["json"]["contexts"] == [
        {"text": "一般説明"},
        {"text": "再リースの注意点"},
    ]
    assert "secret" not in str(calls[0][1]["json"])


def test_reranker_inputs_fit_model_token_window():
    query, contexts = _bounded_reranker_inputs(
        "質" * 300,
        [{"metadata": {"text": "文" * 1200}}],
    )

    assert len(query) <= 200
    assert int((len(query) + len(contexts[0]["text"])) * 1.2) <= 480


def test_reranker_rejects_invalid_response():
    reranker = CloudflareReranker(
        "account",
        "token",
        post_fn=lambda *_args, **_kwargs: _FakeResponse(200, {"result": {"response": []}}),
    )
    with pytest.raises(RuntimeError, match="有効なスコア"):
        reranker.rerank(
            "質問",
            [{"id": "a", "metadata": {"path": "a.md", "text": "本文"}}],
        )
