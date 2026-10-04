from scripts.report_cloudflare_rag_shadow import summarize_rows


def test_summarize_rows_counts_effectiveness_and_errors():
    summary = summarize_rows([
        {"status": "ok", "local_vectorize_overlap_at_5": 2, "vectorize_reranker_changed_top1": True},
        {"status": "ok", "local_vectorize_overlap_at_5": 0, "vectorize_reranker_changed_top1": False},
        {"status": "error", "error_type": "Timeout"},
    ])
    assert summary == {
        "total": 3,
        "ok": 2,
        "errors": 1,
        "reranker_changed_top1": 1,
        "reranker_changed_top1_rate": 0.5,
        "mean_local_vectorize_overlap_at_5": 1.0,
    }
