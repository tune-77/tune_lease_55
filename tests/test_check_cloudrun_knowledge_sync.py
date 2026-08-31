from scripts.check_cloudrun_knowledge_sync import validate_sync_response


def test_validate_sync_response_accepts_ready_status():
    healthy, reason = validate_sync_response(
        {
            "knowledge_sync": {
                "ready": True,
                "state": "ready",
                "reason": "ok",
                "vault_markdown_count": 98,
                "chroma_document_count": 581,
            }
        }
    )

    assert healthy is True
    assert reason == "ok"


def test_validate_sync_response_rejects_partial_status():
    healthy, reason = validate_sync_response(
        {
            "knowledge_sync": {
                "ready": False,
                "state": "partial",
                "reason": "still indexing",
                "vault_markdown_count": 98,
                "chroma_document_count": 12,
            }
        }
    )

    assert healthy is False
    assert "vault=98" in reason
    assert "chroma=12" in reason
