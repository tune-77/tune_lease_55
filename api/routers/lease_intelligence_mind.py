"""Lease-intelligence self-audit endpoints."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/lease-intelligence", tags=["lease-intelligence"])


@router.post("/self-audit")
def post_lease_intelligence_self_audit():
    """紫苑の自律検証ループを即時実行する（REV-080）。週次 cron からも呼ばれる。"""
    from lease_intelligence_mind import run_self_audit
    from lease_news_digest import find_vault

    vault = find_vault()
    if not vault:
        raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")
    try:
        return run_self_audit(vault)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"self-audit 実行エラー: {exc}") from exc


@router.get("/knowledge-gaps")
def get_knowledge_gaps():
    """紫苑の知識ギャップ一覧を返す（REV-082）。"""
    from lease_intelligence_mind import load_lease_intelligence_mind
    from lease_news_digest import find_vault

    vault = find_vault()
    if not vault:
        raise HTTPException(status_code=503, detail="Obsidian Vaultが見つかりません")
    try:
        mind = load_lease_intelligence_mind(vault)
        open_gaps = [gap for gap in mind.get("knowledge_gaps", []) if gap.get("status") == "open"]
        return {"total": len(open_gaps), "gaps": open_gaps}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
