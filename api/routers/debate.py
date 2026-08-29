"""マルチエージェント討論（軍師AI）の結果保存・案件登録エンドポイント。"""
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class DebateCautiousData(BaseModel):
    opinion: str = ""
    reasons: List[str] = []
    key_risks: List[str] = []


class DebateAggressiveData(BaseModel):
    opinion: str = ""
    reasons: List[str] = []
    opportunities: List[str] = []


class SaveDebateToObsidianRequest(BaseModel):
    company_name: str = ""
    score: float = 0.0
    grade: str = ""
    cautious: Optional[DebateCautiousData] = None
    aggressive: Optional[DebateAggressiveData] = None
    arbiter_summary: str = ""
    final_decision: str = ""
    conditions: List[str] = []
    debate_log: Optional[str] = None
    screened_at: Optional[str] = None


@router.post("/api/debate/save-to-obsidian")
def save_debate_to_obsidian(req: SaveDebateToObsidianRequest):
    """討論審査結果を iCloud 上の Obsidian Vault の Debates/ フォルダに保存する。"""
    from api.main import _OBSIDIAN_VAULT_PATH

    vault_root = _OBSIDIAN_VAULT_PATH

    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="iCloud 上の Obsidian Vault が見つかりません。環境変数 OBSIDIAN_VAULT_PATH を設定してください。",
        )

    debates_dir = os.path.join(vault_root, "Debates")
    os.makedirs(debates_dir, exist_ok=True)

    from api.debate_note_render import render_debate_obsidian_note

    note = render_debate_obsidian_note(req)
    filepath = os.path.join(debates_dir, note["filename"])

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(note["content"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル書き込みエラー: {e}")

    relative_path = f"Debates/{note['filename']}"
    return {"path": relative_path}


class RegisterDebateCaseRequest(BaseModel):
    company_name: str = ""
    industry_major: str = ""
    score: float = 0.0
    final_decision: str = ""
    conditions: List[str] = []
    arbiter_summary: str = ""
    inputs: dict = Field(default_factory=dict)


@router.post("/api/debate/register-case")
def register_debate_case(req: RegisterDebateCaseRequest):
    """マルチエージェント討論の判断結果を、結果登録画面（/register）で確定できる未登録案件として保存する。"""
    from data_cases import save_case_log

    hantei = req.final_decision or "要審議"
    case_data = {
        "company_no": "",
        "company_name": req.company_name or "名称未設定",
        "industry_major": req.industry_major,
        "industry_sub": req.industry_major,
        "inputs": req.inputs,
        "result": {
            "score": req.score,
            "score_base": req.score,
            "hantei": hantei,
            "arbiter_reasoning": req.arbiter_summary,
            "arbiter_conditions": req.conditions,
            "engine_source": "multi_agent_debate",
        },
    }
    case_id = save_case_log(case_data)
    if not case_id:
        raise HTTPException(status_code=500, detail="案件登録に失敗しました。")
    from api.prediction_snapshot import record_saved_case_prediction

    record_saved_case_prediction(
        case_id=str(case_id),
        case_data=case_data,
        source="debate_register_case",
    )
    return {"case_id": case_id, "hantei": hantei}
