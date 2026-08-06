"""Vertex AI Search / Vertex Agent Builder 連携エンドポイント。"""
import os
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class VertexSearchDebugRequest(BaseModel):
    query: str
    page_size: int = Field(5, ge=1, le=10)
    include_search: bool = True
    include_answer: bool = True
    include_google_grounding: bool = False
    include_related_questions: bool = True
    include_grounding_supports: bool = True
    grounding_filtering_level: Optional[Literal["FILTERING_LEVEL_UNSPECIFIED", "FILTERING_LEVEL_LOW", "FILTERING_LEVEL_HIGH"]] = None
    filter_expression: Optional[str] = None
    boost_spec: Optional[Dict[str, Any]] = None
    preamble: Optional[str] = None
    max_rephrase_steps: int = Field(3, ge=1, le=5)


class VertexExternalGroundingRequest(BaseModel):
    query: str
    page_size: int = Field(5, ge=1, le=10)


class VertexKnowledgeWorkflowRequest(BaseModel):
    topic: str
    mode: Literal["evidence_support", "judgment_candidates", "knowledge_audit"] = "evidence_support"
    page_size: int = Field(5, ge=1, le=10)
    save_to_obsidian: bool = False


@router.post("/api/vertex-search/debug")
def post_vertex_search_debug(req: VertexSearchDebugRequest):
    """Vertex AI Search/Answer/Google Search grounding を同一質問で比較するデバッグ口。"""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    try:
        from api.vertex_agent_search import (
            answer_vertex_agent,
            build_lease_search_controls,
            get_config,
            google_search_grounding,
            search_vertex_agent,
        )

        config = get_config()
        payload: Dict[str, Any] = {
            "query": query,
            "config": {
                "enabled": config.enabled,
                "project_id": config.project_id,
                "engine_id": config.engine_id,
                "location": config.location,
                "collection": config.collection,
                "page_size": config.page_size,
            },
            "controls": build_lease_search_controls(query),
        }
        if req.include_search:
            payload["search"] = search_vertex_agent(
                query,
                page_size=req.page_size,
                filter_expression=req.filter_expression,
                boost_spec=req.boost_spec,
                apply_controls=True,
            )
        if req.include_answer:
            payload["answer"] = answer_vertex_agent(
                query,
                page_size=req.page_size,
                preamble=req.preamble,
                include_related_questions=req.include_related_questions,
                include_grounding_supports=req.include_grounding_supports,
                grounding_filtering_level=req.grounding_filtering_level,
                filter_expression=req.filter_expression,
                boost_spec=req.boost_spec,
                max_rephrase_steps=req.max_rephrase_steps,
            )
        if req.include_google_grounding:
            payload["google_grounding"] = google_search_grounding(query)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/vertex-search/workflow")
def post_vertex_search_workflow(req: VertexKnowledgeWorkflowRequest):
    """Vertexを用途別に使う実務口。根拠補強・判断資産候補・知識棚卸しに限定する。"""
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="topic is required")
    try:
        from api.vertex_knowledge_workflows import run_vertex_knowledge_workflow

        result = run_vertex_knowledge_workflow(topic, mode=req.mode, page_size=req.page_size)
        if req.save_to_obsidian:
            from api.vertex_distillation import capture_vertex_workflow_result
            from runtime_paths import get_data_dir
            from api.main import _OBSIDIAN_VAULT_PATH

            result["obsidian_capture"] = capture_vertex_workflow_result(
                result,
                vault_path=_OBSIDIAN_VAULT_PATH,
                state_path=get_data_dir() / "vertex_workflow_capture_state.json",
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/vertex-search/external-grounding")
def post_vertex_external_grounding(req: VertexExternalGroundingRequest, request: Request):
    """Gemini Grounding with your own search API 用の `snippet`/`uri` 配列を返す。"""
    expected_key = os.environ.get("VERTEX_SEARCH_EXTERNAL_API_KEY", "").strip()
    if expected_key:
        provided = request.headers.get("X-Vertex-Search-Key", "").strip()
        if provided != expected_key:
            raise HTTPException(status_code=401, detail="invalid external search key")
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    try:
        from api.vertex_agent_search import external_grounding_results

        return external_grounding_results(query, page_size=req.page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/vertex-search/widget-config")
def get_vertex_search_widget_config():
    """Agent Builder search widget を画面に埋め込むための設定を返す。"""
    config_id = os.environ.get("VERTEX_AI_SEARCH_WIDGET_CONFIG_ID", "").strip()
    placeholder = os.environ.get("VERTEX_AI_SEARCH_WIDGET_PLACEHOLDER", "リース知識を検索").strip()
    snippet = ""
    if config_id:
        snippet = (
            f'<gen-search-widget configId="{config_id}" anchorsTarget="_blank" '
            f'placeholder="{placeholder}" alwaysOpened></gen-search-widget>'
        )
    return {
        "configured": bool(config_id),
        "config_id": config_id,
        "placeholder": placeholder,
        "snippet": snippet,
        "setup_note": "Google Cloud Console の AI Applications > Integration > Widget で configId を発行して環境変数に設定します。",
    }
