"""物件ファイナンス審査ルーター (REV-234 Phase2)"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/asset-finance", tags=["asset-finance"])

class AssetFinanceRequest(BaseModel):
    asset_name: str = Field("", max_length=120)
    asset_type: str = Field(..., description="建機 / 工作機械 / PC/IT / 医療機器 / ドローン / 車両")
    term: int = Field(60, ge=12, le=84)
    down_payment: float = Field(0.2, ge=0.0, le=0.5)
    financial_score: str = Field("Medium", description="High / Medium / Low")
    main_bank_support: bool = False
    bank_coordination: bool = False
    core_business: bool = False
    related_assets: bool = False
    annual_km: int = Field(0, ge=0, le=100000)
    has_maintenance_lease: bool = False
    ai_residual_pct: Optional[float] = Field(None, ge=0.0, le=100.0)
    useful_life: Optional[int] = Field(None, ge=1, le=50, description="耐用年数（年）。指定時は r = 2.0 / useful_life で計算。")


class AssetFinanceObsidianContextRequest(BaseModel):
    asset_type: str = Field("", max_length=40)
    asset_name: str = Field("", max_length=120)
    financial_score: str = Field("", max_length=20)
    decision: str = Field("", max_length=40)
    memo_query: str = Field("", max_length=200)


class AssetFinanceSaveToObsidianRequest(BaseModel):
    input: Dict[str, Any]
    result: Dict[str, Any]
    related_paths: List[str] = Field(default_factory=list)


def _build_asset_finance_obsidian_terms(req: AssetFinanceObsidianContextRequest) -> List[str]:
    """物件名・型番からObsidian検索に効く安定語へ展開する。"""
    import re

    raw_parts = [
        "物件ファイナンス",
        "リース",
        "BEP",
        "残価",
        "再販リスク",
        "稟議根拠",
        req.asset_type,
        req.asset_name,
        req.financial_score,
        req.decision,
        req.memo_query,
    ]
    raw = " ".join(str(part or "") for part in raw_parts)
    lower = raw.lower()
    terms: List[str] = [str(part).strip() for part in raw_parts if str(part or "").strip()]

    asset_type_terms = {
        "建機": ["建機", "アワーメーター", "中古相場", "残価"],
        "車両": ["車両", "走行距離", "メンテナンス", "再販リスク"],
        "工作機械": ["工作機械", "中古相場", "制御装置", "保守期限"],
        "医療機器": ["医療機器", "保守期限", "薬機法", "設置撤去費"],
        "PC/IT": ["PC/IT", "陳腐化", "保守", "再販リスク"],
        "ドローン": ["ドローン", "バッテリー", "飛行時間", "法規制"],
    }
    terms.extend(asset_type_terms.get(req.asset_type, []))

    # 型番はハイフン枝番つきでも、親型式で検索できるようにする。
    for token in re.findall(r"[A-Za-z]+[- ]?\d+[A-Za-z0-9-]*|[A-Za-z]{2,}|[0-9]{2,}[A-Za-z0-9-]*", raw):
        clean = token.replace(" ", "").strip()
        if not clean:
            continue
        terms.append(clean)
        parent = clean.split("-", 1)[0]
        if parent != clean and len(parent) >= 3:
            terms.append(parent)

    keyword_groups = [
        (("コマツ", "komatsu", "pc200", "油圧ショベル", "ユンボ"), [
            "建機", "油圧ショベル", "ユンボ", "アワーメーター", "中古相場", "排ガス規制",
        ]),
        (("冷凍車", "冷蔵", "冷凍", "商用車", "メンテリース", "メンテナンスリース"), [
            "車両", "冷蔵冷凍車", "商用車", "メンテリース", "冷凍機", "走行距離", "架装",
        ]),
        (("フォークリフト", "forklift", "リフト", "トヨタl&f", "toyota"), [
            "フォークリフト", "バッテリー劣化", "アワーメーター", "マスト", "定期自主検査",
        ]),
        (("高所作業車", "アイチ", "タダノ", "ブーム", "アウトリガー"), [
            "高所作業車", "年次点検", "安全装置", "アウトリガー", "油圧", "ブーム",
        ]),
        (("発電機", "デンヨー", "denyo", "コンプレッサ", "コンプレッサー", "airman", "北越"), [
            "発電機", "コンプレッサー", "稼働時間", "排ガス規制", "防音型", "中古需要",
        ]),
        (("マシニング", "旋盤", "nc", "制御装置"), [
            "工作機械", "中古相場", "主軸稼働時間", "制御装置", "保守期限", "搬出費",
        ]),
        (("射出成形", "成形機", "型締", "スクリュー", "roboshot"), [
            "射出成形機", "型締力", "制御装置", "スクリュー", "電動式", "油圧式",
        ]),
        (("医療機器", "ct", "mri", "内視鏡", "歯科", "薬機法"), [
            "医療機器", "保守期限", "薬機法", "設置撤去費", "中古医療機器",
        ]),
        (("測定器", "検査装置", "三次元測定", "キーエンス", "ミツトヨ", "東京精密", "島津", "堀場"), [
            "測定器", "検査装置", "校正証明", "保守期限", "ソフトライセンス", "再校正",
        ]),
        (("pc/it", "it機器", "サーバ", "パソコン", "複合機"), [
            "PC/IT", "陳腐化", "保守", "リース", "再販リスク",
        ]),
        (("ドローン", "uav"), [
            "ドローン", "バッテリー", "飛行時間", "法規制", "機体登録",
        ]),
    ]
    for triggers, additions in keyword_groups:
        if any(trigger in lower for trigger in triggers):
            terms.extend(additions)

    seen: set[str] = set()
    deduped: List[str] = []
    for term in terms:
        clean = str(term or "").strip()
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return deduped


def _extract_asset_obsidian_evidence(hits: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Asset Knowledgeノートから審査画面向けの根拠候補を抽出する。"""
    import re

    buckets = {
        "used_market": [],
        "residual_risk": [],
        "approval_basis": [],
        "cautions": [],
    }
    heading_map = {
        "中古相場・再販観点": "used_market",
        "中古相場": "used_market",
        "残価・再販リスク": "residual_risk",
        "残価": "residual_risk",
        "稟議で使えそうな根拠": "approval_basis",
        "稟議根拠": "approval_basis",
        "注意すべき物件特性": "cautions",
        "注意点": "cautions",
    }
    seen: set[str] = set()

    asset_hits_used = 0
    for hit in hits:
        path = str(hit.get("path") or "")
        if "Asset Knowledge" not in path:
            continue
        if path.endswith("物件ファイナンス検索索引.md"):
            continue
        asset_hits_used += 1
        text = str(hit.get("snippet") or "").replace("\r\n", "\n")
        sections = re.split(r"\n(?=##+ .+)", text)
        for section in sections:
            lines = [line.strip() for line in section.splitlines() if line.strip()]
            if not lines:
                continue
            heading = lines[0].lstrip("#").strip()
            bucket = None
            for key, value in heading_map.items():
                if key in heading:
                    bucket = value
                    break
            if not bucket:
                continue
            for line in lines[1:]:
                item = line.lstrip("-・*0123456789. ").strip()
                if not item or item.startswith("http") or item.startswith("[["):
                    continue
                item = item.replace("**", "")
                if len(item) < 8:
                    continue
                dedupe_key = f"{bucket}:{item[:120]}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                buckets[bucket].append(item[:180])
                if len(buckets[bucket]) >= 5:
                    break
        if asset_hits_used >= 1 and any(buckets.values()):
            break

    return buckets


def _classify_asset_obsidian_hit(hit: Dict[str, Any]) -> str:
    """検索ヒットを審査用途の色分けカテゴリへ寄せる。"""
    path = str(hit.get("path") or "")
    text = f"{path}\n{hit.get('snippet') or ''}".lower()
    if "中古相場" in text:
        return "used_market"
    if "残価" in text or "再販" in text:
        return "residual_risk"
    if "稟議" in text or "根拠" in text or "承認" in text:
        return "approval_basis"
    if "注意" in text or "保守" in text or "校正" in text or "期限" in text:
        return "cautions"
    if "asset knowledge" in path.lower():
        return "support"
    if "daily" in path.lower():
        return "context"
    if "generated" in path.lower():
        return "generated"
    return "support"


def _normalize_obsidian_node_label(path_or_label: str, limit: int = 28) -> str:
    text = str(path_or_label or "").strip()
    if not text:
        return "無題"
    if "/" in text:
        text = Path(text).stem
    if len(text) > limit:
        return text[:limit - 1] + "…"
    return text


def _build_asset_finance_obsidian_graph(
    query: str,
    hits: List[Dict[str, Any]],
    generated_terms: List[str],
    evidence: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Obsidianメモの簡易グラフを NEXT 用に返す。"""
    color_map = {
        "used_market": "#2563eb",
        "residual_risk": "#d97706",
        "approval_basis": "#059669",
        "cautions": "#e11d48",
        "support": "#94a3b8",
        "context": "#7c3aed",
        "generated": "#0f766e",
        "query": "#0f172a",
        "focus": "#0f172a",
        "linked": "#cbd5e1",
    }
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    evidence_text = " ".join(" ".join(v) for v in evidence.values()).lower()

    def add_node(node_id: str, label: str, node_type: str, color: str, radius: int, **extra: Any) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({
            "id": node_id,
            "label": label,
            "type": node_type,
            "color": color,
            "radius": radius,
            **extra,
        })

    add_node("focus", "今回の審査", "focus", color_map["focus"], 24, pinned=True, used=True)
    if query.strip():
        add_node("query", "検索語", "query", color_map["query"], 18, used=True)
        edges.append({"source": "focus", "target": "query", "type": "query", "width": 2, "color": "#0ea5e9"})

    for idx, term in enumerate(generated_terms[:10]):
        term_id = f"term_{idx}"
        add_node(term_id, _normalize_obsidian_node_label(term, 24), "generated", color_map["generated"], 14, used=True, term=term)
        edges.append({"source": "query" if query.strip() else "focus", "target": term_id, "type": "term", "width": 1.3, "color": "#14b8a6"})

    for idx, hit in enumerate(hits[:6]):
        path = str(hit.get("path") or "")
        label = _normalize_obsidian_node_label(hit.get("title") or path, 28)
        category = _classify_asset_obsidian_hit(hit)
        used = category in {"used_market", "residual_risk", "approval_basis", "cautions"} or path.lower() in evidence_text
        node_id = f"hit_{idx}"
        color = color_map.get(category, color_map["support"])
        snippet = str(hit.get("snippet") or "").strip().replace("\r\n", "\n")
        add_node(
            node_id,
            label,
            category,
            color,
            19 if used else 14,
            path=path,
            used=used,
            category=category,
            snippet=snippet[:220],
            wikilinks=hit.get("wikilinks") or [],
        )
        edges.append({
            "source": "focus",
            "target": node_id,
            "type": "used" if used else "support",
            "width": 2.5 if used else 1.2,
            "color": color if used else "#cbd5e1",
        })

        linked = hit.get("wikilinks") or []
        if isinstance(linked, str):
            linked = [item.strip() for item in linked.split(",") if item.strip()]
        for link_idx, link in enumerate(list(linked)[:3]):
            link_id = f"{node_id}_link_{link_idx}"
            link_label = _normalize_obsidian_node_label(link, 26)
            add_node(link_id, link_label, "linked", color_map["linked"], 11, used=False, linked_from=path)
            edges.append({
                "source": node_id,
                "target": link_id,
                "type": "wikilink",
                "width": 1,
                "color": "#cbd5e1",
            })

    summary = {
        "total_hits": len(hits),
        "used_hits": sum(1 for hit in hits if _classify_asset_obsidian_hit(hit) != "support"),
        "linked_nodes": sum(1 for node in nodes if node.get("type") == "linked"),
        "generated_terms": len(generated_terms),
    }
    legend = [
        {"label": "今回の審査", "color": color_map["focus"]},
        {"label": "今回使った根拠", "color": color_map["approval_basis"]},
        {"label": "中古相場", "color": color_map["used_market"]},
        {"label": "残価・再販", "color": color_map["residual_risk"]},
        {"label": "注意点", "color": color_map["cautions"]},
        {"label": "関連ノート", "color": color_map["linked"]},
    ]
    return {"nodes": nodes, "edges": edges, "summary": summary, "legend": legend}


@router.post("/evaluate")
def evaluate_asset_finance(req: AssetFinanceRequest):
    """物件保全性・BEP・定性緩和因子を統合した物件ファイナンス審査。"""
    try:
        from components.asset_finance import AssetFinanceEngine
        engine = AssetFinanceEngine()
        if req.asset_type not in engine.ASSET_PARAMS:
            raise HTTPException(status_code=422, detail=f"未対応の物件種別です: {req.asset_type}")
        if req.financial_score not in {"High", "Medium", "Low"}:
            raise HTTPException(status_code=422, detail="financial_score は High / Medium / Low のいずれかです")

        data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        result = engine.run_inference(data)
        params = engine.ASSET_PARAMS[req.asset_type]
        eff_life = req.useful_life if req.useful_life else params["useful_life"]
        eff_r = 2.0 / eff_life
        curve = [
            {"month": i, "asset_value": v, "lease_balance": result["l_curve"][i]}
            for i, v in enumerate(result["v_curve"])
        ]
        return {
            **result,
            "curve": curve,
            "asset_params": {
                "depreciation_rate": eff_r,
                "useful_life": eff_life,
                "priority": params["priority"],
                "priority_score": params["priority_score"],
                "info": params["info"],
            },
            "input": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/obsidian-context")
def get_asset_finance_obsidian_context(req: AssetFinanceObsidianContextRequest):
    """物件ファイナンス審査に関連するObsidianメモを共通検索経路で取得する。"""
    try:
        from mobile_app.obsidian_bridge import build_obsidian_digest, collect_obsidian_context, search_notes

        generated_terms = _build_asset_finance_obsidian_terms(req)
        query = " ".join(generated_terms)
        hits = search_notes(query, limit=5, max_chars=2600)
        if len(hits) < 5:
            seen_paths = {str(hit.get("path") or "") for hit in hits}
            for hit in collect_obsidian_context(query, limit=5 - len(hits)):
                path = str(hit.get("path") or "")
                if path and path not in seen_paths:
                    hits.append(hit)
                    seen_paths.add(path)
        digest = build_obsidian_digest(query, hits) if hits else {"digest": "", "source_count": "0", "links": ""}
        evidence = _extract_asset_obsidian_evidence(hits)
        graph = _build_asset_finance_obsidian_graph(
            query=query,
            hits=hits,
            generated_terms=generated_terms,
            evidence=evidence,
        )
        return {
            "query": query,
            "generated_terms": generated_terms,
            "hits": hits,
            "digest": digest,
            "evidence": evidence,
            "graph": graph,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Obsidian検索エラー: {e}")


@router.post("/similar-notes")
def get_asset_finance_similar_notes(req: AssetFinanceObsidianContextRequest):
    """保存済みの物件ファイナンス・過去案件メモから類似メモを返す。"""
    try:
        from mobile_app.obsidian_bridge import search_notes

        terms = _build_asset_finance_obsidian_terms(req)
        query = " ".join([*terms, "類似", "過去", "案件", "承認", "条件"])
        hits = search_notes(query, limit=12, max_chars=1000)
        filtered = [
            hit for hit in hits
            if "Projects/tune_lease_55/Asset Finance/" in str(hit.get("path") or "")
            or "Projects/tune_lease_55/Cases/" in str(hit.get("path") or "")
        ]
        return {"query": query, "similar_notes": filtered[:5]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"類似メモ検索エラー: {e}")


@router.post("/save-to-obsidian")
def save_asset_finance_to_obsidian(req: AssetFinanceSaveToObsidianRequest):
    """物件ファイナンス審査結果をObsidianへ保存する。"""
    try:
        from mobile_app.obsidian_bridge import append_asset_finance_note, append_asset_knowledge_backlinks

        # クライアント送信の result は信用せず、保存直前にサーバー側で再計算する。
        # Obsidianは後続AI検索の知識源になるため、改ざん済みの判定を残さない。
        asset_req = (
            AssetFinanceRequest.model_validate(req.input)
            if hasattr(AssetFinanceRequest, "model_validate")
            else AssetFinanceRequest.parse_obj(req.input)
        )
        recalculated = evaluate_asset_finance(asset_req)

        saved = append_asset_finance_note(recalculated["input"], recalculated, req.related_paths)
        if saved.get("status") != "saved":
            raise HTTPException(status_code=503, detail=saved.get("reason") or "Obsidian保存をスキップしました")
        backlinks = append_asset_knowledge_backlinks(
            recalculated["input"],
            recalculated,
            req.related_paths,
            saved.get("rel_path"),
        )
        return {
            **saved,
            "score": recalculated.get("score"),
            "decision": recalculated.get("decision"),
            "backlinks": backlinks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Obsidian保存エラー: {e}")

