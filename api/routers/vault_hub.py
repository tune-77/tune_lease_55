"""
api/routers/vault_hub.py  – REV-234 Phase 7
Obsidian Vault / Research Organ / Agent Hub エンドポイント群
Extracted from api/main.py.

Endpoints:
  GET  /api/obsidian/notes               – list_obsidian_notes
  POST /api/obsidian/notes/read          – read_obsidian_notes
  GET  /api/research-organ/topics        – list_research_organ_topics
  GET  /api/research-organ/notes         – list_research_organ_notes
  POST /api/research-organ/run           – run_research_organ
  GET  /api/agent_hub/thoughts           – get_agent_thoughts
  GET  /api/agent_hub/novel/latest       – get_latest_novel_api
  POST /api/agent_hub/script/generate    – generate_script_api
  GET  /api/agent_hub/script/latest      – get_latest_script_api
  POST /api/agent_hub/novel/generate     – generate_novel_api
  GET  /api/agent_hub/novel/episodes     – get_novel_episodes_api
  POST /api/agent_hub/run_agent          – run_agent_api
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from obsidian_query import iter_vault_md_files

router = APIRouter(tags=["vault-hub"])

# Research Organ ノートが置かれる Vault 内フォルダ（vault からの相対パス）
_RESEARCH_ORGAN_FOLDER = "Projects/tune_lease_55/Research"

# _OBSIDIAN_VAULT_PATH: main.py と同じ環境変数から取得
_OBSIDIAN_VAULT_PATH: str = os.environ.get("OBSIDIAN_VAULT_PATH", "")

# section comment removed – content starts below

class ObsidianReadRequest(BaseModel):
    paths: List[str]


class ResearchOrganRunRequest(BaseModel):
    topic: str = Field("", description="Research topic key or free-form theme")
    dry_run: bool = False


def _get_vault_path() -> str:
    """OBSIDIAN_VAULT_PATH 環境変数からvaultパスを取得する。"""
    return os.environ.get("OBSIDIAN_VAULT_PATH", "")


def _read_obsidian_files(vault_path: str, rel_paths: list[str], max_bytes: int = 10_240) -> tuple[str, list[str]]:
    """指定されたObsidianノートを読み込み、結合テキストとファイル名リストを返す。"""
    parts = []
    files_read = []
    try:
        vault_root = Path(vault_path).expanduser().resolve(strict=True)
    except OSError:
        return "", []
    for rel_path in rel_paths:
        try:
            full_path = (vault_root / rel_path).resolve(strict=True)
            full_path.relative_to(vault_root)
        except (OSError, ValueError):
            continue
        if full_path.suffix != ".md" or not full_path.is_file():
            continue
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(max_bytes)
            parts.append(f"=== {rel_path} ===\n{content}")
            files_read.append(rel_path)
        except Exception as e:
            print(f"[API] obsidian read error {rel_path}: {e}")
    return "\n\n".join(parts), files_read


@router.get("/api/obsidian/notes")
def list_obsidian_notes():
    """Obsidian vault 配下の .md ファイルを再帰的に列挙する。
    最大100件・各ファイル10KB以内の情報を返す。
    vaultが設定されていない／存在しない場合は空リストを返す。
    """
    import datetime as _dt

    vault_path = _get_vault_path()
    if not vault_path or not os.path.isdir(vault_path):
        return []

    results = []
    try:
        for dirpath, _dirnames, filenames in os.walk(vault_path):
            for fname in filenames:
                if not fname.endswith(".md"):
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, vault_path)
                try:
                    st = os.stat(full)
                    size = st.st_size
                    modified = _dt.datetime.fromtimestamp(st.st_mtime).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    )
                except Exception:
                    size = 0
                    modified = ""
                title = os.path.splitext(fname)[0]
                if len(results) < 100:
                    results.append(
                        {"path": rel, "title": title, "modified": modified, "size": size}
                    )
            if len(results) >= 100:
                break
    except Exception as e:
        print(f"[API] obsidian/notes walk error: {e}")
        return []

    # 更新日時の降順でソート
    results.sort(key=lambda x: x["modified"], reverse=True)
    return results


@router.post("/api/obsidian/notes/read")
def read_obsidian_notes(req: ObsidianReadRequest):
    """指定された相対パスの .md ファイルを読み込んで結合テキストを返す。
    各ファイル最大10KBを読み込む。
    """
    vault_path = _get_vault_path()
    if not vault_path or not os.path.isdir(vault_path):
        return {"content": "", "files_read": []}

    try:
        content, files_read = _read_obsidian_files(vault_path, req.paths)
        return {"content": content, "files_read": files_read}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


def _research_organ_vault_path() -> Path:
    vault_root = _OBSIDIAN_VAULT_PATH or _get_vault_path()
    if not vault_root:
        try:
            from scripts.auto_research_lease_judgment import DEFAULT_VAULT

            vault_root = str(DEFAULT_VAULT)
        except Exception:
            vault_root = ""
    if not vault_root or not os.path.isdir(vault_root):
        raise HTTPException(
            status_code=503,
            detail="iCloud 上の Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH を設定してください。",
        )
    return Path(vault_root)


@router.get("/api/research-organ/topics")
def list_research_organ_topics():
    """紫苑の外部調査器官で使える定型Researchテーマを返す。"""
    try:
        from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, TOPICS

        return {
            "adapter": "gemini-google-search",
            "label": "Google AI Studio Researcher",
            "default_output_dir": DEFAULT_OUTPUT_DIR,
            "topics": [
                {
                    "key": topic.key,
                    "title": topic.title,
                    "query": topic.query,
                    "validity_days": topic.validity_days,
                    "tags": list(topic.tags),
                }
                for topic in TOPICS
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Researcher設定の読み込みに失敗しました: {exc}")


@router.get("/api/research-organ/notes")
def list_research_organ_notes(limit: int = 20):
    """通常VaultのResearch配下に保存された外部調査ノートを新しい順に返す。"""
    vault = _research_organ_vault_path()
    research_root = vault / _RESEARCH_ORGAN_FOLDER
    if not research_root.exists():
        return {"notes": [], "vault": str(vault), "research_root": str(research_root)}

    notes = []
    try:
        for path in iter_vault_md_files(vault, (_RESEARCH_ORGAN_FOLDER,), (".obsidian",)):
            try:
                rel = path.relative_to(vault).as_posix()
                stat = path.stat()
                head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                title_match = re.search(r"^#\s+(.+)$", head, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else path.stem
                notes.append(
                    {
                        "path": rel,
                        "title": title,
                        "modified": stat.st_mtime,
                        "size": stat.st_size,
                    }
                )
            except Exception:
                continue
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Researchノート一覧の取得に失敗しました: {exc}")

    notes.sort(key=lambda item: item["modified"], reverse=True)
    return {
        "notes": notes[: max(1, min(limit, 100))],
        "vault": str(vault),
        "research_root": str(research_root),
    }


def _research_note_section(markdown: str, title: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(title)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        markdown,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _research_note_bullets(markdown: str, title: str, limit: int = 3) -> list[str]:
    section = _research_note_section(markdown, title)
    bullets: list[str] = []
    for line in section.splitlines():
        text = re.sub(r"^\s*[-*・]\s*", "", line).strip()
        if not text or text.startswith("```") or text.startswith(">"):
            continue
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        bullets.append(text)
        if len(bullets) >= limit:
            break
    return bullets


def _research_run_display(result: dict) -> dict:
    path_value = result.get("path")
    if not path_value:
        return {}
    try:
        note_path = Path(str(path_value))
        if not note_path.exists() or note_path.suffix.lower() != ".md":
            return {}
        markdown = note_path.read_text(encoding="utf-8", errors="ignore")
        summary = _research_note_bullets(markdown, "結論", 3)
        use_cases = _research_note_bullets(markdown, "リース審査への適用", 4)
        questions = _research_note_bullets(markdown, "担当者が確認する質問", 3)
        return {
            "summary": summary,
            "use_cases": use_cases,
            "review_questions": questions,
        }
    except Exception as exc:
        return {"summary_warning": f"保存済みノートの要約読み取りに失敗しました: {exc}"}


@router.post("/api/research-organ/run")
async def run_research_organ(req: ResearchOrganRunRequest):
    """Gemini Google Search researcherで調査し、Obsidian Researchへ保存する。"""
    topic = (req.topic or "").strip()
    if len(topic) > 160:
        raise HTTPException(status_code=400, detail="調査テーマは160文字以内にしてください。")

    vault = _research_organ_vault_path()
    try:
        from scripts.auto_research_lease_judgment import DEFAULT_OUTPUT_DIR, run as run_auto_research

        result = await asyncio.to_thread(
            run_auto_research,
            vault,
            DEFAULT_OUTPUT_DIR,
            topic,
            req.dry_run,
        )
        return {
            "ok": True,
            "adapter": "gemini-google-search",
            "label": "Google AI Studio Researcher",
            "dry_run": req.dry_run,
            **result,
            **({} if req.dry_run else _research_run_display(result)),
        }
    except RuntimeError as exc:
        message = str(exc)
        status = 503 if "GEMINI_API_KEY" in message or "Gemini" in message else 500
        raise HTTPException(status_code=status, detail=message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"外部調査器官の実行に失敗しました: {exc}")


# =============================================================================
# 汎用エージェントハブ & 文明年代記 API (Phase 15.5)
# =============================================================================

# ── エージェントハブ / 文豪AI 関連 ─────────────────────────────────────────────

@router.get("/api/agent_hub/thoughts")
def get_agent_thoughts(limit: int = 50):
    thoughts_path = os.path.join(_REPO_ROOT, "data", "agent_thoughts.jsonl")
    if not os.path.exists(thoughts_path):
        return {"thoughts": []}
    try:
        with open(thoughts_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            results = []
            for line in reversed(lines[-limit:]):
                try: results.append(json.loads(line))
                except: continue
            return {"thoughts": results}
    except Exception as e:
        return {"thoughts": [], "error": str(e)}

@router.get("/api/agent_hub/novel/latest")
def get_latest_novel_api():
    from novelist_agent import get_latest_novel
    try:
        novel = get_latest_novel()
        return {"novel": novel}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")

@router.post("/api/agent_hub/script/generate")
def generate_script_api():
    """脚本家AIが最新ニュースからプロットを生成・保存する。"""
    try:
        import scriptwriter_agent
        plot_data = scriptwriter_agent.generate_weekly_plot()
        return {
            "title": plot_data.get("title"),
            "plot_text": plot_data.get("plot_text"),
            "story_arc": plot_data.get("story_arc"),
            "source_news": plot_data.get("source_news", []),
            "generated_at": plot_data.get("generated_at"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@router.get("/api/agent_hub/script/latest")
def get_latest_script_api():
    """保存済みの最新プロットを返す。"""
    try:
        import scriptwriter_agent
        plot = scriptwriter_agent.get_latest_plot()
        return {"plot": plot}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


class NovelGenerateRequest(BaseModel):
    obsidian_paths: List[str] = Field(default_factory=list)


@router.post("/api/agent_hub/novel/generate")
def generate_novel_api(req: Optional[NovelGenerateRequest] = None):
    """文豪AI「波乱丸」の小説生成エンドポイント（連作対応版・Obsidian素材対応）。
    1. 最新プロットの鮮度確認 → 1時間超なら再取得
    2. 直近3話サマリーを custom_theme に注入
    3. obsidian_paths が指定されていれば素材を custom_theme に追記
    4. novelist_agent.generate_novel() に委譲
    """
    import datetime as _dt
    if req is None:
        req = NovelGenerateRequest()

    try:
        # ── 1. ネット情報取得（1時間キャッシュ）─────────────────────────────
        try:
            import scriptwriter_agent as _sa
            _existing = _sa.get_latest_plot()
            _plot_fresh = False
            if _existing and _existing.get("generated_at"):
                try:
                    _gen_at = _dt.datetime.strptime(
                        _existing["generated_at"], "%Y-%m-%d %H:%M:%S"
                    )
                    if (_dt.datetime.now() - _gen_at).total_seconds() < 3600:
                        _plot_fresh = True
                except Exception:
                    pass
            if not _plot_fresh:
                _sa.generate_weekly_plot()
        except Exception:
            pass  # ネット取得失敗でも小説生成は続行する

        # ── 2. 前話サマリー構築（直近3話、古い順）────────────────────────────
        from novelist_agent import generate_novel, load_novels
        recent = load_novels(limit=3)
        recent_sorted = list(reversed(recent))  # 新→古 → 古→新 に並べ直す

        serial_context = ""
        if recent_sorted:
            lines = [
                "【連作継続】これまでのあらすじ（必ずストーリーを発展させること）:"
            ]
            for ep in recent_sorted:
                body_preview = (ep.get("body") or "")[:300]
                lines.append(
                    f"第{ep['episode_no']}話「{ep['title']}」: {body_preview}..."
                )
            serial_context = "\n".join(lines)

        # ── 3. Obsidian素材の注入 ─────────────────────────────────────────────
        custom_theme = serial_context
        files_read: List[str] = []
        if req.obsidian_paths:
            vault_path = _get_vault_path()
            if vault_path and os.path.isdir(vault_path):
                obsidian_text, files_read = _read_obsidian_files(vault_path, req.obsidian_paths)
                if obsidian_text:
                    obsidian_block = "【Obsidian素材】\n" + obsidian_text
                    if serial_context:
                        custom_theme = obsidian_block + "\n\n" + serial_context
                    else:
                        custom_theme = obsidian_block

        # ── 4. 小説生成 ────────────────────────────────────────────────────────
        result = generate_novel(custom_theme=custom_theme)
        # 使用したObsidianファイル名をレスポンスに付加
        if files_read:
            result["obsidian_files_used"] = files_read
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部エラーが発生しました")


@router.get("/api/agent_hub/novel/episodes")
def get_novel_episodes_api():
    """小説エピソード一覧（バックナンバー）を返す。最大20件。"""
    from novelist_agent import load_novels
    try:
        novels = load_novels(limit=20)
        episodes = [
            {
                "id": n["id"],
                "episode_no": n["episode_no"],
                "title": n["title"],
                "week_label": n["week_label"],
                "ts": n["ts"],
                "body_preview": (n.get("body") or "")[:150],
                "body": n.get("body") or "",
            }
            for n in novels
        ]
        return {"episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AgentRunRequest(BaseModel):
    agent_id: str  # benchmark, market, gunshi, team, slack, anomaly, retrain
    params: Dict[str, Any] = {}

@router.post("/api/agent_hub/run_agent")
def run_agent_api(req: AgentRunRequest):
    agent_id = req.agent_id
    params = req.params

    try:
        if agent_id == "benchmark":
            industry = params.get("industry", "製造業")
            res = _run_benchmark_agent_standalone(industry)
            return {"status": "success", "result": res}

        elif agent_id == "market":
            res = _run_market_agent_standalone()
            return {"status": "success", "result": res}

        elif agent_id == "anomaly":
            from components.agent_hub import _run_anomaly_agent
            from data_cases import load_all_cases
            cases = load_all_cases()
            res = _run_anomaly_agent(cases)
            return {"status": "success", "result": res}

        elif agent_id == "retrain":
            from auto_optimizer import get_training_status
            from components.agent_hub import _run_retrain_trigger
            status = get_training_status()
            if status["count"] < 50:
                return {"status": "skipped", "message": "案件数が50件未満のため再学習はスキップされました。"}
            res = _run_retrain_trigger(threshold=0.02)
            return {"status": "success", "result": res}

        else:
            return {"status": "error", "message": f"Unknown agent: {agent_id}"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── スタンドアロン実行ヘルパー（Streamlit依存回避） ────────────────────────────────

_BENCHMARK_FALLBACK: dict[str, dict] = {
    "製造業":       {"op_margin": 3.5, "equity_ratio": 38.0, "roa": 3.2, "current_ratio": 140.0, "dscr": 1.4},
    "建設業":       {"op_margin": 4.2, "equity_ratio": 32.0, "roa": 3.8, "current_ratio": 130.0, "dscr": 1.5},
    "卸売業":       {"op_margin": 2.1, "equity_ratio": 30.0, "roa": 2.5, "current_ratio": 125.0, "dscr": 1.3},
    "小売業":       {"op_margin": 2.8, "equity_ratio": 28.0, "roa": 3.0, "current_ratio": 115.0, "dscr": 1.2},
    "運輸業":       {"op_margin": 3.0, "equity_ratio": 25.0, "roa": 2.8, "current_ratio": 110.0, "dscr": 1.3},
    "情報通信業":   {"op_margin": 8.5, "equity_ratio": 52.0, "roa": 6.5, "current_ratio": 170.0, "dscr": 2.0},
    "不動産業":     {"op_margin": 12.0,"equity_ratio": 35.0, "roa": 4.0, "current_ratio": 120.0, "dscr": 1.6},
    "医療・福祉":   {"op_margin": 4.5, "equity_ratio": 42.0, "roa": 3.5, "current_ratio": 145.0, "dscr": 1.5},
    "サービス業":   {"op_margin": 5.0, "equity_ratio": 38.0, "roa": 4.2, "current_ratio": 135.0, "dscr": 1.4},
    "飲食業":       {"op_margin": 2.0, "equity_ratio": 18.0, "roa": 2.0, "current_ratio": 90.0,  "dscr": 1.1},
    "農業・漁業":   {"op_margin": 2.5, "equity_ratio": 30.0, "roa": 2.2, "current_ratio": 120.0, "dscr": 1.2},
    "金融・保険業": {"op_margin": 15.0,"equity_ratio": 55.0, "roa": 5.0, "current_ratio": 180.0, "dscr": 2.2},
    "教育・学習支援業": {"op_margin": 5.5, "equity_ratio": 45.0, "roa": 4.0, "current_ratio": 150.0, "dscr": 1.6},
    "宿泊業":       {"op_margin": 3.0, "equity_ratio": 22.0, "roa": 2.5, "current_ratio": 100.0, "dscr": 1.2},
    "その他":       {"op_margin": 4.0, "equity_ratio": 33.0, "roa": 3.0, "current_ratio": 125.0, "dscr": 1.3},
}

def _run_benchmark_agent_standalone(industry: str):
    from ai_chat import _chat_for_thread
    import re
    api_key = os.environ.get("GEMINI_API_KEY", "")
    system = (
        "あなたはリース審査の財務分析専門家です。"
        "指定された業種について、日本の中小企業の財務指標（業界平均）を推定してください。"
        '{"op_margin": <営業利益率%>, "equity_ratio": <自己資本比率%>, '
        '"roa": <ROA%>, "current_ratio": <流動比率%>, "dscr": <DSCR倍>}'
    )
    prompt = f"業種: {industry}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    try:
        res_raw = _chat_for_thread("gemini", "", messages, timeout_seconds=60, api_key=api_key)
        content = (res_raw.get("message") or {}).get("content", "")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            data["_source"] = "ai"
            return data
        # AIレスポンスのパース失敗 → フォールバック
    except Exception:
        pass

    # Gemini 失敗時: 静的ベンチマークを返す
    fallback = _BENCHMARK_FALLBACK.get(industry, _BENCHMARK_FALLBACK["その他"]).copy()
    fallback["_source"] = "static"
    return fallback

def _run_market_agent_standalone():
    from ai_chat import _chat_for_thread
    api_key = os.environ.get("GEMINI_API_KEY", "")
    system = "あなたは経済・金融アナリストです。現在の日本の金利状況を200字程度で報告してください。"
    prompt = "最新の市況を教えてください。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    res_raw = _chat_for_thread("gemini", "", messages, timeout_seconds=60, api_key=api_key)
    return {"content": (res_raw.get("message") or {}).get("content", "")}