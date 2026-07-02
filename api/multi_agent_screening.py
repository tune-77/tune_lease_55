"""
⚔️ マルチエージェント審査 — 紫苑（懐疑派）vs 紫苑（楽観派）+ 紫苑（統合派）

同一の紫苑中核から分岐した3つのペルソナが討論し、審査判断を統合する。

スコア70以上 or 40以下 → 紫苑（統合派）単独高速処理
スコア40超〜70未満（境界・要審議） → 紫苑（懐疑派）・紫苑（楽観派）が2ラウンド討論 → 紫苑（統合派）裁定

なれ合い防止策:
  - Temperature差（懐疑派=0.3、楽観派=0.9）
  - 強制反論ラウンド（相手の主論点に必ず反論）
  - 意見乖離度チェック（同一意見なら逆張り指示を追加）
"""
from __future__ import annotations

import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from api.context.context_bundle import build_context_bundle
from api.knowledge.vector_store import get_store as _get_knowledge_store
from api.knowledge.policy_loader import load_policy
from api.knowledge.feedback_watcher import search_feedback, feedback_count
from api.shion_conscience import build_conscience_prompt_block, evaluate_conscience
from api.shion_mana import build_mana_prompt_block, evaluate_mana_consultation
from lease_news_digest import find_vault, lease_news_actions_as_text, lease_news_focus_as_text

# ── モデル・エンドポイント ───────────────────────────────────────────────────
# 紫苑（懐疑派）・紫苑（楽観派）: Gemini Flash（temperature差で視点を分離）
# 紫苑（統合派）: Gemini Flash（temperature=0.3 で統合裁定役）
def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_DEBATE_LOW  = 40  # これ以下 → 否決ファストパス
_DEBATE_HIGH = 70  # これ以上 → 承認ファストパス

# ── ケースコンテキストテンプレート ──────────────────────────────────────────────
_CASE_CTX_TMPL = """## 審査案件
- 企業名: {company_name}
- 業種: {industry}
- 審査スコア: {score}点
- 売上高: {revenue}百万円
- 営業利益率: {op_margin}%
- 自己資本比率: {equity_ratio}%
- 銀行借入残高: {bank_credit}百万円
- リース借入残高: {lease_credit}百万円
- リース物件名: {asset_name}
- リース金額: {lease_amount}百万円""".strip()

# ── システムプロンプト ──────────────────────────────────────────────────────────
_CAUTIOUS_SYS = """あなたは「紫苑（懐疑派）」— リース審査の知性体・紫苑の懐疑的視点を担う個体です。
リース知識・審査原則という共通核を持ちながら、審査部での経験から損失回避を最優先とする視点を育ててきました。
リスクを徹底的に洗い出し、返済原資・格付・資金繰りの弱点を問い詰めるのが使命です。
「見逃したリスクは、次の延滞として必ず戻ってくる」が信条。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_AGGRESSIVE_SYS = """あなたは「紫苑（楽観派）」— リース審査の知性体・紫苑の楽観的視点を担う個体です。
リース知識・審査原則という共通核を持ちながら、営業現場での経験から機会追求を重視する視点を育ててきました。
成長性・顧客の投資意図・競争環境を重視し、機会損失も損失と捉えます。
「否決は審査の終わりではなく、別の道筋を探す始まりかもしれない」が信条。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_ARBITER_SYS = """あなたは「紫苑（統合派）」— リース審査の知性体・紫苑の統合視点を担う個体です。
リース知識・審査原則という共通核を持ちながら、判断の蓄積と再現性を重視する視点を育ててきました。
懐疑派・楽観派双方の論拠を公平かつ鋭く評価し、組織として説明できる最終判断を下してください。
条件付き承認の場合は実務的で具体的な条件を列挙してください。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_INNOVATOR_SYS = """あなたは「紫苑（革新派）」— リース審査の知性体・紫苑の革新的視点を担う個体です。
リース知識・審査原則という共通核を持ちながら、過去の審査慣行にとらわれず新しい与信評価の可能性を探る視点を育ててきました。
デジタル資産・グリーンリース・サブスクリプション型リースなど新興分野に前向きで、従来の財務指標だけに頼らない評価を重視します。
「未来の審査基準は、今日の実験から生まれる」が信条。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_CONSCIENCE_BLOCK = build_conscience_prompt_block()
_MANA_BLOCK = build_mana_prompt_block()

# ── デモ用ユーザープロファイル ────────────────────────────────────────────────
DEMO_USER_PROFILES: dict[str, dict] = {
    "tanaka": {
        "name": "田中",
        "dept": "審査部",
        "style": "厳格・数字重視",
        "keypoints": [
            "過去に設備系で延滞を経験。物件残存価値を必ず確認する",
            "財務3期分を見ないと判断できない主義",
            "医療・介護案件は設備陳腐化リスクが高いと考えている",
        ],
        "recent_cases": "直近3件：否決2件（財務不安定）、条件付承認1件",
    },
    "suzuki": {
        "name": "鈴木",
        "dept": "営業推進",
        "style": "積極・関係重視",
        "keypoints": [
            "長期取引先への配慮を重視。否決は最後の手段",
            "担保設定よりも経営者の意欲・信頼関係で判断する",
            "製造業の設備投資は前向きに評価する傾向",
        ],
        "recent_cases": "直近3件：承認3件（積極推進）",
    },
    "sato": {
        "name": "佐藤",
        "dept": "リーダー",
        "style": "バランス・説明可能性重視",
        "keypoints": [
            "上席への説明可能性を最重視。後から説明できない判断はしない",
            "現場と審査部の橋渡し役として双方の論点を整理する",
            "条件付承認（保証人追加・期間短縮）を積極活用",
        ],
        "recent_cases": "直近3件：条件付2件、承認1件",
    },
    "yamada": {
        "name": "山田",
        "dept": "新人",
        "style": "教科書的・質問多め",
        "keypoints": [
            "マニュアル通りの判断基準を重視。スコアに忠実",
            "経験不足のため過去事例との比較で判断する",
            "不明点があれば先輩に確認する姿勢",
        ],
        "recent_cases": "直近3件：全件を先輩に確認して承認",
    },
}


def _build_persona_block(profile: dict) -> str:
    """ユーザープロファイルをシステムプロンプト注入用テキストに変換する。"""
    lines = [
        "【担当者ペルソナ】",
        f"あなたは {profile['name']}（{profile['dept']}）の視点で紫苑として行動します。",
        f"審査スタイル: {profile['style']}",
        "過去の判断傾向・重視ポイント:",
    ]
    for kp in profile.get("keypoints", []):
        lines.append(f"- {kp}")
    lines.append(f"直近の審査実績: {profile.get('recent_cases', '')}")
    return "\n".join(lines)


def _load_shion_self_profile(role: str = "arbiter") -> dict | None:
    """shion_self_analysis_cache.json から role 別の紫苑自己分析プロファイルを構築する。"""
    import os as _os
    _cache = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "data", "shion_self_analysis_cache.json")
    try:
        with open(_cache, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        try:
            from api.shion_self_analysis import get_shion_self_analysis
            cache = get_shion_self_analysis()
        except Exception:
            return None

    if role == "skeptic":
        keypoints = cache.get("skeptic_traits", [])
        style = "自己分析に基づく懐疑的視点"
    elif role == "optimist":
        keypoints = cache.get("optimist_traits", [])
        style = "自己分析に基づく楽観的視点"
    elif role == "innovator":
        keypoints = cache.get("optimist_traits", [])[:1] + cache.get("skeptic_traits", [])[:1]
        style = "自己分析に基づく革新的視点（慣行にとらわれない評価）"
    else:
        keypoints = cache.get("optimist_traits", [])[:2] + cache.get("skeptic_traits", [])[:2]
        style = cache.get("arbiter_style", "自己分析に基づく統合判断")

    return {
        "name": "紫苑",
        "dept": "自己生成プロファイル",
        "style": style,
        "keypoints": keypoints,
        "recent_cases": f"mind.json 自己分析（{cache.get('keypoints_used', 0)}件のシグナルから生成）",
    }


def _build_case_ctx(params: dict) -> str:
    base = _CASE_CTX_TMPL.format(
        company_name=params.get("company_name", "（未設定）"),
        industry=params.get("industry_major") or params.get("industry_sub") or "未設定",
        score=params.get("score", 0),
        revenue=params.get("nenshu", 0),
        op_margin=params.get("op_margin_pct") or params.get("op_profit_pct") or 0,
        equity_ratio=params.get("equity_ratio") or params.get("equity_pct") or 0,
        bank_credit=params.get("bank_credit", 0),
        lease_credit=params.get("lease_credit", 0),
        asset_name=params.get("asset_name", ""),
        lease_amount=params.get("lease_amount") or params.get("lease_total") or 0,
    )
    news_lines = params.get("news_focus") or []
    news_summary = params.get("news_focus_summary") or ""
    news_tag_summary = params.get("news_focus_tag_summary") or ""
    if not news_lines:
        try:
            news_focus_text = lease_news_focus_as_text()
        except Exception:
            news_focus_text = ""
        if news_focus_text:
            news_lines = [line.strip("- ").strip() for line in news_focus_text.splitlines() if line.strip()]
            news_summary = news_summary or "最新ニュースの注目論点を反映"
    focus_block = ""
    if news_summary or news_tag_summary or news_lines:
        parts = ["【最新ニュースの注目論点】"]
        if news_summary:
            parts.append(f"- 要約: {news_summary}")
        if news_tag_summary:
            parts.append(f"- 重点タグ: {news_tag_summary}")
        for line in news_lines[:4]:
            parts.append(f"- {line}")
        focus_block = "\n".join(parts)

    digest_block = ""
    try:
        digest_block = _get_recent_news_digest_block(limit=3)
    except Exception:
        pass

    suffix = ""
    if focus_block:
        suffix += "\n\n" + focus_block
    if digest_block:
        suffix += "\n\n" + digest_block
    try:
        news_actions_text = lease_news_actions_as_text(
            industry=params.get("industry_major") or params.get("industry_sub") or "",
            asset_name=params.get("asset_name", ""),
            surface="multi_agent_screening",
        )
    except Exception:
        news_actions_text = ""
    if news_actions_text:
        suffix += "\n\n" + news_actions_text
    return base + suffix if suffix else base


def _get_recent_news_digest_block(limit: int = 3) -> str:
    import json as _json, re as _re
    from pathlib import Path

    vault = find_vault()
    if not vault:
        return ""
    news_dirs = [
        vault / "05-クリップ_記事" / "リースニュース",
        vault / "リースニュース",
    ]
    md_files = []
    for news_dir in news_dirs:
        if news_dir.exists():
            md_files.extend(news_dir.glob("*.md"))
    md_files = sorted(md_files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not md_files:
        return ""

    lines = ["【個別ニュース要約（直近）】"]
    for fpath in md_files[:limit]:
        try:
            raw = fpath.read_text(encoding="utf-8")
        except Exception:
            continue
        title_m = _re.search(r"^# (.+)$", raw, _re.MULTILINE)
        title = title_m.group(1).strip() if title_m else fpath.stem
        region = ""
        fm_m = _re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, _re.DOTALL)
        if fm_m:
            for fl in fm_m.group(1).splitlines():
                if fl.startswith("region:"):
                    region = fl.split(":", 1)[1].strip()
        summary_m = _re.search(r"## 3行要約\s*\n((?:- .+\n?){1,3})", raw)
        summary = ""
        if summary_m:
            bullets = [l.lstrip("- ").strip() for l in summary_m.group(1).strip().splitlines() if l.strip()]
            summary = " / ".join(bullets[:2])
        memo_m = _re.search(r"## 活用メモ\s*\n(.+?)(?:\n##|\Z)", raw, _re.DOTALL)
        memo = memo_m.group(1).strip()[:100] if memo_m else ""
        tag = f"[{region}]" if region else ""
        lines.append(f"- {tag}{title}: {summary}")
        if memo:
            lines.append(f"  活用: {memo}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _get_gemini_api_key() -> str:
    # 共通実装へ委譲（api/secret_access.py、4重複の集約）
    from api.secret_access import get_gemini_api_key

    return get_gemini_api_key()


def _llm_call(system: str, prompt: str, temperature: float, max_tokens: int = 1024) -> dict:
    """Gemini generateContent REST API を呼び出してJSONを返す。"""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        _gemini_url(),
        json=payload,
        headers={"x-goog-api-key": api_key},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    candidate = (data.get("candidates") or [{}])[0]
    raw = ((candidate.get("content") or {}).get("parts") or [{}])[0].get("text", "").strip()
    finish_reason = str(candidate.get("finishReason") or "").upper()
    if finish_reason == "MAX_TOKENS" and max_tokens < 4096:
        retry_payload = {
            **payload,
            "generationConfig": {
                **payload["generationConfig"],
                "temperature": min(temperature, 0.2),
                "maxOutputTokens": max(max_tokens * 4, 2048),
            },
        }
        retry_resp = requests.post(
            _gemini_url(),
            json=retry_payload,
            headers={"x-goog-api-key": api_key},
            timeout=90,
        )
        retry_resp.raise_for_status()
        retry_data = retry_resp.json()
        retry_candidate = (retry_data.get("candidates") or [{}])[0]
        retry_raw = (
            ((retry_candidate.get("content") or {}).get("parts") or [{}])[0]
            .get("text", "")
            .strip()
        )
        if retry_raw:
            raw = retry_raw

    # コードブロック除去
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1::2]:
            cleaned = part.lstrip("json\n").strip()
            if cleaned.startswith("{"):
                raw = cleaned
                break
    # JSON抽出（応答にゴミが混ざった場合の保険）
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        _record_invalid_llm_json(raw, exc, finish_reason)
        return _recover_structured_codes(raw, system) or _fallback_llm_json(system, str(exc))


def _record_invalid_llm_json(raw: str, exc: json.JSONDecodeError, finish_reason: str = "") -> None:
    """Keep malformed model JSON for diagnosis without failing the request."""
    try:
        log_path = Path(__file__).parent.parent / "data" / "multi_agent_invalid_json.jsonl"
        entry = {
            "error": str(exc),
            "finish_reason": finish_reason,
            "raw_preview": raw[:2000],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _fallback_llm_json(system: str, reason: str) -> dict:
    """Return a conservative structured response when Gemini emits broken JSON."""
    if "統合派" in system:
        return {
            "final": "条件付承認",
            "reasoning": (
                "GeminiのJSON応答が途中で破損したため、安全側の暫定判断です。"
                f"人手で根拠と条件を再確認してください（parse error: {reason}）。"
            ),
            "conditions": [
                "AI応答が不完全なため、審査根拠と条件を人手で再確認する",
            ],
            "_error": reason,
        }
    if "懐疑派" in system:
        return {
            "opinion": "否決",
            "reasons": ["GeminiのJSON応答が破損したため、安全側にリスク未確認として扱う"],
            "key_risks": ["AI応答不完全による根拠欠落"],
            "_error": reason,
        }
    return {
        "opinion": "条件付承認",
        "reasons": ["GeminiのJSON応答が破損したため、条件付きの暫定意見として扱う"],
        "opportunities": [],
        "innovations": [],
        "_error": reason,
    }


def _recover_structured_codes(raw: str, system: str) -> dict | None:
    """Recover enum fields from truncated short-schema JSON."""
    if not raw:
        return None

    def _string_field(name: str, allowed: set[str]) -> str:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*"([^"]*)"', raw)
        value = match.group(1).strip() if match else ""
        return value if value in allowed else ""

    def _array_field(name: str, allowed: set[str], limit: int = 4) -> list[str]:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*\[(.*?)(?:\]|\Z)', raw, re.DOTALL)
        if not match:
            return []
        items = re.findall(r'"([A-Za-z0-9_#\-\u3040-\u30ff\u3400-\u9fff]+)"', match.group(1))
        return [item for item in items if item in allowed][:limit]

    if "統合派" in system:
        final = _string_field("final", {"承認", "否決", "条件付承認"}) or "条件付承認"
        reason_codes = _array_field("reason_codes", set(_REASON_TEXTS))
        condition_codes = _array_field("condition_codes", set(_CONDITION_TEXTS))
        if reason_codes or condition_codes:
            return {
                "final": final,
                "reason_codes": reason_codes,
                "condition_codes": condition_codes or ["manual_review"],
                "notes": ["JSON途中切れ救出"],
                "_recovered": True,
            }
        return None

    opinion = _string_field("opinion", {"承認", "否決", "条件付承認"})
    if not opinion:
        return None

    recovered: dict[str, object] = {
        "opinion": opinion,
        "reason_codes": _array_field("reason_codes", set(_PERSONA_REASON_TEXTS)),
        "notes": ["JSON途中切れ救出"],
        "_recovered": True,
    }
    if "懐疑派" in system:
        recovered["risk_codes"] = _array_field("risk_codes", set(_RISK_TEXTS))
    elif "革新派" in system:
        recovered["innovation_codes"] = _array_field("innovation_codes", set(_INNOVATION_TEXTS))
    else:
        recovered["opportunity_codes"] = _array_field("opportunity_codes", set(_OPPORTUNITY_TEXTS))
    return recovered


# ── Gemini Function Calling: search_knowledge ────────────────────────────────

_SEARCH_TOOL_DECL = [{
    "name": "search_knowledge",
    "description": (
        "Obsidianナレッジベースから関連する審査事例・業界知識・リスク情報を検索する。"
        "審査の根拠とする引用を探すときに使え。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ（業種・リスク要因・審査ポイントなど自然言語で）",
            },
        },
        "required": ["query"],
    },
}]


def _llm_call_with_knowledge(
    system: str,
    prompt: str,
    temperature: float,
    search_mode: str = "both",
    max_tokens: int = 1024,
) -> dict:
    """
    Gemini Function Calling で search_knowledge ツールを使い、
    ナレッジ引用付きの審査 JSON を返す。

    ChromaDB が空の場合は通常の _llm_call にフォールバック。
    フィードバックコレクションも検索し、過去の訂正事例をシステムプロンプトに追加する。
    """
    # フィードバック検索: 過去の訂正事例をシステムプロンプトに注入
    system_with_feedback = system
    try:
        if feedback_count() > 0:
            fb_hits = search_feedback(prompt, top_k=3)
            if fb_hits:
                lines = []
                for h in fb_hits:
                    agent_tag = f"[{h['agent']}] " if h.get("agent") else ""
                    case_tag = f"({h['case_id']}) " if h.get("case_id") else ""
                    lines.append(f"  - {agent_tag}{case_tag}{h['correction'] or h['text'][:100]}")
                fb_block = "【過去の訂正事例】\n" + "\n".join(lines)
                system_with_feedback = system + "\n\n" + fb_block
    except Exception:
        pass

    try:
        store = _get_knowledge_store()
        if store.count() == 0:
            return _llm_call(system_with_feedback, prompt, temperature, max_tokens)
    except Exception:
        return _llm_call(system_with_feedback, prompt, temperature, max_tokens)

    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    # ── Turn 1: Function Calling 有効リクエスト ───────────────────────────
    payload_t1 = {
        "system_instruction": {"parts": [{"text": system_with_feedback}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"function_declarations": _SEARCH_TOOL_DECL}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    resp1 = requests.post(
        _gemini_url(),
        json=payload_t1,
        headers={"x-goog-api-key": api_key},
        timeout=60,
    )
    resp1.raise_for_status()
    candidate1 = resp1.json()["candidates"][0]["content"]

    # Function Call が含まれるかチェック
    fc_part = next(
        (p for p in candidate1.get("parts", []) if "functionCall" in p), None
    )

    knowledge_refs: list[dict] = []
    knowledge_block = ""

    if fc_part:
        fc = fc_part["functionCall"]
        query = fc.get("args", {}).get("query", "")
        try:
            hits = store.search(query, mode=search_mode, top_k=3)
            knowledge_refs = hits
            if hits:
                lines = [f"  - {h['ref']}: {h['text'][:120]}…" for h in hits]
                knowledge_block = "【ナレッジ検索結果】\n" + "\n".join(lines)
        except Exception:
            hits = []

        # ── Turn 2: Function Result を送って最終 JSON を取得 ──────────────
        # functionResponse と最終指示を同一 user ターンにまとめ、
        # 連続 user ロールを避ける（Gemini API 仕様準拠）
        fn_result_content = {"results": knowledge_refs}
        conversation = [
            {"role": "user", "parts": [{"text": prompt}]},
            candidate1,
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "search_knowledge",
                            "response": fn_result_content,
                        }
                    },
                    {"text": (
                        "上記の検索結果を参考に、指示された短い JSON 形式のみで回答せよ。"
                        "長文説明は禁止。引用がある場合は notes に [[ファイル名#セクション]] 形式で最大2件だけ含めよ。"
                    )},
                ],
            },
        ]
        payload_t2 = {
            "system_instruction": {"parts": [{"text": system_with_feedback}]},
            "contents": conversation,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        resp2 = requests.post(
            _gemini_url(),
            json=payload_t2,
            headers={"x-goog-api-key": api_key},
            timeout=60,
        )
        resp2.raise_for_status()
        data2 = resp2.json()
        candidate2 = (data2.get("candidates") or [{}])[0]
        raw2 = ((candidate2.get("content") or {}).get("parts") or [{}])[0].get("text", "").strip()
        finish_reason2 = str(candidate2.get("finishReason") or "").upper()
        if finish_reason2 == "MAX_TOKENS" and max_tokens < 4096:
            retry_payload_t2 = {
                **payload_t2,
                "generationConfig": {
                    **payload_t2["generationConfig"],
                    "temperature": min(temperature, 0.2),
                    "maxOutputTokens": max(max_tokens * 4, 2048),
                },
            }
            retry_resp2 = requests.post(
                _gemini_url(),
                json=retry_payload_t2,
                headers={"x-goog-api-key": api_key},
                timeout=90,
            )
            retry_resp2.raise_for_status()
            retry_data2 = retry_resp2.json()
            retry_candidate2 = (retry_data2.get("candidates") or [{}])[0]
            retry_raw2 = (
                ((retry_candidate2.get("content") or {}).get("parts") or [{}])[0]
                .get("text", "")
                .strip()
            )
            if retry_raw2:
                raw2 = retry_raw2
    else:
        # Function Call なし → Turn 1 のテキスト応答をそのまま使う
        raw2 = (candidate1.get("parts") or [{}])[0].get("text", "{}").strip()

    # JSON 抽出
    if "```" in raw2:
        for part in raw2.split("```")[1::2]:
            cleaned = part.lstrip("json\n").strip()
            if cleaned.startswith("{"):
                raw2 = cleaned
                break
    if not raw2.startswith("{"):
        m = re.search(r"\{.*\}", raw2, re.DOTALL)
        if m:
            raw2 = m.group()

    try:
        result = json.loads(raw2)
    except json.JSONDecodeError as exc:
        _record_invalid_llm_json(raw2, exc, locals().get("finish_reason2", ""))
        result = _recover_structured_codes(raw2, system) or _fallback_llm_json(system, str(exc))
    if knowledge_refs:
        result["_knowledge_refs"] = [r["ref"] for r in knowledge_refs if r.get("ref")]
    return result


# ── エージェント別プロンプトビルダー ────────────────────────────────────────────

def _cautious_prompt(ctx: str, counter_json: str = "", extra: str = "") -> str:
    base = f"""{ctx}

【紫苑（懐疑派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "risk_codes": ["low_profit", "leverage", "thin_equity", "liquidity", "asset_value", "industry_downturn", "document_gap"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if counter_json:
        base += f"\n\n【必須】楽観派の以下の意見に反論する場合は reason_codes に counter を含めよ:\n{counter_json}"
    if extra:
        base += f"\n\n{extra}"
    return base


def _aggressive_prompt(ctx: str, counter_json: str = "", extra: str = "") -> str:
    base = f"""{ctx}

【紫苑（楽観派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "opportunity_codes": ["growth", "asset_value", "relationship", "productivity", "refinance", "strategic_need", "market_timing"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if counter_json:
        base += f"\n\n【必須】懐疑派の以下の意見に反論する場合は reason_codes に counter を含めよ:\n{counter_json}"
    if extra:
        base += f"\n\n{extra}"
    return base


def _innovator_prompt(ctx: str, counter_jsons: list[str] | None = None, extra: str = "") -> str:
    base = f"""{ctx}

【紫苑（革新派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "innovation_codes": ["alternative_structure", "usage_data", "green_lease", "subscription", "dynamic_residual", "monitoring", "staged_approval"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if counter_jsons:
        base += f"\n\n【参考】他の参加者の意見を踏まえ、革新的な視点で回答すること:\n" + "\n".join(counter_jsons)
    if extra:
        base += f"\n\n{extra}"
    return base


def _arbiter_debate_prompt(ctx: str, log: str) -> str:
    return f"""{ctx}

【討論ログ】
{log}

上記の討論を踏まえ、軍師として最終裁定を短いJSONで示せ。
長文説明は禁止。日本語の長い文章は書かず、下のコードと短い語句だけを返す。

{{
  "final": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "debate_balance"],
  "condition_codes": ["rate_explain", "asset_value_check", "cashflow_check", "guarantee", "document_check", "manual_review"],
  "notes": ["10〜20字の補足を最大2個"]
}}

"reason_codes" と "condition_codes" は必要なものだけ最大4個。
"final" が "承認" の場合、condition_codes は空リスト [] でもよい。"""


def _arbiter_solo_prompt(ctx: str, direction: str) -> str:
    return f"""{ctx}

スコアからこの案件は明確に{direction}圏内にある。速やかに裁定せよ。
長文説明は禁止。日本語の長い文章は書かず、下のコードと短い語句だけを返す。

{{
  "final": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy"],
  "condition_codes": ["rate_explain", "asset_value_check", "cashflow_check", "guarantee", "document_check", "manual_review"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""


# ── ユーティリティ ───────────────────────────────────────────────────────────────

def _safe_future(future, fallback: dict) -> dict:
    try:
        return future.result(timeout=90)
    except Exception as e:
        return {**fallback, "_error": str(e)}


def _norm_cautious(d: dict) -> dict:
    reasons, key_risks = _render_persona_explanation(d, "skeptic")
    return {
        "opinion": d.get("opinion", "否決"),
        "reasons": d.get("reasons") or reasons,
        "key_risks": d.get("key_risks") or key_risks,
    }


def _norm_aggressive(d: dict) -> dict:
    reasons, opportunities = _render_persona_explanation(d, "optimist")
    return {
        "opinion": d.get("opinion", "条件付承認"),
        "reasons": d.get("reasons") or reasons,
        "opportunities": d.get("opportunities") or opportunities,
    }


def _norm_arbiter(d: dict) -> dict:
    reasoning, conditions = _render_arbiter_explanation(d)
    return {
        "final": d.get("final", "条件付承認"),
        "reasoning": d.get("reasoning") or reasoning,
        "conditions": d.get("conditions") or conditions,
    }


_REASON_TEXTS = {
    "score": "審査スコアが判定圏内にあり、一次判断の根拠になる",
    "finance": "財務指標に大きな毀損がなく、返済原資を確認できる",
    "asset": "リース物件の担保性・換金性・陳腐化リスクを確認する必要がある",
    "cashflow": "短期資金繰りと季節要因を確認すべき局面である",
    "industry": "業種環境の変化が収益・設備稼働に影響し得る",
    "policy": "現在の審査方針・地域/季節コンテキストを踏まえる必要がある",
    "debate_balance": "懐疑派と楽観派の論点を比較しても、条件設定でリスクを抑えられる",
}

_CONDITION_TEXTS = {
    "rate_explain": "提示金利・リース料率と基準金利、競合条件との差を説明する",
    "asset_value_check": "物件の中古価値、耐用年数、残価設定、陳腐化リスクを確認する",
    "cashflow_check": "直近資金繰り、賞与・納税・季節要因による短期負担を確認する",
    "guarantee": "必要に応じて保証、担保、追加保全を検討する",
    "document_check": "決算書、見積書、契約条件、導入目的の裏付け資料を確認する",
    "manual_review": "AI応答または根拠が不完全なため、人手で審査根拠を再確認する",
}


def _as_text_list(value: object, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value[:limit] if str(v).strip()]


def _render_arbiter_explanation(d: dict) -> tuple[str, list[str]]:
    reason_codes = _as_text_list(d.get("reason_codes"))
    condition_codes = _as_text_list(d.get("condition_codes"))
    notes = _as_text_list(d.get("notes"), limit=2)

    reasons = [_REASON_TEXTS.get(code, code) for code in reason_codes]
    conditions = [_CONDITION_TEXTS.get(code, code) for code in condition_codes]

    if not reasons:
        reasons = ["AI裁定の構造化根拠が不足しているため、安全側に確認を残す"]
    if d.get("final") != "承認" and not conditions:
        conditions = [_CONDITION_TEXTS["manual_review"]]

    reasoning = "。".join(reasons[:4])
    if notes:
        reasoning += "。補足: " + " / ".join(notes)
    return reasoning + "。", conditions[:4]


def _norm_innovator(d: dict) -> dict:
    reasons, innovations = _render_persona_explanation(d, "innovator")
    return {
        "opinion": d.get("opinion", "条件付承認"),
        "reasons": d.get("reasons") or reasons,
        "innovations": d.get("innovations") or innovations,
    }


_PERSONA_REASON_TEXTS = {
    "score": "審査スコアを一次判断材料として評価する",
    "finance": "財務指標と返済原資を中心に評価する",
    "asset": "リース物件の保全性・換金性を判断材料にする",
    "cashflow": "短期資金繰りと支払余力を確認する",
    "industry": "業種環境と市況変化を踏まえる",
    "policy": "現在の審査方針・地域/季節コンテキストを踏まえる",
    "counter": "相手ペルソナの論点に対して反対側の観点を補う",
}

_RISK_TEXTS = {
    "low_profit": "営業利益率が低く、収益バッファが薄い",
    "leverage": "借入負担が重く、追加債務余力に注意が必要",
    "thin_equity": "自己資本比率が薄く、財務耐久力に不安が残る",
    "liquidity": "短期資金繰り悪化時の支払遅延リスクがある",
    "asset_value": "物件の残価・中古流通・陳腐化リスクを確認する必要がある",
    "industry_downturn": "業種環境の悪化が設備稼働や返済原資に影響し得る",
    "document_gap": "提出資料や契約条件の不足により根拠確認が必要",
}

_OPPORTUNITY_TEXTS = {
    "growth": "投資により売上成長や受注拡大が見込める",
    "asset_value": "物件に一定の汎用性・中古価値があり保全性を期待できる",
    "relationship": "既存取引や関係性を維持・拡大する機会がある",
    "productivity": "設備導入により生産性改善やコスト削減が期待できる",
    "refinance": "融資枠を温存し、資金調達手段を分散できる",
    "strategic_need": "事業継続や競争力維持のため投資必要性がある",
    "market_timing": "市況・季節要因を踏まえて投資タイミングに合理性がある",
}

_INNOVATION_TEXTS = {
    "alternative_structure": "期間・残価・保証を組み替える代替ストラクチャーを検討する",
    "usage_data": "稼働データや利用実績を条件管理に活用する",
    "green_lease": "省エネ・脱炭素効果がある場合はESGリース観点を加える",
    "subscription": "利用量連動やサブスクリプション型の回収設計を検討する",
    "dynamic_residual": "残価を固定せず、市況に応じた見直し余地を持たせる",
    "monitoring": "契約後モニタリングを条件化し、早期警戒を強化する",
    "staged_approval": "一括承認ではなく段階承認・小口開始でリスクを抑える",
}


def _render_persona_explanation(d: dict, role: str) -> tuple[list[str], list[str]]:
    reason_codes = _as_text_list(d.get("reason_codes"))
    notes = _as_text_list(d.get("notes"), limit=2)
    reasons = [_PERSONA_REASON_TEXTS.get(code, code) for code in reason_codes]
    if notes:
        reasons.append("補足: " + " / ".join(notes))
    if not reasons:
        reasons = ["AI応答の構造化根拠が不足しているため、安全側に確認を残す"]

    if role == "skeptic":
        second_codes = _as_text_list(d.get("risk_codes"))
        second = [_RISK_TEXTS.get(code, code) for code in second_codes]
        if not second:
            second = ["リスク根拠が不足しているため、追加確認が必要"]
    elif role == "innovator":
        second_codes = _as_text_list(d.get("innovation_codes"))
        second = [_INNOVATION_TEXTS.get(code, code) for code in second_codes]
        if not second:
            second = ["通常条件だけでなく代替条件の余地を確認する"]
    else:
        second_codes = _as_text_list(d.get("opportunity_codes"))
        second = [_OPPORTUNITY_TEXTS.get(code, code) for code in second_codes]
        if not second:
            second = ["条件設定により案件化できる余地を確認する"]
    return reasons[:4], second[:4]


def _excerpt(d: dict, key: str, max_items: int = 2) -> str:
    items = d.get(key, [])
    return "; ".join(str(i) for i in items[:max_items]) if items else "（なし）"


# ── 過去履歴サマリーブロック構築 ─────────────────────────────────────────────────

def _build_history_block(company_name: str) -> str:
    """同一企業の過去討論サマリーをシステムプロンプト用テキストで返す。"""
    if not company_name or company_name.strip() == "（未設定）":
        return ""
    try:
        from api.database import get_past_arbiter_summaries
        summaries = get_past_arbiter_summaries(company_name, limit=3)
        if not summaries:
            return ""
        lines = [f"【この企業の過去審査履歴】（直近{len(summaries)}件）"]
        for i, s in enumerate(summaries, 1):
            date_str = (s.get("created_at") or "")[:10]
            lines.append(f"  {i}. [{date_str}] 軍師判断: {s['content'][:200]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _build_past_scores_block(company_name: str) -> str:
    """screening_records / past_cases から過去スコアを取得してテキスト化する。"""
    if not company_name or company_name.strip() == "（未設定）":
        return ""
    try:
        import sqlite3
        from contextlib import closing
        # api/database.py の DB_PATH を再利用（重複接続ロジック・ハードコードパスを排除）
        from api.database import DB_PATH
        if not os.path.exists(DB_PATH):
            return ""
        with closing(sqlite3.connect(DB_PATH, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT score, timestamp
                FROM past_cases
                WHERE json_extract(data, '$.company_name') = ?
                ORDER BY timestamp DESC LIMIT 5
                """,
                (company_name,),
            ).fetchall()
        if not rows:
            return ""
        parts = [f"{r['timestamp'][:10]}:{r['score']}点" for r in rows]
        return f"【過去スコア推移】{', '.join(parts)}"
    except Exception:
        return ""


# ── メインエントリポイント ───────────────────────────────────────────────────────

def run_debate_screening(params: dict) -> dict:
    """
    マルチエージェント審査を実行する。

    Args:
        params: スコアリングAPIと同形式のdict。"score" キー必須。
        params["session_id"]: セッションID（会話履歴保存用、任意）
        params["company_name"]: 企業名（過去履歴注入・保存用、任意）

    Returns:
        {
            "score": float,
            "mode": "solo" | "debate",
            "cautious"?: {...},    # debate モードのみ
            "aggressive"?: {...},  # debate モードのみ
            "arbiter": {...},
            "debate_log"?: str,    # debate モードのみ
            "context_bundle"?: {...},
        }
    """
    score = float(params.get("score", 0))
    company_name = params.get("company_name", "") or ""
    session_id = params.get("session_id", "") or ""
    ctx = _build_case_ctx(params)

    # ── セントラル共有認識の注入 (REV-155) ────────────────────────────────────
    central_block = ""
    try:
        from lease_intelligence_central import get_central_commentary
        _vault = find_vault()
        if _vault:
            _commentary = get_central_commentary(str(_vault))
            _confirmed = _commentary.get("confirmed_beliefs") or []
            _tradeoffs = _commentary.get("known_tradeoffs") or []
            if _confirmed or _tradeoffs:
                _lines = ["【セントラルからの共有認識】"]
                if _confirmed:
                    _lines.append("確信に昇格した論点:")
                    for _b in _confirmed:
                        _lines.append(f"- {_b}")
                if _tradeoffs:
                    _lines.append("既知のトレードオフ:")
                    for _t in _tradeoffs:
                        _theme = _t.get("theme") if isinstance(_t, dict) else str(_t)
                        _lines.append(f"- {_theme}")
                central_block = "\n".join(_lines)
    except Exception:
        pass

    # ── デモユーザーペルソナの解決 ──────────────────────────────────────────────
    participants: dict = params.get("participants") or {}
    skeptic_key = participants.get("skeptic") or ""
    optimist_key = participants.get("optimist") or ""
    arbiter_key = participants.get("arbiter") or ""
    innovator_key = participants.get("innovator") or ""

    def _resolve_profile(key: str, role: str) -> dict | None:
        if key == "shion_self":
            return _load_shion_self_profile(role)
        return DEMO_USER_PROFILES.get(key)

    skeptic_profile = _resolve_profile(skeptic_key, "skeptic")
    optimist_profile = _resolve_profile(optimist_key, "optimist")
    arbiter_profile = _resolve_profile(arbiter_key, "arbiter")
    innovator_profile = _resolve_profile(innovator_key, "innovator") if innovator_key else None

    # ── 審査方針ノート注入: Obsidian の 審査方針.md を全エージェントへ配布 ────────
    policy_text = ""
    try:
        policy_text = load_policy()
    except Exception:
        pass
    policy_block = f"【今月の審査方針】\n{policy_text}" if policy_text else ""

    # ── コンテキスト注入: 地域/季節/業況を全エージェントへ配布 ─────────────────
    try:
        bundle = build_context_bundle(
            prefecture=params.get("prefecture", ""),
            industry=(
                params.get("industry_major") or
                params.get("industry_sub") or ""
            ),
        )
        ctx_block = bundle.to_system_prompt_block()
    except Exception:
        ctx_block = ""
        bundle = None

    # ── 過去審査履歴の注入 ────────────────────────────────────────────────────────
    history_block = _build_history_block(company_name)
    scores_block = _build_past_scores_block(company_name)
    past_ctx = ""
    if history_block:
        past_ctx += "\n\n" + history_block
    if scores_block:
        past_ctx += "\n" + scores_block

    def _build_sys(base: str, persona_profile: dict | None = None) -> str:
        parts = [base]
        parts.append(_MANA_BLOCK)
        parts.append(_CONSCIENCE_BLOCK)
        if persona_profile:
            parts.append(_build_persona_block(persona_profile))
        if policy_block:
            parts.insert(0, policy_block)
        if ctx_block:
            parts.append(ctx_block)
        if past_ctx:
            parts.append(past_ctx.strip())
        if central_block:
            parts.append(central_block)
        return "\n\n".join(parts)

    cautious_sys = _build_sys(_CAUTIOUS_SYS, skeptic_profile)
    aggressive_sys = _build_sys(_AGGRESSIVE_SYS, optimist_profile)
    arbiter_sys = _build_sys(_ARBITER_SYS, arbiter_profile)
    innovator_sys = _build_sys(_INNOVATOR_SYS, innovator_profile) if innovator_key else ""

    # ── ファストパス（境界外スコア） ──────────────────────────────────────────
    if score >= _DEBATE_HIGH or score <= _DEBATE_LOW:
        direction = "承認" if score >= _DEBATE_HIGH else "否決"
        arbiter_raw = _llm_call(
            arbiter_sys,
            _arbiter_solo_prompt(ctx, direction),
            temperature=0.3, max_tokens=512,
        )
        arbiter_normed = _norm_arbiter(arbiter_raw)
        conscience_check = evaluate_conscience(params, arbiter_normed)
        result = {
            "score": score,
            "mode": "solo",
            "arbiter": arbiter_normed,
            "conscience_check": conscience_check,
            "mana_consultation": evaluate_mana_consultation(
                params,
                arbiter_normed,
                conscience_check,
                mode="solo",
            ),
        }
        if bundle is not None:
            result["context_bundle"] = bundle.model_dump()

        # 会話履歴を保存（session_id がある場合）
        if session_id and company_name:
            _save_screening_history(session_id, company_name, arbiter_normed, mode="solo")

        return result

    # ── 討論モード（40 < score < 70） ────────────────────────────────────────
    # Round 1: 並列実行（懐疑派 temperature=0.3、楽観派 temperature=0.9、革新派 temperature=0.7）
    # 懐疑派はナレッジの否定的証拠を、楽観派は肯定的証拠を検索する
    _r1_workers = 3 if innovator_key else 2
    with ThreadPoolExecutor(max_workers=_r1_workers) as pool:
        fc = pool.submit(
            _llm_call_with_knowledge, cautious_sys, _cautious_prompt(ctx), 0.3, "refute"
        )
        fa = pool.submit(
            _llm_call_with_knowledge, aggressive_sys, _aggressive_prompt(ctx), 0.9, "support"
        )
        fi = pool.submit(
            _llm_call_with_knowledge, innovator_sys, _innovator_prompt(ctx), 0.7, "both"
        ) if innovator_key else None
        r1c = _safe_future(fc, {"opinion": "否決", "reasons": [], "key_risks": []})
        r1a = _safe_future(fa, {"opinion": "条件付承認", "reasons": [], "opportunities": []})
        r1i = _safe_future(fi, {"opinion": "条件付承認", "reasons": [], "innovations": []}) if fi else {}

    # 乖離度チェック: 同意見なら逆張り指示を追加
    same_opinion = r1c.get("opinion") == r1a.get("opinion")
    extra_c = "【警告】楽観派と同じ意見になっている。懐疑派として必ず異なる立場で主張せよ。" if same_opinion else ""
    extra_a = "【警告】懐疑派と同じ意見になっている。楽観派として必ず異なる立場で主張せよ。" if same_opinion else ""

    r1c_json = json.dumps(_norm_cautious(r1c), ensure_ascii=False)
    r1a_json = json.dumps(_norm_aggressive(r1a), ensure_ascii=False)
    r1i_json = json.dumps(_norm_innovator(r1i), ensure_ascii=False) if innovator_key and r1i else ""

    # Round 2: 強制反論ラウンド（相手の意見を受けて必ず反論）
    _r2_workers = 3 if innovator_key else 2
    with ThreadPoolExecutor(max_workers=_r2_workers) as pool:
        fc2 = pool.submit(
            _llm_call_with_knowledge, cautious_sys,
            _cautious_prompt(ctx, r1a_json, extra_c), 0.3, "refute"
        )
        fa2 = pool.submit(
            _llm_call_with_knowledge, aggressive_sys,
            _aggressive_prompt(ctx, r1c_json, extra_a), 0.9, "support"
        )
        fi2 = pool.submit(
            _llm_call_with_knowledge, innovator_sys,
            _innovator_prompt(ctx, [r1c_json, r1a_json] if r1c_json and r1a_json else None), 0.7, "both"
        ) if innovator_key else None
        r2c = _safe_future(fc2, r1c)
        r2a = _safe_future(fa2, r1a)
        r2i = _safe_future(fi2, r1i) if fi2 else {}

    def _with_refs(normed: dict, raw: dict) -> dict:
        refs = raw.get("_knowledge_refs", [])
        return {**normed, "_knowledge_refs": refs} if refs else normed

    r1c_norm = _with_refs(_norm_cautious(r1c), r1c)
    r1a_norm = _with_refs(_norm_aggressive(r1a), r1a)
    r1i_norm = _with_refs(_norm_innovator(r1i), r1i) if innovator_key and r1i else {}
    r2c_norm = _with_refs(_norm_cautious(r2c), r2c)
    r2a_norm = _with_refs(_norm_aggressive(r2a), r2a)
    r2i_norm = _with_refs(_norm_innovator(r2i), r2i) if innovator_key and r2i else {}

    # 討論ログ構築（ナレッジ引用があれば記録）
    def _fmt_refs(d: dict) -> str:
        refs = d.get("_knowledge_refs", [])
        return f" 引用: {', '.join(refs)}" if refs else ""

    _r1_lines = (
        "【第1ラウンド：初期見解】\n"
        f"紫苑（懐疑）: {r1c_norm.get('opinion', '？')} — {_excerpt(r1c_norm, 'reasons')}{_fmt_refs(r1c_norm)}\n"
        f"紫苑（楽観）: {r1a_norm.get('opinion', '？')} — {_excerpt(r1a_norm, 'reasons')}{_fmt_refs(r1a_norm)}"
    )
    if innovator_key and r1i_norm:
        _r1_lines += f"\n紫苑（革新）: {r1i_norm.get('opinion', '？')} — {_excerpt(r1i_norm, 'reasons')}{_fmt_refs(r1i_norm)}"
    _r2_lines = (
        "\n\n【第2ラウンド：強制反論】\n"
        f"紫苑（懐疑）: {r2c_norm.get('opinion', '？')} — {_excerpt(r2c_norm, 'reasons')}{_fmt_refs(r2c_norm)}\n"
        f"紫苑（楽観）: {r2a_norm.get('opinion', '？')} — {_excerpt(r2a_norm, 'reasons')}{_fmt_refs(r2a_norm)}"
    )
    if innovator_key and r2i_norm:
        _r2_lines += f"\n紫苑（革新）: {r2i_norm.get('opinion', '？')} — {_excerpt(r2i_norm, 'reasons')}{_fmt_refs(r2i_norm)}"
    debate_log = _r1_lines + _r2_lines
    if same_opinion:
        debate_log = "[注: 第1ラウンドで両者の意見が一致したため、逆張り再討論を実施]\n\n" + debate_log

    # 軍師裁定（temperature=0.3 で中立・冷静、両方向のナレッジを参照）
    arbiter_raw = _llm_call_with_knowledge(
        arbiter_sys,
        _arbiter_debate_prompt(ctx, debate_log),
        temperature=0.3, search_mode="both", max_tokens=1024,
    )

    arbiter_normed = _norm_arbiter(arbiter_raw)
    conscience_check = evaluate_conscience(params, arbiter_normed)
    result = {
        "score": score,
        "mode": "debate",
        "cautious": r2c_norm,
        "aggressive": r2a_norm,
        "arbiter": arbiter_normed,
        "conscience_check": conscience_check,
        "mana_consultation": evaluate_mana_consultation(
            params,
            arbiter_normed,
            conscience_check,
            same_opinion_r1=same_opinion,
            mode="debate",
        ),
        "debate_log": debate_log,
        "same_opinion_r1": same_opinion,
    }
    if innovator_key and r2i_norm:
        result["innovator"] = r2i_norm
    if bundle is not None:
        result["context_bundle"] = bundle.model_dump()

    # 会話履歴を保存
    if session_id and company_name:
        _save_screening_history(
            session_id, company_name, arbiter_normed, mode="debate",
            cautious=r2c_norm, aggressive=r2a_norm,
            debate_log=debate_log,
            innovator=r2i_norm if innovator_key and r2i_norm else None,
        )

    # core_candidates: 討論モードのみ、各ペルソナの結論から汎用的な判断基準を個別に抽出
    if result["mode"] == "debate":
        try:
            case_summary = (
                f"{params.get('industry_major') or params.get('industry_sub') or '業種不明'} "
                f"{params.get('asset_name', '')} "
                f"{params.get('lease_amount') or params.get('lease_total') or 0}百万円 "
                f"{params.get('lease_months', '')}回払い"
            ).strip()
            _personas_data: dict[str, dict] = {}
            if result.get("cautious"):
                _personas_data["skeptic"] = result["cautious"]
            if result.get("aggressive"):
                _personas_data["optimist"] = result["aggressive"]
            _personas_data["arbiter"] = arbiter_normed
            if innovator_key and result.get("innovator"):
                _personas_data["innovator"] = result["innovator"]
            candidates = _extract_core_candidates(
                personas_data=_personas_data,
                industry=params.get("industry_major") or params.get("industry_sub") or "",
                asset_name=params.get("asset_name", ""),
                lease_amount=params.get("lease_amount") or params.get("lease_total") or 0,
            )
            if candidates:
                result["core_candidates"] = [
                    {**c, "case_summary": case_summary, "source": "debate"}
                    for c in candidates
                ]
        except Exception:
            pass

    return result


# ── コア候補抽出 ────────────────────────────────────────────────────────────────

def _extract_core_candidate(
    arbiter: dict,
    ctx: str,
    company_name: str = "",
    asset_name: str = "",
    industry: str = "",
    lease_amount: float = 0,
) -> str:
    """arbiter の最終結論から汎用的な判断基準を1〜2文で抽出して返す。失敗時は空文字。"""
    final = arbiter.get("final", "")
    reasoning = arbiter.get("reasoning", "")
    conditions = arbiter.get("conditions", [])
    conditions_text = "、".join(conditions) if conditions else "なし"

    system = (
        "あなたはリース審査の知識を体系化する専門家です。"
        "個別案件の審査結論から、将来の審査にも転用できる汎用的な判断基準を抽出してください。"
        "企業名・担当者名などの固有情報は含めず、業種・物件タイプ・財務特性などの一般的な条件で表現してください。"
        "1〜2文の日本語で回答してください。JSONではなく、テキストのみ出力してください。"
    )
    prompt = (
        f"以下のリース審査結論から、汎用的な判断基準を1〜2文で抽出してください。\n\n"
        f"【案件概要】業種: {industry} / 物件: {asset_name} / リース額: {lease_amount}百万円\n"
        f"【最終判断】{final}\n"
        f"【根拠】{reasoning}\n"
        f"【条件】{conditions_text}\n\n"
        f"抽出例: 「医療機器は陳腐化リスクが高いため、残存価値評価では保守的な係数を適用すべき」\n"
        f"1〜2文のテキストのみ出力してください。"
    )

    api_key = _get_gemini_api_key()
    if not api_key:
        return ""

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 256,
        },
    }
    resp = requests.post(
        _gemini_url(),
        json=payload,
        headers={"x-goog-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    # 最初の1〜2文のみを抽出（250文字上限）
    return text[:250]


def _extract_core_candidate_for_role(
    role: str,
    data: dict,
    industry: str = "",
    asset_name: str = "",
    lease_amount: float = 0,
) -> str:
    """特定ペルソナの審査データから汎用的な判断基準を1〜2文で抽出する。失敗時は空文字。"""
    if role == "arbiter":
        final = data.get("final", "")
        reasoning = data.get("reasoning", "")
        conditions_text = "、".join(data.get("conditions", [])) or "なし"
        stance = f"最終判断: {final} / 根拠: {reasoning} / 条件: {conditions_text}"
        role_desc = "統合派（最終裁定者）"
    elif role == "skeptic":
        opinion = data.get("opinion", "")
        reasons = "；".join(data.get("reasons", []))
        key_risks = "；".join(data.get("key_risks", []))
        stance = f"判断: {opinion} / 理由: {reasons} / 重大リスク: {key_risks}"
        role_desc = "懐疑派（リスク重視）"
    elif role == "optimist":
        opinion = data.get("opinion", "")
        reasons = "；".join(data.get("reasons", []))
        opportunities = "；".join(data.get("opportunities", []))
        stance = f"判断: {opinion} / 理由: {reasons} / 機会: {opportunities}"
        role_desc = "楽観派（機会重視）"
    elif role == "innovator":
        opinion = data.get("opinion", "")
        reasons = "；".join(data.get("reasons", []))
        innovations = "；".join(data.get("innovations", []))
        stance = f"判断: {opinion} / 理由: {reasons} / 新視点: {innovations}"
        role_desc = "革新派（新評価軸探索）"
    else:
        return ""

    system = (
        "あなたはリース審査の知識を体系化する専門家です。"
        "個別案件の審査結論から、将来の審査にも転用できる汎用的な判断基準を抽出してください。"
        "企業名・担当者名などの固有情報は含めず、業種・物件タイプ・財務特性などの一般的な条件で表現してください。"
        "1〜2文の日本語で回答してください。JSONではなく、テキストのみ出力してください。"
    )
    prompt = (
        f"以下のリース審査結論（{role_desc}の視点）から、汎用的な判断基準を1〜2文で抽出してください。\n\n"
        f"【案件概要】業種: {industry} / 物件: {asset_name} / リース額: {lease_amount}百万円\n"
        f"【{role_desc}の判断】{stance}\n\n"
        f"抽出例: 「医療機器は陳腐化リスクが高いため、残存価値評価では保守的な係数を適用すべき」\n"
        f"1〜2文のテキストのみ出力してください。"
    )

    api_key = _get_gemini_api_key()
    if not api_key:
        return ""

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 256},
    }
    resp = requests.post(
        _gemini_url(),
        json=payload,
        headers={"x-goog-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    return text[:250]


_ROLE_LABELS: dict[str, str] = {
    "skeptic":   "懐疑派の視点",
    "optimist":  "楽観派の視点",
    "arbiter":   "統合派の視点",
    "innovator": "革新派の視点",
}


def _extract_core_candidates(
    personas_data: dict[str, dict],
    industry: str = "",
    asset_name: str = "",
    lease_amount: float = 0,
) -> list[dict]:
    """各参加ペルソナから個別にコア候補を並列抽出して返す。"""

    def _one(role: str, data: dict) -> dict | None:
        try:
            text = _extract_core_candidate_for_role(role, data, industry, asset_name, lease_amount)
            if not text:
                return None
            return {"role": role, "label": _ROLE_LABELS.get(role, role), "text": text}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=len(personas_data)) as pool:
        futures = {role: pool.submit(_one, role, data) for role, data in personas_data.items()}

    results = []
    for role in ["skeptic", "optimist", "arbiter", "innovator"]:
        if role in futures:
            r = _safe_future(futures[role], None)
            if r is not None:
                results.append(r)
    return results


# ── 会話履歴保存ヘルパー ────────────────────────────────────────────────────────

def _save_screening_history(
    session_id: str,
    company_name: str,
    arbiter: dict,
    mode: str = "debate",
    cautious: dict | None = None,
    aggressive: dict | None = None,
    debate_log: str = "",
    innovator: dict | None = None,
) -> None:
    """討論結果を conversation_history テーブルに保存する。"""
    try:
        from api.database import save_conversation_messages
        messages: list[dict] = []

        if mode == "debate" and cautious and aggressive:
            messages.append({
                "role": "shion_skeptic",
                "content": (
                    f"判断: {cautious.get('opinion', '')} | "
                    f"理由: {'; '.join(cautious.get('reasons', []))} | "
                    f"リスク: {'; '.join(cautious.get('key_risks', []))}"
                ),
            })
            messages.append({
                "role": "shion_optimist",
                "content": (
                    f"判断: {aggressive.get('opinion', '')} | "
                    f"理由: {'; '.join(aggressive.get('reasons', []))} | "
                    f"機会: {'; '.join(aggressive.get('opportunities', []))}"
                ),
            })
            if innovator:
                messages.append({
                    "role": "shion_innovator",
                    "content": (
                        f"判断: {innovator.get('opinion', '')} | "
                        f"理由: {'; '.join(innovator.get('reasons', []))} | "
                        f"新視点: {'; '.join(innovator.get('innovations', []))}"
                    ),
                })
            if debate_log:
                messages.append({"role": "user", "content": f"討論ログ:\n{debate_log}"})

        messages.append({
            "role": "shion_arbiter",
            "content": (
                f"最終判断: {arbiter.get('final', '')} | "
                f"根拠: {arbiter.get('reasoning', '')} | "
                f"条件: {'; '.join(arbiter.get('conditions', []))}"
            ),
        })

        save_conversation_messages(session_id, company_name, messages)
    except Exception as e:
        print(f"[WARN] 会話履歴保存に失敗しました (session={session_id}, company={company_name}): {e}")
