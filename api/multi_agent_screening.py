"""
⚔️ マルチエージェント審査 — 紫苑（懐疑派）vs 紫苑（楽観派）+ 紫苑（統合派）

同一の紫苑中核から分岐した3つのペルソナが討論し、審査判断を統合する。

スコアが承認ライン（scoring_core.APPROVAL_LINE、既定71）以上 or 40以下 → 紫苑（統合派）単独高速処理
スコア40超〜承認ライン未満（境界・要審議） → 紫苑（懐疑派）・紫苑（楽観派）が2ラウンド討論 → 紫苑（統合派）裁定

なれ合い防止策:
  - Temperature差（懐疑派=0.3、楽観派=0.9）
  - 強制反論ラウンド（相手の主論点に必ず反論。賛同のみの回答は不可）
  - 第2ラウンドには自分の第1ラウンド意見も渡し、立場の維持/変更を明示させる
  - 意見乖離度チェック（第1ラウンドで同一意見ならスチールマン指示
    =相手の結論への最強の反対論拠を挙げさせる。無理な逆張りは強制しない。
    第2ラウンド後も一致していれば裁定役へ「討論による対立は限定的」と明示）
  - アンカリング防止: 懐疑派・楽観派・革新派には審査スコアを見せない
    （生の財務指標だけで討論させ、スコアは裁定役のみ参照）

実行制御:
  - 討論全体に時間バジェット（DEBATE_TIME_BUDGET_SEC、既定240秒）
  - ナレッジ・フィードバック検索は討論冒頭に1回だけ実行し全員に配布
  - Gemini 429/5xx は指数バックオフでリトライ
  - iter_debate_screening() でラウンドごとの途中経過をストリーミング可能
  - 討論メトリクスを data/multi_agent_debate_metrics.jsonl に追記（効果測定用）
"""
from __future__ import annotations

import json
import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterator

from api.context.context_bundle import build_context_bundle
from api.knowledge.vector_store import get_store as _get_knowledge_store
from api.knowledge.policy_loader import load_policy
from api.knowledge.feedback_watcher import search_feedback, feedback_count
from api.shion_conscience import build_conscience_prompt_block, evaluate_conscience
from api.shion_mana import build_mana_prompt_block, evaluate_mana_consultation
from lease_news_digest import find_vault, lease_news_actions_as_text, lease_news_focus_as_text
from scoring_core import APPROVAL_LINE

# ── モデル・エンドポイント ───────────────────────────────────────────────────
# 紫苑（懐疑派）・紫苑（楽観派）: Gemini Flash（temperature差で視点を分離）
# 紫苑（統合派）: Gemini Flash（temperature=0.3 で統合裁定役）
def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_DEBATE_LOW  = 40  # これ以下 → 否決ファストパス
# 承認ファストパスは scoring_core.APPROVAL_LINE（既定71）を単一ソースとして参照する。
# ハードコードすると審査結果がモジュールごとに食い違う（CLAUDE.md 参照）
_DEBATE_HIGH = APPROVAL_LINE

# 討論全体の時間バジェット（秒）。フロントの proxyTimeout（300秒）より短くし、
# 途中失敗時はフォールバック意見で必ず応答を返す。
_TOTAL_BUDGET_SEC = float(os.environ.get("DEBATE_TIME_BUDGET_SEC", "240"))

# 討論の効果測定用メトリクスログ（data/ 配下なのでコミット対象外）
_METRICS_PATH = Path(__file__).parent.parent / "data" / "multi_agent_debate_metrics.jsonl"

# ── ケースコンテキストテンプレート ──────────────────────────────────────────────
_CASE_CTX_TMPL = """## 審査案件
- 企業名: {company_name}
- 業種: {industry}
{score_line}- 売上高: {revenue}百万円
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


def _build_case_ctxs(params: dict) -> tuple[str, str]:
    """(裁定役用ctx, 討論者用ctx) を返す。

    討論者用には審査スコアを含めない。スコアを見せると討論がスコアの追認に
    寄りやすい（アンカリング）ため、懐疑派・楽観派・革新派は生の財務指標
    だけで論じ、スコアは裁定役のみが参照する。
    """
    def _base(include_score: bool) -> str:
        return _CASE_CTX_TMPL.format(
            company_name=params.get("company_name", "（未設定）"),
            industry=params.get("industry_major") or params.get("industry_sub") or "未設定",
            score_line=f"- 審査スコア: {params.get('score', 0)}点\n" if include_score else "",
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
    return _base(include_score=True) + suffix, _base(include_score=False) + suffix


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


def _remaining(deadline: float | None, cap: float = 60.0, floor: float = 5.0) -> float:
    """時間バジェットの残りを返す。残りが floor 未満なら TimeoutError。

    deadline が None の場合はバジェット管理なし（cap をそのまま返す）。
    """
    if deadline is None:
        return cap
    rem = deadline - time.monotonic()
    if rem < floor:
        raise TimeoutError("討論の時間バジェットを使い切りました")
    return min(cap, rem)


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _post_gemini(payload: dict, timeout: float, deadline: float | None = None) -> requests.Response:
    """Gemini REST API を呼ぶ。429/5xx・接続エラーは指数バックオフで最大2回リトライ。"""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(
                _gemini_url(),
                json=payload,
                headers={"x-goog-api-key": api_key},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.ConnectionError, requests.Timeout)) or (
                status in _RETRYABLE_STATUS
            )
            if not retryable or attempt >= 2:
                raise
            last_exc = exc
            backoff = 2 ** attempt  # 1秒 → 2秒
            # バジェット残りが少ないときはリトライせず即座に諦める
            if deadline is not None and time.monotonic() + backoff + 10 > deadline:
                raise
            time.sleep(backoff)
    raise last_exc  # 到達しない（防御）


def _llm_call(
    system: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 1024,
    deadline: float | None = None,
) -> dict:
    """Gemini generateContent REST API を呼び出してJSONを返す。"""
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = _post_gemini(payload, timeout=_remaining(deadline, cap=60.0), deadline=deadline)
    data = resp.json()
    candidate = (data.get("candidates") or [{}])[0]
    raw = ((candidate.get("content") or {}).get("parts") or [{}])[0].get("text", "").strip()
    finish_reason = str(candidate.get("finishReason") or "").upper()
    if finish_reason == "MAX_TOKENS" and max_tokens < 4096:
        try:
            retry_timeout = _remaining(deadline, cap=90.0, floor=15.0)
        except TimeoutError:
            retry_timeout = 0.0
        if retry_timeout:
            retry_payload = {
                **payload,
                "generationConfig": {
                    **payload["generationConfig"],
                    "temperature": min(temperature, 0.2),
                    "maxOutputTokens": max(max_tokens * 4, 2048),
                },
            }
            retry_resp = _post_gemini(retry_payload, timeout=retry_timeout, deadline=deadline)
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


# ── 共有ナレッジ検索（討論冒頭に1回だけ実行して全員に配布） ────────────────────

def _kb_entry(hits: list[dict]) -> dict:
    """検索ヒットをプロンプト注入用ブロックと引用refリストに変換する。"""
    if not hits:
        return {"block": "", "refs": []}
    lines = [f"  - {h['ref']}: {h['text'][:120]}…" for h in hits]
    block = (
        "【ナレッジ検索結果】\n" + "\n".join(lines)
        + "\n引用する場合は notes に [[ファイル名#セクション]] 形式で最大2件含めよ。"
    )
    return {"block": block, "refs": [h["ref"] for h in hits if h.get("ref")]}


def _prepare_shared_knowledge(params: dict) -> dict:
    """ナレッジ・フィードバック検索を討論全体で1回だけ実行する。

    以前は各ペルソナ×各ラウンドで Gemini Function Calling 経由の検索を
    行っていた（最大7回のFCラウンドトリップ）。案件コンテキストは全員同じ
    なので、冒頭に support/refute の2モードで検索して配布する方が
    レイテンシ・API代ともに大幅に小さい。

    Returns:
        {"support": {"block", "refs"}, "refute": {...}, "both": {...},
         "feedback_block": str}
    """
    empty = {"block": "", "refs": []}
    kb: dict = {"support": empty, "refute": empty, "both": empty, "feedback_block": ""}

    query = " ".join(filter(None, [
        params.get("industry_major") or params.get("industry_sub") or "",
        params.get("asset_name", ""),
        "リース審査 リスク 判断ポイント",
    ]))

    # 過去の訂正フィードバック（全ペルソナのシステムプロンプトに注入）
    try:
        if feedback_count() > 0:
            fb_hits = search_feedback(query, top_k=3)
            if fb_hits:
                fb_lines = []
                for h in fb_hits:
                    agent_tag = f"[{h['agent']}] " if h.get("agent") else ""
                    case_tag = f"({h['case_id']}) " if h.get("case_id") else ""
                    fb_lines.append(f"  - {agent_tag}{case_tag}{h['correction'] or h['text'][:100]}")
                kb["feedback_block"] = "【過去の訂正事例】\n" + "\n".join(fb_lines)
    except Exception:
        pass

    try:
        store = _get_knowledge_store()
        if store.count() == 0:
            return kb
        support_hits = store.search(query, mode="support", top_k=3, surface="multi_agent_screening")
        refute_hits = store.search(query, mode="refute", top_k=3, surface="multi_agent_screening")
        kb["support"] = _kb_entry(support_hits)
        kb["refute"] = _kb_entry(refute_hits)
        # both: 両モードをrefで重複排除して結合（裁定役用）
        seen: set[str] = set()
        both_hits = []
        for h in (refute_hits or []) + (support_hits or []):
            ref = h.get("ref", "")
            if ref and ref in seen:
                continue
            seen.add(ref)
            both_hits.append(h)
        kb["both"] = _kb_entry(both_hits[:4])
    except Exception:
        pass
    return kb


def _persona_call(
    system: str,
    prompt: str,
    temperature: float,
    kb: dict,
    kb_mode: str,
    deadline: float | None = None,
    max_tokens: int = 1024,
) -> dict:
    """共有ナレッジ・フィードバックを注入してペルソナ1体のLLM呼び出しを行う。"""
    if kb.get("feedback_block"):
        system = system + "\n\n" + kb["feedback_block"]
    entry = kb.get(kb_mode) or {}
    if entry.get("block"):
        prompt = prompt + "\n\n" + entry["block"]
    result = _llm_call(system, prompt, temperature, max_tokens=max_tokens, deadline=deadline)
    if entry.get("refs") and "_knowledge_refs" not in result:
        result["_knowledge_refs"] = entry["refs"]
    return result



# ── エージェント別プロンプトビルダー ────────────────────────────────────────────

def _own_opinion_block(own_json: str) -> str:
    return (
        "\n\n【あなたの第1ラウンド意見】\n" + own_json
        + "\n上記の自分の意見を踏まえ、結論を維持するか変更するか判断せよ。"
        "変更する場合はその理由を notes に書け。"
    )


def _cautious_prompt(ctx: str, counter_json: str = "", extra: str = "", own_json: str = "") -> str:
    base = f"""{ctx}

【紫苑（懐疑派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "risk_codes": ["low_profit", "leverage", "thin_equity", "liquidity", "asset_value", "industry_downturn", "document_gap"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if own_json:
        base += _own_opinion_block(own_json)
    if counter_json:
        base += (
            "\n\n【強制反論ラウンド】以下は楽観派の意見である。最も重要な論点を1つ選んで必ず反論し、"
            "reason_codes に counter を含め、反論の要点を notes に書け。賛同のみの回答は不可:\n"
            + counter_json
        )
    if extra:
        base += f"\n\n{extra}"
    return base


def _aggressive_prompt(ctx: str, counter_json: str = "", extra: str = "", own_json: str = "") -> str:
    base = f"""{ctx}

【紫苑（楽観派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "opportunity_codes": ["growth", "asset_value", "relationship", "productivity", "refinance", "strategic_need", "market_timing"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if own_json:
        base += _own_opinion_block(own_json)
    if counter_json:
        base += (
            "\n\n【強制反論ラウンド】以下は懐疑派の意見である。最も重要な論点を1つ選んで必ず反論し、"
            "reason_codes に counter を含め、反論の要点を notes に書け。賛同のみの回答は不可:\n"
            + counter_json
        )
    if extra:
        base += f"\n\n{extra}"
    return base


def _innovator_prompt(ctx: str, counter_jsons: list[str] | None = None, extra: str = "", own_json: str = "") -> str:
    base = f"""{ctx}

【紫苑（革新派）の立場】この案件を審査し、短いJSONのみで回答せよ。
長文説明は禁止。日本語の長い文章は書かず、コードと短い語句だけを返す。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reason_codes": ["score", "finance", "asset", "cashflow", "industry", "policy", "counter"],
  "innovation_codes": ["alternative_structure", "usage_data", "green_lease", "subscription", "dynamic_residual", "monitoring", "staged_approval"],
  "notes": ["10〜20字の補足を最大2個"]
}}"""
    if own_json:
        base += _own_opinion_block(own_json)
    if counter_jsons:
        base += (
            "\n\n【必須】以下は他の参加者の意見である。両者のどちらも挙げていない評価軸・条件設計を"
            " innovation_codes から最低1つ提示せよ。両者と同じ結論に安易に合流しないこと:\n"
            + "\n".join(counter_jsons)
        )
    if extra:
        base += f"\n\n{extra}"
    return base


def _arbiter_debate_prompt(ctx: str, log: str) -> str:
    return f"""{ctx}

【討論ログ】
{log}

上記の討論を踏まえ、統合派として最終裁定を短いJSONで示せ。
討論ログに「結論が一致」の注記がある場合、debate_balance を根拠にせず案件事実と条件設定で判断せよ。
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

def _safe_future(future, fallback: dict | None, deadline: float | None = None) -> dict | None:
    try:
        timeout = max(1.0, deadline - time.monotonic()) if deadline is not None else 90
        return future.result(timeout=timeout)
    except Exception as e:
        if fallback is None:
            return None
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
    マルチエージェント審査を実行する（同期版）。

    iter_debate_screening() を最後まで消費して最終結果を返す。
    引数・返却フィールドは iter_debate_screening() の docstring を参照
    （core_candidates は同期版では結果 dict にマージされる）。
    """
    result: dict = {}
    for stage, payload in iter_debate_screening(params):
        if stage == "result":
            result = payload
        elif stage == "core_candidates":
            result["core_candidates"] = payload.get("core_candidates", [])
    return result


def iter_debate_screening(params: dict) -> Iterator[tuple[str, dict]]:
    """
    マルチエージェント審査をステージごとに実行し、(stage, payload) を yield する。

    SSE（/api/multi-agent-screening/stream）で途中経過を配信するための
    ジェネレータ。UI は第1ラウンドの意見が出た時点で中間表示できる。

    ステージ:
      ("start",  {"score", "mode"})
      ("round1", {"cautious", "aggressive", "innovator"?})  # debate のみ
      ("round2", {"cautious", "aggressive", "innovator"?})  # debate のみ
      ("result", 最終結果dict)   # core_candidates を除く全フィールド
      ("core_candidates", {"core_candidates": [...]})       # debate のみ・空なら省略

    Args:
        params: スコアリングAPIと同形式のdict。"score" キー必須。
        params["session_id"]: セッションID（会話履歴保存用、任意）
        params["company_name"]: 企業名（過去履歴注入・保存用、任意）
    """
    t0 = time.monotonic()
    deadline = t0 + _TOTAL_BUDGET_SEC
    score = float(params.get("score", 0))
    company_name = params.get("company_name", "") or ""
    session_id = params.get("session_id", "") or ""
    # 討論者用 ctx には審査スコアを含めない（アンカリング防止）。裁定役のみスコアを見る
    ctx_arbiter, ctx_debater = _build_case_ctxs(params)

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
        yield ("start", {"score": score, "mode": "solo"})
        direction = "承認" if score >= _DEBATE_HIGH else "否決"
        try:
            arbiter_raw = _llm_call(
                arbiter_sys,
                _arbiter_solo_prompt(ctx_arbiter, direction),
                temperature=0.3, max_tokens=512, deadline=deadline,
            )
        except Exception as e:
            # LLM不通でも安全側の暫定裁定で必ず応答を返す
            arbiter_raw = _fallback_llm_json(_ARBITER_SYS, str(e))
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

        _log_debate_metrics(result, t0)
        yield ("result", result)
        return

    # ── 討論モード（_DEBATE_LOW < score < _DEBATE_HIGH） ─────────────────────
    yield ("start", {"score": score, "mode": "debate"})

    # ナレッジ・フィードバック検索は討論全体で1回だけ実行して全員に配布
    kb = _prepare_shared_knowledge(params)

    def _with_refs(normed: dict, raw: dict) -> dict:
        refs = raw.get("_knowledge_refs", [])
        return {**normed, "_knowledge_refs": refs} if refs else normed

    # Round 1: 並列実行（懐疑派 temperature=0.3、楽観派 temperature=0.9、革新派 temperature=0.7）
    # 懐疑派にはナレッジの否定的証拠を、楽観派には肯定的証拠を配布する
    _r1_workers = 3 if innovator_key else 2
    pool = ThreadPoolExecutor(max_workers=_r1_workers)
    try:
        fc = pool.submit(
            _persona_call, cautious_sys, _cautious_prompt(ctx_debater), 0.3, kb, "refute", deadline
        )
        fa = pool.submit(
            _persona_call, aggressive_sys, _aggressive_prompt(ctx_debater), 0.9, kb, "support", deadline
        )
        fi = pool.submit(
            _persona_call, innovator_sys, _innovator_prompt(ctx_debater), 0.7, kb, "both", deadline
        ) if innovator_key else None
        r1c = _safe_future(fc, {"opinion": "否決", "reasons": [], "key_risks": []}, deadline)
        r1a = _safe_future(fa, {"opinion": "条件付承認", "reasons": [], "opportunities": []}, deadline)
        r1i = _safe_future(fi, {"opinion": "条件付承認", "reasons": [], "innovations": []}, deadline) if fi else {}
    finally:
        # タイムアウト済みスレッドの完了を待たない（時間バジェット厳守）
        pool.shutdown(wait=False, cancel_futures=True)

    r1c_norm = _with_refs(_norm_cautious(r1c), r1c)
    r1a_norm = _with_refs(_norm_aggressive(r1a), r1a)
    r1i_norm = _with_refs(_norm_innovator(r1i), r1i) if innovator_key and r1i else {}

    round1_payload: dict = {"cautious": r1c_norm, "aggressive": r1a_norm}
    if innovator_key and r1i_norm:
        round1_payload["innovator"] = r1i_norm
    yield ("round1", round1_payload)

    # 乖離度チェック: 同意見ならスチールマン指示
    # （無理な逆張りで偽の対立を作らせず、相手の結論への最強の反対論拠を挙げさせる）
    same_opinion = r1c.get("opinion") == r1a.get("opinion")
    extra_c = (
        "【注意】第1ラウンドで楽観派と同じ結論だった。結論を無理に変える必要はないが、"
        "懐疑派として楽観派の結論に対する最強の反対論拠を1つ挙げ、notes に含めよ。"
    ) if same_opinion else ""
    extra_a = (
        "【注意】第1ラウンドで懐疑派と同じ結論だった。結論を無理に変える必要はないが、"
        "楽観派として懐疑派の結論に対する最強の反対論拠を1つ挙げ、notes に含めよ。"
    ) if same_opinion else ""

    r1c_json = json.dumps(_norm_cautious(r1c), ensure_ascii=False)
    r1a_json = json.dumps(_norm_aggressive(r1a), ensure_ascii=False)
    r1i_json = json.dumps(_norm_innovator(r1i), ensure_ascii=False) if innovator_key and r1i else ""

    # Round 2: 強制反論ラウンド（自分のR1意見＋相手の意見を受けて必ず反論）
    _r2_workers = 3 if innovator_key else 2
    pool = ThreadPoolExecutor(max_workers=_r2_workers)
    try:
        fc2 = pool.submit(
            _persona_call, cautious_sys,
            _cautious_prompt(ctx_debater, r1a_json, extra_c, own_json=r1c_json),
            0.3, kb, "refute", deadline
        )
        fa2 = pool.submit(
            _persona_call, aggressive_sys,
            _aggressive_prompt(ctx_debater, r1c_json, extra_a, own_json=r1a_json),
            0.9, kb, "support", deadline
        )
        fi2 = pool.submit(
            _persona_call, innovator_sys,
            _innovator_prompt(
                ctx_debater,
                [r1c_json, r1a_json] if r1c_json and r1a_json else None,
                own_json=r1i_json,
            ),
            0.7, kb, "both", deadline
        ) if innovator_key else None
        r2c = _safe_future(fc2, r1c, deadline)
        r2a = _safe_future(fa2, r1a, deadline)
        r2i = _safe_future(fi2, r1i, deadline) if fi2 else {}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    r2c_norm = _with_refs(_norm_cautious(r2c), r2c)
    r2a_norm = _with_refs(_norm_aggressive(r2a), r2a)
    r2i_norm = _with_refs(_norm_innovator(r2i), r2i) if innovator_key and r2i else {}

    round2_payload: dict = {"cautious": r2c_norm, "aggressive": r2a_norm}
    if innovator_key and r2i_norm:
        round2_payload["innovator"] = r2i_norm
    yield ("round2", round2_payload)

    # 討論ログ構築（ナレッジ引用があれば記録）
    # 裁定役が判断材料にできるよう、理由に加えてリスク/機会/新視点も含める
    def _fmt_refs(d: dict) -> str:
        refs = d.get("_knowledge_refs", [])
        return f" 引用: {', '.join(refs)}" if refs else ""

    def _fmt_line(label: str, d: dict, second_key: str, second_label: str) -> str:
        line = f"紫苑（{label}）: {d.get('opinion', '？')} — {_excerpt(d, 'reasons')}"
        second = _excerpt(d, second_key)
        if second != "（なし）":
            line += f" / {second_label}: {second}"
        return line + _fmt_refs(d)

    _r1_lines = (
        "【第1ラウンド：初期見解】\n"
        + _fmt_line("懐疑", r1c_norm, "key_risks", "リスク") + "\n"
        + _fmt_line("楽観", r1a_norm, "opportunities", "機会")
    )
    if innovator_key and r1i_norm:
        _r1_lines += "\n" + _fmt_line("革新", r1i_norm, "innovations", "新視点")
    _r2_lines = (
        "\n\n【第2ラウンド：強制反論】\n"
        + _fmt_line("懐疑", r2c_norm, "key_risks", "リスク") + "\n"
        + _fmt_line("楽観", r2a_norm, "opportunities", "機会")
    )
    if innovator_key and r2i_norm:
        _r2_lines += "\n" + _fmt_line("革新", r2i_norm, "innovations", "新視点")
    debate_log = _r1_lines + _r2_lines
    if same_opinion:
        debate_log = (
            "[注: 第1ラウンドで両者の意見が一致したため、スチールマン再討論"
            "（相手の結論への最強の反対論拠の提示）を実施]\n\n"
        ) + debate_log

    # 強制反論後も懐疑派・楽観派の結論が一致したままなら、裁定役に明示する
    # （見かけの「討論バランス」を根拠にした裁定を防ぐ）
    same_opinion_r2 = r2c_norm.get("opinion") == r2a_norm.get("opinion")
    if same_opinion_r2:
        debate_log += (
            f"\n\n[注: 強制反論後も懐疑派・楽観派の結論が「{r2c_norm.get('opinion', '？')}」で一致。"
            "討論による対立は限定的のため、裁定は討論バランスではなく案件事実と条件設定を根拠にすること]"
        )

    # 統合派裁定（temperature=0.3 で中立・冷静、両方向のナレッジを参照）
    try:
        arbiter_raw = _persona_call(
            arbiter_sys,
            _arbiter_debate_prompt(ctx_arbiter, debate_log),
            0.3, kb, "both", deadline, max_tokens=1024,
        )
    except Exception as e:
        # 討論結果まで出ているのに裁定LLMの不通で全体を失敗させない
        arbiter_raw = _fallback_llm_json(_ARBITER_SYS, str(e))

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
        "same_opinion_r2": same_opinion_r2,
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

    _log_debate_metrics(result, t0)

    # 最終結果を先に届け、core_candidates は後続イベントとして配信する
    # （UIの任意機能のために全ユーザーの応答を10〜20秒ブロックしない）
    yield ("result", result)

    # core_candidates: 各ペルソナの結論から汎用的な判断基準を個別に抽出
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
            yield ("core_candidates", {"core_candidates": [
                {**c, "case_summary": case_summary, "source": "debate"}
                for c in candidates
            ]})
    except Exception:
        pass


def _log_debate_metrics(result: dict, t0: float) -> None:
    """討論の効果測定用メトリクスを JSONL に追記する（失敗しても審査は継続）。

    「討論モードが単独裁定と違う結論を出す割合」「一致率と最終判断の関係」を
    後から監査できるようにする。judgment-feedback（人間の判断変更）と
    突き合わせれば討論の価値を検証できる。
    """
    try:
        opinions = {}
        for key, role in (("cautious", "skeptic"), ("aggressive", "optimist"), ("innovator", "innovator")):
            persona = result.get(key)
            if isinstance(persona, dict) and persona:
                opinions[role] = persona.get("opinion", "")
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "score": result.get("score"),
            "mode": result.get("mode"),
            "final": (result.get("arbiter") or {}).get("final", ""),
            "opinions": opinions,
            "same_opinion_r1": result.get("same_opinion_r1"),
            "same_opinion_r2": result.get("same_opinion_r2"),
            "duration_sec": round(time.monotonic() - t0, 1),
        }
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _METRICS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── コア候補抽出 ────────────────────────────────────────────────────────────────

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

    if not _get_gemini_api_key():
        return ""

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 256},
    }
    resp = _post_gemini(payload, timeout=30)
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
