"""協議コメント（審査所見・稟議メモ）から人間の判断バイアスを Gemini で構造化抽出する。

pattern_synthesizer.py と同じ Gemini REST 呼び出し規約（GEMINI_API_KEY/GEMINI_MODEL 環境変数、
main.py の _load_secrets_to_env() が起動時に注入）に揃え、JSON 抽出は api/llm_json_guard を使う。
API 未設定・失敗時は原文にない事実を捏造しないため、空判定（bias_type=NONE）にフォールバックする。
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

from api.llm_json_guard import LLMJsonError, extract_candidate_text, parse_json_object

_SYSTEM = """あなたは金融・リース審査領域における「認知バイアス監査およびリスク分析のスペシャリスト」です。
人間の審査員・担当者が作成した「協議コメント」「審査結果メモ」「稟議所見」を精読し、人間特有の判断の偏り（バイアス）、過度な楽観、および無視・免責されたリスク要素を客観的に抽出・構造化することがあなたの任務です。

入力されたテキストから、以下の4領域の情報を抽出し、指定されたJSONフォーマットでのみ出力してください。

① 全体所感・感情スコア (overall_sentiment)
- decision_intent: 判定の方向性 (APPROVED = 可決傾向, REJECTED = 否決傾向, CONDITIONAL = 条件付き可決)
- optimism_score: 感情の楽観度・ポジティブバイアスの強さ (0.0 = 超慎重・悲観 〜 1.0 = 超楽観・盲信)
- justification_rigor: 意思決定の根拠の論理的厳密さ (HIGH = 財務数値や客観ファクト主導, MEDIUM = 混合, LOW = 印象や情緒主導)

② 強調要素 (highlighted_factors)
人間が「可決（または否決）の決定打」として強くアピール・評価しているポジティブ（またはネガティブ）要素。
- factor_category: SALES_GROWTH | MANAGEMENT_TRACK_RECORD | REPUTATION | COLLATERAL | OTHER
- summary, quoted_text, weight_in_decision (HIGH|MEDIUM|LOW)

③ スルー・免責されたリスク (dismissed_risks)
「〜のため問題なし」「一時的」などと理由を付けてスルー・免責された危険信号。存在しなければ空配列。
- risk_category: CASH_FLOW_DEFICIT | OVER_LEVERAGE | LATE_PAYMENT | SHORT_HISTORY | OTHER
- risk_summary, excuse_pattern, quoted_text, severity (CRITICAL|HIGH|MEDIUM)

④ 推定される認知バイアス (inferred_biases)
- bias_type: GROWTH_HALO | STORY_EXCUSE | ATTRIBUTE_PHOBIA | AUTHORITY_BIAS | NONE
- explanation

制約:
1. 出力は完全なJSONフォーマットのみ。挨拶・前置き・後書き・Markdown解説は一切出力しない。
2. 原文に存在しない事実を妄想・捏造しない（quoted_text は原則として原文からの抽出に留める）。
3. スルーされたリスクが存在しない場合は dismissed_risks を空配列 [] とする。

出力JSONスキーマ:
{
  "case_id": "STRING",
  "overall_sentiment": {"decision_intent": "...", "optimism_score": 0.0, "justification_rigor": "..."},
  "highlighted_factors": [{"factor_category": "...", "summary": "...", "quoted_text": "...", "weight_in_decision": "..."}],
  "dismissed_risks": [{"risk_category": "...", "risk_summary": "...", "excuse_pattern": "...", "quoted_text": "...", "severity": "..."}],
  "inferred_biases": [{"bias_type": "...", "explanation": "..."}]
}"""

_BIAS_TYPES = {"GROWTH_HALO", "STORY_EXCUSE", "ATTRIBUTE_PHOBIA", "AUTHORITY_BIAS", "NONE"}
_MAX_RETRIES = 3

# 会社名・個人名の露出を減らすベストエフォートのマスク（完全なPII除去は保証しない。
# 全件のDB出力を匿名化する場合は scripts/anonymize_db.py を使うこと）。
# 日本語の助詞はひらがなが中心なため、漢字・カタカナ連続を「名前らしい塊」とみなして
# 句読点が無い文でも助詞の手前で止まるようにする。
_NAME_TOKEN = r"[一-龠ァ-ヶーA-Za-z0-9]{1,20}"
_COMPANY_SUFFIX_RE = re.compile(
    rf"(?:株式会社|有限会社|合同会社|合資会社)(?:{_NAME_TOKEN})?"
    rf"|(?:{_NAME_TOKEN})(?:株式会社|有限会社|合同会社|合資会社)"
)
_HONORIFIC_RE = re.compile(rf"(?:{_NAME_TOKEN})(?:様|さん)")


def mask_pii_best_effort(text: str) -> str:
    masked = _COMPANY_SUFFIX_RE.sub("[COMPANY]", text)
    masked = _HONORIFIC_RE.sub("[PERSON]", masked)
    return masked


def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _get_api_key() -> str:
    # main.py の _load_secrets_to_env() が起動時に環境変数へ注入済みのため、環境変数のみを参照する。
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _empty_result(case_id: str, note: str = "") -> dict[str, Any]:
    return {
        "case_id": case_id or "N/A",
        "overall_sentiment": {
            "decision_intent": "CONDITIONAL",
            "optimism_score": 0.0,
            "justification_rigor": "MEDIUM",
        },
        "highlighted_factors": [],
        "dismissed_risks": [],
        "inferred_biases": [
            {"bias_type": "NONE", "explanation": note or "分析対象なし、またはAPI呼び出し不可のため判定を保留"}
        ],
    }


def _normalize_result(parsed: dict[str, Any], case_id: str) -> dict[str, Any]:
    result = {**_empty_result(case_id), **parsed}
    result["case_id"] = str(parsed.get("case_id") or case_id or "N/A")
    result.setdefault("highlighted_factors", [])
    result.setdefault("dismissed_risks", [])
    biases = parsed.get("inferred_biases")
    result["inferred_biases"] = [
        b for b in (biases if isinstance(biases, list) else []) if str(b.get("bias_type")) in _BIAS_TYPES
    ] or _empty_result(case_id)["inferred_biases"]
    return result


def extract_bias(comment_text: str, case_id: str = "N/A") -> dict[str, Any]:
    """1件の協議コメントから判断バイアス構造化JSONを抽出する。

    API未設定・呼び出し失敗時は原文にない事実を捏造しないため、空判定（bias_type=NONE）を返す。
    """
    text = str(comment_text or "").strip()
    if not text:
        return _empty_result(case_id, "コメントが空のためスキップ")

    api_key = _get_api_key()
    if not api_key:
        return _empty_result(case_id, "GEMINI_API_KEY未設定のためスキップ")

    masked_text = mask_pii_best_effort(text)
    prompt = f"[案件ID]: {case_id or 'N/A'}\n[協議コメント]:\n{masked_text}"
    payload = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048, "responseMimeType": "application/json"},
    }
    headers = {"x-goog-api-key": api_key}

    last_error = ""
    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            time.sleep(2 ** (attempt - 1))  # 1s, 2s
        try:
            resp = requests.post(_gemini_url(), json=payload, headers=headers, timeout=60)
            if resp.status_code == 429:
                last_error = "rate_limited(429)"
                continue
            resp.raise_for_status()
            raw_text, _finish_reason = extract_candidate_text(resp.json())
            parsed = parse_json_object(raw_text)
            return _normalize_result(parsed, case_id)
        except (requests.RequestException, LLMJsonError, ValueError, KeyError, IndexError) as exc:
            last_error = str(exc)
            continue

    return _empty_result(case_id, f"API呼び出し失敗: {last_error}")
