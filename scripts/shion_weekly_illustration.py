#!/usr/bin/env python3
"""週1回、その週のチャットから題材を選んで「今日の紫苑」のイラストを1枚作る（REV-490）。

launchd: com.tunelease.shion-weekly-illustration（日曜 07:30）。毎朝の生成は止めたまま
（SHION_DAILY_ILLUSTRATION_ENABLED 既定オフ）。

1. 直近7日の紫苑との会話（/chat・紫苑対話室）のユーザー発話を読み取り専用で集める
2. プライバシー: 審査・財務・会社名に触れた発話は除外、数字は伏せ、人名・社名は
   mask_for_jev で伏せる（伏せきれない発話は捨てる）
3. flash-lite で「印象的な出来事」を1つ選び、場面・構図・表情・一言を作る（過去の題材と重複させない）
4. キャラクター参照画像で見た目を揃えて gemini-3.1-flash-image で1枚生成
5. data/lease_grumble_gallery/<日付>.webp に保存（ランダム候補に入る）し、一言を captions.json に、
   題材の履歴を data/shion_weekly_illustration_history.jsonl に残す。Obsidian のアーカイブにも写す

予算ガード（ai_budget）では自発系（proactive）。1回 約14円。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_runtime_client import _MAIN_ROOT  # noqa: E402

USER_IDS = ("lease-intelligence-dialogue", "shion-default", "default")
HISTORY_PATH = _MAIN_ROOT / "data" / "shion_weekly_illustration_history.jsonl"
CHARACTER_PATH = _MAIN_ROOT / "frontend" / "public" / "lease-grumble" / "characters" / "lease-intelligence-girl.jpg"
TOPIC_FEATURE = "shion_weekly_illustration_topic"
IMAGE_FEATURE = "shion_weekly_illustration"
IMAGE_MODEL = "gemini-3.1-flash-image"
MAX_SNIPPETS = 150
MAX_CHARS = 6000

# 審査・業務・財務・会社に触れた発話は題材に使わない
_BUSINESS_RE = re.compile(
    r"審査|案件|スコア|与信|財務|決算|売上|利益|稟議|融資|リース料|金利|格付|承認|否決|物件|借手|取引先|"
    r"株式会社|有限会社|合同会社|\(株\)|（株）|㈱|顧客|社長|代表取締役|口座|住所|電話|メール|"
    # 仕事・システムの話題も絵の題材にしない（日常の出来事を優先）
    r"リース|業界|人手不足|省力化|システム|稼働|改善|パイプライン|REV|エラー|バグ|実装|デプロイ|コード|"
    r"判断資産|調査|計算|データ|テンプレート|依頼|有料先|優良先|スコアリング|スクアリング"
)
_DIGITS_RE = re.compile(r"[0-9０-９][0-9０-９,，.．]*")


def collect_snippets(db_path: Path, *, days: int = 7) -> list[str]:
    """直近 days 日のユーザー発話から、題材に使ってよい短い文だけを返す。"""
    from api.chat_judgment_asset_capture import mask_for_jev

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT content FROM chat_messages WHERE role = 'user' AND user_id IN ({','.join('?' * len(USER_IDS))}) "
            "AND created_at >= datetime('now', ?) ORDER BY created_at DESC",
            (*USER_IDS, f"-{int(days)} days"),
        ).fetchall()
    finally:
        conn.close()
    # 新しい順に集め、同じ発話は1回だけ
    snippets: list[str] = []
    seen: set[str] = set()
    total = 0
    for (content,) in rows:
        text = " ".join(str(content or "").split())
        if len(text) < 4 or _BUSINESS_RE.search(text):
            continue
        masked = mask_for_jev(_DIGITS_RE.sub("〈数〉", text)[:200])
        if not masked or masked in seen:
            continue
        seen.add(masked)
        snippets.append(masked)
        total += len(masked)
        if len(snippets) >= MAX_SNIPPETS or total >= MAX_CHARS:
            break
    return snippets


def load_history(path: Path = HISTORY_PATH, limit: int = 12) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items[-limit:]


def topic_prompt(snippets: list[str], history: list[dict[str, Any]]) -> str:
    past = "\n".join(f"- {h.get('topic', '')}（構図: {h.get('composition', '')}）" for h in history) or "- なし"
    talk = "\n".join(f"- {s}" for s in snippets)
    return f"""あなたはイラストの題材を選ぶ編集者です。以下は、この1週間にユーザーが AI キャラクター「紫苑」に話した内容の抜粋です（数字や固有名は伏せてあります）。
この中から、絵にすると楽しい「印象的な出来事・話題」を1つ選び、紫苑がその場面にいる1枚絵の設計を JSON で返してください。

条件:
- 日常の出来事・食べ物・散歩・季節・趣味・歌など、仕事以外の明るい話題を優先する。審査・会社・お金・数字・実在の人物や場所の特定につながる要素は入れない
- 机の前で唸る・書類に埋もれる構図は禁止。場所・カメラアングル・時間帯・表情に変化をつける
- 過去の題材と同じ話題・似た構図は避ける
- 登場する人物は紫苑1人だけ。動物・食べ物・小物は出してよい（ユーザー本人は描かない）
- caption は日本語20字以内の一言（画像と一緒に小さく表示する）

過去の題材:
{past}

今週の会話の抜粋:
{talk}

JSON（キーはこの5つだけ）: {{"topic": "選んだ話題（日本語・短く）", "caption": "一言", "scene": "場面の説明（英語）", "composition": "構図・カメラアングル（英語）", "expression": "表情・しぐさ（英語）"}}"""


def validate_topic(data: dict[str, Any]) -> dict[str, str] | None:
    """題材に仕事・数字・固有名が混じっていれば使わない。"""
    from api.chat_judgment_asset_capture import mask_for_jev

    keys = ("topic", "caption", "scene", "composition", "expression")
    topic = {key: " ".join(str(data.get(key) or "").split())[:400] for key in keys}
    if not all(topic.values()):
        return None
    for key in ("topic", "caption"):
        value = topic[key]
        if _BUSINESS_RE.search(value) or _DIGITS_RE.search(value) or mask_for_jev(value) != value:
            return None
    topic["caption"] = topic["caption"][:20]
    return topic


def image_prompt(topic: dict[str, str]) -> str:
    return f"""Edit the supplied reference character into a 16:9 illustration of a scene from her week.
Preserve her identity exactly: long silver-white hair, looped ahoge, large purple eyes,
flower hair ornament, rear bow, red-white-pink floral kimono, dark patterned obi,
and cute chibi proportions.

Scene: {topic['scene']}
Composition / camera: {topic['composition']}
Expression and pose: {topic['expression']}

Rules:
- Exactly ONE instance of the girl. No other human. Animals, food and props are fine.
- She is NOT at an office desk, NOT surrounded by documents, NOT grumbling. Make the setting lively and specific.
- Polished kawaii chibi anime linework, soft cel shading, natural lighting that suits the scene.
- No logos. Do not render any readable letters, words, numbers, captions, signs or watermark."""


def _call_topic_model(prompt: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    from ai_runtime_client import google_genai_client
    from config import get_gemini_model
    from novelist_agent import _get_daily_gemini_api_key

    client = google_genai_client(feature=TOPIC_FEATURE, client_factory=genai.Client, api_key=_get_daily_gemini_api_key())
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.9, max_output_tokens=800, response_mime_type="application/json"),
    )
    return json.loads(response.text or "{}")


def _call_image_model(prompt: str, target: Path) -> bool:
    from google import genai
    from google.genai import types
    from PIL import Image

    from ai_runtime_client import google_genai_client
    from novelist_agent import _get_daily_gemini_api_key, _save_gemini_image

    client = google_genai_client(feature=IMAGE_FEATURE, client_factory=genai.Client, api_key=_get_daily_gemini_api_key())
    with Image.open(CHARACTER_PATH) as reference:
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=[prompt, reference.convert("RGB")],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
            ),
        )
    for part in response.parts or ():
        if getattr(part, "inline_data", None) is not None:
            _save_gemini_image(part.as_image(), target)
            return True
    return False


def _archive_copy(image: Path, day: dt.date) -> None:
    """Obsidian の Lease Grumble アーカイブにも写す（他の過去イラストと同じ場所で保管）。"""
    try:
        from api.shion_illustration_gallery import _archive_dir

        archive = _archive_dir()
        if archive is None:
            return
        target_dir = archive / day.strftime("%Y") / day.strftime("%m")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_dir / image.name)
    except Exception as exc:  # noqa: BLE001 - アーカイブ失敗で生成結果を捨てない
        print(f"[weekly_illustration] archive copy skipped: {type(exc).__name__}")


def run(
    *,
    db_path: Path,
    day: dt.date,
    gallery_dir: Path | None = None,
    history_path: Path = HISTORY_PATH,
    topic_fn: Callable[[str], dict[str, Any]] = _call_topic_model,
    image_fn: Callable[[str, Path], bool] = _call_image_model,
    archive: bool = True,
) -> dict[str, Any]:
    from api import shion_illustration_gallery as gallery

    gallery_dir = gallery_dir or gallery.GALLERY_DIR
    target = gallery_dir / f"{day.isoformat()}.webp"
    if target.exists():
        return {"status": "skipped", "reason": "already_exists", "path": str(target)}
    snippets = collect_snippets(db_path)
    if not snippets:
        return {"status": "skipped", "reason": "no_usable_chat"}
    history = load_history(history_path)
    topic = validate_topic(topic_fn(topic_prompt(snippets, history)))
    if topic is None:
        return {"status": "skipped", "reason": "topic_rejected"}
    gallery_dir.mkdir(parents=True, exist_ok=True)
    if not image_fn(image_prompt(topic), target):
        return {"status": "error", "reason": "no_image"}
    gallery.save_caption(target.name, topic["caption"], gallery_dir)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        record = {"date": day.isoformat(), "file": target.name, "snippets": len(snippets), **topic}
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if archive:
        _archive_copy(target, day)
    return {"status": "ok", "path": str(target), "topic": topic["topic"], "caption": topic["caption"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="週1回、その週のチャットから題材を選んで紫苑のイラストを作る")
    parser.add_argument("--db", type=Path, default=_MAIN_ROOT / "data" / "lease_data.db")
    parser.add_argument("--date", default="", help="保存名の日付（既定は今日）")
    args = parser.parse_args(argv)
    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    try:
        result = run(db_path=args.db, day=day)
    except Exception as exc:  # noqa: BLE001 - 予算ガード停止・API失敗は記録して終了
        result = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
