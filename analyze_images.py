#!/usr/bin/env python3
"""
画像フォルダ内の全画像を分析し、特徴をファイルごとにJSONで保存する。
"""
import json
import os
from pathlib import Path

ASSETS_DIR = Path("/Users/kobayashiisaoryou/.cursor/projects/Users-kobayashiisaoryou-clawd/assets")
OUTPUT_DIR = Path(__file__).resolve().parent / "image_analysis_results"

# 会話で得られた画像の特徴説明（ファイル名ベースのキー）
FEATURE_DESCRIPTIONS = {
    "______-fe3eb438-36a6-4842-9359-254247925b3b.png": "白背景・着物のちびキャラ。困惑・混乱の表情、頭上に黄色い疑問符。銀白長髪、赤基調の着物。両手を腰に当て前かがみ。",
    "______-ce4d90f7-0277-4df5-ac9a-025441cabbc9.png": "ちびキャラ。黄色ヘルメット・デニムオーバーオール、右手に金槌。建設作業員風、元気な表情。",
    "______-656792f0-615d-4c28-811d-916519005827.png": "ちびキャラ・ナース帽と白い制服、注射器を持った看護師。白髪、紫の瞳、笑顔。",
    "______-3b519bb4-385d-4687-b2e9-969594fbfe3a.png": "着物のちびキャラ。怒り表情、腕組み、緑色の怒り絵文字。白地に桜模様の着物。",
    "______-69d90af5-fd7b-4cf1-b50e-ce702ce37618.png": "ちびキャラ。正座して赤い椀のご飯を箸で食べている。満面の笑み、和装。",
    "______-c71696ae-1339-4a33-add9-46ee31c1f52c.png": "ちびキャラ。布団で眠っている。白枕、桜模様の布団、穏やかな表情。",
    "IMG_1788-70983e57-d965-4752-bebb-3e0c4e16798c.png": "着物ちびキャラ。頬に手を当て困惑・照れ。疑問符とハート、横にカラーボール。",
    "IMG_1793-152eae6e-9149-4c8e-91b6-c570711199bf.png": "ちび・看護師。白いナース帽に赤十字、注射器。白い制服、紫の瞳で笑顔。",
    "IMG_1754-cc58ef0c-3f27-4ebd-b33b-81b57f1fb833.png": "ちび・建設風。黄色ヘルメット、金槌、オーバーオール。両腕を広げたポーズ。",
    "1849E856-971D-4B79-AD5E-E1074D93B043-55ad16b8-11ff-4717-8e5d-5a920fecae0d.png": "桜満開の春の屋外。白髪の少女が子犬を抱き、ゴールデンレトリバーがボールを咥えている。",
    "72603010-1AA5-4BEA-824C-DC847E2CF765-7e30894e-bac6-4875-b652-b23064d771b4.png": "白いレクサスLC500オープンカー。白髪アニメ少女が運転、山道と青空。",
    "IMG_1792-ada87e96-4147-4768-9ad5-6c0c24c8b37c.png": "ちび・メイド。黒カチューシャ、白いエプロン、ショートケーキの皿。笑顔で片手を広げ。",
    "IMG_1790-fe822488-7416-4aaa-8f52-6704440f2c17.png": "緑のゼリー状キャラ4種「めう!」・喜び/怒り/悲しみ/疲労の表情。",
    "IMG_1797-e8623f08-8e0f-4467-b718-82b43a6a9c69.png": "着物ちび。悲しそうに膝を抱えて座る。緑の球に「めっ!」の文字。",
}

def get_image_metadata(img_path: Path) -> dict:
    """PILで画像の寸法・形式を取得。失敗時はNoneで返す。"""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return {
                "width": im.width,
                "height": im.height,
                "format": im.format,
                "mode": im.mode,
            }
    except Exception as e:
        return {"error": str(e)}


def analyze_and_save():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ASSETS_DIR.exists():
        print(f"Assets dir not found: {ASSETS_DIR}")
        return

    extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    files = sorted([f for f in ASSETS_DIR.iterdir() if f.suffix.lower() in extensions])

    for path in files:
        name = path.name
        size_bytes = path.stat().st_size
        meta = get_image_metadata(path)
        features = FEATURE_DESCRIPTIONS.get(name, "（メタデータのみ取得）")

        analysis = {
            "filename": name,
            "path": str(path),
            "file_size_bytes": size_bytes,
            "file_size_kb": round(size_bytes / 1024, 2),
            "metadata": meta,
            "feature_summary": features,
        }

        out_name = path.stem + "_analysis.json"
        out_path = OUTPUT_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_name}")


if __name__ == "__main__":
    analyze_and_save()
