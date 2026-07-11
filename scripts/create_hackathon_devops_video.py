#!/usr/bin/env python3
"""Create a short silent hackathon explainer video for the DevOps loop."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports" / "hackathon_video"
VIDEO_PATH = OUT_DIR / "devops_ai_agent_loop.mp4"
FALLBACK_VIDEO_PATH = OUT_DIR / "devops_ai_agent_loop.avi"
GIF_PATH = OUT_DIR / "devops_ai_agent_loop.gif"
HTML_PATH = OUT_DIR / "devops_ai_agent_loop.html"
THUMB_PATH = OUT_DIR / "devops_ai_agent_loop_thumbnail.png"
SCRIPT_PATH = OUT_DIR / "devops_ai_agent_loop_script.md"

W, H = 1280, 720
FPS = 24
FONT_JP = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_EN = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # Hiragino W3 is readable enough for slides; keep one font to avoid missing glyphs.
    return ImageFont.truetype(FONT_JP, size=size)


def wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for part in text.split("\n"):
        if not part:
            if current:
                lines.append(current)
                current = ""
            lines.append("")
            continue
        for ch in part:
            current += ch
            if len(current) >= max_chars:
                lines.append(current)
                current = ""
        if current:
            lines.append(current)
            current = ""
    return lines


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: tuple[int, int, int] = (245, 248, 252),
    max_chars: int = 28,
    line_gap: int = 10,
) -> int:
    x, y = xy
    f = font(size)
    for line in wrap_text(text, max_chars):
        draw.text((x, y), line, font=f, fill=fill)
        y += size + line_gap
    return y


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 2,
    radius: int = 8,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: tuple[int, int, int]) -> None:
    draw.line([start, end], fill=fill, width=4)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = 0.55
    p1 = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    p2 = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    draw.polygon([end, p1, p2], fill=fill)


def base_slide(title: str, subtitle: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), (16, 22, 31))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 82), fill=(24, 34, 48))
    draw_text(draw, (48, 22), title, 34, max_chars=32)
    if subtitle:
        draw_text(draw, (48, 100), subtitle, 24, fill=(174, 194, 214), max_chars=44)
    draw.text((1060, 32), "AI DevOps", font=font(24), fill=(121, 214, 199))
    return img


def slide_intro() -> Image.Image:
    img = base_slide("紫苑システム", "業務AIエージェントDevOpsの実証プロトタイプ")
    draw = ImageDraw.Draw(img)
    draw_text(
        draw,
        (76, 172),
        "私はプロのエンジニアではありません。",
        34,
        fill=(255, 255, 255),
        max_chars=24,
    )
    draw_text(
        draw,
        (76, 260),
        "実務で使うAIを、AIと一緒に作り、毎日直しながら運用しています。",
        34,
        fill=(255, 255, 255),
        max_chars=28,
    )
    rounded_rect(draw, (76, 430, 1204, 560), (30, 47, 61), (121, 214, 199), radius=8)
    draw_text(
        draw,
        (110, 456),
        "作って終わりではなく、使いながら育てる業務AI。",
        38,
        fill=(121, 214, 199),
        max_chars=28,
    )
    return img


def slide_loop() -> Image.Image:
    img = base_slide("AIエージェントDevOpsループ", "使われた結果を、次の改善へ戻す")
    draw = ImageDraw.Draw(img)
    nodes = [
        ("Use", "業務で使う", 78, 170),
        ("Observe", "ログを見る", 310, 170),
        ("Detect", "劣化を見つける", 542, 170),
        ("Reflect", "振り返る", 774, 170),
        ("Improve", "直す", 1006, 170),
        ("Verify", "検証する", 650, 455),
        ("Deploy", "本番へ戻す", 382, 455),
    ]
    for label, sub, x, y in nodes:
        rounded_rect(draw, (x, y, x + 180, y + 94), (33, 49, 63), (121, 214, 199), radius=8)
        draw.text((x + 18, y + 14), label, font=font(28), fill=(255, 255, 255))
        draw.text((x + 18, y + 54), sub, font=font(20), fill=(190, 205, 220))
    for i in range(4):
        arrow(draw, (78 + i * 232 + 180, 217), (310 + i * 232 - 14, 217), (121, 214, 199))
    arrow(draw, (1096, 264), (760, 455), (121, 214, 199))
    arrow(draw, (650, 502), (562, 502), (121, 214, 199))
    arrow(draw, (382, 455), (160, 264), (121, 214, 199))
    rounded_rect(draw, (78, 602, 1204, 666), (54, 39, 55), (221, 171, 93), radius=8)
    draw_text(draw, (106, 618), "Evidence: refs / memory recall / response quality / improvement logs", 24, fill=(255, 226, 170), max_chars=68)
    return img


def slide_observe() -> Image.Image:
    img = base_slide("実際に運用中", "デモではなく、日々の違和感が改善材料になる")
    draw = ImageDraw.Draw(img)
    items = [
        ("会話ログ", "何を聞かれ、どう答えたかを残す"),
        ("RAG参照", "どの知識を使ったかを確認する"),
        ("改善ログ", "違和感を次の修正候補に変える"),
        ("環境差分", "Cloud Run / Cloudflare / ローカルを比べる"),
    ]
    y = 190
    for title, body in items:
        rounded_rect(draw, (90, y, 1190, y + 78), (31, 45, 58), (65, 88, 108), radius=8)
        draw.text((124, y + 18), title, font=font(28), fill=(121, 214, 199))
        draw.text((342, y + 21), body, font=font(26), fill=(242, 247, 250))
        y += 92
    return img


def slide_reflect() -> Image.Image:
    img = base_slide("Reflect / Improve", "AIの見落としを、次の変更へ変える")
    draw = ImageDraw.Draw(img)
    left = "Private Reflection\n- 今日の観察\n- 私の見落とし\n- 仮説の更新\n- 次回の小さな実験"
    right = "改善対象\n- プロンプト\n- RAG / 記憶\n- UI\n- API"
    rounded_rect(draw, (90, 170, 590, 555), (31, 45, 58), (121, 214, 199), radius=8)
    rounded_rect(draw, (690, 170, 1190, 555), (31, 45, 58), (221, 171, 93), radius=8)
    draw_text(draw, (128, 205), left, 34, fill=(245, 248, 252), max_chars=16)
    draw_text(draw, (728, 205), right, 34, fill=(245, 248, 252), max_chars=16)
    arrow(draw, (600, 362), (680, 362), (221, 171, 93))
    return img


def slide_verify() -> Image.Image:
    img = base_slide("Verify / Deploy", "実務AIだから、直したら必ず確かめる")
    draw = ImageDraw.Draw(img)
    checks = [
        "pytest",
        "typecheck",
        "memory_debug",
        "Cloud Run / Cloudflare比較",
        "回答品質評価",
    ]
    x = 118
    for i, check in enumerate(checks):
        y = 210 + (i % 3) * 110
        xx = x + (i // 3) * 520
        rounded_rect(draw, (xx, y, xx + 440, y + 82), (31, 45, 58), (121, 214, 199), radius=8)
        draw.text((xx + 28, y + 24), check, font=font(30), fill=(255, 255, 255))
    draw_text(
        draw,
        (118, 590),
        "検証してから Cloud Run / Cloudflare / ローカル運用へ戻す。",
        32,
        fill=(221, 171, 93),
        max_chars=32,
    )
    return img


def slide_close() -> Image.Image:
    img = base_slide("結論", "リース審査から始めた、業務AI DevOpsの実証")
    draw = ImageDraw.Draw(img)
    draw_text(
        draw,
        (92, 190),
        "今回はリース審査AIで実装しています。",
        34,
        fill=(255, 255, 255),
        max_chars=28,
    )
    draw_text(
        draw,
        (92, 310),
        "でも中核は、業務AIを運用しながら改善し続ける仕組みです。",
        38,
        fill=(121, 214, 199),
        max_chars=26,
    )
    rounded_rect(draw, (92, 500, 1188, 608), (54, 39, 55), (221, 171, 93), radius=8)
    draw_text(draw, (126, 528), "Shion System: AI Agent DevOps Prototype", 36, fill=(255, 226, 170), max_chars=48)
    return img


SLIDES = [
    (slide_intro, 10),
    (slide_loop, 14),
    (slide_observe, 12),
    (slide_reflect, 14),
    (slide_verify, 12),
    (slide_close, 12),
]


def fade_frame(img: Image.Image, frame_idx: int, total: int) -> Image.Image:
    fade = FPS // 2
    alpha = 1.0
    if frame_idx < fade:
        alpha = frame_idx / max(1, fade)
    elif total - frame_idx < fade:
        alpha = (total - frame_idx) / max(1, fade)
    if alpha >= 0.995:
        return img
    black = Image.new("RGB", img.size, (16, 22, 31))
    return Image.blend(black, img, alpha)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slide_images: list[Image.Image] = []
    durations_ms: list[int] = []
    for index, (slide_fn, seconds) in enumerate(SLIDES, start=1):
        img = slide_fn()
        slide_images.append(img)
        durations_ms.append(int(seconds * 1000))
        img.save(OUT_DIR / f"slide_{index:02d}.png")
    if slide_images:
        slide_images[0].save(
            GIF_PATH,
            save_all=True,
            append_images=slide_images[1:],
            duration=durations_ms,
            loop=0,
            optimize=True,
        )
        slide_images[0].save(THUMB_PATH)
    HTML_PATH.write_text(
        """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI DevOps Hackathon Video</title>
<style>
  html, body { margin: 0; width: 100%; height: 100%; background: #10161f; overflow: hidden; }
  .stage { width: 100vw; height: 100vh; display: grid; place-items: center; background: #10161f; }
  img { width: 100vw; height: 100vh; object-fit: contain; background: #10161f; }
  .timer { position: fixed; right: 18px; bottom: 14px; color: rgba(255,255,255,.65); font: 16px system-ui, sans-serif; }
</style>
</head>
<body>
<div class="stage"><img id="slide" src="slide_01.png" alt="slide"></div>
<div class="timer" id="timer"></div>
<script>
const slides = [
  ["slide_01.png", 10000],
  ["slide_02.png", 14000],
  ["slide_03.png", 12000],
  ["slide_04.png", 14000],
  ["slide_05.png", 12000],
  ["slide_06.png", 12000],
];
let i = 0;
let elapsed = 0;
const img = document.getElementById("slide");
const timer = document.getElementById("timer");
function show() {
  img.src = slides[i][0];
  timer.textContent = `${i + 1}/${slides.length}`;
  setTimeout(() => {
    elapsed += slides[i][1];
    i = (i + 1) % slides.length;
    show();
  }, slides[i][1]);
}
show();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )

    selected_path = None
    if os.environ.get("CREATE_MP4") != "1":
        writer = None
    else:
        writer = None
        selected_path = VIDEO_PATH
        for path, codec in (
            (VIDEO_PATH, "mp4v"),
            (VIDEO_PATH, "avc1"),
            (FALLBACK_VIDEO_PATH, "MJPG"),
            (FALLBACK_VIDEO_PATH, "XVID"),
        ):
            candidate = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), FPS, (W, H))
            if candidate.isOpened():
                writer = candidate
                selected_path = path
                break
            candidate.release()
        if writer is None:
            raise RuntimeError("Failed to open video writer with mp4v/avc1/MJPG/XVID")

    if writer is not None:
        for img, (_, seconds) in zip(slide_images, SLIDES):
            frames = int(seconds * FPS)
            for i in range(frames):
                frame = fade_frame(img, i, frames)
                arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                writer.write(arr)
        writer.release()

    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size == 0:
        VIDEO_PATH.unlink()

    SCRIPT_PATH.write_text(
        """# DevOps Hackathon Video Script

私はプロのエンジニアではありません。
これは、実務者が自分の業務AIをAIと一緒に作り、毎日使いながら改善しているプロジェクトです。

その中で生まれたのが、業務AIエージェントDevOpsの実証プロトタイプ、紫苑システムです。

DevOpsとは、作って終わりではなく、使われた結果を観測し、問題を検知し、改善し、検証して、また本番へ戻す継続的な運用ループです。

紫苑システムでは、そのループをAIエージェント自身に適用しています。

会話ログ、RAG参照、改善ログ、Cloud RunとCloudflareの環境差分を観測します。
記憶抜け、浅い内省、回答品質の劣化を検知します。

そしてPrivate Reflectionや改善レポートで振り返り、プロンプト、RAG、記憶、UI、APIを直します。
直したあとはpytest、typecheck、memory_debug、環境比較で検証し、また運用へ戻します。

今回はリース審査AIで実装しています。
でも中核は、業務AIを実際に運用しながら改善し続けるための仕組みです。
""",
        encoding="utf-8",
    )
    if selected_path:
        print(selected_path)
    print(GIF_PATH)
    print(HTML_PATH)
    print(THUMB_PATH)
    print(SCRIPT_PATH)


if __name__ == "__main__":
    main()
