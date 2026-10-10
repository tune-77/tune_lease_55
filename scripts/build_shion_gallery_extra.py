#!/usr/bin/env python3
"""既存の紫苑イラストを「今日の紫苑」の候補（紫苑ギャラリー）へ写す（REV-600, Gemini 呼び出しなし）。

- 出どころ: mebuki/ の画像と、frontend/public/lease-grumble/characters/ の紫苑の動画から1コマ
- 元ファイルは動かさない。16:9（1280x720 まで）に収めて webp で
  data/shion_gallery_extra/gallery-<10桁>.webp に書く（候補の本体は git に入れない）
- 選んだ絵は下の SOURCES に固定してある。除いた絵（軍師の衣装・素材シート・紫苑が写っていない絵・
  ほぼ同じ絵の重複・moods/ にある mebuki と同じ絵）は入れていない
- 動画のコマは macOS の AVFoundation（swift）で切り出す（ffmpeg は使わない）
- 週1回の生成（REV-490）の題材履歴・captions.json とは別の置き場で、混ざらない

使い方: python scripts/build_shion_gallery_extra.py [--out DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageFilter, ImageOps  # noqa: E402

from api.shion_illustration_gallery import EXTRA_DIR  # noqa: E402

CANVAS = (1280, 720)
QUALITY = 85
VIDEO_DIR = "frontend/public/lease-grumble/characters"

# (リポジトリ内の相対パス, 動画なら切り出す秒数)。ほぼ同じ絵は画素数・容量の大きい方を残した
SOURCES: list[tuple[str, float | None]] = [
    ("mebuki/72603010-1AA5-4BEA-824C-DC847E2CF765.png", None),  # 両手を広げる（0・6・7・guide 等と同じ絵）
    *((f"mebuki/{name}", None) for name in (
        "IMG_1762.PNG", "IMG_1763.PNG", "IMG_1764.PNG", "IMG_1765.PNG", "IMG_1766.jpeg", "IMG_1767.jpeg",
        "IMG_1777.PNG", "IMG_1778.PNG", "IMG_1779.PNG", "IMG_1781.PNG",
        "IMG_1788.jpeg", "IMG_1789.jpeg", "IMG_1790.PNG", "IMG_1791.jpeg", "IMG_1792.PNG",
        "IMG_1793.jpeg", "IMG_1794.jpeg", "IMG_1796.jpeg",
        "IMG_1799.PNG", "IMG_1800.PNG", "IMG_1802.PNG",
        "IMG_1910.PNG", "IMG_1911.PNG", "IMG_1912.PNG",
        "IMG_1919.jpeg", "IMG_1920.PNG", "IMG_1945.PNG", "IMG_1946.PNG", "IMG_2181.jpeg",
        "キャラクター (1).jpg",  # challenge.jpg と同じ絵
        "キャラクター (2).jpg",  # reject.jpg と同じ絵
        "キャラクター (3).jpg",  # approve.jpg と同じ絵
        "キャラクター (5).jpg",  # キャラクター.jpg とほぼ同じ絵
    )),
    # 動画は表情が分かる1コマ（shion-loop-01 は intro と同じ中身、04 は 03 と同じ動き）
    (f"{VIDEO_DIR}/shion-intro-loop.mp4", 0.6),
    (f"{VIDEO_DIR}/shion-loop-02.mp4", 1.8),
    (f"{VIDEO_DIR}/shion-loop-03.mp4", 5.4),
    (f"{VIDEO_DIR}/shion-loop-05.mp4", 1.8),
    (f"{VIDEO_DIR}/shion-loop-06.mp4", 3.0),
    (f"{VIDEO_DIR}/shion-loop-07.mp4", 1.8),
    (f"{VIDEO_DIR}/shion-loop-08.mp4", 3.0),
]

_SWIFT_FRAME = r"""
import AVFoundation
import AppKit
let a = CommandLine.arguments
let gen = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: a[1])))
gen.appliesPreferredTrackTransform = true
gen.requestedTimeToleranceBefore = .zero
gen.requestedTimeToleranceAfter = .zero
let cg = try gen.copyCGImage(at: CMTime(seconds: Double(a[2])!, preferredTimescale: 600), actualTime: nil)
try NSBitmapImageRep(cgImage: cg).representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[3]))
"""


def output_name(source: str, at: float | None) -> str:
    key = source if at is None else f"{source}@{at}"
    return f"gallery-{hashlib.sha1(key.encode()).hexdigest()[:10]}.webp"


def video_frame(path: Path, at: float, workdir: Path) -> Image.Image:
    script = workdir / "frame.swift"
    script.write_text(_SWIFT_FRAME, encoding="utf-8")
    png = workdir / "frame.png"
    subprocess.run(["swift", str(script), str(path), str(at), str(png)], check=True, capture_output=True, timeout=180)
    with Image.open(png) as im:
        return im.convert("RGB")


def fit_16x9(im: Image.Image) -> Image.Image:
    """切り抜かずに 16:9 へ収める（余白は同じ絵をぼかして敷く。白地の絵は白のまま）。拡大はしない。"""
    im = ImageOps.exif_transpose(im).convert("RGB")
    scale = min(CANVAS[0] / im.width, CANVAS[1] / im.height, 1.0)
    fg = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    width, height = (max(fg.width, round(fg.height * 16 / 9)), fg.height)
    bg = ImageOps.fit(im, (width, height), Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
    bg = Image.blend(bg, Image.new("RGB", bg.size, "white"), 0.35)
    bg.paste(fg, ((width - fg.width) // 2, 0))
    return bg


def build(out_dir: Path, *, root: Path = REPO_ROOT, dry_run: bool = False) -> list[Path]:
    written = []
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for source, at in SOURCES:
            path = root / source
            target = out_dir / output_name(source, at)
            if dry_run:
                print(f"{source}{'' if at is None else f' @{at}s'} -> {target.name}")
                continue
            if at is None:
                with Image.open(path) as im:
                    image = fit_16x9(im)
            else:
                image = fit_16x9(video_frame(path, at, Path(tmp)))
            image.save(target, "WEBP", quality=QUALITY, method=6)
            written.append(target)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXTRA_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    written = build(args.out, dry_run=args.dry_run)
    if not args.dry_run:
        size = sum(p.stat().st_size for p in written)
        print(f"{len(written)} files, {size / 1024 / 1024:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
