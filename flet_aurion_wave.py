"""
AURION CORE — Flet スタンドアロン波形ビジュアライザー (AV.4 / AV.5 / AV.6)

起動:
    python3 flet_aurion_wave.py

機能:
- 2変数のサイン波 + 合成波をリアルタイムアニメーション描画（60fps 目標）
- v1 / v2 スライダーで振幅変化を即時反映
- 位相差 > π*0.7 でデコヒーレンス警告表示（「⚠ データの歪みを検出」）
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import flet as ft
import flet.canvas as cv
import numpy as np

# プロジェクトルートを sys.path に追加（スタンドアロン実行用）
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aurion_wave_engine import compute_wave

# ── 定数 ─────────────────────────────────────────────────────────────────────
CANVAS_W = 640
CANVAS_H = 300
N_POINTS  = 200
DECOHERENCE_THRESHOLD = math.pi * 0.7   # AV.6 警告閾値

COLOR_A       = ft.Colors.CYAN_400
COLOR_B       = ft.Colors.AMBER_400
COLOR_COMP    = ft.Colors.GREEN_400
COLOR_WARNING = ft.Colors.RED_400
COLOR_BG      = ft.Colors.with_opacity(0.08, ft.Colors.WHITE)

ANIMATION_INTERVAL_MS = 16   # ~60 fps


def _wave_to_path(
    ys: np.ndarray,
    offset_x: float,
    mid_y: float,
    amp_px: float,
    width: float,
) -> list[cv.Path.PathElement]:
    """正規化 y値配列 → Canvas Path 要素リスト。x 方向に波を流す。"""
    n = len(ys)
    elems: list[cv.Path.PathElement] = []
    for i, y in enumerate(ys):
        x = (i / (n - 1)) * width
        # offset_x でスクロール（x 軸方向に流れる）
        px = (x + offset_x) % width
        py = mid_y - float(y) * amp_px
        if i == 0:
            elems.append(cv.Path.MoveTo(px, py))
        else:
            elems.append(cv.Path.LineTo(px, py))
    return elems


def main(page: ft.Page) -> None:
    page.title = "AURION CORE — 量子波形ビジュアライザー"
    page.bgcolor = ft.Colors.GREY_900
    page.padding = 20
    page.theme_mode = ft.ThemeMode.DARK

    # ── 状態 ─────────────────────────────────────────────────────────────────
    state = {
        "v1": 0.5,
        "v2": 0.5,
        "t_offset": 0.0,
        "running": True,
    }

    # ── Canvas ───────────────────────────────────────────────────────────────
    canvas = cv.Canvas(
        width=CANVAS_W,
        height=CANVAS_H,
        shapes=[],
        content=ft.GestureDetector(mouse_cursor=ft.MouseCursor.BASIC),
    )

    # ── スコア・警告テキスト ──────────────────────────────────────────────────
    score_text = ft.Text(
        "振幅スコア: —",
        color=ft.Colors.WHITE,
        size=16,
        weight=ft.FontWeight.BOLD,
    )
    phase_text = ft.Text(
        "位相差: —",
        color=ft.Colors.WHITE70,
        size=13,
    )
    warning_text = ft.Text(
        "",
        color=COLOR_WARNING,
        size=20,
        weight=ft.FontWeight.BOLD,
        visible=False,
    )

    # ── スライダー ────────────────────────────────────────────────────────────
    def _make_slider(label: str, key: str, color: str) -> tuple[ft.Text, ft.Slider]:
        lbl = ft.Text(f"{label}: {state[key]:.2f}", color=color, size=13, width=160)

        def on_change(e: ft.ControlEvent) -> None:
            state[key] = float(e.control.value)
            lbl.value = f"{label}: {state[key]:.2f}"
            _redraw()
            page.update()

        slider = ft.Slider(
            value=state[key],
            min=0.0,
            max=1.0,
            divisions=100,
            active_color=color,
            on_change=on_change,
            expand=True,
        )
        return lbl, slider

    lbl_v1, slider_v1 = _make_slider("変数A (v1)", "v1", str(COLOR_A))
    lbl_v2, slider_v2 = _make_slider("変数B (v2)", "v2", str(COLOR_B))

    # ── 描画ロジック ──────────────────────────────────────────────────────────
    def _redraw() -> None:
        v1 = state["v1"]
        v2 = state["v2"]
        t_off = state["t_offset"]

        result = compute_wave(v1, v2, n_points=N_POINTS)
        decoherent = result.phase_diff > DECOHERENCE_THRESHOLD

        score_text.value = f"振幅スコア: {result.amplitude_score:.3f}"
        phase_text.value = f"位相差: {result.phase_diff / math.pi:.2f}π rad"

        warning_text.visible = decoherent
        warning_text.value = "⚠ データの歪みを検出" if decoherent else ""

        canvas.shapes.clear()

        # 背景
        canvas.shapes.append(cv.Rect(
            x=0, y=0, width=CANVAS_W, height=CANVAS_H,
            paint=ft.Paint(color=ft.Colors.with_opacity(0.15, ft.Colors.BLUE_GREY_900)),
        ))

        # 中心線
        mid = CANVAS_H / 2.0
        canvas.shapes.append(cv.Line(
            x1=0, y1=mid, x2=CANVAS_W, y2=mid,
            paint=ft.Paint(color=ft.Colors.with_opacity(0.2, ft.Colors.WHITE), stroke_width=1),
        ))

        amp_px = CANVAS_H * 0.38
        x_shift = (t_off % 1.0) * CANVAS_W

        if decoherent:
            # AV.6: 位相差 > 閾値 → 波形を消す
            canvas.shapes.append(cv.Text(
                x=CANVAS_W / 2 - 120,
                y=mid - 14,
                text="⚠ データの歪みを検出",
                style=ft.TextStyle(
                    color=ft.Colors.RED_400,
                    size=22,
                    weight=ft.FontWeight.BOLD,
                ),
            ))
        else:
            # 波 A
            path_a = _wave_to_path(result.wave_a, x_shift, mid, amp_px, CANVAS_W)
            if len(path_a) > 1:
                canvas.shapes.append(cv.Path(
                    elements=path_a,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.STROKE,
                        color=COLOR_A,
                        stroke_width=2,
                    ),
                ))

            # 波 B
            path_b = _wave_to_path(result.wave_b, x_shift, mid, amp_px, CANVAS_W)
            if len(path_b) > 1:
                canvas.shapes.append(cv.Path(
                    elements=path_b,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.STROKE,
                        color=COLOR_B,
                        stroke_width=2,
                    ),
                ))

            # 合成波
            path_c = _wave_to_path(result.composite, x_shift, mid, amp_px, CANVAS_W)
            if len(path_c) > 1:
                canvas.shapes.append(cv.Path(
                    elements=path_c,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.STROKE,
                        color=COLOR_COMP,
                        stroke_width=3,
                    ),
                ))

    # ── アニメーションループ ──────────────────────────────────────────────────
    def on_tick(e: ft.ControlEvent) -> None:
        if not state["running"]:
            return
        state["t_offset"] += 0.006   # スクロール速度
        _redraw()
        page.update()

    page.on_resized = None

    # ── レイアウト ────────────────────────────────────────────────────────────
    legend = ft.Row([
        ft.Container(width=20, height=4, bgcolor=str(COLOR_A)),
        ft.Text("波A (v1)", color=str(COLOR_A), size=12),
        ft.Container(width=12),
        ft.Container(width=20, height=4, bgcolor=str(COLOR_B)),
        ft.Text("波B (v2)", color=str(COLOR_B), size=12),
        ft.Container(width=12),
        ft.Container(width=20, height=4, bgcolor=str(COLOR_COMP)),
        ft.Text("合成波", color=str(COLOR_COMP), size=12),
    ], spacing=6)

    controls_panel = ft.Container(
        content=ft.Column([
            ft.Text("🎛 パラメータ調整", color=ft.Colors.WHITE, size=14, weight=ft.FontWeight.BOLD),
            ft.Row([lbl_v1, slider_v1]),
            ft.Row([lbl_v2, slider_v2]),
        ]),
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.BLUE_GREY),
        border_radius=8,
        padding=12,
    )

    score_panel = ft.Container(
        content=ft.Column([
            score_text,
            phase_text,
            warning_text,
        ], spacing=4),
        bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.BLUE_GREY),
        border_radius=8,
        padding=12,
    )

    page.add(
        ft.Text("⚛ AURION CORE — 量子波形干渉ビジュアライザー",
                color=ft.Colors.CYAN_200, size=18, weight=ft.FontWeight.BOLD),
        ft.Divider(color=ft.Colors.WHITE12),
        legend,
        ft.Container(content=canvas, border_radius=8, clip_behavior=ft.ClipBehavior.HARD_EDGE),
        ft.Row([score_panel, controls_panel], spacing=16, alignment=ft.MainAxisAlignment.START),
    )

    # 初回描画
    _redraw()

    # タイマー起動（Flet 0.x: page.run_task / on_event ではなく add ループ）
    async def _animation_loop() -> None:
        while state["running"]:
            state["t_offset"] += 0.006
            _redraw()
            page.update()
            await asyncio.sleep(ANIMATION_INTERVAL_MS / 1000.0)

    import asyncio
    page.run_task(_animation_loop)


if __name__ == "__main__":
    ft.app(target=main)
