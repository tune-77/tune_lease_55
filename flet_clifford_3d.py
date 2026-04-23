import flet as ft
import flet.canvas as cv
import numpy as np
import time
import threading
import math

from clifford_poc import CliffordSuccessPredictor

class Projection3D:
    def __init__(self, width, height, scale=150):
        self.width = width
        self.height = height
        self.scale = scale
        self.distance = 4.0

    def project(self, x, y, z):
        # シンプルな透視投影 (Perspective Projection)
        factor = self.scale / (z + self.distance)
        px = x * factor + self.width / 2
        py = -y * factor + self.height / 2
        return px, py

def rotate_y(points, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for p in points:
        x, y, z = p[0], p[1], p[2]
        nx = x * cos_a + z * sin_a
        nz = -x * sin_a + z * cos_a
        rotated.append((nx, y, nz))
    return rotated

def rotate_x(points, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = []
    for p in points:
        x, y, z = p[0], p[1], p[2]
        ny = y * cos_a - z * sin_a
        nz = y * sin_a + z * cos_a
        rotated.append((x, ny, nz))
    return rotated

def main(page: ft.Page):
    page.title = "AURION CORE - 3D CliffordNet Engine"
    page.padding = 30
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0B0F19" # 深いダークブルー（SF風）

    predictor = CliffordSuccessPredictor()

    # 3D空間上の各指標の基本軸ベクトル (正八面体を形成: 6頂点)
    base_vectors = {
        'sales_growth':      np.array([ 1.5,  0.0,  0.0]),
        'capital_ratio':     np.array([-1.5,  0.0,  0.0]),
        'liquidity':         np.array([ 0.0,  1.5,  0.0]),
        'years_in_business': np.array([ 0.0, -1.5,  0.0]),
        'operating_margin':  np.array([ 0.0,  0.0,  1.5]),
        'asset_turnover':    np.array([ 0.0,  0.0, -1.5])
    }

    # 正八面体の面を構成するインデックス (8面)
    faces = [
        (0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
        (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)
    ]

    features = {
        'sales_growth': 1.0,
        'capital_ratio': 1.0,
        'liquidity': 1.0,
        'years_in_business': 1.0,
        'operating_margin': 1.0,
        'asset_turnover': 1.0
    }

    sliders = {}
    
    canvas_width = 600
    canvas_height = 500
    canvas = cv.Canvas(width=canvas_width, height=canvas_height, shapes=[])
    proj = Projection3D(canvas_width, canvas_height, scale=300)

    angle_y = 0.0
    angle_x = 0.3 # 少し上から見下ろす角度
    
    is_running = True

    def draw_3d_frame():
        nonlocal angle_y
        
        # UIの各スライダー値を反映
        for k in features:
            features[k] = sliders[k].value

        result = predictor.predict(features)
        prob = result['success_probability']
        
        # 確率表示の更新
        prob_text.value = f"3D GEOMETRIC PROBABILITY: {prob:.2%}"
        prob_text.color = ft.Colors.CYAN_ACCENT_400 if prob >= 0.5 else ft.Colors.RED_ACCENT_400
        
        # 黄金面積（理想の形: 全て1.0）の頂点
        golden_verts = [base_vectors[k] * 1.0 for k in features.keys()]
        # 現在の案件の頂点
        curr_verts = [base_vectors[k] * features[k] for k in features.keys()]

        # 回転適用
        golden_rot = rotate_x(rotate_y(golden_verts, angle_y), angle_x)
        curr_rot = rotate_x(rotate_y(curr_verts, angle_y), angle_x)

        # 2D投影
        golden_2d = [proj.project(*v) for v in golden_rot]
        curr_2d = [proj.project(*v) for v in curr_rot]

        canvas.shapes.clear()

        # 1. 黄金の四面体（ワイヤーフレーム表示: サイバーな黄色）
        for face in faces:
            path_elements = [cv.Path.MoveTo(*golden_2d[face[0]])]
            for idx in face[1:]:
                path_elements.append(cv.Path.LineTo(*golden_2d[idx]))
            path_elements.append(cv.Path.Close())
            
            canvas.shapes.append(
                cv.Path(
                    elements=path_elements,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.STROKE,
                        color=ft.Colors.with_opacity(0.3, ft.Colors.AMBER_400),
                        stroke_width=1,
                        stroke_dash_pattern=[4, 4]
                    )
                )
            )

        # 2. 現在の案件（ソリッド面とエッジ: シアンブルー）
        base_color = ft.Colors.CYAN_ACCENT_400 if prob >= 0.5 else ft.Colors.RED_400
        
        # Zソート（奥の面から描画するための簡易ソート）
        face_depths = []
        for i, face in enumerate(faces):
            # 面の中心のZ座標（回転後）を計算
            avg_z = sum(curr_rot[idx][2] for idx in face) / 3.0
            face_depths.append((avg_z, face))
        
        # 奥(Zが小さい)から手前へソート
        face_depths.sort(key=lambda x: x[0], reverse=True)

        for _, face in face_depths:
            path_elements = [cv.Path.MoveTo(*curr_2d[face[0]])]
            for idx in face[1:]:
                path_elements.append(cv.Path.LineTo(*curr_2d[idx]))
            path_elements.append(cv.Path.Close())
            
            # 面の塗りつぶし（半透明）
            canvas.shapes.append(
                cv.Path(
                    elements=path_elements,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.FILL,
                        color=ft.Colors.with_opacity(0.15, base_color)
                    )
                )
            )
            # エッジ（太線）
            canvas.shapes.append(
                cv.Path(
                    elements=path_elements,
                    paint=ft.Paint(
                        style=ft.PaintingStyle.STROKE,
                        color=base_color,
                        stroke_width=2
                    )
                )
            )

        # 3. ガイドベクトル（赤のネオン線：中心点または各頂点からの引力）
        for g_pt, c_pt in zip(golden_2d, curr_2d):
            canvas.shapes.append(
                cv.Line(
                    c_pt[0], c_pt[1], g_pt[0], g_pt[1],
                    paint=ft.Paint(color=ft.Colors.PINK_ACCENT_400, stroke_width=2)
                )
            )
            # 頂点に光るドット
            canvas.shapes.append(
                cv.Circle(x=c_pt[0], y=c_pt[1], radius=4, paint=ft.Paint(color=ft.Colors.WHITE))
            )

        try:
            page.update()
        except:
            pass # ページ終了時エラー回避

    def animation_loop():
        nonlocal angle_y
        while is_running:
            angle_y += 0.02 # 回転速度
            draw_3d_frame()
            time.sleep(0.03) # 約30FPS

    def on_slider_change(e):
        # 描画はアニメーションループに任せる
        pass

    # UIコンポーネント
    for k in features:
        sliders[k] = ft.Slider(
            min=0.1, max=2.5, value=1.0, divisions=24,
            active_color=ft.Colors.CYAN_ACCENT_700,
            label="{value}", width=350, on_change=on_slider_change
        )
        
    prob_text = ft.Text(size=28, weight=ft.FontWeight.W_900, font_family="monospace")

    controls_panel = ft.Column([
        ft.Text("HUD: 3D GEOMETRIC ANALYSIS", size=20, weight="bold", color=ft.Colors.WHITE54, font_family="monospace"),
        prob_text,
        ft.Divider(color=ft.Colors.WHITE12),
        ft.Text("Sales Growth (売上成長率)", color=ft.Colors.WHITE70), sliders['sales_growth'],
        ft.Text("Capital Ratio (自己資本比率)", color=ft.Colors.WHITE70), sliders['capital_ratio'],
        ft.Text("Liquidity (流動性)", color=ft.Colors.WHITE70), sliders['liquidity'],
        ft.Text("Years in Business (業歴)", color=ft.Colors.WHITE70), sliders['years_in_business'],
        ft.Text("Operating Margin (営業利益率)", color=ft.Colors.WHITE70), sliders['operating_margin'],
        ft.Text("Asset Turnover (総資産回転率)", color=ft.Colors.WHITE70), sliders['asset_turnover'],
        ft.Container(height=10),
        ft.Text("■ YELLOW FRAME : Ideal Success Tensor", color=ft.Colors.AMBER_400, size=12, font_family="monospace"),
        ft.Text("■ CYAN/RED AREA: Current Deal Tensor", color=ft.Colors.CYAN_ACCENT_400, size=12, font_family="monospace"),
        ft.Text("■ PINK LINES   : Required Distortion Vector", color=ft.Colors.PINK_ACCENT_400, size=12, font_family="monospace"),
    ], width=400)

    # レイアウト
    main_row = ft.Row([
        controls_panel,
        ft.Container(
            content=canvas,
            border=ft.border.all(1, ft.Colors.WHITE12),
            border_radius=16,
            bgcolor="#111827", # ダークパネル
            shadow=ft.BoxShadow(spread_radius=1, blur_radius=15, color=ft.Colors.CYAN_ACCENT_700, offset=ft.Offset(0,0))
        )
    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)

    page.add(main_row)

    # アニメーションスレッド開始
    anim_thread = threading.Thread(target=animation_loop, daemon=True)
    anim_thread.start()

    def on_disconnect(e):
        nonlocal is_running
        is_running = False
    page.on_disconnect = on_disconnect

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
