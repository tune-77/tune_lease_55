import flet as ft
import flet.canvas as cv
import numpy as np
import time

from clifford_poc import CliffordSuccessPredictor, CliffordVisualizerLogic

def main(page: ft.Page):
    page.title = "AURION CORE - CliffordNet Geometric Analytics"
    page.padding = 20
    page.theme_mode = ft.ThemeMode.DARK

    predictor = CliffordSuccessPredictor()
    vis_logic = CliffordVisualizerLogic(center_x=300, center_y=300, scale=150)

    # 黄金の成約面積の座標
    golden_points = vis_logic.calculate_golden_area()
    
    # 描画キャンバス
    canvas = cv.Canvas(
        width=600,
        height=600,
        shapes=[]
    )
    
    # 財務指標の初期値
    features = {
        'sales_growth': 1.0,
        'capital_ratio': 1.0,
        'liquidity': 1.0,
        'years_in_business': 1.0
    }

    sliders = {}
    
    def update_canvas():
        # Canvas上の図形をクリア
        canvas.shapes.clear()
        
        # 現在の値を更新
        for k in features:
            features[k] = sliders[k].value

        result = predictor.predict(features)
        prob = result['success_probability']
        prob_text.value = f"幾何的成約確率 (Clifford Probability): {prob:.2%}"
        
        current_points = vis_logic.calculate_current_state(result['vectors'])
        guides = vis_logic.calculate_guide_vectors(current_points, golden_points)
        
        # 1. 黄金の成約面積 (基準面: ゴースト描画)
        path_elements = [cv.Path.MoveTo(*golden_points[0])]
        for p in golden_points[1:]:
            path_elements.append(cv.Path.LineTo(*p))
        path_elements.append(cv.Path.Close())
        
        canvas.shapes.append(
            cv.Path(
                elements=path_elements,
                paint=ft.Paint(
                    style=ft.PaintingStyle.FILL,
                    color=ft.Colors.with_opacity(0.1, ft.Colors.AMBER)
                )
            )
        )
        canvas.shapes.append(
            cv.Path(
                elements=path_elements,
                paint=ft.Paint(
                    style=ft.PaintingStyle.STROKE,
                    color=ft.Colors.AMBER_500,
                    stroke_width=1,
                    stroke_dash_pattern=[5, 5]
                )
            )
        )

        # 2. 現在の案件の歪み形状 (動的描画)
        curr_elements = [cv.Path.MoveTo(*current_points[0])]
        for p in current_points[1:]:
            curr_elements.append(cv.Path.LineTo(*p))
        curr_elements.append(cv.Path.Close())
        
        # 成約確率に応じて色を変化
        base_color = ft.Colors.BLUE if prob < 0.6 else ft.Colors.GREEN
        canvas.shapes.append(
            cv.Path(
                elements=curr_elements,
                paint=ft.Paint(
                    style=ft.PaintingStyle.FILL,
                    color=ft.Colors.with_opacity(0.3, base_color)
                )
            )
        )
        canvas.shapes.append(
            cv.Path(
                elements=curr_elements,
                paint=ft.Paint(
                    style=ft.PaintingStyle.STROKE,
                    color=base_color,
                    stroke_width=2
                )
            )
        )
        
        # 3. ガイドベクトル（どの方向に引っ張るべきか）
        for g in guides:
            # 線の描画
            canvas.shapes.append(
                cv.Line(
                    g['start'][0], g['start'][1],
                    g['end'][0], g['end'][1],
                    paint=ft.Paint(color=ft.Colors.RED_ACCENT, stroke_width=1)
                )
            )
            # 点（現在位置）
            canvas.shapes.append(
                cv.Circle(
                    x=g['start'][0], y=g['start'][1], radius=4,
                    paint=ft.Paint(color=ft.Colors.WHITE)
                )
            )
            
        page.update()

    def on_slider_change(e):
        update_canvas()

    # UIコンポーネントの構築
    for k in features:
        sliders[k] = ft.Slider(
            min=0.1, max=2.5, value=1.0, divisions=24,
            label="{value}", width=300, on_change=on_slider_change
        )
        
    prob_text = ft.Text(size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_400)

    controls_panel = ft.Column([
        ft.Text("幾何積・歪みパラメータ調整", size=24, weight="bold"),
        prob_text,
        ft.Divider(),
        ft.Text("Sales Growth (売上成長率):"), sliders['sales_growth'],
        ft.Text("Capital Ratio (自己資本比率):"), sliders['capital_ratio'],
        ft.Text("Liquidity (流動性):"), sliders['liquidity'],
        ft.Text("Years in Business (業歴):"), sliders['years_in_business'],
        ft.Text("\n【ガイドライン】", color=ft.Colors.GREY_400),
        ft.Text("黄色破線：成約クラスターの「黄金面積」", color=ft.Colors.AMBER_400),
        ft.Text("青/緑面：現在の案件の「幾何学的形状」", color=ft.Colors.BLUE_200),
        ft.Text("赤い線：黄金面積に重なるために必要な「ガイドベクトル（補正力）」", color=ft.Colors.RED_ACCENT),
    ])

    page.add(
        ft.Row([
            controls_panel,
            ft.Container(
                content=canvas,
                border=ft.border.all(1, ft.Colors.WHITE24),
                border_radius=8,
                bgcolor=ft.Colors.BLACK45
            )
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START)
    )

    # 初回描画
    update_canvas()

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
