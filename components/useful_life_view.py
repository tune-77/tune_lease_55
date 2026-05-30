"""
法定耐用年数 一覧ビュー（REV-085/121）
useful_life_equipment.json から全品目の耐用年数を表示する。
"""
import streamlit as st
import json
import os

_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "static_data", "useful_life_equipment.json")

# カテゴリアイコン
_CAT_ICONS: dict[str, str] = {
    "建設・土木機械": "🏗️",
    "製造・加工機械": "⚙️",
    "情報通信・オフィス": "💻",
    "運輸・車両": "🚛",
    "空調・電気・設備": "🌡️",
    "宿泊・飲食・小売": "🍳",
    "農業・林業・漁業": "🌾",
    "医療・福祉": "🏥",
}


def _load_data() -> dict:
    try:
        with open(_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _years_badge(years: int) -> str:
    if years <= 4:
        color, label = "#3b82f6", f"{years}年"
    elif years <= 6:
        color, label = "#16a34a", f"{years}年"
    elif years <= 10:
        color, label = "#d97706", f"{years}年"
    else:
        color, label = "#dc2626", f"{years}年"
    return (
        f'<span style="display:inline-block;padding:.15rem .5rem;'
        f'border-radius:9999px;background:{color}22;border:1px solid {color};'
        f'color:{color};font-weight:700;font-size:.82rem;">{label}</span>'
    )


def render_useful_life_view() -> None:
    st.header("📋 法定耐用年数 一覧")
    st.caption(
        "国税庁 耐用年数表（令和5年）をベースにした主要品目の法定耐用年数。"
        "詳細は[国税庁サイト](https://www.keisan.nta.go.jp/r5yokuaru/aoiroshinkoku/hitsuyokeihi/genkashokyakuhi/taiyonensuhyo.html)を参照。"
    )

    data = _load_data()
    categories = data.get("categories", [])
    if not categories:
        st.warning("データファイルが読み込めません。")
        return

    # ─── 検索 ─────────────────────────────────────────────────
    search = st.text_input("🔍 品目名で検索", placeholder="例: サーバー、トラック、クレーン", key="useful_life_search")

    # ─── 耐用年数フィルタ ──────────────────────────────────────
    max_years = st.slider("最大耐用年数で絞り込み（年以下を表示）", 3, 20, 20, step=1, key="useful_life_max_years")

    all_items = [(cat["name"], it) for cat in categories for it in cat["items"]]

    # 検索フィルタ
    if search:
        all_items = [(cat, it) for cat, it in all_items if search.lower() in it["name"].lower()]

    # 年数フィルタ
    all_items = [(cat, it) for cat, it in all_items if it.get("years", 99) <= max_years]

    if search or max_years < 20:
        # 検索/フィルタ結果をフラットテーブルで表示
        st.markdown(f"**{len(all_items)} 件ヒット**")
        if not all_items:
            st.info("条件に合う品目がありません。")
            return
        rows = ""
        for cat_name, it in sorted(all_items, key=lambda x: x[1]["years"]):
            icon = _CAT_ICONS.get(cat_name, "📦")
            rows += (
                f"<tr>"
                f"<td>{icon} {cat_name}</td>"
                f"<td><b>{it['name']}</b></td>"
                f"<td style='text-align:center;'>{_years_badge(it['years'])}</td>"
                f"<td style='font-size:.78rem;color:#64748b;'>{it.get('note','')}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
            f"<thead><tr style='background:#f1f5f9;'>"
            f"<th style='text-align:left;padding:.4rem .6rem;'>カテゴリ</th>"
            f"<th style='text-align:left;'>品目</th>"
            f"<th>耐用年数</th><th>備考</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        # カテゴリ別に折りたたみ表示
        for cat in categories:
            icon = _CAT_ICONS.get(cat["name"], "📦")
            items = [it for it in cat["items"] if it.get("years", 99) <= max_years]
            if not items:
                continue
            with st.expander(f"{icon} {cat['name']}（{len(items)} 品目）", expanded=True):
                rows = ""
                for it in items:
                    rows += (
                        f"<tr>"
                        f"<td style='padding:.3rem .5rem;'><b>{it['name']}</b></td>"
                        f"<td style='text-align:center;'>{_years_badge(it['years'])}</td>"
                        f"<td style='font-size:.78rem;color:#64748b;'>{it.get('note','')}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
                    f"<thead><tr style='background:#f8fafc;'>"
                    f"<th style='text-align:left;padding:.3rem .5rem;'>品目</th>"
                    f"<th>耐用年数</th><th>備考</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ─── リース期間の目安 ──────────────────────────────────────
    with st.expander("💡 リース期間の設定目安", expanded=False):
        st.markdown("""
| 法定耐用年数 | 推奨リース期間 | 備考 |
|------------|-------------|------|
| 4年（PC等） | 2〜4年 | 耐用年数内に収める |
| 5年（トラック・建機等） | 3〜5年 | 60〜100%が標準 |
| 6〜8年 | 4〜6年 | 耐用年数の60〜80%が安全圏 |
| 10年以上 | 5〜8年 | 耐用年数の50〜70%推奨 |

> ⚠️ リース期間が耐用年数の **70%超** になると、満了時の残余価値が低下し、物件スコアが下がる可能性があります。
""")
