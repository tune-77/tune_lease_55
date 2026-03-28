"""
基準金利マスタ管理モジュール
- 月次基準金利をlease_data.dbで管理
- 審査時に当月の基準金利を自動取得
- Streamlit UI: render_base_rate_manager()
"""
import sqlite3
import datetime
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lease_data.db")


def _get_conn():
    return sqlite3.connect(_DB_PATH)


def get_base_rate(month: str | None = None) -> float | None:
    """
    指定月（YYYY-MM）の基準金利を返す。
    省略時は当月。登録がなければ None を返す。
    """
    if month is None:
        month = datetime.date.today().strftime("%Y-%m")
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT rate FROM base_rate_master WHERE month = ?", (month,)
        ).fetchone()
    return float(row[0]) if row else None


def get_current_base_rate(fallback: float = 2.1) -> float:
    """
    当月の基準金利を返す。未登録なら fallback を返す。
    """
    rate = get_base_rate()
    return rate if rate is not None else fallback


def upsert_base_rate(month: str, rate: float, note: str = "") -> None:
    """月次基準金利を登録／更新する"""
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO base_rate_master (month, rate, note)
            VALUES (?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                rate = excluded.rate,
                note = excluded.note
            """,
            (month, rate, note),
        )


def list_base_rates(limit: int = 24) -> list[dict]:
    """直近N件の基準金利一覧を返す（新しい順）"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT month, rate, note FROM base_rate_master ORDER BY month DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"month": r[0], "rate": r[1], "note": r[2]} for r in rows]


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def render_base_rate_manager():
    """基準金利マスタ管理パネルをStreamlitで描画する"""
    import streamlit as st

    st.subheader("📅 基準金利マスタ")

    current_month = datetime.date.today().strftime("%Y-%m")
    next_month = (datetime.date.today().replace(day=1) + datetime.timedelta(days=32)).strftime("%Y-%m")
    current_rate = get_base_rate(current_month)
    next_rate = get_base_rate(next_month)

    # 当月・来月の状況表示
    c1, c2 = st.columns(2)
    with c1:
        if current_rate is not None:
            st.metric("当月基準金利", f"{current_rate:.2f}%", f"{current_month}")
        else:
            st.warning(f"⚠️ 当月（{current_month}）の基準金利が未登録です")
    with c2:
        if next_rate is not None:
            st.metric("来月基準金利（登録済）", f"{next_rate:.2f}%", f"{next_month}")
        else:
            st.info(f"来月（{next_month}）はまだ未登録です")

    st.divider()

    # 登録フォーム
    with st.form("base_rate_form"):
        st.markdown("#### 基準金利を登録／更新")
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            input_month = st.text_input(
                "対象月 (YYYY-MM)",
                value=next_month,
                placeholder="例: 2026-05",
            )
        with col2:
            input_rate = st.number_input(
                "基準金利 (%)",
                value=next_rate if next_rate else (current_rate or 2.1),
                step=0.01,
                format="%.2f",
            )
        with col3:
            input_note = st.text_input("メモ（任意）", placeholder="例: 金利変更なし")

        submitted = st.form_submit_button("登録", type="primary", use_container_width=True)
        if submitted:
            # 入力チェック
            try:
                datetime.datetime.strptime(input_month, "%Y-%m")
            except ValueError:
                st.error("月の形式が正しくありません（例: 2026-05）")
                return
            if input_rate <= 0:
                st.error("基準金利は0より大きい値を入力してください")
                return
            upsert_base_rate(input_month, float(input_rate), input_note)
            st.success(f"✅ {input_month} の基準金利 {input_rate:.2f}% を登録しました")
            st.rerun()

    # 履歴一覧
    st.markdown("#### 登録履歴")
    records = list_base_rates()
    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        df.columns = ["月", "基準金利 (%)", "メモ"]
        df["基準金利 (%)"] = df["基準金利 (%)"].map(lambda x: f"{x:.2f}%")
        # 当月行をハイライト
        def _highlight(row):
            return ["background-color: #fef9c3" if row["月"] == current_month else "" for _ in row]
        st.dataframe(df.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("登録データがありません")
