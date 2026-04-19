"""
基準金利マスタ管理モジュール
- 月次基準金利をlease_data.dbで管理（リース期間別9区分）
- 審査時に当月・期間別の基準金利を自動取得
- Streamlit UI: render_base_rate_manager()
"""
import sqlite3
import datetime
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lease_logic_sumaho12", "data", "lease_data.db")

# リース期間列の定義（月数上限, 列名）
_TERM_MAP = [
    (24,  "r_2y"),
    (36,  "r_3y"),
    (48,  "r_4y"),
    (60,  "r_5y"),
    (72,  "r_6y"),
    (84,  "r_7y"),
    (96,  "r_8y"),
    (108, "r_9y"),
]
TERM_COLS = ["r_2y", "r_3y", "r_4y", "r_5y", "r_6y", "r_7y", "r_8y", "r_9y", "r_over9y"]

# 列名 → 表示ラベル
_COL_LABELS = {
    "r_2y":     "2年以内",
    "r_3y":     "3年以内",
    "r_4y":     "4年以内",
    "r_5y":     "5年以内",
    "r_6y":     "6年以内",
    "r_7y":     "7年以内",
    "r_8y":     "8年以内",
    "r_9y":     "9年以内",
    "r_over9y": "9年超",
}

# 初期データ（2022/11〜2026/4、42ヶ月分）
# 出典: 基準金利マスタ表
_INITIAL_DATA: list[tuple] = [
    # (month,    r_2y, r_3y, r_4y, r_5y, r_6y, r_7y, r_8y, r_9y, r_over9y)
    ("2026-04", 2.19, 2.37, 2.56, 2.66, 2.85, 3.08, 3.15, 3.27, 3.54),
    ("2026-03", 2.16, 2.30, 2.46, 2.54, 2.72, 2.93, 3.00, 3.11, 3.37),
    ("2026-02", 1.98, 2.14, 2.30, 2.40, 2.58, 2.81, 2.88, 3.00, 3.27),
    ("2026-01", 1.83, 1.98, 2.14, 2.23, 2.41, 2.64, 2.71, 2.83, 3.09),
    ("2025-12", 1.65, 1.78, 1.93, 2.01, 2.18, 2.40, 2.46, 2.57, 2.83),
    ("2025-11", 1.63, 1.74, 1.87, 1.93, 2.09, 2.29, 2.35, 2.45, 2.70),
    ("2025-10", 1.62, 1.74, 1.87, 1.93, 2.09, 2.29, 2.34, 2.44, 2.69),
    ("2025-09", 1.57, 1.66, 1.77, 1.83, 1.97, 2.17, 2.22, 2.32, 2.57),
    ("2025-08", 1.57, 1.65, 1.76, 1.81, 1.96, 2.15, 2.20, 2.29, 2.54),
    ("2025-07", 1.48, 1.55, 1.64, 1.69, 1.83, 2.02, 2.06, 2.15, 2.40),
    ("2025-06", 1.48, 1.54, 1.65, 1.69, 1.84, 2.03, 2.07, 2.17, 2.42),
    ("2025-05", 1.48, 1.51, 1.60, 1.63, 1.76, 1.94, 1.98, 2.07, 2.31),
    ("2025-04", 1.53, 1.61, 1.72, 1.76, 1.90, 2.09, 2.13, 2.22, 2.46),
    ("2025-03", 1.48, 1.54, 1.64, 1.67, 1.80, 1.99, 2.02, 2.10, 2.34),
    ("2025-02", 1.43, 1.50, 1.60, 1.64, 1.77, 1.95, 1.98, 2.06, 2.29),
    ("2025-01", 1.25, 1.30, 1.39, 1.42, 1.55, 1.73, 1.76, 1.84, 2.08),
    ("2024-12", 1.25, 1.31, 1.41, 1.44, 1.57, 1.75, 1.79, 1.87, 2.11),
    ("2024-11", 1.11, 1.16, 1.25, 1.28, 1.41, 1.60, 1.63, 1.72, 1.96),
    ("2024-10", 1.07, 1.12, 1.21, 1.24, 1.37, 1.55, 1.59, 1.68, 1.92),
    ("2024-09", 1.07, 1.13, 1.22, 1.25, 1.39, 1.57, 1.61, 1.69, 1.93),
    ("2024-08", 1.07, 1.15, 1.26, 1.31, 1.45, 1.64, 1.68, 1.78, 2.02),
    ("2024-07", 1.00, 1.09, 1.20, 1.26, 1.41, 1.61, 1.66, 1.76, 2.01),
    ("2024-06", 1.00, 1.10, 1.22, 1.28, 1.43, 1.63, 1.68, 1.79, 2.04),
    ("2024-05", 0.94, 1.04, 1.13, 1.21, 1.33, 1.55, 1.63, 1.70, 1.96),
    ("2024-04", 0.88, 0.98, 1.06, 1.14, 1.25, 1.47, 1.54, 1.61, 1.86),
    ("2024-03", 0.80, 0.90, 0.98, 1.07, 1.19, 1.41, 1.49, 1.56, 1.82),
    ("2024-02", 0.76, 0.86, 0.94, 1.04, 1.17, 1.40, 1.47, 1.55, 1.81),
    ("2024-01", 0.76, 0.86, 0.94, 1.02, 1.15, 1.37, 1.44, 1.51, 1.77),
    ("2023-12", 0.76, 0.86, 0.94, 1.02, 1.15, 1.37, 1.44, 1.51, 1.77),
    ("2023-11", 0.76, 0.86, 0.94, 1.02, 1.15, 1.37, 1.44, 1.51, 1.77),
    ("2023-10", 0.73, 0.83, 0.90, 0.97, 1.08, 1.29, 1.34, 1.40, 1.65),
    ("2023-09", 0.68, 0.75, 0.81, 0.87, 0.98, 1.19, 1.25, 1.30, 1.56),
    ("2023-08", 0.60, 0.66, 0.71, 0.77, 0.87, 1.08, 1.14, 1.20, 1.45),
    ("2023-07", 0.60, 0.65, 0.70, 0.74, 0.84, 1.05, 1.10, 1.15, 1.40),
    ("2023-06", 0.60, 0.65, 0.69, 0.73, 0.83, 1.02, 1.07, 1.11, 1.36),
    ("2023-05", 0.61, 0.66, 0.71, 0.75, 0.85, 1.06, 1.11, 1.16, 1.40),
    ("2023-04", 0.63, 0.67, 0.71, 0.74, 0.83, 1.02, 1.06, 1.11, 1.35),
    ("2023-03", 0.69, 0.76, 0.82, 0.88, 0.99, 1.20, 1.26, 1.32, 1.57),
    ("2023-02", 0.70, 0.77, 0.84, 0.90, 1.01, 1.23, 1.29, 1.35, 1.60),
    ("2023-01", 0.77, 0.87, 0.94, 1.00, 1.10, 1.30, 1.34, 1.39, 1.63),
    ("2022-12", 0.63, 0.68, 0.73, 0.77, 0.85, 1.04, 1.09, 1.13, 1.36),
    ("2022-11", 0.64, 0.70, 0.75, 0.80, 0.90, 1.09, 1.14, 1.19, 1.43),
]


def _term_to_col(lease_term_months: int) -> str:
    """月数から対応する期間列名を返す。9年超は r_over9y。"""
    for months, col in _TERM_MAP:
        if lease_term_months <= months:
            return col
    return "r_over9y"


def _ensure_term_columns(conn: sqlite3.Connection) -> None:
    """期間別列が未存在なら ALTER TABLE で追加する。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(base_rate_master)")}
    for col in TERM_COLS:
        if col not in existing:
            conn.execute(f"ALTER TABLE base_rate_master ADD COLUMN {col} REAL")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS base_rate_master (
            month TEXT PRIMARY KEY,
            rate  REAL NOT NULL,
            note  TEXT DEFAULT ''
        )
        """
    )
    _ensure_term_columns(conn)
    return conn


# ── 読み取りAPI ───────────────────────────────────────────────────────────────

def get_base_rate(month: str | None = None) -> float | None:
    """
    指定月（YYYY-MM）の基準金利（旧 rate 列）を返す。
    省略時は当月。登録がなければ None を返す。
    """
    if month is None:
        month = datetime.date.today().strftime("%Y-%m")
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT rate FROM base_rate_master WHERE month = ?", (month,)
        ).fetchone()
    return float(row[0]) if row else None


def get_base_rate_by_term(
    month: str | None = None,
    lease_term_months: int = 60,
) -> float | None:
    """
    期間月数に対応した基準金利を返す。未登録なら None。
    例: lease_term_months=84 → r_7y（7年以内）
    """
    if month is None:
        month = datetime.date.today().strftime("%Y-%m")
    col = _term_to_col(lease_term_months)
    with _get_conn() as conn:
        row = conn.execute(
            f"SELECT {col} FROM base_rate_master WHERE month = ?", (month,)
        ).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    # 期間列が未登録なら旧 rate 列にフォールバック
    return get_base_rate(month)


def get_current_base_rate(fallback: float = 2.1) -> float:
    """
    当月の基準金利（5年以内=r_5y）を返す。未登録なら fallback を返す。
    シグネチャは旧バージョンと互換。
    """
    rate = get_base_rate_by_term(lease_term_months=60)
    return rate if rate is not None else fallback


# ── 書き込みAPI ───────────────────────────────────────────────────────────────

def upsert_base_rate(
    month: str,
    rate: float = 0.0,
    note: str = "",
    *,
    r_2y: float | None = None,
    r_3y: float | None = None,
    r_4y: float | None = None,
    r_5y: float | None = None,
    r_6y: float | None = None,
    r_7y: float | None = None,
    r_8y: float | None = None,
    r_9y: float | None = None,
    r_over9y: float | None = None,
) -> None:
    """月次基準金利を登録／更新する。期間別列はキーワード引数で指定。"""
    # rate 列は r_5y を優先して同期
    effective_rate = r_5y if r_5y is not None else rate
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO base_rate_master
                (month, rate, note, r_2y, r_3y, r_4y, r_5y, r_6y, r_7y, r_8y, r_9y, r_over9y)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                rate     = excluded.rate,
                note     = excluded.note,
                r_2y     = COALESCE(excluded.r_2y,     r_2y),
                r_3y     = COALESCE(excluded.r_3y,     r_3y),
                r_4y     = COALESCE(excluded.r_4y,     r_4y),
                r_5y     = COALESCE(excluded.r_5y,     r_5y),
                r_6y     = COALESCE(excluded.r_6y,     r_6y),
                r_7y     = COALESCE(excluded.r_7y,     r_7y),
                r_8y     = COALESCE(excluded.r_8y,     r_8y),
                r_9y     = COALESCE(excluded.r_9y,     r_9y),
                r_over9y = COALESCE(excluded.r_over9y, r_over9y)
            """,
            (month, effective_rate, note,
             r_2y, r_3y, r_4y, r_5y, r_6y, r_7y, r_8y, r_9y, r_over9y),
        )


def list_base_rates(limit: int = 60) -> list[dict]:
    """直近N件の基準金利一覧を返す（新しい順）。期間別9列を含む。"""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT month, rate, r_2y, r_3y, r_4y, r_5y, r_6y, r_7y, r_8y, r_9y, r_over9y, note
            FROM base_rate_master
            ORDER BY month DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "month": r[0], "rate": r[1],
            "r_2y": r[2], "r_3y": r[3], "r_4y": r[4], "r_5y": r[5],
            "r_6y": r[6], "r_7y": r[7], "r_8y": r[8], "r_9y": r[9],
            "r_over9y": r[10], "note": r[11],
        }
        for r in rows
    ]


def seed_initial_data(overwrite: bool = False) -> tuple[int, int]:
    """
    初期データ（42ヶ月分）を一括投入する。
    overwrite=False の場合、既存月はスキップ。
    Returns: (inserted, skipped)
    """
    inserted = skipped = 0
    with _get_conn() as conn:
        for row in _INITIAL_DATA:
            month, r2, r3, r4, r5, r6, r7, r8, r9, rp = row
            exists = conn.execute(
                "SELECT 1 FROM base_rate_master WHERE month = ?", (month,)
            ).fetchone()
            if exists and not overwrite:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO base_rate_master
                    (month, rate, note, r_2y, r_3y, r_4y, r_5y, r_6y, r_7y, r_8y, r_9y, r_over9y)
                VALUES (?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(month) DO UPDATE SET
                    rate=excluded.rate, r_2y=excluded.r_2y, r_3y=excluded.r_3y,
                    r_4y=excluded.r_4y, r_5y=excluded.r_5y, r_6y=excluded.r_6y,
                    r_7y=excluded.r_7y, r_8y=excluded.r_8y, r_9y=excluded.r_9y,
                    r_over9y=excluded.r_over9y
                """,
                (month, r5, r2, r3, r4, r5, r6, r7, r8, r9, rp),
            )
            inserted += 1
    return inserted, skipped


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def render_base_rate_manager() -> None:
    """基準金利マスタ管理パネルをStreamlitで描画する"""
    import streamlit as st
    import pandas as pd

    st.subheader("📅 基準金利マスタ")

    today = datetime.date.today()
    current_month = today.strftime("%Y-%m")
    next_month = (today.replace(day=1) + datetime.timedelta(days=32)).strftime("%Y-%m")

    # ── 月次金利更新フォーム（最上部・常時表示） ──────────────────────────────
    # 更新対象月: 未登録の月を優先（当月未登録 → 当月、来月未登録 → 来月、それ以外 → 来月）
    current_rate_5y = get_base_rate_by_term(current_month, 60)
    next_rate_5y    = get_base_rate_by_term(next_month, 60)
    default_target  = current_month if current_rate_5y is None else next_month

    # 直近登録データを前月比のデフォルト値として使用
    recent = list_base_rates(limit=2)
    latest = recent[0] if recent else {}          # 最新登録月のデータ
    prev   = recent[1] if len(recent) > 1 else {} # その前月のデータ

    def _fv(col: str, src: dict) -> float:
        """辞書から float 値を取り出す。なければ 1.00。"""
        v = src.get(col)
        return float(v) if v is not None else 1.00

    with st.container(border=True):
        st.markdown("### 月次金利更新")

        # 対象月 + ステータスバッジ
        hc1, hc2 = st.columns([3, 5])
        with hc1:
            if current_rate_5y is None:
                st.warning(f"⚠️ 当月（{current_month}）未登録")
            else:
                st.metric("当月（5年以内）", f"{current_rate_5y:.2f}%", current_month)
        with hc2:
            if next_rate_5y is None:
                st.info(f"来月（{next_month}）はまだ未登録です")
            else:
                st.metric("来月（5年以内）", f"{next_rate_5y:.2f}%", next_month)

        st.divider()

        with st.form("monthly_rate_form", border=False):
            fc1, fc2 = st.columns([2, 4])
            with fc1:
                target_month = st.text_input(
                    "適用月 (YYYY-MM)",
                    value=default_target,
                    placeholder="例: 2026-05",
                )
            with fc2:
                input_note = st.text_input(
                    "メモ（任意）",
                    placeholder="例: 日銀利上げにより改定",
                )

            st.markdown("**リース期間別基準金利 (%)**　※前月値をデフォルトで表示")

            # 3列 × 3行レイアウト
            row1 = st.columns(3)
            row2 = st.columns(3)
            row3 = st.columns(3)

            with row1[0]:
                v_2y = st.number_input(
                    "2年以内",
                    value=_fv("r_2y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_2y', prev):.2f}%" if prev else None,
                )
            with row1[1]:
                v_3y = st.number_input(
                    "3年以内",
                    value=_fv("r_3y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_3y', prev):.2f}%" if prev else None,
                )
            with row1[2]:
                v_4y = st.number_input(
                    "4年以内",
                    value=_fv("r_4y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_4y', prev):.2f}%" if prev else None,
                )
            with row2[0]:
                v_5y = st.number_input(
                    "5年以内",
                    value=_fv("r_5y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_5y', prev):.2f}%" if prev else None,
                )
            with row2[1]:
                v_6y = st.number_input(
                    "6年以内",
                    value=_fv("r_6y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_6y', prev):.2f}%" if prev else None,
                )
            with row2[2]:
                v_7y = st.number_input(
                    "7年以内",
                    value=_fv("r_7y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_7y', prev):.2f}%" if prev else None,
                )
            with row3[0]:
                v_8y = st.number_input(
                    "8年以内",
                    value=_fv("r_8y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_8y', prev):.2f}%" if prev else None,
                )
            with row3[1]:
                v_9y = st.number_input(
                    "9年以内",
                    value=_fv("r_9y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_9y', prev):.2f}%" if prev else None,
                )
            with row3[2]:
                v_over = st.number_input(
                    "9年超",
                    value=_fv("r_over9y", latest),
                    step=0.01, format="%.2f",
                    help=f"前月: {_fv('r_over9y', prev):.2f}%" if prev else None,
                )

            submitted = st.form_submit_button(
                "登録する", type="primary", use_container_width=True
            )
            if submitted:
                try:
                    datetime.datetime.strptime(target_month, "%Y-%m")
                except ValueError:
                    st.error("月の形式が正しくありません（例: 2026-05）")
                else:
                    upsert_base_rate(
                        month=target_month,
                        note=input_note,
                        r_2y=v_2y, r_3y=v_3y, r_4y=v_4y, r_5y=v_5y,
                        r_6y=v_6y, r_7y=v_7y, r_8y=v_8y, r_9y=v_9y,
                        r_over9y=v_over,
                    )
                    st.success(
                        f"✅ {target_month} の基準金利を登録しました"
                        f"（5年以内: {v_5y:.2f}%）"
                    )
                    st.rerun()

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 登録一覧・グリッド編集", "📥 初期データ一括投入", "📈 金利推移グラフ"])

    # ── タブ1: グリッド編集 ──────────────────────────────────────────────────
    with tab1:
        records = list_base_rates(limit=60)
        if records:
            df = pd.DataFrame(records)
            display_cols = ["month"] + TERM_COLS + ["note"]
            df = df[display_cols]

            column_config: dict = {
                "month": st.column_config.TextColumn("適用月", disabled=True, width="small"),
                "note":  st.column_config.TextColumn("メモ", width="medium"),
            }
            for col, label in _COL_LABELS.items():
                column_config[col] = st.column_config.NumberColumn(
                    label, format="%.2f", step=0.01, min_value=0.0, width="small"
                )

            edited_df = st.data_editor(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="grid_editor",
            )

            if st.button("変更を保存", type="primary", key="save_grid"):
                saved = 0
                for _, row in edited_df.iterrows():
                    orig_rows = [r for r in records if r["month"] == row["month"]]
                    orig = orig_rows[0] if orig_rows else {}
                    changed = any(
                        abs((row.get(c) or 0) - (orig.get(c) or 0)) > 1e-9
                        for c in TERM_COLS
                    ) or row.get("note", "") != orig.get("note", "")
                    if changed:
                        upsert_base_rate(
                            month=row["month"],
                            note=row.get("note") or "",
                            r_2y=row.get("r_2y"),   r_3y=row.get("r_3y"),
                            r_4y=row.get("r_4y"),   r_5y=row.get("r_5y"),
                            r_6y=row.get("r_6y"),   r_7y=row.get("r_7y"),
                            r_8y=row.get("r_8y"),   r_9y=row.get("r_9y"),
                            r_over9y=row.get("r_over9y"),
                        )
                        saved += 1
                if saved:
                    st.success(f"✅ {saved}件を保存しました")
                    st.rerun()
                else:
                    st.info("変更はありませんでした")
        else:
            st.info("登録データがありません。「初期データ一括投入」タブからデータを投入してください。")

    # ── タブ2: 初期データ一括投入 ─────────────────────────────────────────────
    with tab2:
        st.markdown(f"#### 初期データ一括投入（{len(_INITIAL_DATA)}件）")
        st.info(
            "基準金利テーブル（2022/11〜2026/4）を一括登録します。\n"
            "「上書きなし」は既存月をスキップ、「上書きあり」は全件更新します。"
        )

        records_count = len(list_base_rates(limit=100))
        st.metric("現在の登録件数", f"{records_count}件")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("上書きなし投入", type="primary", use_container_width=True):
                ins, skp = seed_initial_data(overwrite=False)
                st.success(f"✅ {ins}件投入、{skp}件スキップ（既存）")
                st.rerun()
        with col_b:
            if st.button("上書きあり投入（全件更新）", use_container_width=True):
                ins, _ = seed_initial_data(overwrite=True)
                st.success(f"✅ {ins}件を上書き更新しました")
                st.rerun()

    # ── タブ3: 金利推移グラフ ────────────────────────────────────────────────
    with tab3:
        import math
        import plotly.graph_objects as go

        records_for_chart = list_base_rates(limit=60)
        if not records_for_chart:
            st.info("データがありません。「初期データ一括投入」タブからデータを投入してください。")
        else:
            records_for_chart.sort(key=lambda r: r["month"])
            months = [r["month"] for r in records_for_chart]

            COLORS = [
                "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444",
                "#ec4899", "#06b6d4", "#84cc16", "#f97316",
            ]

            fig = go.Figure()
            for i, col in enumerate(TERM_COLS):
                y_vals = [r.get(col) for r in records_for_chart]
                fig.add_trace(go.Scatter(
                    x=months,
                    y=y_vals,
                    mode="lines",
                    name=_COL_LABELS[col],
                    line=dict(color=COLORS[i], width=2),
                    connectgaps=True,
                ))

            all_vals = [
                r.get(col)
                for r in records_for_chart
                for col in TERM_COLS
                if r.get(col) is not None
            ]
            if all_vals:
                y_min = math.floor(min(all_vals) * 10) / 10
                y_max = math.ceil(max(all_vals) * 10) / 10
            else:
                y_min, y_max = 0, 5

            fig.update_layout(
                title=f"期間別基準金利 推移グラフ（{len(records_for_chart)}ヶ月）",
                xaxis_title="適用月",
                yaxis_title="金利 (%)",
                xaxis=dict(
                    tickangle=-45,
                    dtick=max(1, len(months) // 10),
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    tickformat=".2f",
                    ticksuffix="%",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                height=480,
                margin=dict(l=60, r=20, t=60, b=80),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True, key="rate_trend_chart")
