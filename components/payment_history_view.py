"""
支払い実績管理画面。

成約後の実際の支払い状況を時系列で蓄積し、将来の与信モデル再学習
（現在の成約/失注ラベル → 支払い実績ラベル）に使える基盤を提供する。
"""
import datetime
import json
import os
import sqlite3
from contextlib import closing
from typing import Optional

import pandas as pd
import streamlit as st
from runtime_paths import get_data_path

_LEASE_DB_PATH = get_data_path("lease_data.db")

PAYMENT_STATUS_OPTIONS = ["正常", "延滞", "デフォルト", "完済"]
_STATUS_ICON = {"正常": "🟢", "延滞": "🟡", "デフォルト": "🔴", "完済": "🔵"}


# ── DB ヘルパー ───────────────────────────────────────────────────────────────

def _ensure_table() -> None:
    """payment_history テーブルが存在しなければ作成する（冪等）"""
    if not os.path.exists(_LEASE_DB_PATH):
        return
    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS payment_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_id     TEXT    NOT NULL,
                check_date      TEXT    NOT NULL,
                payment_status  TEXT    NOT NULL,
                overdue_amount  INTEGER DEFAULT 0,
                model_version   TEXT    DEFAULT '',
                screening_score REAL,
                notes           TEXT    DEFAULT '',
                created_at      TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ph_contract_id "
            "ON payment_history(contract_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ph_check_date "
            "ON payment_history(check_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ph_payment_status "
            "ON payment_history(payment_status)"
        )
        conn.commit()


def _get_contracted_cases() -> list[dict]:
    """成約・検収完了案件の一覧を返す（フォーム選択肢用）"""
    if not os.path.exists(_LEASE_DB_PATH):
        return []
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, industry_sub, score, final_status, data "
                "FROM past_cases "
                "WHERE final_status IN ('成約', '検収完了', '検収') "
                "ORDER BY timestamp DESC"
            ).fetchall()
    except Exception:
        return []

    result = []
    for r in rows:
        try:
            data = json.loads(r["data"] or "{}")
        except Exception:
            data = {}
        company_name = data.get("company_name", "")
        company_no = data.get("company_no", "")
        industry = r["industry_sub"] or data.get("industry_sub", "")
        label = company_name or industry or "—"
        if company_no:
            label = f"[{company_no}] {label}"
        result.append({
            "id": r["id"],
            "display": f"[{str(r['id'])[:8]}] {label}",
            "score": r["score"],
            "final_status": r["final_status"],
        })
    return result


def _add_record(
    contract_id: str,
    check_date: str,
    payment_status: str,
    overdue_amount: int,
    model_version: str,
    screening_score: Optional[float],
    notes: str,
) -> int:
    _ensure_table()
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
        cur = conn.execute(
            """INSERT INTO payment_history
               (contract_id, check_date, payment_status, overdue_amount,
                model_version, screening_score, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (contract_id, check_date, payment_status, overdue_amount,
             model_version, screening_score, notes, created_at),
        )
        new_id = cur.lastrowid
        conn.commit()
    return new_id


def _get_records(
    contract_id: str = "",
    payment_status: str = "",
    limit: int = 300,
) -> list[dict]:
    _ensure_table()
    if not os.path.exists(_LEASE_DB_PATH):
        return []
    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        where, params = [], []
        if contract_id:
            where.append("contract_id = ?")
            params.append(contract_id)
        if payment_status:
            where.append("payment_status = ?")
            params.append(payment_status)
        sql = "SELECT * FROM payment_history"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY check_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _delete_record(record_id: int) -> bool:
    _ensure_table()
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            conn.execute("DELETE FROM payment_history WHERE id = ?", (record_id,))
            conn.commit()
        return True
    except Exception:
        return False


# ── リトレーニング用ヘルパー ──────────────────────────────────────────────────

def extract_training_data() -> Optional[pd.DataFrame]:
    """
    支払い実績からリトレーニング用の学習データセットを生成する。

    Returns
    -------
    pd.DataFrame | None
        契約ごとの最悪ステータスをラベルとして付与したデータフレーム。
        columns: contract_id, check_date, payment_status, overdue_amount,
                 screening_score, model_version, label,
                 nenshu, item4_ord_profit, equity_ratio, lease_credit, industry_sub

        label: 0 = 正常 or 完済, 1 = デフォルト
               延滞は label=0 だが overdue_amount > 0 で区別可能。
    """
    _ensure_table()
    if not os.path.exists(_LEASE_DB_PATH):
        return None

    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        ph_rows = conn.execute("SELECT * FROM payment_history ORDER BY check_date").fetchall()
        if not ph_rows:
            return None

        ph_df = pd.DataFrame([dict(r) for r in ph_rows])

        # 元の入力データを past_cases から結合
        unique_ids = list(ph_df["contract_id"].unique())
        pc_rows = conn.execute(
            "SELECT id, data FROM past_cases WHERE id IN ({})".format(
                ",".join("?" * len(unique_ids))
            ),
            unique_ids,
        ).fetchall()

    pc_map: dict[str, dict] = {}
    for r in pc_rows:
        try:
            pc_map[r["id"]] = json.loads(r["data"] or "{}")
        except Exception:
            pc_map[r["id"]] = {}

    def _label(status: str) -> int:
        return 1 if status == "デフォルト" else 0

    ph_df["label"] = ph_df["payment_status"].apply(_label)

    def _expand_inputs(row: pd.Series) -> pd.Series:
        raw = pc_map.get(row["contract_id"], {})
        inp = raw.get("inputs") or raw
        return pd.Series({
            "nenshu": inp.get("nenshu"),
            "item4_ord_profit": inp.get("item4_ord_profit") or inp.get("rieki"),
            "equity_ratio": inp.get("equity_ratio"),
            "lease_credit": inp.get("lease_credit"),
            "industry_sub": raw.get("industry_sub", ""),
        })

    extra = ph_df.apply(_expand_inputs, axis=1)
    return pd.concat(
        [
            ph_df[[
                "contract_id", "check_date", "payment_status",
                "overdue_amount", "screening_score", "model_version", "label",
            ]],
            extra,
        ],
        axis=1,
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def render_payment_history() -> None:
    """支払い実績管理画面を描画する"""
    st.title("💳 支払い実績管理")
    st.info(
        "成約後の実際の支払い状況を記録・管理します。"
        "蓄積データは将来の与信モデル再学習（デフォルト予測）に活用できます。"
    )

    if not os.path.exists(_LEASE_DB_PATH):
        st.error(f"DBファイルが見つかりません: `{_LEASE_DB_PATH}`")
        return

    _ensure_table()

    tab_list, tab_add, tab_export = st.tabs(
        ["📋 実績一覧", "➕ 実績登録", "🔬 リトレーニング用エクスポート"]
    )

    # ── 実績一覧 ──────────────────────────────────────────────────────────────
    with tab_list:
        st.subheader("📋 支払い実績一覧")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_status = st.selectbox(
                "ステータスで絞り込み",
                ["全て"] + PAYMENT_STATUS_OPTIONS,
                key="ph_filter_status",
            )
        with col_f2:
            filter_contract = st.text_input(
                "契約ID で絞り込み（完全一致）",
                placeholder="past_cases.id をそのまま入力",
                key="ph_filter_contract",
            )

        records = _get_records(
            contract_id=filter_contract.strip(),
            payment_status="" if filter_status == "全て" else filter_status,
        )

        if not records:
            st.caption("実績データがありません。「実績登録」タブから追加してください。")
        else:
            df = pd.DataFrame(records)
            df["状態"] = df["payment_status"].map(
                lambda s: f"{_STATUS_ICON.get(s, '⚪')} {s}"
            )
            df["延滞金額（千円）"] = df["overdue_amount"].fillna(0).astype(int)
            disp = df.rename(columns={
                "id": "ID", "contract_id": "契約ID", "check_date": "記録日",
                "screening_score": "審査スコア", "model_version": "モデルVer.",
                "notes": "備考",
            })[["ID", "契約ID", "記録日", "状態", "延滞金額（千円）",
                 "審査スコア", "モデルVer.", "備考"]]
            st.dataframe(disp, use_container_width=True, hide_index=True)
            st.caption(f"合計 {len(records)} 件")

            with st.expander("📊 ステータス集計", expanded=False):
                summary = df["payment_status"].value_counts().reset_index()
                summary.columns = ["ステータス", "件数"]
                summary["割合(%)"] = (summary["件数"] / len(df) * 100).round(1)
                st.dataframe(summary, use_container_width=True, hide_index=True)

                n_default = int((df["payment_status"] == "デフォルト").sum())
                c1, c2 = st.columns(2)
                c1.metric("デフォルト件数", f"{n_default} 件")
                c2.metric("デフォルト率", f"{n_default / len(df) * 100:.1f} %")

            with st.expander("🗑️ レコード削除", expanded=False):
                del_id = st.number_input(
                    "削除する ID", min_value=1, step=1, key="ph_del_id"
                )
                if st.button("削除する", type="secondary", key="ph_del_btn"):
                    if _delete_record(int(del_id)):
                        st.success(f"ID {int(del_id)} を削除しました。")
                        st.rerun()
                    else:
                        st.error("削除に失敗しました。")

    # ── 実績登録 ──────────────────────────────────────────────────────────────
    with tab_add:
        st.subheader("➕ 支払い実績を登録")

        contracted = _get_contracted_cases()
        if not contracted:
            st.warning(
                "成約・検収完了の案件がありません。"
                "先に「📝 結果登録 (成約/失注)」で成約を登録してください。"
            )
        else:
            case_map = {c["display"]: c for c in contracted}

            with st.form("payment_history_form"):
                selected_disp = st.selectbox(
                    "契約案件を選択",
                    list(case_map.keys()),
                    help="成約・検収完了済みの案件のみ表示されます",
                )
                sel = case_map[selected_disp]

                col1, col2 = st.columns(2)
                with col1:
                    check_date = st.date_input(
                        "記録日", value=datetime.date.today()
                    )
                    payment_status = st.selectbox(
                        "支払いステータス",
                        PAYMENT_STATUS_OPTIONS,
                        help=(
                            "正常: 期日通り支払い済み  /  延滞: 遅延あり  /  "
                            "デフォルト: 債務不履行  /  完済: 全額完済"
                        ),
                    )
                with col2:
                    overdue_amount = st.number_input(
                        "延滞金額（千円）",
                        min_value=0,
                        value=0,
                        step=10,
                        help="正常・完済の場合は 0 のまま",
                    )
                    screening_score = st.number_input(
                        "審査スコア（スナップショット）",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(sel.get("score") or 0.0),
                        step=0.1,
                        format="%.1f",
                        help="審査時のスコアが自動入力されます",
                    )

                model_version = st.text_input(
                    "モデルバージョン",
                    value="v1.0",
                    placeholder="例: v1.0, v2025-05",
                )
                notes = st.text_area(
                    "備考（自由記述）",
                    placeholder="例: 3ヶ月延滞後に回収完了、担当者変更あり 等",
                    max_chars=500,
                )

                if st.form_submit_button("💾 実績を保存", type="primary"):
                    new_id = _add_record(
                        contract_id=sel["id"],
                        check_date=check_date.isoformat(),
                        payment_status=payment_status,
                        overdue_amount=int(overdue_amount),
                        model_version=model_version.strip(),
                        screening_score=float(screening_score),
                        notes=notes.strip(),
                    )
                    st.success(f"✅ 実績を保存しました（ID: {new_id}）")
                    st.rerun()

    # ── リトレーニング用エクスポート ──────────────────────────────────────────
    with tab_export:
        st.subheader("🔬 リトレーニング用データエクスポート")
        st.info(
            "支払い実績（デフォルト=1, 正常/完済=0）をラベルとして付与した"
            "学習データセットを生成します。将来の与信モデル再学習に使用します。"
        )

        if st.button("📊 学習データを生成", key="ph_gen_train"):
            with st.spinner("データを生成中..."):
                df_train = extract_training_data()

            if df_train is None or df_train.empty:
                st.warning("支払い実績データがまだありません。「実績登録」で追加してください。")
            else:
                st.success(f"✅ {len(df_train)} 件の学習データを生成しました")

                c1, c2, c3 = st.columns(3)
                c1.metric("総件数", len(df_train))
                c2.metric(
                    "デフォルト (label=1)",
                    int((df_train["label"] == 1).sum()),
                )
                c3.metric(
                    "正常/完済 (label=0)",
                    int((df_train["label"] == 0).sum()),
                )

                st.dataframe(df_train, use_container_width=True, hide_index=True)

                csv = df_train.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 CSV をダウンロード",
                    data=csv,
                    file_name=f"payment_history_train_{datetime.date.today()}.csv",
                    mime="text/csv",
                )

        with st.expander("ℹ️ リトレーニング手順（将来の実装ガイド）", expanded=False):
            st.markdown("""
### リトレーニング手順

1. **データ収集**: 本画面で支払い実績を蓄積（目安: デフォルト事例30件以上）
2. **エクスポート**: 「学習データを生成」でCSVを取得
3. **特徴量選定**: `nenshu`・`equity_ratio`・`lease_credit`・`screening_score` 等
4. **ラベル設定**: `label` 列（0=正常, 1=デフォルト）を目的変数に使用
5. **再学習**: `train_quantum.py` または新規スクリプトで LightGBM / LogisticRegression を再学習
6. **モデル保存**: `data/payment_risk_model.joblib` として保存
7. **スコアリング連携**: `scoring_core.py` の `payment_risk_score` フィールドに組み込む

> **注意**: デフォルト事例は稀なため、クラス不均衡補正（`class_weight='balanced'`）を推奨します。
            """)
