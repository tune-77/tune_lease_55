"""
secondary_review.py — 二次審査チェックリスト

ボーダーライン案件（スコア 60〜74）向けに追加確認項目を管理する。
チェック項目は secondary_review_items.json で管理し、UI から追加・削除可能。

expose:
  render_secondary_review_ui(res)   ← Streamlit UI
  load_items() / save_items(items)  ← JSON 操作
"""

from __future__ import annotations
import json
import os
import uuid
from typing import Optional

try:
    from config import BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ITEMS_FILE = os.path.join(BASE_DIR, "data", "secondary_review_items.json")

# ── カテゴリーの表示順 ──────────────────────────────────────
CATEGORY_ORDER = ["財務確認", "信用調査", "担保・保証", "物件・契約", "事業性確認", "コンプライアンス"]
CATEGORY_ICONS = {
    "財務確認":       "📊",
    "信用調査":       "🔍",
    "担保・保証":     "🛡️",
    "物件・契約":     "📋",
    "事業性確認":     "🏭",
    "コンプライアンス": "⚖️",
}


# ─────────────────────────────────────────────────────────────
# JSON 操作
# ─────────────────────────────────────────────────────────────

def load_items() -> list[dict]:
    """チェック項目を JSON から読み込む"""
    if not os.path.isfile(ITEMS_FILE):
        return []
    try:
        with open(ITEMS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_items(items: list[dict]) -> bool:
    """チェック項目を JSON に保存する"""
    try:
        with open(ITEMS_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _get_categories(items: list[dict]) -> list[str]:
    """存在するカテゴリを順序つきで返す"""
    seen = []
    for cat in CATEGORY_ORDER:
        if any(i["category"] == cat for i in items):
            seen.append(cat)
    for item in items:
        cat = item.get("category", "その他")
        if cat not in seen:
            seen.append(cat)
    return seen


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

def render_secondary_review_ui(res: Optional[dict] = None):
    """二次審査チェックリストの Streamlit UI"""
    import streamlit as st

    score = float((res or {}).get("score", 0))
    industry = (res or {}).get("industry_sub", "")
    hantei   = (res or {}).get("hantei", "")

    st.subheader("🔎 二次審査チェックリスト")

    # スコア帯の説明
    if score == 0:
        st.info("審査を実行するとスコアが反映されます。チェック項目の管理・追加は下記から行えます。")
    elif score >= 75:
        st.success(f"スコア {score:.1f} — 通常審査範囲です。二次審査チェックは任意です。")
    elif score >= 60:
        st.warning(
            f"スコア **{score:.1f}** — ボーダーライン案件です。"
            "下記チェックリストを全て確認し、条件付き承認の根拠を整備してください。"
        )
    else:
        st.error(
            f"スコア **{score:.1f}** — 否決圏内です。"
            "重大なリスク項目の確認のみ実施し、否決理由を記録してください。"
        )

    if industry:
        st.caption(f"対象業種：{industry}")

    items = load_items()
    categories = _get_categories(items)

    # セッション初期化（チェック状態）
    if "sr_checks" not in st.session_state:
        st.session_state["sr_checks"] = {}
    if "sr_notes" not in st.session_state:
        st.session_state["sr_notes"] = {}

    # ── チェックリスト表示 ────────────────────────────────────
    total_required = sum(1 for i in items if i.get("required"))
    checked_required = 0
    total_all = len(items)
    checked_all = 0

    for cat in categories:
        cat_items = [i for i in items if i.get("category") == cat]
        if not cat_items:
            continue
        icon = CATEGORY_ICONS.get(cat, "📌")
        cat_checked = sum(1 for i in cat_items if st.session_state["sr_checks"].get(i["id"]))
        label_color = "green" if cat_checked == len(cat_items) else ("orange" if cat_checked > 0 else "gray")

        with st.expander(
            f"{icon} {cat}　　{cat_checked}/{len(cat_items)} 完了",
            expanded=(score >= 60 and score < 75)
        ):
            for item in cat_items:
                iid = item["id"]
                required_tag = " 🔴" if item.get("required") else ""
                checked = st.checkbox(
                    f"{item['text']}{required_tag}",
                    value=st.session_state["sr_checks"].get(iid, False),
                    key=f"sr_chk_{iid}",
                )
                st.session_state["sr_checks"][iid] = checked

                if item.get("note"):
                    st.caption(f"　　💡 {item['note']}")

                # メモ欄
                note_val = st.text_input(
                    "メモ",
                    value=st.session_state["sr_notes"].get(iid, ""),
                    key=f"sr_note_{iid}",
                    placeholder="確認内容・根拠を記入",
                    label_visibility="collapsed",
                )
                st.session_state["sr_notes"][iid] = note_val

                if item.get("required"):
                    if checked:
                        checked_required += 1

    # 全チェック数は items 全体から一括集計（ループ内の中間集計は削除）
    checked_all = sum(1 for i in items if st.session_state["sr_checks"].get(i["id"]))

    # ── 進捗サマリー ──────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        pct_req = int(checked_required / total_required * 100) if total_required else 100
        st.metric("必須項目", f"{checked_required}/{total_required}", f"{pct_req}%")
    with c2:
        pct_all = int(checked_all / total_all * 100) if total_all else 100
        st.metric("全項目", f"{checked_all}/{total_all}", f"{pct_all}%")
    with c3:
        unchecked_required = [
            i["text"] for i in items
            if i.get("required") and not st.session_state["sr_checks"].get(i["id"])
        ]
        if unchecked_required:
            st.metric("未完了必須", len(unchecked_required), delta="要対応", delta_color="inverse")
        else:
            st.metric("必須完了", "✅ 全完了", delta="承認根拠が整いました", delta_color="normal")

    st.progress(pct_req / 100, text=f"必須チェック進捗 {pct_req}%")

    # 最終判定コメント
    if checked_required == total_required and total_required > 0:
        st.success("✅ 必須チェック項目が全て完了しました。審査委員会への提出が可能です。")
    elif unchecked_required:
        st.warning(f"未完了の必須項目：{', '.join(unchecked_required[:3])}{'…' if len(unchecked_required) > 3 else ''}")

    # ── PDF 印刷用メモ出力 ──────────────────────────────────
    with st.expander("📄 チェック結果テキスト出力（印刷・コピー用）"):
        lines = [f"【二次審査チェックリスト】　業種：{industry}　スコア：{score:.1f}"]
        lines.append("=" * 60)
        for cat in categories:
            lines.append(f"\n▼ {cat}")
            for item in [i for i in items if i["category"] == cat]:
                mark = "☑" if st.session_state["sr_checks"].get(item["id"]) else "☐"
                req  = "【必須】" if item.get("required") else "　　　 "
                note = st.session_state["sr_notes"].get(item["id"], "")
                lines.append(f"  {mark} {req} {item['text']}")
                if note:
                    lines.append(f"       → {note}")
        st.text_area("", "\n".join(lines), height=300, key="sr_text_out")

    # ── 項目管理（追加・削除） ────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ チェック項目の管理")
    st.caption("項目を追加・削除できます。変更は `secondary_review_items.json` に保存されます。")

    # 新規追加フォーム
    with st.expander("➕ 新しいチェック項目を追加", expanded=False):
        with st.form("sr_add_form"):
            new_cat = st.selectbox(
                "カテゴリ",
                options=CATEGORY_ORDER + ["その他"],
                key="sr_new_cat",
            )
            new_text = st.text_input("チェック内容 *", key="sr_new_text")
            new_note = st.text_input("補足説明（任意）", key="sr_new_note")
            new_required = st.checkbox("必須項目にする", value=False, key="sr_new_required")
            submitted = st.form_submit_button("追加する", type="primary")
            if submitted:
                if not new_text.strip():
                    st.error("チェック内容を入力してください。")
                else:
                    new_item = {
                        "id":       f"sr_custom_{uuid.uuid4().hex[:8]}",
                        "category": new_cat,
                        "text":     new_text.strip(),
                        "required": new_required,
                        "note":     new_note.strip(),
                    }
                    items.append(new_item)
                    if save_items(items):
                        st.success(f"「{new_text[:30]}」を追加しました。")
                        st.rerun()
                    else:
                        st.error("保存に失敗しました。")

    # 削除フォーム
    with st.expander("🗑️ 項目を削除", expanded=False):
        del_options = {f"[{i['category']}] {i['text'][:50]}": i["id"] for i in items}
        if del_options:
            del_label = st.selectbox("削除する項目を選択", list(del_options.keys()), key="sr_del_select")
            if st.button("この項目を削除", type="secondary", key="sr_del_btn"):
                del_id = del_options[del_label]
                items = [i for i in items if i["id"] != del_id]
                if save_items(items):
                    # セッションからも削除
                    st.session_state["sr_checks"].pop(del_id, None)
                    st.session_state["sr_notes"].pop(del_id, None)
                    st.success("削除しました。")
                    st.rerun()
                else:
                    st.error("保存に失敗しました。")
        else:
            st.caption("登録済みの項目がありません。")

    # チェック状態リセット
    st.divider()
    if st.button("🔄 チェック状態をリセット（次の案件用）", key="sr_reset"):
        st.session_state["sr_checks"] = {}
        st.session_state["sr_notes"]  = {}
        st.rerun()
