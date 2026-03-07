"""
report_generator.py — 審査結果レポート自動生成モジュール

テンプレート方式（LLM不要）でレポートを生成する。
ユーモアコメントは humor_comments.json からマッチング方式で取得。
スタイルを HUMOR_STYLE_STANDARD / HUMOR_STYLE_YANAMI で切り替え可能。

想定する result オブジェクト: montecarlo.SimResult
  result.company.name          : 企業名 (str)
  result.company.industry      : 業種名 (str)  例: "製造業", "小売業"
  result.company.revenue       : 売上高 (円)
  result.company.lease_amount  : リース希望額 (円)
  result.company.lease_months  : リース期間 (月)
  result.default_prob          : デフォルト確率 (0〜1)
  result.risk_level            : リスクレベル ("低リスク"/"中リスク"/"高リスク"/"極高リスク")
  result.score_median          : 総合スコア中央値 (0〜100)
  result.time_series_default_prob : 累積デフォルト確率の時系列 (np.ndarray)
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

# ── データファイルのデフォルトパス ──────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── ユーモアスタイル定数 ────────────────────────────────────────────
HUMOR_STYLE_STANDARD = "standard"
HUMOR_STYLE_YANAMI   = "yanami"

HUMOR_FILES = {
    HUMOR_STYLE_STANDARD: os.path.join(_DATA_DIR, "humor_comments.json"),
    HUMOR_STYLE_YANAMI:   os.path.join(_DATA_DIR, "humor_comments_yanami.json"),
}

_DEFAULT_TRENDS_FILE = os.path.join(_DATA_DIR, "industry_trends.json")


# =====================
# 1. 指標サマリー
# =====================
def generate_metrics_summary(result) -> str:
    """デフォルト確率・財務スコア・リース依存度等を文章化する（テンプレート方式）。"""
    level = result.risk_level
    prob  = result.default_prob
    score = result.score_median

    # リース依存度: リース希望額 ÷ 売上高
    rev = result.company.revenue
    lease_ratio = (result.company.lease_amount / rev) if rev > 0 else 0.0

    return (
        "\n【指標サマリー】\n"
        f"  - リスクレベル  : {level}\n"
        f"  - デフォルト確率: {prob:.2%}\n"
        f"  - 総合スコア   : {score:.1f}点\n"
        f"  - リース依存度  : {lease_ratio:.1%}（リース額／売上高）\n"
    )


# =====================
# 2. 業界動向（RAG方式）
# =====================
def load_industry_trends(
    industry: str,
    trends_file: str = _DEFAULT_TRENDS_FILE,
) -> str:
    """
    業種名をキーに industry_trends.json から動向テキストを返す。
    キーは CompanyData.industry（モンテカルロ業種名）と一致させること。
    """
    try:
        with open(trends_file, encoding="utf-8") as f:
            trends = json.load(f)
        return trends.get(industry, "（該当業種の動向データなし）")
    except FileNotFoundError:
        return "（業界動向データファイルが見つかりません）"
    except Exception as e:
        return f"（業界動向データ読み込みエラー: {e}）"


# =====================
# 3. 承認に必要な追加情報
# =====================
# 各項目末尾の (BN: ○○) はベイジアンネットワークの対応証拠変数または二次審査コード。
# BN逆転提案の優先順位: Main_Bank_Support > Co_Lease > Parent_Guarantor >
#                        Related_Assets > Related_Bank_Status >
#                        Shorter_Lease_Term > One_Time_Deal
APPROVAL_CHECKLIST: dict[str, list[str]] = {
    "極高リスク": [
        # ── 書類確認（sr001〜sr004） ──────────────────────────────
        "直近3期分の決算書・試算表（原本確認必須）(sr001/sr002)",
        "負債内訳・返済条件の精査（銀行借入・リース・その他）(sr004)",
        "直近3ヶ月の預金残高推移または資金繰り表 (sr003)",
        # ── BN逆転提案：信用補完（優先度順） ────────────────────
        "【BN最優先】メイン銀行支援の確約取得（単独で承認確率を大幅改善） (BN: Main_Bank_Support)",
        "【BN逆転】銀行との50%協調リース設定の検討 (BN: Co_Lease)",
        "【BN逆転】親会社・グループ会社による連帯保証の取得 (BN: Parent_Guarantor)",
        # ── BN逆転提案：関係者・ヘッジ ──────────────────────────
        "関係者（代表者・親族等）の個人資産・路線価確認 (BN: Related_Assets)",
        "関係者のメイン銀行取引状況・信用力確認 (BN: Related_Bank_Status)",
        # ── BN逆転提案：取引条件 ─────────────────────────────────
        "リース期間の大幅短縮（60ヶ月→36ヶ月等）の検討 (BN: Shorter_Lease_Term)",
        "業況改善まで本件限りの設定・確認 (BN: One_Time_Deal)",
        # ── 物件・信用調査 ───────────────────────────────────────
        "債務超過の有無・規模の精査（BNシミュレーターへ入力）(BN: Insolvent_Status)",
        "物件が本業に不可欠かの確認（BN中間スコア: 物件価値）(BN: Core_Business_Use)",
        "物件の中古市場流動性・残存価値の確認 (BN: Asset_Liquidity)",
        "信用情報機関の照会・延滞事故情報の確認 (sr005/sr007)",
        "反社会的勢力チェックの実施 (sr018)",
        # ── 組織対応 ─────────────────────────────────────────────
        "上位役員によるダブルチェック（承認確率 < 40% の場合は必須）",
        "早期警戒リストへの登録および四半期モニタリングの設定",
    ],
    "高リスク": [
        # ── 書類確認（sr001〜sr004） ──────────────────────────────
        "直近2期分の決算書・試算表（原本確認）(sr001/sr002)",
        "負債内訳・返済条件の確認 (sr004)",
        # ── BN逆転提案：信用補完（優先度順） ────────────────────
        "【BN推奨】メイン銀行支援の有無・取得可否を確認 (BN: Main_Bank_Support)",
        "銀行との協調リース設定の検討 (BN: Co_Lease)",
        "親会社・上位法人による連帯保証の確認 (BN: Parent_Guarantor)",
        # ── BN逆転提案：関係者・ヘッジ ──────────────────────────
        "関係者の個人資産・銀行取引状況の確認 (BN: Related_Assets / Related_Bank_Status)",
        "リース期間短縮の交渉・検討 (BN: Shorter_Lease_Term)",
        "本件限り設定の検討（業況回復後に再申請前提）(BN: One_Time_Deal)",
        # ── 物件・信用調査 ───────────────────────────────────────
        "物件が本業に不可欠かの確認 (BN: Core_Business_Use)",
        "物件の中古流動性・残価確認 (BN: Asset_Liquidity)",
        "債務超過の有無確認（BNシミュレーターへ入力）(BN: Insolvent_Status)",
        "信用調査機関（帝国DB等）による与信確認 (sr005/sr006)",
        "反社会的勢力チェックの実施 (sr018)",
        # ── 組織対応 ─────────────────────────────────────────────
        "上位者へのエスカレーションおよび半期モニタリングの設定",
    ],
    "中リスク": [
        # ── 書類確認（sr001〜sr002） ──────────────────────────────
        "直近1期分の決算書・試算表の確認 (sr001/sr002)",
        # ── BN確認項目 ───────────────────────────────────────────
        "関係者の銀行取引状況・個人資産の確認（BN中間スコア: 信用力に影響）(BN: Related_Bank_Status / Related_Assets)",
        "物件が本業に不可欠かの確認（BN中間スコア: 物件価値）(BN: Core_Business_Use)",
        "物件の中古市場流動性の確認 (BN: Asset_Liquidity)",
        "リース期間短縮の余地確認 (BN: Shorter_Lease_Term)",
        # ── 任意強化 ─────────────────────────────────────────────
        "事業計画書または主要取引先状況の提出依頼（任意）(sr015/sr016)",
        "半期ごとの財務状況モニタリングの設定",
    ],
    "低リスク": [
        # ── 標準書類（sr001・sr011・sr014・sr017・sr018） ─────────
        "標準書類一式の確認（決算書・登記簿謄本・印鑑証明）(sr001/sr017)",
        "物件の用途・設置場所・管理方法の確認 (sr011/sr014)",
        "反社会的勢力チェックの実施 (sr018)",
        # ── 通常モニタリング ─────────────────────────────────────
        "年次モニタリング計画の設定",
    ],
}


def generate_approval_checklist(risk_level: str) -> str:
    """リスクレベルに応じた承認チェックリストを生成する（テンプレート方式）。"""
    items    = APPROVAL_CHECKLIST.get(risk_level, ["（リスクレベル不明）"])
    checklist = "\n".join(f"  □ {item}" for item in items)
    return f"\n【承認に必要な追加情報】\n{checklist}\n"


# =====================
# 4. 今後の見込み
# =====================
def generate_outlook(result) -> str:
    """モンテカルロシミュレーション結果（デフォルト確率）を文章化する（テンプレート方式）。"""
    prob = result.default_prob

    if prob < 0.05:
        outlook = (
            "今後も安定的な契約継続が見込まれます。"
            "定期モニタリングで対応可能な水準です。"
        )
    elif prob < 0.15:
        outlook = (
            "一定のリスクが存在するため、"
            "半期ごとの財務状況確認を推奨します。"
        )
    elif prob < 0.30:
        outlook = (
            "業績悪化の兆候が見られます。"
            "早期警戒リストへの登録を検討し、"
            "四半期ごとの財務モニタリングを実施してください。"
        )
    else:
        outlook = (
            "高確率でデフォルトリスクがあります。"
            "契約条件の見直しまたは保全措置を至急検討してください。"
            "継続審査と担保充実が不可欠です。"
        )

    # 時系列が利用可能な場合は最終月の確率も補足
    ts = getattr(result, "time_series_default_prob", None)
    if ts is not None and len(ts) > 1:
        months = len(ts) - 1
        final_prob = float(ts[-1])
        outlook += (
            f"\n  （参考）シミュレーション期間 {months}ヶ月後の累積デフォルト確率: "
            f"{final_prob:.2%}"
        )

    return f"\n【今後の見込み】\n{outlook}\n"


# =====================
# 5. ユーモアコメント（テンプレートマッチング方式）
# =====================
def generate_humor_comment(result, style: str = HUMOR_STYLE_STANDARD) -> str:
    """
    humor_comments.json（または yanami版）からマッチングしてコメントを返す。

    マッチング優先順位:
      1. リスクレベル × 業種（完全一致）
      2. リスクレベル × 全業種
      3. リスクレベルのみ
      4. 上記すべて不一致 → エラーメッセージ

    Args:
        result: SimResult オブジェクト
        style:  HUMOR_STYLE_STANDARD または HUMOR_STYLE_YANAMI
    """
    humor_file = HUMOR_FILES.get(style, HUMOR_FILES[HUMOR_STYLE_STANDARD])
    try:
        with open(humor_file, encoding="utf-8") as f:
            data = json.load(f)
        comments = data.get("comments", [])

        risk     = result.risk_level
        industry = result.company.industry

        # 1. リスクレベル × 業種（完全一致）
        matched = [
            c for c in comments
            if c.get("risk") == risk and c.get("industry") == industry
        ]
        # 2. リスクレベル × 全業種
        if not matched:
            matched = [
                c for c in comments
                if c.get("risk") == risk and c.get("industry") == "全業種"
            ]
        # 3. リスクレベルのみ
        if not matched:
            matched = [c for c in comments if c.get("risk") == risk]

        if matched:
            chosen = random.choice(matched)
            return f"\n【ひとことコメント】\n{chosen.get('comment', '')}\n"
        else:
            return "\n【ひとことコメント】\n（該当するコメントが見つかりませんでした）\n"

    except FileNotFoundError:
        return f"\n【ひとことコメント】\n（{os.path.basename(humor_file)} が見つかりません）\n"
    except Exception as e:
        return f"\n【ひとことコメント】\n（エラー: {e}）\n"


# =====================
# 統合レポート生成
# =====================
def generate_full_report(
    result,
    style: str = HUMOR_STYLE_STANDARD,
    industry_trends_file: str = _DEFAULT_TRENDS_FILE,
) -> str:
    """
    審査結果レポートを生成して文字列で返す。

    Args:
        result:               SimResult オブジェクト
        style:                ユーモアコメントスタイル (HUMOR_STYLE_STANDARD / HUMOR_STYLE_YANAMI)
        industry_trends_file: 業界動向JSONファイルのパス
    """
    sep = "=" * 50
    report  = f"{sep}\n"
    report += f"  審査結果レポート：{result.company.name}\n"
    report += f"{sep}\n"

    report += generate_metrics_summary(result)
    report += (
        f"\n【業界動向】\n"
        f"{load_industry_trends(result.company.industry, industry_trends_file)}\n"
    )
    report += generate_approval_checklist(result.risk_level)
    report += generate_outlook(result)
    report += generate_humor_comment(result, style=style)

    return report


# =====================
# 通常審査結果 dict アダプター
# =====================
def generate_full_report_from_res(res: dict, session_state) -> str:
    """
    通常の審査結果 dict (st.session_state['last_result']) から
    generate_full_report と同等のレポートを生成する。

    Args:
        res:          st.session_state['last_result'] の dict
        session_state: st.session_state（企業名・入力値の取得に使用）
    """
    from types import SimpleNamespace
    from montecarlo import map_industry_from_major

    # 業種名変換（"D 建設業" → "建設業" 等）
    industry_major = res.get("industry_major", "")
    industry = map_industry_from_major(industry_major)

    # 企業名取得
    company_name = session_state.get("rep_company") or "（企業名未入力）"

    # 入力値取得（万円 → 円）
    inputs = session_state.get("last_submitted_inputs") or {}
    revenue      = float(inputs.get("nenshu",          0) or 0) * 10_000
    lease_amount = float(inputs.get("acquisition_cost", 0) or 0) * 10_000
    lease_months = int(inputs.get("lease_term", 36) or 36)
    nenshu_val   = float(inputs.get("nenshu", 1) or 1) or 1.0
    rieki_val    = float(inputs.get("rieki",  0) or 0)
    op_margin    = rieki_val / nenshu_val
    equity_ratio = float(res.get("user_eq", 0) or 0) / 100
    total_debt   = float(inputs.get("bank_credit", 0) or 0) * 10_000

    # デフォルト確率・リスクレベル（montecarlo._risk_level と同基準）
    default_prob = float(res.get("pd_percent", 0) or 0) / 100
    if default_prob < 0.05:
        risk_level = "低リスク"
    elif default_prob < 0.15:
        risk_level = "中リスク"
    elif default_prob < 0.30:
        risk_level = "高リスク"
    else:
        risk_level = "極高リスク"

    score_median = float(res.get("score", 0) or 0)

    company = SimpleNamespace(
        name=company_name,
        industry=industry,
        revenue=revenue,
        operating_margin=op_margin,
        equity_ratio=equity_ratio,
        total_debt=total_debt,
        lease_amount=lease_amount,
        lease_months=lease_months,
    )
    result = SimpleNamespace(
        company=company,
        default_prob=default_prob,
        risk_level=risk_level,
        score_median=score_median,
        time_series_default_prob=None,
    )

    style = session_state.get("humor_style", HUMOR_STYLE_STANDARD)
    return generate_full_report(result, style=style)
