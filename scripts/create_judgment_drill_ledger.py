#!/usr/bin/env python3
"""Create a 1000-row user judgment drill ledger.

The generated rows are synthetic case prompts with blank user judgment fields.
They are input stock, not validated field feedback.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "judgment_drills" / "judgment_drill_1000_20260725.csv"
DEFAULT_GUIDE = PROJECT_ROOT / "reports" / "judgment_drill_1000_input_ledger_20260725.md"
DEFAULT_LATEST_GUIDE = PROJECT_ROOT / "reports" / "judgment_drill_1000_input_ledger_latest.md"

FIELDNAMES = [
    "schema_version",
    "case_id",
    "source_kind",
    "created_date",
    "batch_no",
    "status",
    "case_theme",
    "industry",
    "company_stage",
    "bank_is_main",
    "bank_credit_balance_million_yen",
    "lease_customer_status",
    "existing_lease_credit_balance_million_yen",
    "equity_ratio_percent",
    "asset_type",
    "amount_million_yen",
    "lease_term_years",
    "purpose",
    "financial_snapshot",
    "case_context",
    "positive_points",
    "risk_points",
    "credit_score_20",
    "repayment_source_score_20",
    "asset_exit_score_20",
    "plan_specificity_score_20",
    "uncertainty_control_score_20",
    "total_score_100",
    "user_decision",
    "heaviest_issue",
    "additional_checks",
    "ringi_sentence",
    "score_decision_gap_note",
    "ai_feedback_outcome",
    "ai_feedback_note",
    "okf_candidate_tags",
    "presentation_use_ok",
    "reviewer",
    "judged_at",
]


@dataclass(frozen=True)
class Pattern:
    theme: str
    industry: str
    asset: str
    purpose: str
    positive: str
    risk: str
    context: str


PATTERNS = [
    Pattern(
        "補助金前提",
        "製造業",
        "CNC工作機械",
        "老朽設備更新と増産対応",
        "既存主力先からの受注残があり、設備必要性は明確。",
        "補助金採択前にリース開始予定で、未採択時の資金繰り説明が弱い。",
        "社長は採択前提で説明しているが、銀行支援は口頭段階。",
    ),
    Pattern(
        "新店舗計画",
        "飲食業",
        "厨房設備一式",
        "新店舗出店",
        "既存店舗は黒字で、代表に運営経験がある。",
        "新店舗売上計画が既存店実績より強く、人員確保も未確定。",
        "立地評価は高いが、回転率・客単価・原価率の資料が薄い。",
    ),
    Pattern(
        "増額更新",
        "食品製造業",
        "包装ライン",
        "既存設備更新",
        "故障頻度と停止時間が増え、更新合理性はある。",
        "月額負担が上がる一方、改善効果がメーカー資料中心。",
        "旧設備の廃棄費用・下取り可否が未確認。",
    ),
    Pattern(
        "新規契約前提",
        "運送業",
        "大型トラック",
        "新規荷主ルート対応",
        "既存車両の稼働率は高く、返済履歴も良好。",
        "荷主契約が内諾段階で、燃料費上昇時の採算余力が薄い。",
        "契約不成立時は既存業務に転用可能だが採算は下がる。",
    ),
    Pattern(
        "購入選択権",
        "卸売業",
        "フォークリフト",
        "倉庫能力維持",
        "倉庫稼働は安定し、物件の必要性は明確。",
        "購入選択権価格が低く、残価・満了後出口の説明が弱い。",
        "使用時間が長く、バッテリー交換費用の負担確認が必要。",
    ),
    Pattern(
        "医療機器更新",
        "医療・福祉",
        "画像診断装置",
        "診療能力維持と待ち時間短縮",
        "地域需要があり、既存患者基盤も安定している。",
        "診療報酬改定や保守費用を織り込んだ採算表が不足。",
        "メーカー保守契約は提示済みだが、更新後の稼働計画は概算。",
    ),
    Pattern(
        "建機高稼働",
        "建設業",
        "油圧ショベル",
        "受注現場対応",
        "公共工事の受注残があり、機械稼働の見込みはある。",
        "工期後の転用先と中古価格下落時の出口説明が弱い。",
        "代表は中古売却可能と説明するが、査定根拠は未取得。",
    ),
    Pattern(
        "IT設備刷新",
        "情報通信業",
        "サーバー・ネットワーク機器",
        "顧客案件増加に伴う基盤増強",
        "継続契約が複数あり、粗利率は比較的高い。",
        "技術陳腐化が早く、期間と経済的寿命の整合確認が必要。",
        "保守契約と入替計画はあるが、解約時の転用価値は限定的。",
    ),
    Pattern(
        "美容店舗拡張",
        "生活関連サービス業",
        "美容機器",
        "新メニュー開始",
        "既存顧客があり、単価上昇の余地がある。",
        "新メニュー売上は口頭見込み中心で、解約率・稼働率が未整理。",
        "機器は専門性が高く、撤退時の換金性に幅がある。",
    ),
    Pattern(
        "農業設備",
        "農業",
        "選果機",
        "出荷能力向上",
        "出荷先との取引は長く、季節需要も見込める。",
        "収穫量・天候影響に左右され、年間資金繰りの波が大きい。",
        "補助金と組合支援の説明はあるが、入金時期が未確定。",
    ),
]

STAGES = ["安定黒字", "低利益", "赤字改善中", "急成長", "事業承継直後"]
BANK_MAIN_STATUS = ["メイン", "非メイン"]
LEASE_CUSTOMER_STATUS = ["既存先", "新規先"]


def _amount(index: int, pattern_index: int) -> int:
    base = [12, 16, 24, 32, 36, 48, 62, 85]
    return base[(index + pattern_index) % len(base)]


def _term(index: int) -> int:
    return [4, 5, 6, 7][index % 4]


def _bank_credit_balance(index: int) -> int:
    return [0, 20, 35, 50, 80, 120, 180, 250][index % 8]


def _lease_credit_balance(index: int, customer_status: str) -> int:
    if customer_status == "新規先":
        return 0
    return [5, 12, 18, 25, 40, 60, 90][index % 7]


def _equity_ratio(index: int) -> int:
    return [8, 12, 16, 20, 24, 30, 36][index % 7]


def _financial_snapshot(index: int, stage: str, equity_ratio: int) -> str:
    sales = 90 + (index % 17) * 35
    profit = [-800, 300, 900, 1600, 2400, 3600][index % 6]
    debt = 2500 + (index % 13) * 900
    return (
        f"直近期売上{sales}百万円、営業利益{profit}万円、借入金{debt}万円、"
        f"自己資本比率{equity_ratio}%、ステージ={stage}。"
    )


def build_rows(count: int, created: date) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for zero_based in range(count):
        number = zero_based + 1
        pattern = PATTERNS[zero_based % len(PATTERNS)]
        stage = STAGES[(zero_based // len(PATTERNS)) % len(STAGES)]
        bank_is_main = BANK_MAIN_STATUS[(zero_based // 3) % len(BANK_MAIN_STATUS)]
        lease_customer_status = LEASE_CUSTOMER_STATUS[(zero_based // 5) % len(LEASE_CUSTOMER_STATUS)]
        bank_credit_balance = _bank_credit_balance(zero_based)
        lease_credit_balance = _lease_credit_balance(zero_based, lease_customer_status)
        equity_ratio = _equity_ratio(zero_based)
        amount = _amount(zero_based, zero_based % len(PATTERNS))
        term = _term(zero_based)
        case_id = f"CASE-JD-{created:%Y%m%d}-{number:04d}"
        rows.append(
            {
                "schema_version": "1",
                "case_id": case_id,
                "source_kind": "demo_user_judgment_drill",
                "created_date": created.isoformat(),
                "batch_no": str((zero_based // 50) + 1),
                "status": "pending_user_judgment",
                "case_theme": pattern.theme,
                "industry": pattern.industry,
                "company_stage": stage,
                "bank_is_main": bank_is_main,
                "bank_credit_balance_million_yen": str(bank_credit_balance),
                "lease_customer_status": lease_customer_status,
                "existing_lease_credit_balance_million_yen": str(lease_credit_balance),
                "equity_ratio_percent": str(equity_ratio),
                "asset_type": pattern.asset,
                "amount_million_yen": str(amount),
                "lease_term_years": str(term),
                "purpose": pattern.purpose,
                "financial_snapshot": _financial_snapshot(zero_based, stage, equity_ratio),
                "case_context": pattern.context,
                "positive_points": pattern.positive,
                "risk_points": pattern.risk,
                "credit_score_20": "",
                "repayment_source_score_20": "",
                "asset_exit_score_20": "",
                "plan_specificity_score_20": "",
                "uncertainty_control_score_20": "",
                "total_score_100": "",
                "user_decision": "",
                "heaviest_issue": "",
                "additional_checks": "",
                "ringi_sentence": "",
                "score_decision_gap_note": "",
                "ai_feedback_outcome": "",
                "ai_feedback_note": "",
                "okf_candidate_tags": "",
                "presentation_use_ok": "yes",
                "reviewer": "",
                "judged_at": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_guide(path: Path, csv_path: Path, rows: list[dict[str, str]], created: date) -> None:
    rel_csv = csv_path.relative_to(PROJECT_ROOT)
    sample = rows[0]
    text = f"""# 1000本判断ドリル 入力台帳

- Date: {created.isoformat()}
- CSV: `{rel_csv}`
- Input UI: `/judgment-drill`
- Rows: {len(rows)}
- Source: synthetic demo cases
- Status: pending_user_judgment
- Guardrail: User採点前は実戦検証・成果指標・RAG強化に使わない。採点後は human_judgment_asset として扱い、実案件結果検証とは分ける。

## 使い方

通常は `/judgment-drill` の入力画面から1件ずつ保存する。CSVを直接編集する場合は、次の空欄を埋める。

- `credit_score_20`: 信用力 0-20
- `repayment_source_score_20`: 返済原資 0-20
- `asset_exit_score_20`: 物件性・出口 0-20
- `plan_specificity_score_20`: 計画具体性 0-20
- `uncertainty_control_score_20`: 未確定リスク管理 0-20
- `total_score_100`: 合計 0-100
- `user_decision`: 承認 / 条件付き承認 / 追加確認 / 否認
- `heaviest_issue`: 一番重い論点
- `additional_checks`: 追加確認事項。複数ある場合は `;` 区切り
- `ringi_sentence`: 稟議に残す一文
- `score_decision_gap_note`: 点数と結論がズレた理由
- `ai_feedback_outcome`: helped / neutral / challenged / rejected
- `ai_feedback_note`: AI案へのダメ出し、または採用理由

銀行・既存取引の見方:

- `bank_is_main`: 銀行がメイン / 非メイン
- `bank_credit_balance_million_yen`: その銀行の与信残高 百万円
- `lease_customer_status`: リース既存先 / 新規先
- `existing_lease_credit_balance_million_yen`: 既存先の場合のリース与信残高 百万円。新規先は0
- `equity_ratio_percent`: 自己資本比率 %

## 判定目安

- 80点以上: 承認候補
- 65-79点: 条件付き承認候補
- 50-64点: 追加確認・保留候補
- 49点以下: 否認候補

点数は適当でよい。重要なのは、点数よりも「なぜその点にしたか」「どの条件なら通すか」「稟議にどう残すか」。

## 先頭ケース例

- case_id: `{sample["case_id"]}`
- 業種: {sample["industry"]}
- 銀行: {sample["bank_is_main"]} / 与信残高 {sample["bank_credit_balance_million_yen"]}百万円
- リース取引: {sample["lease_customer_status"]} / 既存リース与信残高 {sample["existing_lease_credit_balance_million_yen"]}百万円
- 自己資本比率: {sample["equity_ratio_percent"]}%
- 物件: {sample["asset_type"]}
- 申込: {sample["amount_million_yen"]}百万円 / {sample["lease_term_years"]}年
- 目的: {sample["purpose"]}
- 財務: {sample["financial_snapshot"]}
- 良い材料: {sample["positive_points"]}
- 懸念: {sample["risk_points"]}

## プレゼンで出す時の言い方

この台帳は実案件結果を当てたデータではなく、開発者本人が1000本の模擬案件に対して採点・条件・稟議文面・AI案へのダメ出しを入れた human_judgment_asset である。したがって、アウトカム検証ではなく、現場判断の形式知化・AI共同作業による判断資産化の証跡として提示する。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--date", default="2026-07-25")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--latest-guide", type=Path, default=DEFAULT_LATEST_GUIDE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created = date.fromisoformat(args.date)
    rows = build_rows(args.count, created)
    write_csv(args.output, rows)
    write_guide(args.guide, args.output, rows, created)
    write_guide(args.latest_guide, args.output, rows, created)
    print(f"wrote {len(rows)} rows -> {args.output}")
    print(f"wrote guide -> {args.guide}")
    print(f"wrote latest guide -> {args.latest_guide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
