"""
判断資産ルール v2 生成スクリプト
型: 条件 → 見る論点 → 稟議に残す一文テンプレ → 使わない条件
効果記録: worked / marginal / missed / revised / none
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── スキーマ v2 の型 ────────────────────────────────────────────────────────
# condition        : この判断型を適用する「どんな案件か」
# check_points     : 実際に確認する論点のリスト（3〜5項目）
# memo_sentence    : 稟議・営業メモに残す一文テンプレ（{} でプレースホルダ）
# exclusion_conditions : この型を使わない / 使うと危ない条件
# is_common        : True=業種共通 / False=特定業種向け
# industry_tags    : 該当業種（空=全業種共通）
# effectiveness    : none / worked / marginal / missed / revised
# effectiveness_log: [{date, case_id, result, note}]

RULES: list[dict] = []

# ══════════════════════════════════════════════════════════════════════════════
# 共通ルール（is_common=True）
# ══════════════════════════════════════════════════════════════════════════════

RULES.append({
    "id": "b259411afb954d6d",
    "concept": "business_plan_specificity",
    "label": "事業計画の具体性確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "事業拡大・新規先獲得・新規出店など、将来見込みに依存した設備投資申込。または変動費が大きい業種（運送・飲食・製造）。",
    "check_points": [
        "受注根拠：既存先からの増量か新規先か、契約書・発注書の有無",
        "稼働計画：想定稼働率、シフト体制、ピーク・閑散の見通し",
        "資金繰り：売掛金回収サイトと返済タイミングのズレ",
        "返済原資の説明可能性：利益からの返済か、補助金・融資頼みか",
    ],
    "memo_sentence": "{業種}・{投資目的}の設備投資。受注根拠は{既存先増量/新規先}、返済原資は{自社利益/補助金/融資}で{整合あり/要確認}。",
    "exclusion_conditions": [
        "売上見込みだけで返済原資の根拠が薄い（「増える予定」のみ）",
        "稼働計画が抽象的で実績も契約見込みもない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk", "asset_life"],
    "canonical_statement": "事業計画は売上見込みだけでなく、受注根拠、稼働計画、資金繰り、返済原資の説明可能性で確認する。",
    "confidence": 0.94,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "a1b2c3d4e5f6a7b8",
    "concept": "asset_life_and_residual",
    "label": "物件寿命・残価・出口の確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "リース期間・残価・購入選択権に関する判断が必要な場合。特殊物件・陳腐化しやすい設備・中古流動性が低い物件。",
    "check_points": [
        "法定耐用年数と実際の経済的寿命の差（どちらが短いか）",
        "稼働状況・メンテナンス履歴・改造の有無",
        "中古市場での需要と換金性（汎用機か特殊機か）",
        "満了後の出口：再販・再リース・廃棄・返還のどれが現実的か",
    ],
    "memo_sentence": "物件：{物件名}（法定{N}年、経済的寿命{M}年見込み）。中古市場{あり/薄い}。満了後出口：{再販/再リース/廃棄}。",
    "exclusion_conditions": [
        "法定耐用年数のみで残価を機械的に設定する（特殊・陳腐化物件は別途検討）",
        "一品物・特注品で換金先がなく残価が取れない物件を高残価設定",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["asset_life", "cash_flow"],
    "canonical_statement": "リース期間・残価判断では、法定耐用年数だけでなく、実際の使用状況、経済的寿命、換金性、満了後の出口を合わせて確認する。",
    "confidence": 0.91,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "c3d4e5f6a7b8c9d0",
    "concept": "support_specificity",
    "label": "銀行支援・補助金の実効性確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "銀行支援、補助金、融資とセットでリース物件を申し込む場合。または「支援があるから大丈夫」という説明が出てきた場合。",
    "check_points": [
        "支援・補助金が対象リースへ直接使えるか（使途制限の確認）",
        "入金時期：採択から実際の入金まで何ヶ月か",
        "返済原資への効き方：リース料の何ヶ月分をカバーするか",
        "不採択・融資否決時の代替資金・代替計画の有無",
    ],
    "memo_sentence": "{支援名}は{物件}に直接充当可。入金予定{N}ヶ月後。不採択時代替資金：{確認済み/不十分}。",
    "exclusion_conditions": [
        "補助金採択が唯一の返済原資で代替がない",
        "支援の対象期間・使途がリースと合致しない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "support_specificity"],
    "canonical_statement": "銀行支援や補助金は、対象リースへの直接性、入金時期、返済原資への効き方を具体的に確認する。",
    "confidence": 0.89,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "d4e5f6a7b8c9d0e1",
    "concept": "renewal_increase_check",
    "label": "更新・増額申込の整合確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "既存リースを更新、または増額して申し込む場合。",
    "check_points": [
        "既存設備の稼働率・実績（稼働低迷中の増額は要注意）",
        "増額の根拠：受注増・粗利改善の具体的根拠",
        "旧設備の処分予定・返還見通し",
        "増額後もリース料支払いが返済原資から説明できるか",
    ],
    "memo_sentence": "更新増額申込。既存設備稼働率{%}、増額根拠{受注増/粗利改善}、旧設備処分{済み/予定N月}。増額後返済原資{整合あり/要確認}。",
    "exclusion_conditions": [
        "既存設備の稼働率が低いのに増額（使いこなせていない）",
        "旧設備処分の見通しが全くない（二重返済リスク）",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "asset_life"],
    "canonical_statement": "更新設備の増額申込は、既存設備の稼働率、粗利改善見込み、旧設備の処分予定が説明と整合する時だけ前向きに見る。",
    "confidence": 0.87,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "e5f6a7b8c9d0e1f2",
    "concept": "conditional_approval_checks",
    "label": "条件付き承認の明文化",
    "is_common": True,
    "industry_tags": [],
    "condition": "数字は基準を満たすが未確認リスクがあり、否認まではいかない案件。追加資料で解消できる可能性がある場合。",
    "check_points": [
        "未確認リスクを列挙し「追加資料で解消できるか」を判断",
        "実行条件：何が揃ったら実行してよいか",
        "撤退条件：何が出てきたら止めるか",
        "条件が守られなかった場合の対応（回収方針）",
    ],
    "memo_sentence": "条件付き承認。未確認リスク：{X}。実行条件：{Y}が揃ったら実行。撤退条件：{Z}の場合は中止。",
    "exclusion_conditions": [
        "撤退条件が明文化できない（曖昧な条件は条件付き承認にしない）",
        "追加資料で解消できないリスクがある場合は否認へ",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "条件付き承認では、未確認リスクを追加資料・実行条件・撤退条件に分けて明文化する。",
    "confidence": 0.88,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "f6a7b8c9d0e1f2a3",
    "concept": "intuition_gap",
    "label": "違和感の論点化",
    "is_common": True,
    "industry_tags": [],
    "condition": "数字は基準内だが業種・物件・申込背景・説明の一貫性に違和感がある場合。",
    "check_points": [
        "違和感を言語化する：何が気になるのか（物件目的？申込人の態度？数字の整合？）",
        "違和感を追加確認事項に変換できるか",
        "稟議補足として書き添えられるか",
        "条件設定（担保・実行条件）として使えるか",
    ],
    "memo_sentence": "数字は基準内。{違和感の内容}が気になる。稟議補足：{一文}。追加確認：{項目}。",
    "exclusion_conditions": [
        "違和感を言語化せず「なんとなく」で否認する（根拠化できないなら保留・確認に変換）",
        "違和感だけを理由に否認する（数字が基準内なら条件付き承認かコメント付き通過を検討）",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": [],
    "canonical_statement": "数字が悪くない案件でも、違和感は追加確認事項・稟議補足・条件設定のいずれかに変換する。",
    "confidence": 0.85,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "a7b8c9d0e1f2a3b4",
    "concept": "purchase_option_guidance",
    "label": "購入選択権の確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "購入選択権の有無・価格について判断が必要な場合。または申込人から購入選択権について質問が出た場合。",
    "check_points": [
        "購入選択権の価格設定（5〜30%が一般的な目安）",
        "残価との整合（残価設定と矛盾しないか）",
        "再リース条件との整合（再リースか購入かの出口を確認）",
        "満了後出口計画全体との整合",
    ],
    "memo_sentence": "購入選択権{有/無}。選択権価格{N%}（相場5〜30%）。残価・再リースとの整合：{確認済み/要調整}。",
    "exclusion_conditions": [
        "購入選択権を設定するが残価との整合が取れていない",
        "購入選択権価格が5%未満（みなし売買の懸念）",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["asset_life"],
    "canonical_statement": "購入選択権は5〜30%程度が多い前提で、残価・再リース・満了後出口と整合するか確認する。",
    "confidence": 0.82,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 共通ルール（新規追加）─────────────────────────────────────────────────────

RULES.append({
    "id": "b8c9d0e1f2a3b4c5",
    "concept": "equity_ratio_quality",
    "label": "自己資本比率・純資産の実質確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "自己資本比率や純資産がボーダーライン付近、または急に改善している場合。",
    "check_points": [
        "純資産の中身：資本金・利益剰余金・評価差額のどれが主体か",
        "含み損・引当金不足・役員借入金の影響",
        "直近決算が期末だけ黒字で実態が赤字に近くないか",
        "3期トレンド：増加基調か減少傾向か",
    ],
    "memo_sentence": "自己資本比率{N%}。純資産の中身は{利益剰余金主体/含み益あり/役員借入あり}。3期トレンド：{改善/横ばい/悪化}。",
    "exclusion_conditions": [
        "純資産が役員借入金の相殺で成り立っている（引き出すと実質債務超過）",
        "期末だけ黒字で実態の資金繰りが厳しい場合は数字を鵜呑みにしない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "自己資本比率は数字だけでなく、純資産の中身・3期トレンド・含み損を合わせて実質評価する。",
    "confidence": 0.86,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "c9d0e1f2a3b4c5d6",
    "concept": "existing_credit_exposure",
    "label": "既存与信・残債の確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "既存のリース・ローン残高がある状態での追加申込。または複数社へのリース申込が確認された場合。",
    "check_points": [
        "既存リース・ローンの残高合計と今回追加後の総与信額",
        "月次リース料・返済額の合計が月次キャッシュフローに対して無理のない比率か",
        "他社リース・銀行借入の状況（申告漏れがないか）",
        "与信枠内に収まるか（社内基準との照合）",
    ],
    "memo_sentence": "既存残債合計：{N}百万円。今回追加後の月次返済合計：{M}万円。対月次CF比率：{%}。与信枠：{枠内/要検討}。",
    "exclusion_conditions": [
        "月次返済合計が月次CFの50%を超える（一般的な上限目安）",
        "他社リース状況が不明・申告不一致の場合は追加調査",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "既存リース・ローン残高と今回追加後の総返済額を月次CFと対比し、与信枠との整合を確認する。",
    "confidence": 0.90,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "d0e1f2a3b4c5d6e7",
    "concept": "three_year_financial_trend",
    "label": "3期財務トレンドの読み方",
    "is_common": True,
    "industry_tags": [],
    "condition": "決算書3期分を評価する場面全般。特に直近1期が突出して良い・悪い場合。",
    "check_points": [
        "売上・営業利益の増減トレンド（増収増益か、一時的な変動か）",
        "直近期の改善が一過性か（補助金・資産売却など特殊要因の有無）",
        "在庫・売掛金の急増（水増し・回収遅延の兆候）",
        "営業CFがプラス基調か（利益と乖離していないか）",
    ],
    "memo_sentence": "3期トレンド：売上{増/横ばい/減}、営業利益{改善/悪化}。直近好転の要因：{特殊なし/補助金/資産売却あり}。営業CF：{プラス/マイナス}。",
    "exclusion_conditions": [
        "直近1期だけ黒字で2期赤字の場合を「業績改善」と評価しない",
        "特殊要因（補助金・資産売却）を除いたら実質赤字の場合",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "3期財務トレンドは増減の方向性と特殊要因の有無を分けて読み、直近1期だけで判断しない。",
    "confidence": 0.88,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "e1f2a3b4c5d6e7f8",
    "concept": "guarantor_and_collateral",
    "label": "保証・担保の確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "個人保証・連帯保証・物的担保を条件として検討する場合。または財務が弱い申込人への与信判断。",
    "check_points": [
        "代表者個人保証の有効性（資産状況・他の保証の有無）",
        "連帯保証人がいる場合：その保証人の財務状況",
        "物的担保がある場合：担保評価額と被担保債権の比率",
        "担保が取れない場合の代替ヘッジ（信用保証協会・保険等）",
    ],
    "memo_sentence": "個人保証：{あり/なし}（代表者資産{確認済み/要確認}）。担保：{物的担保/なし}。担保カバー率：{%}。",
    "exclusion_conditions": [
        "保証人の財務状況が未確認のまま保証を与信判断の根拠にする",
        "担保評価が古く実勢価格と乖離している可能性がある",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "保証・担保は有無だけでなく、実効性（資産状況・評価額・カバー率）を合わせて確認する。",
    "confidence": 0.85,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "f2a3b4c5d6e7f8a9",
    "concept": "special_asset_liquidation_risk",
    "label": "特殊物件・一品物の換金リスク",
    "is_common": True,
    "industry_tags": [],
    "condition": "汎用性の低い特殊機械・特注設備・一品物のリース申込。",
    "check_points": [
        "物件の汎用性（他業者・他業種でも使えるか）",
        "中古市場の存在と流動性（買取業者・オークション等の有無）",
        "メーカーによる下取り保証の有無",
        "デフォルト時の回収シナリオ（引き上げ・売却・廃棄コスト）",
    ],
    "memo_sentence": "物件：{名称}（汎用性：{高/中/低}）。中古市場：{活発/限定/なし}。デフォルト時回収見込み：{N%}。",
    "exclusion_conditions": [
        "特注・一品物で中古市場がなく、回収シナリオが廃棄のみ",
        "物件の担保価値に大きく依存した与信構造",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["asset_life"],
    "canonical_statement": "特殊物件は汎用性・中古流動性・デフォルト時回収シナリオを確認し、担保価値依存の与信を避ける。",
    "confidence": 0.84,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ══════════════════════════════════════════════════════════════════════════════
# 業種別ルール
# ══════════════════════════════════════════════════════════════════════════════

# ── 運送業 ────────────────────────────────────────────────────────────────────

RULES.append({
    "id": "a3b4c5d6e7f8a9b0",
    "concept": "transport_shipper_concentration",
    "label": "荷主集中リスク（運送業）",
    "is_common": False,
    "industry_tags": ["運送業"],
    "condition": "運送業・物流業の車両リース申込。特定荷主への依存度が高い場合。",
    "check_points": [
        "主要荷主上位3社の売上比率（1社50%超は要注意）",
        "荷主との契約形態（長期契約か随時発注か）",
        "荷主の業況・業界動向（発注減少・物流見直しリスク）",
        "荷主喪失時の代替路線・代替荷主の見通し",
    ],
    "memo_sentence": "主要荷主：{社名}（売上比率{%}）。契約：{長期/随時}。荷主業況：{良好/要注意}。荷主喪失時代替：{あり/なし}。",
    "exclusion_conditions": [
        "1社荷主依存100%で代替なし（荷主が傾けば即リース料不払い）",
        "荷主との契約が口頭のみで証拠がない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk", "cash_flow"],
    "canonical_statement": "運送業は荷主集中リスクを確認し、主要荷主喪失時の代替可能性を評価する。",
    "confidence": 0.87,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "b4c5d6e7f8a9b0c1",
    "concept": "transport_variable_cost",
    "label": "燃料費・人件費の変動リスク（運送業）",
    "is_common": False,
    "industry_tags": ["運送業"],
    "condition": "運送業の大型・中型車両リース。燃料費・ドライバー人件費が大きなコスト構造の場合。",
    "check_points": [
        "燃料費が月次収支に占める比率（燃料高騰時のシミュレーション）",
        "運賃への燃料サーチャージ転嫁の有無と実績",
        "ドライバー不足の状況・採用コスト・離職率",
        "燃料費・人件費上昇時でも返済原資が残る粗利水準か",
    ],
    "memo_sentence": "燃料費比率：{%}。サーチャージ転嫁：{できている/できていない}。ドライバー確保：{安定/不足傾向}。粗利水準：{十分/ボーダー}。",
    "exclusion_conditions": [
        "燃料費転嫁ができず燃料高騰時に粗利が消える構造",
        "ドライバー確保が不安定で稼働率維持に懸念がある",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk", "cash_flow"],
    "canonical_statement": "運送業は燃料費・人件費の変動と転嫁力を確認し、コスト上昇時でも返済原資が残るかを評価する。",
    "confidence": 0.86,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 製造業 ────────────────────────────────────────────────────────────────────

RULES.append({
    "id": "c5d6e7f8a9b0c1d2",
    "concept": "manufacturing_customer_concentration",
    "label": "主要取引先集中度（製造業）",
    "is_common": False,
    "industry_tags": ["製造業"],
    "condition": "製造業の設備投資申込。特定取引先・元請けへの依存度が高い場合。",
    "check_points": [
        "上位取引先3社の売上比率",
        "元請けとの取引継続性（長期取引か新規発注か）",
        "元請けの業況・業界動向（自動車・半導体等のサイクルリスク）",
        "取引先喪失時の代替受注の見通し",
    ],
    "memo_sentence": "主要取引先：{社名}（{%}）。取引継続性：{長期/新規}。元請け業況：{良好/要注意}。代替先：{あり/なし}。",
    "exclusion_conditions": [
        "単一元請け100%依存で設備投資額が大きい（元請けの方針変更で即破綻）",
        "元請けが業界再編・撤退を検討している",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk", "cash_flow"],
    "canonical_statement": "製造業は主要取引先の集中度と取引継続性を確認し、元請け喪失時の代替可能性を評価する。",
    "confidence": 0.87,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "d6e7f8a9b0c1d2e3",
    "concept": "subsidy_machinery_check",
    "label": "補助金前提の工作機械申込（製造業）",
    "is_common": False,
    "industry_tags": ["製造業"],
    "condition": "工作機械をものづくり補助金・IT導入補助金等の前提で申し込む場合。",
    "check_points": [
        "採択前の返済原資（補助金なしで返済できるか）",
        "未採択時の代替資金（自己資金・融資への切り替え可否）",
        "補助金入金スケジュールと返済タイミングの整合",
        "採択率の実績・申請書類の完成度",
    ],
    "memo_sentence": "補助金前提申込（{補助金名}）。採択前返済原資：{自社資金{N}百万円/不十分}。未採択時代替：{あり/なし}。入金予定：{N}ヶ月後。",
    "exclusion_conditions": [
        "補助金採択が唯一の返済原資で代替がない",
        "採択見通しが根拠なく楽観的",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "support_specificity"],
    "canonical_statement": "工作機械を補助金前提で導入する案件は、採択前の返済原資と未採択時の代替資金を分けて確認する。",
    "confidence": 0.89,
    "created_at": "2026-07-12T07:23:23",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 飲食業 ────────────────────────────────────────────────────────────────────

RULES.append({
    "id": "e7f8a9b0c1d2e3f4",
    "concept": "food_service_business_plan",
    "label": "飲食業の事業計画の実現性",
    "is_common": False,
    "industry_tags": ["飲食業"],
    "condition": "飲食業（レストラン・カフェ・ラーメン等）の厨房設備・什器リース申込。新規開業または既存店増設の場合。",
    "check_points": [
        "立地・客層・競合状況（商圏分析の有無）",
        "席数・客単価・回転数の収支シミュレーション（損益分岐点）",
        "仕入先の多様性と食材コストの変動リスク",
        "既存店舗がある場合：直近の売上・利益実績",
    ],
    "memo_sentence": "立地：{エリア}（競合{あり/少ない}）。損益分岐点：月{N}万円。食材コスト比率：{%}。既存店実績：{黒字/赤字/新規}。",
    "exclusion_conditions": [
        "損益分岐点の計算がなく「流行っている」だけの根拠",
        "開業資金が手元資金だけで運転資金余力がない",
        "食材コストが高く損益分岐点が売上目標に対して厳しすぎる",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk", "cash_flow"],
    "canonical_statement": "飲食業は立地・損益分岐点・食材コスト・既存店実績で事業計画の実現性を確認する。",
    "confidence": 0.85,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "f8a9b0c1d2e3f4a5",
    "concept": "food_service_equipment_liquidation",
    "label": "飲食設備の換金性確認",
    "is_common": False,
    "industry_tags": ["飲食業"],
    "condition": "飲食業の厨房設備・特注什器のリース申込。物件が店舗特化型の場合。",
    "check_points": [
        "厨房設備の汎用性（他業態・他店舗でも使えるか）",
        "中古業者・オークションでの流通状況",
        "設置工事費が高く移設コストがかかる場合の回収シナリオ",
        "物件を店舗に固定する改造の有無（取り外し可能か）",
    ],
    "memo_sentence": "厨房設備：{汎用/店舗特化}。中古市場：{あり/限定}。移設コスト：{低/高}。デフォルト時回収：{N%相当}。",
    "exclusion_conditions": [
        "特注厨房設備で移設不可・中古市場なし・廃棄費用が残る構造",
        "回収見込みが著しく低く担保価値ゼロに近い",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["asset_life"],
    "canonical_statement": "飲食設備は汎用性・移設可否・中古流動性を確認し、デフォルト時の回収シナリオを事前に想定する。",
    "confidence": 0.82,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 医療・介護 ────────────────────────────────────────────────────────────────

RULES.append({
    "id": "a9b0c1d2e3f4a5b6",
    "concept": "medical_reimbursement_risk",
    "label": "診療報酬・介護報酬の変動リスク（医療・介護）",
    "is_common": False,
    "industry_tags": ["医療業", "介護業"],
    "condition": "医療・介護施設の医療機器・介護設備リース申込。",
    "check_points": [
        "診療報酬・介護報酬改定スケジュールと改定幅の影響",
        "自由診療・保険外収入の比率（報酬依存度）",
        "施設認可・指定の更新状況と法令遵守状況",
        "稼働率（病床・定員に対する実稼働率）",
    ],
    "memo_sentence": "診療・介護報酬依存度：{%}。稼働率：{%}。認可更新：{問題なし/要確認}。自由診療比率：{%}。",
    "exclusion_conditions": [
        "報酬改定で収入が大幅減となる場合に返済原資の手当がない",
        "施設認可・指定に問題があり取り消しリスクがある",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk", "cash_flow"],
    "canonical_statement": "医療・介護業は診療介護報酬の改定リスク・稼働率・施設認可状況を返済原資の変動要因として確認する。",
    "confidence": 0.83,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 建設・工事 ────────────────────────────────────────────────────────────────

RULES.append({
    "id": "b0c1d2e3f4a5b6c7",
    "concept": "construction_order_timing",
    "label": "工事受注・入金タイミング（建設業）",
    "is_common": False,
    "industry_tags": ["建設業", "工事業"],
    "condition": "建設・工事業者の重機・特殊車両・足場等のリース申込。受注型ビジネスの場合。",
    "check_points": [
        "受注残高・受注パイプラインの状況（工事案件名・発注元）",
        "工期と入金タイミング（出来高払い・竣工後払いのどちらか）",
        "リース料支払いと工事入金のタイミングのズレ",
        "未受注期間（閑散期）のリース料支払い原資",
    ],
    "memo_sentence": "受注残：{N}百万円（工期{M}ヶ月）。入金：{出来高払い/竣工後}。閑散期のリース料原資：{手元資金/融資枠}。",
    "exclusion_conditions": [
        "受注がないと返済できない構造で受注パイプラインが薄い",
        "入金が竣工後一括で工期が長く資金繰りが厳しい",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "建設業は受注残高・入金タイミング・閑散期の資金繰りをセットで確認し、工事サイクルとリース料支払いの整合を取る。",
    "confidence": 0.85,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── IT・ソフトウェア ─────────────────────────────────────────────────────────

RULES.append({
    "id": "c1d2e3f4a5b6c7d8",
    "concept": "it_recurring_revenue_check",
    "label": "ストック収益・受注残の確認（IT業）",
    "is_common": False,
    "industry_tags": ["IT業", "ソフトウェア業"],
    "condition": "IT・ソフトウェア企業のサーバー・開発機器リース申込。SaaS・保守契約等のストック収益がある場合。",
    "check_points": [
        "ストック収益（保守・SaaS）の月次安定額",
        "プロジェクト型収益の受注残と完遂リスク",
        "エンジニア確保状況と開発遅延リスク",
        "主要顧客の解約率（チャーン）と新規獲得パイプライン",
    ],
    "memo_sentence": "ストック収益：月{N}万円。受注残：{M}百万円。エンジニア充足率：{%}。主要顧客：{集中/分散}。",
    "exclusion_conditions": [
        "プロジェクト型収益のみでストックがなく受注が途切れると即厳しい",
        "エンジニア不足でプロジェクト遅延・赤字転落リスクが高い",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "IT業はストック収益の安定額とプロジェクト型収益の受注残を分けて評価し、エンジニア確保リスクを合わせて見る。",
    "confidence": 0.82,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 共通（物件系・追加）────────────────────────────────────────────────────────

RULES.append({
    "id": "d2e3f4a5b6c7d8e9",
    "concept": "lease_period_vs_usage_period",
    "label": "リース期間と実使用期間の整合",
    "is_common": True,
    "industry_tags": [],
    "condition": "リース期間の設定について判断する場面全般。または「短すぎる・長すぎる」と感じた場合。",
    "check_points": [
        "申込人が実際に使う予定期間（事業計画との整合）",
        "法定耐用年数に対するリース期間の比率（一般的に70〜120%の範囲）",
        "途中解約リスク：事業縮小・倒産・物件不要になる可能性",
        "リース期間終了後の再リース・購入・返還のどれを想定するか",
    ],
    "memo_sentence": "リース期間：{N}ヶ月（法定耐用{M}年の{%}）。実使用予定：{K}年。途中解約リスク：{低/中/高}。満了後：{再リース/購入/返還}。",
    "exclusion_conditions": [
        "リース期間が法定耐用年数の70%未満（短すぎる・月リース料が高くなりすぎる）",
        "リース期間が経済的寿命を超えて設定されている",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["asset_life"],
    "canonical_statement": "リース期間は法定耐用年数・実使用予定期間・途中解約リスク・満了後出口と整合させて設定する。",
    "confidence": 0.84,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

# ── 業種共通・審査運用 ────────────────────────────────────────────────────────

RULES.append({
    "id": "e3f4a5b6c7d8e9f0",
    "concept": "risk_origin_separation",
    "label": "信用リスクと競合・成約リスクの分離",
    "is_common": True,
    "industry_tags": [],
    "condition": "案件の懸念が「返済できない（信用リスク）」なのか「受注が取れない（競合・成約リスク）」なのかが混ざっている場合。",
    "check_points": [
        "懸念の本質：財務の問題か、事業・競合の問題か",
        "信用リスク：返済能力そのもの（財務比率・CF）",
        "競合・成約リスク：売上・受注が計画通りにいかないリスク（市場・競合）",
        "どちらのリスクか分けて、稟議で別々に明記する",
    ],
    "memo_sentence": "主リスク：{信用リスク/競合・成約リスク}。信用面：{財務評価}。事業面：{競合・受注状況}。",
    "exclusion_conditions": [
        "信用リスクと競合リスクを一緒にして「全体的に不安」という評価にする",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "案件の懸念を信用リスク（財務・返済能力）と競合・成約リスク（事業・受注）に分けて評価し、それぞれ別に稟議に明記する。",
    "confidence": 0.86,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})



RULES.append({
    "id": "f4a5b6c7d8e9f0a1",
    "concept": "seasonal_fluctuation",
    "label": "季節変動・繁閑期の資金繰り",
    "is_common": True,
    "industry_tags": [],
    "condition": "売上・利益に季節変動がある業種（観光・農業・小売・建設等）のリース申込。",
    "check_points": [
        "繁閑の月次売上・利益の振れ幅",
        "閑散期でもリース料が払えるだけのCFがあるか",
        "閑散期のつなぎ資金の手当（融資枠・内部留保）",
        "繁忙期に一括回収するビジネスモデルかどうか",
    ],
    "memo_sentence": "繁閑差：ピーク月{N}万円 vs 閑散期{M}万円。閑散期リース料原資：{手元資金{K}万円/融資枠あり}。",
    "exclusion_conditions": [
        "閑散期に返済原資がなく繁忙期一括回収だけに依存（1ヶ月の不振で即延滞）",
        "年度末の決算が良く見えるが月次CFが赤字の月がある",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "季節変動がある業種は繁閑期の月次CFを確認し、閑散期でもリース料が払えるかをつなぎ資金込みで評価する。",
    "confidence": 0.84,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "a5b6c7d8e9f0a1b2",
    "concept": "owner_succession_risk",
    "label": "代表者依存・後継者リスク",
    "is_common": True,
    "industry_tags": [],
    "condition": "代表者が高齢または高スキル職人系の中小企業。代表者1人に依存した収益構造の場合。",
    "check_points": [
        "代表者の年齢と健康状態（リース期間と照合）",
        "後継者・後継候補の有無",
        "代表者が不在になった場合の事業継続可否",
        "生命保険・役員保険等のヘッジの有無",
    ],
    "memo_sentence": "代表者：{N}歳。後継者：{あり/なし/検討中}。代表不在時の事業継続：{可能/困難}。リスクヘッジ：{保険/なし}。",
    "exclusion_conditions": [
        "代表者が高齢かつ後継者なし・リース期間が長い（期間中に廃業リスク）",
        "代表者の技術・人脈が全売上を支えており法人としての信用がない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["industry_risk"],
    "canonical_statement": "代表者依存の中小企業はリース期間内の後継者リスクと事業継続性を確認し、必要に応じてヘッジを条件にする。",
    "confidence": 0.82,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "b6c7d8e9f0a1b2c3",
    "concept": "new_vs_existing_customer",
    "label": "新規顧客 vs 既存顧客の与信水準",
    "is_common": True,
    "industry_tags": [],
    "condition": "初めてリースを申し込む新規顧客。または取引実績が浅い顧客への再申込。",
    "check_points": [
        "取引履歴の有無（既存なら返済実績を確認）",
        "新規の場合：財務書類の精査水準を既存より上げる",
        "紹介元・銀行リレーション・業界での評判",
        "初回申込の物件規模が与信水準に対して過大でないか",
    ],
    "memo_sentence": "取引：{初回/N回目・返済実績良好}。紹介元：{銀行/既存取引先/なし}。初回物件規模：{与信水準内/やや大きい}。",
    "exclusion_conditions": [
        "新規で財務書類なし・口頭のみの案件（最低でも直近決算1期は必要）",
        "新規で物件規模が大きく与信水準に見合わない",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "新規顧客は財務精査水準を上げ、紹介元・取引実績・物件規模が与信水準に見合うかを確認する。",
    "confidence": 0.83,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "c7d8e9f0a1b2c3d4",
    "concept": "multi_business_revenue_separation",
    "label": "複数事業の収益分離",
    "is_common": True,
    "industry_tags": [],
    "condition": "複数の事業を営む申込人。物件が特定事業向けだが財務は全社合算の場合。",
    "check_points": [
        "リース物件が使われる事業セグメントの収益（全社合算ではなく事業別）",
        "赤字事業が黒字事業の利益を食っていないか",
        "物件使用事業が縮小・撤退予定でないか",
        "事業ごとのCF（事業別損益の開示を求めることも検討）",
    ],
    "memo_sentence": "対象事業：{事業名}（全社売上比率{%}）。対象事業単体損益：{黒字/赤字}。他事業への依存：{なし/あり}。",
    "exclusion_conditions": [
        "全社合算は黒字だが対象事業単体は赤字で事業継続が不透明",
        "収益事業の利益で不採算事業のリース料を補填している",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "複数事業を営む申込人はリース物件の使用事業単体の収益を把握し、事業間補填依存の構造を避ける。",
    "confidence": 0.83,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "d8e9f0a1b2c3d4e5",
    "concept": "construction_subcontract_risk",
    "label": "下請け依存度（建設業）",
    "is_common": False,
    "industry_tags": ["建設業", "工事業"],
    "condition": "建設業の下請け比率が高く、外注費が売上の大半を占める場合。",
    "check_points": [
        "外注費比率（売上に対して何%か）",
        "主要下請け業者の確保状況（人手不足・価格高騰の影響）",
        "元請け交渉力：外注費上昇を元請けに転嫁できるか",
        "下請け比率が高い場合の実質粗利と返済原資の水準",
    ],
    "memo_sentence": "外注費比率：{%}。主要下請け：{確保済み/不足傾向}。価格転嫁：{できている/困難}。実質粗利：{%}。",
    "exclusion_conditions": [
        "外注費比率が90%超で自社施工能力がほぼない（コスト上昇に無防備）",
        "下請け確保ができず工期遅延・赤字工事のリスクが顕在化",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "建設業の下請け依存度を確認し、外注費上昇を価格転嫁できるかと実質粗利水準を評価する。",
    "confidence": 0.83,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "e9f0a1b2c3d4e5f6",
    "concept": "retail_inventory_turnover",
    "label": "在庫回転・客単価・粗利（小売業）",
    "is_common": False,
    "industry_tags": ["小売業"],
    "condition": "小売業の陳列棚・冷蔵設備・POS端末等のリース申込。",
    "check_points": [
        "在庫回転率（回転が遅いと不良在庫リスク）",
        "客単価と客数のトレンド（下落傾向は収益悪化のサイン）",
        "粗利率（仕入コスト上昇の転嫁力）",
        "競合（近隣大型店・EC）の影響と差別化ポイント",
    ],
    "memo_sentence": "在庫回転：{N}回/年。客単価：{N}円（{上昇/横ばい/下落}）。粗利率：{%}。競合影響：{低/中/高}。",
    "exclusion_conditions": [
        "在庫回転が極端に遅く不良在庫が積み上がっている",
        "客単価・客数が継続的に下落し粗利が縮小傾向",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow", "industry_risk"],
    "canonical_statement": "小売業は在庫回転・客単価トレンド・粗利率・競合影響を確認し、収益基盤が安定しているかを評価する。",
    "confidence": 0.81,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})

RULES.append({
    "id": "f0a1b2c3d4e5f6a7",
    "concept": "working_capital_adequacy",
    "label": "運転資金余力の確認",
    "is_common": True,
    "industry_tags": [],
    "condition": "中小企業の設備投資リース申込全般。特に資産規模に対して申込金額が大きい場合。",
    "check_points": [
        "手元現預金の水準（月次固定費の何ヶ月分あるか）",
        "当座比率・流動比率（短期支払い能力）",
        "設備投資後の手元資金が運転資金として十分か",
        "緊急時の融資枠の有無（当座貸越・コミットメントライン）",
    ],
    "memo_sentence": "手元現預金：{N}ヶ月分の固定費相当。当座比率：{%}。緊急時融資枠：{あり{M}百万円/なし}。",
    "exclusion_conditions": [
        "手元資金が月次固定費の1ヶ月分未満で緊急時の余力がない",
        "設備投資後に運転資金がなくなる（設備は動くが仕入れができない）",
    ],
    "effectiveness": "none",
    "effectiveness_log": [],
    "risk_axis": ["cash_flow"],
    "canonical_statement": "設備リース申込では手元資金の水準・当座比率・緊急融資枠を確認し、投資後も運転資金が確保されているか評価する。",
    "confidence": 0.87,
    "created_at": "2026-07-30T00:00:00",
    "updated_at": "2026-07-30T00:00:00",
})


# ── 出力 ─────────────────────────────────────────────────────────────────────

def main():
    out = {
        "schema_version": 2,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": "判断資産ルール v2 — 型: 条件→見る論点→稟議の一文→使わない条件",
        "total_rules": len(RULES),
        "common_rules": sum(1 for r in RULES if r["is_common"]),
        "industry_rules": sum(1 for r in RULES if not r["is_common"]),
        "rules": RULES,
    }
    out_path = REPO_ROOT / "data" / "judgment_asset_rules_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Written: {out_path}")
    print(f"Total: {out['total_rules']} rules  "
          f"(共通: {out['common_rules']}, 業種別: {out['industry_rules']})")

if __name__ == "__main__":
    main()
