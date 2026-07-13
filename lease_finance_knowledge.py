"""Canonical lease-finance knowledge prompt block.

This module is the single in-code source for compact lease-finance baseline
knowledge injected into Shion/Mebuki prompts. Detailed, fresh, or corrected
knowledge should live in Obsidian/RAG or verified external sources.
"""

from __future__ import annotations

LEASE_FINANCE_KNOWLEDGE_REVIEWED_ON = "2026-06-25"


def build_basic_lease_question_block(question: str, heading: str = "基本リースQA") -> str:
    """Return short canonical facts for common lease questions.

    This block is intentionally deterministic. It prevents simple baseline
    questions from being treated as unanswerable just because RAG has no hit.
    """
    text = (question or "").lower()
    if not text.strip():
        return ""

    rows: list[str] = []

    if any(term in text for term in ("法定耐用年数", "耐用年数", "減価償却")):
        if any(term in text for term in ("トラック", "貨物自動車", "商用車", "配送車")):
            rows.extend(
                [
                    "- トラック一般・平ボディ・ウイング車・大型トラック: 5年（車両運搬具/貨物自動車）",
                    "- 中型トラック（3.5t以下）・軽トラック/軽バン: 4年（車両運搬具/貨物自動車）",
                    "- 冷凍冷蔵車、ダンプ、タンクローリー、ユニック車など特殊車両: 5年を目安にする",
                ]
            )
        else:
            try:
                from useful_life_lookup import get_legal_useful_life

                years = get_legal_useful_life(question)
                rows.append(f"- 質問文から推定した物件の法定耐用年数: {years}年")
            except Exception:
                pass

    if "ファイナンス" in text or "所有権移転" in text or "フルペイアウト" in text:
        rows.append(
            "- ファイナンス・リースは、原則中途解約不可で、リース料総額が物件価額等をおおむね回収するリース。所有権移転/所有権移転外に分かれる。"
        )

    if "オペレーティング" in text or "オペリース" in text or "残価設定" in text:
        rows.append(
            "- オペレーティング・リースは、残価を見込んで月額負担を抑える設計。満了時の返却・再リース・買取りや残価精算条件を確認する。"
        )

    if "残価" in text or "再販" in text or "中古" in text:
        rows.append(
            "- 残価は満了時に見込む物件価値。審査では中古市場、汎用性、メンテ状況、使用時間、モデル陳腐化、処分ルートを見る。"
        )

    if "動産保険" in text or "保険" in text:
        rows.append(
            "- 動産総合保険は、リース物件の盗難・火災・破損等に備える保全手段。対象外事故、免責、保険金額、付保期間を確認する。"
        )

    if "固定資産税" in text or "償却資産税" in text:
        rows.append(
            "- 一般にリース物件の固定資産税/償却資産税申告はリース会社側で扱うことが多い。契約形態により異なるため契約条件で確認する。"
        )

    if "銀行融資" in text or "自己資金" in text or "現金購入" in text or "比較" in text:
        rows.append(
            "- 銀行融資は金利が低くなりやすい一方で融資枠や担保管理を使う。リースは初期費用を抑え、事務・保険・税務管理を簡素化しやすいが料率は高く見えやすい。"
        )

    if "リース期間" in text or "期間設定" in text:
        rows.append(
            "- リース期間は法定耐用年数、経済的使用可能年数、残価、月額負担、税務・会計処理、顧客の更新予定を合わせて決める。"
        )

    if not rows:
        return ""

    return "\n".join(
        [
            f"【{heading}】",
            *dict.fromkeys(rows),
            "回答規則: RAGやObsidian検索で直接ノートが0件でも、この基本QAが該当する場合は先に結論を答える。",
            "注意: 制度・税務・会計・個別契約条件で変わる点は、最新公式情報や契約書/見積書で最終確認する。",
        ]
    )


def build_lease_finance_knowledge_block(heading: str = "リースファイナンス基礎知識") -> str:
    """Return the shared baseline knowledge block for system prompts."""
    return f"""【{heading}】
参照方針:
- このブロックは短い基礎知識の正本であり、同じ内容を他のシステムプロンプトへ直書きしない。
- 個別案件・最新運用・ユーザー訂正は、関連するObsidian知識、保存済み業務メモ、現行コード仕様を優先する。
- 税制・会計・法務は改正で変わり得る。具体的な適用可否、控除率、期限、要件は最新の公式情報または専門家確認を前提にし、古い断定を避ける。
- 文書と実装が食い違う場合、現在の動作説明では実装を優先し、知識更新が必要な差分として明示する。
- 最終棚卸日: {LEASE_FINANCE_KNOWLEDGE_REVIEWED_ON}

1. ファイナンス・リース:
   フルペイアウトかつ原則中途解約不可（ノンキャンセラブル）が要件。所有権移転ファイナンス・リースと所有権移転外ファイナンス・リースに分かれる。
   所有権移転外リースは、リース期間を耐用年数として「リース期間定額法」で減価償却できる場合がある。
2. 中小企業の特例（SME会計指針）:
   中小企業では、一定のファイナンス・リース取引について賃貸借処理に準じ、支払リース料を費用処理できる扱いがある。
   ただし会計基準、税務、会社規模、契約条件で扱いが変わるため、適用判断は最新基準で確認する。
3. オペレーティング・リース:
   残価設定により月額負担を抑えやすい。終了時に返却・再リース・買取り（残価精算）を選択する設計があり、技術陳腐化が早い設備に向くことがある。
4. 税制優遇制度:
   - 中小企業投資促進税制: 特別償却または税額控除の対象になり得る。対象設備、資本金区分、控除率、適用期限、リース証書などの要件確認が必要。
   - 中小企業経営強化税制（A類型・B類型等）: 経営力向上計画の認定等により即時償却または税額控除を選択できる場合がある。リース取引への適用可否と計算基礎は最新制度で確認する。
5. 補助金・助成金とリース:
   補助金は制度ごとに「リース契約が対象になるか」「契約・発注・納品・支払のタイミング」「補助対象経費」「補助金入金までのつなぎ資金」が異なる。
   審査では、補助金を返済原資として過信せず、採択前・未採択時・入金遅延時でもリース料を払えるかを見る。
   ものづくり補助金、省力化投資補助金、IT導入補助金、省エネ系補助金などは候補になり得るが、対象設備・申請枠・公募回・交付決定前発注可否は最新の公式公募要領で確認する。
   回答時は「使えそうな制度名」だけで終えず、対象設備、導入目的、契約時期、採択前提の資金繰り、未採択時の代替策をセットで確認する。
6. リース vs 銀行融資 vs 自己資金:
   - 自己資金: 金利負担はないが、資金流動性を圧迫する。
   - 銀行融資: 金利は低くなりやすい一方、融資枠、担保、保証、資産管理、固定資産税、保険、減価償却事務の負担が残ることがある。
   - リース: 初期費用を抑え、銀行融資枠を温存し、固定資産税申告や動産総合保険をリース会社側で扱える場合がある。一方でリース料率は融資金利より高く見えることがある。
7. 業種・決算状況別の調達方針:
   創業期・赤字企業では銀行融資が厳しい一方、リース会社が物件担保価値や用途を評価できるケースがある。
   長期利用前提の設備はファイナンス・リースまたは融資購入、陳腐化が早い設備はオペレーティング・リースが候補になりやすい。
   税制適用の最終判断は顧問税理士等の専門家確認を推奨する。"""
