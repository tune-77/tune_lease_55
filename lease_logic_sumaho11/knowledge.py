"""
ナレッジベース（リース審査マニュアル・業種別ガイド・FAQ・事例集）の読み込みと検索。
PDF「1 リース審査システム概要書」「2 審査マニュアル」「3 業種別ガイド」「4 FAQ集」「5 審査事例集」
の内容を構造化したJSONを読み込み、AIチャットのコンテキスト生成に使用する。
"""
import json
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge_base.json")

_knowledge_cache: dict | None = None


def load_knowledge() -> dict:
    """knowledge_base.json を読み込む（キャッシュ付き）。"""
    global _knowledge_cache
    if _knowledge_cache is not None:
        return _knowledge_cache
    try:
        with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
            _knowledge_cache = json.load(f)
    except Exception:
        _knowledge_cache = {}
    return _knowledge_cache


# ─── スコア・判定ルール ─────────────────────────────────────────────────────────

def get_scoring_overview() -> str:
    """スコアリング概要テキストを返す（AIシステムプロンプト用）。"""
    kb = load_knowledge()
    ss = kb.get("scoring_system", {})
    lines = [
        "【スコアリング概要】",
        ss.get("overview", ""),
        f"計算式: {ss.get('formula', '')}",
        f"承認ライン: {ss.get('approval_line', 71)}点以上",
        "",
        "【ランク境界】",
    ]
    for g in ss.get("grade_boundaries", []):
        lines.append(f"  {g['grade']}ランク（{g['min']}点以上）: {g['label']} — {g['policy']}")
    return "\n".join(lines)


def get_qualitative_items_text() -> str:
    """定性スコアリング6項目の説明テキストを返す。"""
    kb = load_knowledge()
    items = kb.get("scoring_system", {}).get("qualitative_items", [])
    lines = ["【定性スコアリング6項目（各0〜4点）】"]
    for item in items:
        lines.append(f"\n■ {item['label']}（重み{item['weight']}%）")
        for opt in item.get("options", []):
            lines.append(f"  {opt['score']}点: {opt['label']} — {opt.get('detail','')}")
    return "\n".join(lines)


def get_manual_text() -> str:
    """審査マニュアルのテキストを返す。"""
    kb = load_knowledge()
    manual = kb.get("manual", {})
    lines = ["【審査マニュアル】"]

    pre = manual.get("pre_screening", {})
    if pre:
        lines.append("\n■ 審査前チェックリスト")
        for item in pre.get("required_items", []):
            lines.append(f"  ✓ {item}")
        lines.append("\n■ 自動否決条件")
        for item in pre.get("auto_reject_conditions", []):
            lines.append(f"  ✗ {item}")

    bh = manual.get("borderline_handling", {})
    if bh:
        lines.append("\n■ ボーダーライン処理")
        for key, val in bh.items():
            lines.append(f"  {val.get('condition','')}")
            lines.append(f"    必要条件: {val.get('required','')}")
            for ex in val.get("examples", []):
                lines.append(f"    例: {ex}")
    return "\n".join(lines)


def get_industry_guide_text(industry_name: str = "") -> str:
    """
    業種別ガイドのテキストを返す。
    industry_name を指定すると該当業種のみ、空なら全業種を返す。
    """
    kb = load_knowledge()
    cats = kb.get("industry_guide", {}).get("categories", [])
    if industry_name:
        cats = [c for c in cats if industry_name in c.get("name", "")]
        if not cats:
            cats = kb.get("industry_guide", {}).get("categories", [])
    lines = [f"【業種別ガイド】{('（' + industry_name + '）') if industry_name else ''}"]
    for cat in cats[:3]:
        lines.append(f"\n■ {cat['name']}")
        ki = cat.get("key_indicators", {})
        if ki:
            lines.append("  主要指標目安:")
            for k, v in ki.items():
                lines.append(f"    {k}: {v}")
        tips = cat.get("scoring_tips", [])
        if tips:
            lines.append("  審査ポイント:")
            for t in tips:
                lines.append(f"    ・{t}")
        risks = cat.get("risks", [])
        if risks:
            lines.append("  主なリスク: " + " / ".join(risks))
    return "\n".join(lines)


def get_faq_text(category: str = "", max_items: int = 10) -> str:
    """FAQ テキストを返す。category 指定でフィルタリング。"""
    kb = load_knowledge()
    faqs = kb.get("faq", [])
    if category:
        faqs = [f for f in faqs if category in f.get("category", "")]
    faqs = faqs[:max_items]
    lines = ["【FAQ】"]
    for faq in faqs:
        lines.append(f"\nQ（{faq.get('category','')}）: {faq.get('q','')}")
        lines.append(f"A: {faq.get('a','')}")
    return "\n".join(lines)


def get_cases_text(max_cases: int = 3) -> str:
    """審査事例集テキストを返す。"""
    kb = load_knowledge()
    cases = kb.get("cases", [])[:max_cases]
    lines = ["【審査事例集】"]
    for case in cases:
        lines.append(f"\n▶ 事例: {case.get('title','')}")
        lines.append(f"  業種: {case.get('industry','')} / 設立: {case.get('company_age','')}年 / 従業員: {case.get('employees','')}名")
        q = case.get("quant_score") or case.get("quant_score_initial", "")
        ql = case.get("qual_total", "")
        t = case.get("total_score") or case.get("total_score_final", "")
        g = case.get("grade") or case.get("grade_final", "")
        lines.append(f"  定量スコア: {q} / 定性スコア: {ql} / 総合: {t} → {g}ランク")
        lines.append(f"  判定: {case.get('decision','')}")
        for lesson in case.get("lessons", [])[:2]:
            lines.append(f"  学び: {lesson}")
    return "\n".join(lines)


def get_improvement_guide_text() -> str:
    """スコア改善ガイドを返す。"""
    kb = load_knowledge()
    guide = kb.get("improvement_guide", {})
    lines = ["【スコア改善ガイド】"]
    lines.append("\n■ すぐにできること（クイックウィン）")
    for item in guide.get("quick_wins", []):
        lines.append(f"  ・{item.get('action','')}: {item.get('impact','')}（{item.get('timeline','')}）")
    lines.append("\n■ 中期的取り組み")
    for item in guide.get("medium_term", []):
        lines.append(f"  ・{item.get('action','')}: {item.get('impact','')}（{item.get('timeline','')}）")
    return "\n".join(lines)


# ─── キーワード検索でFAQを返す ────────────────────────────────────────────────

def search_faq(query: str) -> list[dict]:
    """
    クエリに関連するFAQを返す（簡易キーワードマッチ）。
    各FAQ の q・a・category を対象に検索し、マッチしたものを最大5件返す。
    """
    kb = load_knowledge()
    faqs = kb.get("faq", [])
    keywords = [w for w in query.split() if len(w) >= 2]
    if not keywords:
        return faqs[:3]

    scored: list[tuple[int, dict]] = []
    for faq in faqs:
        text = faq.get("q", "") + faq.get("a", "") + faq.get("category", "")
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, faq))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:5]]


def search_cases(query: str) -> list[dict]:
    """
    クエリに関連する審査事例を返す（簡易キーワードマッチ）。
    """
    kb = load_knowledge()
    cases = kb.get("cases", [])
    keywords = [w for w in query.split() if len(w) >= 2]
    if not keywords:
        return cases[:2]

    scored: list[tuple[int, dict]] = []
    for case in cases:
        text = (
            case.get("title", "")
            + case.get("industry", "")
            + case.get("decision", "")
            + case.get("notes", "")
            + " ".join(case.get("lessons", []))
        )
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, case))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:3]]


# ─── チャット用コンテキスト生成 ───────────────────────────────────────────────

def build_knowledge_context(
    query: str = "",
    industry: str = "",
    use_faq: bool = True,
    use_cases: bool = True,
    use_manual: bool = True,
    use_industry_guide: bool = True,
    use_improvement: bool = False,
    max_tokens_approx: int = 3000,
) -> str:
    """
    チャットのシステムプロンプトに挿入するナレッジコンテキストを生成する。

    Parameters
    ----------
    query : ユーザーの質問テキスト（FAQ・事例の検索に使用）
    industry : 業種名（業種別ガイドのフィルタに使用）
    use_faq : FAQ を含めるか
    use_cases : 事例集を含めるか
    use_manual : 審査マニュアルを含めるか
    use_industry_guide : 業種別ガイドを含めるか
    use_improvement : スコア改善ガイドを含めるか
    max_tokens_approx : 目安の最大文字数（超えたら後ろを切る）

    Returns
    -------
    str : ナレッジコンテキスト文字列
    """
    sections = [
        "=== リース審査ナレッジベース ===",
        get_scoring_overview(),
    ]

    if use_manual:
        sections.append(get_manual_text())

    if use_industry_guide:
        sections.append(get_industry_guide_text(industry_name=industry))

    if use_faq:
        if query:
            matched = search_faq(query)
            if matched:
                lines = ["【関連FAQ】"]
                for faq in matched:
                    lines.append(f"\nQ: {faq.get('q','')}")
                    lines.append(f"A: {faq.get('a','')}")
                sections.append("\n".join(lines))
            else:
                sections.append(get_faq_text(max_items=5))
        else:
            sections.append(get_faq_text(max_items=5))

    if use_cases:
        if query:
            matched_cases = search_cases(query)
            if matched_cases:
                lines = ["【関連審査事例】"]
                for case in matched_cases:
                    lines.append(f"\n▶ {case.get('title','')}")
                    q = case.get("quant_score") or case.get("quant_score_initial", "")
                    ql = case.get("qual_total", "")
                    t = case.get("total_score") or case.get("total_score_final", "")
                    g = case.get("grade") or case.get("grade_final", "")
                    lines.append(f"  スコア: 定量{q} / 定性{ql} / 総合{t} → {g}")
                    lines.append(f"  判定: {case.get('decision','')}")
                    for lesson in case.get("lessons", [])[:2]:
                        lines.append(f"  ・{lesson}")
                sections.append("\n".join(lines))
            else:
                sections.append(get_cases_text(max_cases=2))
        else:
            sections.append(get_cases_text(max_cases=2))

    if use_improvement:
        sections.append(get_improvement_guide_text())

    context = "\n\n".join(sections)

    if len(context) > max_tokens_approx:
        context = context[:max_tokens_approx] + "\n…（省略）"

    return context


def get_system_prompt_with_knowledge(
    base_system: str,
    query: str = "",
    industry: str = "",
    use_faq: bool = True,
    use_cases: bool = True,
    use_manual: bool = True,
    use_industry_guide: bool = True,
    use_improvement: bool = False,
) -> str:
    """
    ベースのシステムプロンプトにナレッジベースを追記して返す。
    AIチャット呼び出し時にシステムプロンプトとして使用する。
    """
    kb_context = build_knowledge_context(
        query=query,
        industry=industry,
        use_faq=use_faq,
        use_cases=use_cases,
        use_manual=use_manual,
        use_industry_guide=use_industry_guide,
        use_improvement=use_improvement,
    )
    return f"{base_system}\n\n{kb_context}"
