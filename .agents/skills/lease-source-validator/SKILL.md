---
name: lease-source-validator
description: リース審査AIで外部情報を使う前に情報源の信頼性・鮮度・偏り・審査適用可否を検証するスキル。リースニュース、業界動向、補助金、倒産/景気/金利情報、法令・制度、Web調査、Auto Research素材について「ソース確認」「信頼できる？」「根拠チェック」「情報健康」「使っていい情報か」を求められた時に使用する。
---

# Lease Source Validator

## Purpose

Validate sources before they influence lease screening comments, Obsidian notes, research briefs, or judgment-asset candidates.

## Workflow

Reason: source freshness and reliability directly affect screening explanations and knowledge health.
Scope: use for external news, research, law/regulation, market indicators, subsidy information, and industry benchmarks.
Retirement: remove if source validation becomes enforced by the research ingestion pipeline.

1. Classify each source.
   - `primary`: government, regulator, company disclosure, court/official filing, original statistics.
   - `specialist_secondary`: industry body, reputable trade publication, analyst report with methodology.
   - `general_secondary`: newspaper, general web article, blog summary.
   - `weak`: anonymous, no date, no author, SEO aggregate, unverifiable social post.

2. Check five dimensions.
   - `date`: publication date and event date.
   - `authority`: author, publisher, institutional position.
   - `evidence`: data, citations, documents, named facts.
   - `bias`: sales motive, political angle, sponsored content, single-company framing.
   - `lease_relevance`: how directly it affects repayment source, asset value, customer demand, funding cost, regulation, or competition.

3. Assign a use level.
   - `usable_as_fact`: primary or strong source; can support factual statements.
   - `usable_as_signal`: plausible but not definitive; use as a caution or question.
   - `background_only`: useful context but not enough for case judgment.
   - `do_not_use`: stale, unsupported, conflicted, or irrelevant.

4. State what would change in screening.
   - If nothing changes, say so.
   - Prefer concise outputs: confirmed fact, uncertainty, recommended next check.

## Output

Use this table:

| Source | Type | Freshness | Reliability | Bias/Risk | Lease relevance | Use level |
|---|---|---|---|---|---|---|

Then add:

- `使ってよい結論`
- `審査コメントに入れるなら`
- `追加確認が必要な点`
- `使わない方がよい点`

## Constraints

- Browse or verify when the fact may have changed recently, especially laws, rates, news, executives, subsidies, market data, and regulations.
- Do not treat a single weak source as a judgment asset.
- Do not overstate broad macro news as proof for an individual borrower.
- Keep primary-source links or local note paths in the final answer when available.
