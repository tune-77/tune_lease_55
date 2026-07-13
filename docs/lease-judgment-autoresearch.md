# Lease Judgment Auto Research

`scripts/auto_research_lease_judgment.py` researches knowledge needed for
lease-underwriting decisions and saves it to the normal Obsidian Vault.

Default destination:

`Projects/tune_lease_55/Research/Auto Research/`

The daily rotation covers:

- cash flow and repayment capacity
- residual value and used-asset liquidity
- lease accounting, tax, and regulation
- contract, ownership, inspection, and fraud risk
- industry default indicators
- asset utilization, maintenance, and replacement
- subsidies and payment timing
- pricing, rates, and competitive conditions
- bank support and guarantees
- arrears, collection, and asset preservation

Each note must include:

- source-quality assessment
- confirmed facts
- lease-screening application
- questions for the customer or sales representative
- signals that can change approval conditions
- counter-evidence and limitations
- conditions requiring a knowledge refresh

Gemini Google Search is used for research. The script retries when grounding
URLs are missing and refuses to save a new note if no verifiable source URL is
returned. Sources are labeled as `primary`, `recognized`, or `supplementary`.
All generated notes remain `needs_human_review`.

Judgment asset use is a separate, review-only step. Auto Research output is not
promoted directly into long-term memory or scoring rules. Run
`scripts/build_autoresearch_judgment_asset_candidates.py` to extract only the
substantive sections into candidates:

- `リース審査への適用` -> application rules
- `担当者が確認する質問` -> confirmation questions
- `承認条件を変える兆候` -> condition signals
- `反証・過信してはいけない点` -> cautions

The candidate report stays `not_promoted` until the item is used in a real case,
gets human usefulness feedback, and is later checked against the outcome.
Candidate feedback is stored by candidate id in
`data/autoresearch_judgment_asset_candidate_state.json` and overlaid on each
regenerated report:

- `use_count`: how many case reviews used the candidate
- `useful_count`: human marked it useful
- `rejected_count`: human marked it wrong or noisy
- `neutral_count`: used but not clearly useful
- `verified_status`: `unverified`, `supported`, `contradicted`, or `unclear`

A candidate becomes `ready_for_promotion` only when it has at least one useful
human feedback entry and `verified_status=supported`. The script still does not
promote it automatically.

Textbook generalities are explicitly blocked. A candidate that only says obvious
things such as "財務内容を確認する" or "業界動向を確認する" is marked
`asset_quality=textbook_general` and `promotion_status=not_promoted_textbook_general`.
The operating rule is: `当たり前なこと言ってやった気になるな`. Judgment assets
must change a case action, approval condition, rebuttal, or rejection reason.

The dedicated Auto Research job refreshes candidates automatically after a
research note is saved. The refresh uses the last 14 days by default; override
with `AUTORESEARCH_CANDIDATE_DAYS` if needed. Candidate refresh errors are
reported in the script output but do not delete or invalidate the research note
that was already saved.

Similar candidates are compressed in the refreshed report. The script keeps one
representative candidate for the same research topic and candidate type, records
the absorbed items in `similar_candidates`, and reports the number as
`deduped_similar_candidates`. This keeps daily growth from turning into a noisy
list of near-duplicate questions.

Screening reviews can now fetch up to three candidates from
`/api/judgment-asset-candidates/screening`. The screening page passes those
candidates into the Shion review prompt and shows a small feedback card. Human
feedback uses `効いた / 微妙 / 外した`, updating the candidate state counts. This
is still a trial path: candidates are not promoted merely because they appeared
in a review.

Commands:

```bash
python scripts/auto_research_lease_judgment.py --dry-run
python scripts/auto_research_lease_judgment.py
python scripts/auto_research_lease_judgment.py --topic residual-value
python scripts/build_autoresearch_judgment_asset_candidates.py --days 14
python daily_knowledge_feed.py --only research
```

The dedicated `com.tunelease.lease-judgment-autoresearch` LaunchAgent runs
every day at 06:10. The monthly knowledge feed remains monthly, while its task
registry also supports `--only research`. Saved notes are immediately indexed
into the Obsidian RAG store, then the judgment asset candidate report is
regenerated with the existing state metrics overlaid.
