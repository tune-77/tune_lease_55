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

Commands:

```bash
python scripts/auto_research_lease_judgment.py --dry-run
python scripts/auto_research_lease_judgment.py
python scripts/auto_research_lease_judgment.py --topic residual-value
python daily_knowledge_feed.py --only research
```

The dedicated `com.tunelease.lease-judgment-autoresearch` LaunchAgent runs
every day at 06:10. The monthly knowledge feed remains monthly, while its task
registry also supports `--only research`. Saved notes are immediately indexed
into the Obsidian RAG store.
