# SHION (紫苑) — The AI That Doesn't Throw Away "Something Feels Off"

*[日本語版 README はこちら](README.md)*

[![SHION demo video](https://img.youtube.com/vi/KWLbWEHHn-E/hqdefault.jpg)](https://youtu.be/KWLbWEHHn-E)

**SHION is an AI for lease-financing credit review that captures the "something feels off" a human reviewer senses in the field — instead of letting it evaporate — and turns it into checkpoints, conditions, and material for the next decision.**

A financial score alone rarely explains everything. Revenue looks fine, but collection feels risky. The asset isn't bad, but handing it to this company feels uneasy. The deal looks subsidy-dependent. It might pass with the right conditions. These vague, half-formed remarks are exactly what a normal database or rule engine tends to drop.

SHION doesn't throw that remark away. It keeps the AI dialogue, the human's correction, the approval condition, the counter-argument, and the eventual outcome in a form that can be traced later and recalled on the next similar deal.

SHION isn't a magic AI that "solves" credit decisions. It's an AI that preserves the discomfort reviewers tend to overlook, makes it explainable, and hands it back to the next reviewer on the next deal.

> Instead of letting AI make the call, SHION keeps a history of how humans corrected the AI — and feeds that history into the next decision.

## In engineering terms: GitHub for human judgment

Code has GitHub. Human business judgment doesn't. SHION preserves the correction history that emerges from AI/human dialogue and reuses it in the next decision — the way GitHub preserves the history of code.

| GitHub | SHION |
|---|---|
| issue | an oversight, a judgment bug, something that feels off |
| commit | a human's correction, confirmation condition, approval condition |
| diff | the gap between the AI's answer and the human's correction |
| pull request | review of a candidate judgment asset |
| merge | only human-approved judgment assets get promoted to the main store |
| revert | a wrong judgment asset gets rolled back / reconsidered |
| regression test | checking whether the same issue is recalled on the next deal |

SHION is not a plain RAG bot. Rather than only retrieving documents and answering, it **quarantines, gets human approval, and promotes** human discomfort, corrections, approval conditions, counter-evidence, and outcome data — reusing them on the next deal and later verifying whether they actually helped. It's a prototype for operating judgment, not just answering questions.

By design, SHION separates its **body** from its **brain**:

- **Body**: the UI / API / scoring / chat / SHION review running on Cloud Run
- **Brain**: judgment assets, past decisions, discomfort notes, and improvement logs stored in an Obsidian / Markdown vault

Cloud Run is the runtime that drives SHION, but it never directly rewrites the brain's source of truth. Improvement candidates first enter a quarantine queue and only get promoted to the brain after human approval. Because of this separation, the "brain" behind lease-financing review could in principle be swapped for a different domain — legal review, sales support, CS quality audits, and so on — while keeping the same body.

## What we built, in one table

| Angle | Content |
|---|---|
| One line | An AI that doesn't throw away "something feels off" |
| Problem it solves | A veteran reviewer's discomfort, checkpoints, and approval conditions disappear deal by deal |
| What's new | Turns a human's vague discomfort into a checkpoint, condition, or explanation reusable later |
| What the demo shows | discomfort → AI puts it into words → human corrects it → recalled on the next deal / evaluation GUI |
| Core tech | Next.js / FastAPI / Gemini / ADK / Cloud Run / Obsidian-RAG / quarantine DB / judgment-asset DevOps / brain swapping |

SHION is not a one-shot chatbot meant to "produce the right answer." It's an operating platform that stores a reviewer's on-the-ground discomfort, corrections, approval conditions, and counter-evidence as reusable **judgment parts**, recombines them for the next deal, and lets humans re-evaluate them again. In other words, it turns prompt engineering from an individual's craft into a PDCA process for business judgment.

SHION does not aim to replace human judgment. It aims to amplify human judgment performance in lease-financing credit review.

## Judgment lifecycle management

SHION's core isn't summarizing approval memos — it's **managing the flow by which a judgment is born, used, corrected, and verified against outcomes**:

```text
Deal input
-> SHION's checkpoints / discomfort / approval conditions
-> Human correction, hold, or adoption decision
-> Quarantined as a judgment-asset candidate
-> Promoted to a judgment asset after human approval
-> Reused on the next deal
-> Verified against outcomes: closed, lost, terms changed, delinquent, etc.
```

Comments left in approval memos or a CRM are the fuel for this flow. SHION goes one step further and treats pre-memo hesitation, the reasoning behind a conditional approval, the reasons behind a lost deal worth revisiting later, and reusable checkpoints all as first-class judgment events.

So SHION isn't aiming for fully automated credit decisions — it supports review that can be **explained, revisited, and continuously improved**.

## How this differs from a plain RAG bot

A plain RAG bot searches for relevant documents and answers:

```text
question -> document search -> answer
```

SHION doesn't stop at searching and answering. It turns the human judgment produced on one deal into a judgment asset usable on the next:

```text
deal
-> recall past judgments, sales notes, discomfort, conditions, counter-evidence
-> recompose them into judgment syntax for this deal
-> human evaluates and corrects
-> quarantine improvement candidates
-> only approved ones get promoted to judgment assets
-> reuse on the next deal
-> verify against the recorded outcome
```

In short, SHION isn't an AI that searches knowledge and answers — it's an **AI that operates judgment**.

## Separating body and brain

SHION separates the AI runtime from the source of truth for judgment assets:

| Element | Role | Example implementation |
|---|---|---|
| Body | Runs the UI, API, scoring, chat, and SHION review | Cloud Run / Next.js / FastAPI / Gemini / ADK |
| Brain | Holds judgment assets, past decisions, discomfort notes, conditional-approval reasoning, improvement logs | Obsidian / Markdown vault / GCS-synced copy / memory index |
| Safety layer | Never writes improvement candidates straight into the brain — always through quarantine, human approval, and promotion | quarantine DB / approval workflow / promotion scripts |

Because of this, the same Cloud Run runtime can, in principle, be pointed at a different domain simply by swapping which "brain" (vault) it's connected to:

```text
Lease Brain  -> checkpoints, approval conditions, memo drafting for lease-financing review
Legal Brain  -> contract-review issues, risk clauses, redline drafts
Sales Brain  -> next actions, loss reasons, proposal conditions from deal reviews
CS Brain     -> inquiry quality, recurrence prevention, response policy
```

This isn't a one-click magic transplant — domain-specific inputs, decision logic, evaluation metrics, and output templates still need to be adapted per domain. AI can generate a fast first draft, but a human still has to quarantine, approve, and update the final judgment criteria from real operating results.

SHION tends to transplant well into domains that share these traits: outcomes aren't decided by numbers alone; past judgment matters; human discomfort or exceptions matter; there's accountability; approval/rejection/conditional-approval history is kept; and outcomes can be verified after the fact.

## Why it matters for enterprises

SHION's value isn't just about answering cleverly in the moment. What actually matters for a business is whether it can later explain *why* a decision came out the way it did, whether it can trace what a human changed, and whether that correction gets carried back into the next case.

SHION treats this as **GitHub for judgment**:

| What enterprises need | What SHION preserves |
|---|---|
| Auditability | The AI's first-pass judgment, the human's correction, and why it became a judgment asset |
| Accountability | The source behind why a given checkpoint or approval condition exists |
| Governance | An operating model where only human-approved judgment assets get promoted to the main store |
| Knowledge continuity | A veteran's discomfort and conditions carried forward as checkpoints for the next deal |
| Verifying reuse | A history of where a preserved judgment reappeared on later deals |

## Reviewer's guide

Read this repository as a **business AI agent that captures on-the-ground checkpoints, judgments, and outcomes and carries them back into the next deal** — not as a document summarizer or an automated approval engine.

SHION's operational role is not to replace an absolute approve/reject decision. It's to narrow down what needs to be checked before a lease-financing deal moves forward, stop dangerous deals early, and accumulate human confirmation, judgment, and outcomes as judgment assets. This structure generalizes beyond lease-specific scoring into a general B2B sales judgment pattern: *initial deal info → key questions → answers collected → assessment updated → outcome → reuse*.

To review the technical core quickly, this order works well:

| Area | Key implementation |
|---|---|
| Screening API, outcome registration, judgment-asset candidates | `api/main.py`, `api/schemas.py` |
| ADK / SHION agent entry point | `api/shion_agent.py` |
| Screening input & demo screens | `frontend/src/app/screening/page.tsx`, `frontend/src/app/demo/page.tsx` |
| Q_risk and human-judgment feedback | `frontend/src/components/analysis/QRiskPanel.tsx` |
| Cloud Run input return & quarantine | `scripts/sync_cloudrun_inputs_from_gcs.py`, `scripts/promote_cloudrun_return_data.py` |
| Judgment-material extraction & canonicalization | `scripts/build_judgment_materials_preview.py`, `scripts/build_canonical_judgment_rules.py` |
| SHION's improvement PM & system monitoring | `frontend/src/app/lease-intelligence/page.tsx`, `/api/improvement-log`, `/api/improvement-pipeline/summary`, `/api/lease-system-gaps` |
| Memory, reflection, judgment-asset wiring | `lease_intelligence_dialogue.py`, `lease_intelligence_reflection.py`, `scripts/build_shion_memory_index.py` |
| Regression tests | `tests/` |

What's worth evaluating isn't how clever the underlying model is on its own — it's the **operating loop that observes the AI's answers and human judgment and outcome data, quarantines them, and converts them into reusable judgment assets**.

## Quick start

```bash
bash run_next_stable.sh
```

After it starts:

- Next.js: `http://127.0.0.1:3000`
- FastAPI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

The stack is **Next.js + FastAPI + SQLite/PostgreSQL**, with Gemini + ADK (Agent Development Kit) for the agent layer and Cloud Run for deployment (API and Web deployed separately). See the Japanese README for full deployment, tunneling, and Obsidian/RAG setup instructions.

## Main screens

| Screen | Role |
|---|---|
| `/` | SHION concierge — routes you to the right screen based on prior activity |
| `/home` | Home: KPIs, hot topics, news, SHION's status |
| `/screening` | Screening input and analysis: numbers on the left, the "strategist AI" on the right |
| `/quantitative` / `/qualitative` | Quantitative / qualitative model comparison (LR / RandomForest / LightGBM) |
| `/history-dash` | Past deals, deal-closing drivers, tag trends |
| `/chat` | AI chat grounded in the Obsidian knowledge base |
| `/chat-compare` | SHION vs. a generic AI on the same question, to visualize the effect of memory/identity/experience loops |
| `/lease-intelligence` | Dedicated dialogue with SHION |
| `/voice-chat` | Real-time voice conversation with SHION |
| `/shion-core` | SHION's "core" — observe its experience, mood, confidence, and practical-knowledge map |
| `/debate` | Multi-persona debate: cautious, optimistic, innovator, and arbiter |
| `/report` | Screening report export |
| `/improvement-log` | Improvement candidates, AI-proposed rules, and auto-fix suggestions |

## Learn more

This file is an entry point for a global, English-reading audience. The full design doc — judgment-asset lifecycle, the DevOps loop for judgment quality, memory architecture, safety gates for memory promotion, and the complete system diagrams — lives in the primary [Japanese README](README.md). Machine translation of that document works well if you want the full depth.

---

*SHION was built for Findy's "DevOps × AI Agent Hackathon" (sponsored by Google Cloud Japan), and continues to evolve as a lease-financing credit-review AI agent.*
