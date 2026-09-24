# TypeSafe/Jev Routing and RAG Pilot

The shared Obsidian RAG path can ask Jev to classify retrieved passages before
they are added to an AI prompt. Every runtime keeps the feature disabled unless
an operator explicitly configures it.

The main `/api/chat` retrieval path uses the same gate. It first retrieves a
larger local shortlist, sends only the current query and compact passage text to
TypeSafe, and then restores the configured prompt limit. The local file path is
never included in that request.

## Enable

Set these server-side environment variables:

```bash
TYPESAFE_RAG_ENABLED=1
TYPESAFE_API_KEY=...
```

On macOS, a local operator can use an existing Keychain item without copying the
secret into the launcher environment:

```bash
TYPESAFE_RAG_ENABLED=1
TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key
```

Disable it for a local run with `TYPESAFE_RAG_ENABLED=0`. This is also the
application default.

Question-category routing can be evaluated in shadow mode with:

```bash
TYPESAFE_ROUTING_MODE=shadow
```

Shadow mode compares Jev's typed Choice with the existing classifier and writes
only category, agreement, confidence, model, and token usage to the server log.
It does not change the selected category and does not log the user message.

Available routing modes are:

- `off`: do not call Jev for routing
- `shadow`: compare Jev with the existing classifier but keep existing behavior
- `enforce`: use Jev only when its confidence reaches the configured threshold;
  otherwise keep the existing classifier result

`enforce` is implemented for controlled evaluation but should not become the
default until the shadow results have been measured.

Optional settings:

- `TYPESAFE_MODEL` (default: `jev-latest`)
- `TYPESAFE_ENDPOINT` (default: `https://api.typesafe.ai/v1/systemone`)
- `TYPESAFE_RAG_TIMEOUT_SECONDS` (default: `8`)
- `TYPESAFE_RAG_MAX_CANDIDATES` (default: `8`, maximum: `20`)
- `TYPESAFE_ROUTING_MODE` (default: `off`)
- `TYPESAFE_ROUTING_CONFIDENCE` (default: `0.85`, only used by `enforce`)
- `TYPESAFE_ALLOW_SCREENING` (default: unset/false)
- `TYPESAFE_ALLOW_SHARED_CONTEXT` (default: unset/false; required by the shared Obsidian helper)

Do not put the API key in source code. Enabling the pilot sends the user query
and candidate snippets to TypeSafe. Local file paths are deliberately excluded.
Use only data approved for external processing; keep the feature disabled for
confidential screening cases until the applicable retention terms are approved.
Questions already classified as `lease_screening` therefore stay entirely on
the existing local/Gemini path by default. Routing and RAG can process them only
when an operator explicitly sets `TYPESAFE_ALLOW_SCREENING=1`.
The code also applies a conservative pre-routing sensitivity check, so this
protection does not depend only on the probabilistic baseline category. Shared
Obsidian callers have no reliable category contract and therefore require the
additional `TYPESAFE_ALLOW_SHARED_CONTEXT=1` opt-in.

## Judgments and routing

Each passage receives four independent Noul probabilities:

- relevant to the query
- contains usable answer evidence
- contradicts a premise in the query
- appears to contain an instruction or prompt injection

Thresholds remain explicit in `typesafe_rag_guard.py`. Jev supplies semantic
judgments; code makes the inclusion decision. Injection detection is an
additional signal, not a security boundary.

If TypeSafe is disabled, times out, returns an error, or returns malformed
probabilities, the existing retrieval candidates are used unchanged. The shared
Obsidian context records the outcome under `retrieval_boundary.typesafe`; the
main chat exposes the same safe metadata as `memory_debug.typesafe_rag` when
debug metadata is requested.

`verify_citation_support()` is also available for checking one claim against one
source. It is not wired into answer generation yet because claim extraction and
the action threshold need separate evaluation.

## Obsidian curation / ChromaDB reduction shadow tests

Two more shadow-only tests reuse the same Jev gate to measure whether it is
worth wiring Jev into Obsidian vault curation or ChromaDB reduction later.
Neither ever writes to the Vault, reindexes ChromaDB, or changes the
improvement ledger — they only append a jsonl log for offline agreement
analysis via `scripts/typesafe_curation_shadow_report.py`.

- `TYPESAFE_CURATION_MODE=shadow` — in `api/shion_obsidian_curator.py`,
  `review_obsidian_vault_health()` asks Jev whether each `connect_used_isolate`
  proposal's suggested hub notes are actually related to the isolated note
  (title only, no note body), and logs the routing to
  `data/obsidian_curation_shadow_log.jsonl`.
- `TYPESAFE_STALENESS_MODE=shadow` — in `scripts/analyze_rag_staleness.py`,
  `shadow_log_jev_relevance()` asks Jev whether each top stale /
  important_but_unused candidate (title + tags only) is still valuable, and
  logs agreement with the existing keyword heuristic to
  `data/rag_staleness_shadow_log.jsonl`.

Both additionally require `TYPESAFE_ALLOW_SHARED_CONTEXT=1` (same opt-in the
shared Obsidian context helper already requires) since they read vault
metadata. Neither mode has an `enforce` option yet — promoting either to
influence real curator auto-actions or reduction candidates needs a
follow-up decision once the agreement report has enough samples.

## Verification

```bash
pytest -q tests/test_typesafe_rag_guard.py tests/test_obsidian_ai_context.py \
  tests/test_chat_routing.py tests/test_chat_retrieval.py \
  tests/test_chat_architecture_helpers.py
```

The tests use injected fake responses and never contact TypeSafe.
