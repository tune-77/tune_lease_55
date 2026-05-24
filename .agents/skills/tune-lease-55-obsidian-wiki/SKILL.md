---
name: tune-lease-55-obsidian-wiki
description: Obsidian wiki/linking and maintenance workflow for tune_lease_55. Use when the user asks to "obdisian wiki", "Obsidian wiki", "wiki", "Obsidian", "connect notes", "search terms", "Related links", "related sections", or to make the project easier to search across daily notes and project notes.
---

# tune-lease-55 Obsidian Wiki

Use this skill to turn project work into searchable Obsidian notes and keep the wiki structure tidy over time.

## When to use

- The user asks to "wikiして", "Obsidianにまとめて", "検索しやすくして", or "関連ノートを繋げて".
- The work spans multiple project notes and should be navigable later.
- The user wants searchability, bidirectional links, or a reusable note structure.
- The user wants to normalize note entrances, Related sections, or search-term indexes.
- The user wants periodic wiki maintenance, not just a one-off note update.

## Core workflow

1. Identify the canonical hub note for the topic.
   - Prefer `Projects/tune_lease_55/tune_lease_55 Wiki.md`.
   - Add a dedicated project note when the topic is substantial.

2. Add a search-first note structure.
   - Create or update a `検索語インデックス.md` when searchability matters.
   - Include synonyms, English names, abbreviations, file names, and API paths.
   - Group terms by theme: data/analysis, knowledge/rules, AI/agents, operations.

3. Add bidirectional links.
   - Every major topic note should have a `## Related` section.
   - The wiki hub should link out to the topic note.
   - The topic note should link back to the wiki hub and sibling notes.

4. Keep notes easy to find.
   - Use concrete titles and consistent naming.
   - Add search aliases near the top of the note body.
   - Prefer short, repeated anchors over long prose.

5. Capture the work safely.
   - Summarize decisions, changes, and verification.
   - Omit secrets, raw DB rows, tokens, and private data.
   - If the task is large, also append a daily log entry.

6. Maintain the wiki shape.
   - Keep `Wiki -> 検索語インデックス -> 元ノート` as the default reading path.
   - Normalize `## Related` blocks so the first links are the hub, the search index, then the source note.
   - Keep only high-signal links in each note; do not widen the graph unless the user asks.

## Preferred note targets

- `Projects/tune_lease_55/tune_lease_55 Wiki.md`
- `Projects/tune_lease_55/検索語インデックス.md`
- `Projects/tune_lease_55/2026-05-DD_*.md`
- `Daily/YYYY-MM-DD.md`

## Practical rules

- When the topic is about analysis, add synonyms for the model names and metrics.
- When the topic is about AI agents, add synonyms for each agent name and endpoint.
- When the topic is about operations, add tool names like `uv`, `py_compile`, or `tsc`.
- When the user says "もっと繋げて", expand the hub and the `Related` sections first.
- When the user asks to "整える", "見直す", "揃える", or "粒度を揃える", normalize the note graph and trim duplicates before adding new links.

## Link curation rules

Avoid turning the wiki into an all-purpose link dump. Prefer links that help the next real reading or decision step.

- Put only curated, high-signal links in hub notes.
- Aggregate detailed date logs in `Projects/tune_lease_55/日付別DATA連携インデックス.md` instead of adding every date note to every hub.
- Keep each note's `## Related` section to roughly 5-8 links unless the user explicitly asks for a wider index.
- Promote `AI Chat` / `Improvement Log` notes into project hubs only when they are accepted, planned for implementation, or clearly reusable.
- Leave ordinary conversation notes in `Daily/` or `Projects/tune_lease_55/AI Chat/` without broad wiki promotion.
- Link by purpose, not by keyword alone: connect notes that answer the same decision, share a data lineage, or map directly to an implementation file, API, UI, or source dataset.

## Use the helper

Use `python3 .agents/skills/obsidian/scripts/obsidian_note.py` for daily notes, project notes, and search.
