---
name: tune-lease-55-obsidian-wiki
description: Obsidian wiki/linking workflow for tune_lease_55. Use when the user asks to "obdisian wiki", "Obsidian wiki", "wiki", "Obsidian", "connect notes", "search terms", "Related links", or to make the project easier to search across daily notes and project notes.
---

# tune-lease-55 Obsidian Wiki

Use this skill to turn project work into searchable Obsidian notes.

## When to use

- The user asks to "wikiして", "Obsidianにまとめて", "検索しやすくして", or "関連ノートを繋げて".
- The work spans multiple project notes and should be navigable later.
- The user wants searchability, bidirectional links, or a reusable note structure.

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

## Use the helper

Use `python3 .agents/skills/obsidian/scripts/obsidian_note.py` for daily notes, project notes, and search.
