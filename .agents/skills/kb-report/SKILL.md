---
name: kb-report
description: Obsidian wiki/knowledge notesを根拠に、問いへの回答をreportsへMarkdown保存する。ユーザーが「kb-report」「wikiだけを使って」「根拠付きレポート」「reportsに残す」と言った時に使用する。
---

# kb-report

Use this skill to answer a focused question from promoted Obsidian knowledge and save the result as a traceable report.

## When to use

- The user asks for a report based on Obsidian wiki/knowledge notes.
- The answer should remain reusable after the chat ends.
- The user needs source traceability for lease screening judgment.
- The task asks to avoid raw chat logs and use curated knowledge.

## Workflow

1. Build search terms with the shared project path.
   - Use `obsidian_query.py` for query splitting when running local code.
   - Use `obsidian_ai_context.py` or `mobile_app/obsidian_bridge.py` for context.
   - Do not scan the Vault directly with `vault.rglob("*.md")`.

2. Prefer promoted knowledge.
   - Prefer `Projects/tune_lease_55/tune_lease_55 Wiki.md`.
   - Prefer `Projects/tune_lease_55/Asset Knowledge/`.
   - Prefer `Projects/tune_lease_55/Cases/` when the question is about past screening cases.
   - Treat `AI Chat`, `Daily`, `Improvement Log`, and `Weekly Review` as candidate material, not final authority.

3. Write a report under the Vault.
   - Default path: `Projects/tune_lease_55/reports/{YYYY-MM-DD}_{topic-slug}.md`.
   - Include a frontmatter block:
     - `created`
     - `source: codex`
     - `type: kb_report`
     - `generated_from_query`
     - `used_wiki_pages`
     - `source_notes`

4. Report format.
   - Start with a 3-line lead.
   - Then write concise sections for findings, implications, and next actions.
   - End with source links to the used wiki pages.

## Constraints

- Do not save secrets, raw DB rows, API keys, or private customer details.
- Do not use unpromoted chat logs as the only basis for a conclusion.
- If the source set is weak, say that in the report and mark the answer as tentative.
- Keep the final chat response short: saved path, 3-line lead, source count.

