<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.

## Taste Skill Usage

Reason: frontend work can drift into generic or over-animated UI quickly, so a lightweight taste check helps keep layout and hierarchy intentional.
Scope: apply only when creating or substantially revising frontend screens in this repository.
Retirement: remove if a project-specific design review pipeline replaces manual UI taste checks.

- Use `taste-skill` only for new screens, major redesigns, or visual review.
- Focus it on layout, spacing, typography, color, density, and distinctiveness.
- Do not use it for small bug fixes or backend/API work.
- Preserve the existing design language when one already exists; do not relitigate the whole UI on every change.
<!-- END:nextjs-agent-rules -->
