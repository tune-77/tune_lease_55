# HEARTBEAT.md

## Heartbeat Checklist

- Check urgent unread email only if the email source is actually connected.
- Check calendar events in the next 24 to 48 hours only if calendar access is available.
- Ignore weather and social mentions by default unless the user explicitly asked for them.
- If no configured source has changed, reply `HEARTBEAT_OK`.

## Memory Maintenance

- The daily improvement pipeline now auto-promotes durable items from `memory/YYYY-MM-DD.md` into `MEMORY.md`.
- Every few days, review recent `memory/YYYY-MM-DD.md` notes for items the auto-promotion missed.
- Keep long-term memory concise: fact, impact, next action.
