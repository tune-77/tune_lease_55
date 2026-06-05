---
name: restart-next-cloudflare
description: Restart the tune_lease_55 Next/FastAPI production-style launcher with Cloudflare Tunnel. Use when the user says "再起動して", "Cloudflareも同時に", "public tunnel", "外部URL", "NextをCloudflare付きで起動", "run_next_stable", or asks to restart the lease app and expose it publicly.
---

# Restart Next With Cloudflare

Use this skill to keep the current app stack available and publish the Next UI through Cloudflare Tunnel. Prefer health checks and targeted restarts before a full restart.

## Fast Path

Run from the repository root:

```bash
PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH RESTART_SCOPE=status PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

If API and Next are OK, do not restart. Report the existing URL.

## Targeted Restarts

Use these before a full restart:

```bash
PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH RESTART_SCOPE=api PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

```bash
PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH RESTART_SCOPE=next PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

```bash
PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH RESTART_SCOPE=tunnel PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

These nudge the supervised child process and let the existing launcher restart only that service.

## Full Restart

Use this only when targeted restart is insufficient or the launcher/supervisors are stale:

```bash
FORCE_RESTART=1 PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

## Workflow

1. Run status first.
2. If only API is down, run `RESTART_SCOPE=api`.
3. If only Next is down, run `RESTART_SCOPE=next`.
4. If the public URL is dead but local Next is OK, run `RESTART_SCOPE=tunnel`.
5. Use full restart only if the launcher itself is stale or targeted restarts do not recover the app.
6. Keep the launcher session running; it supervises FastAPI and Next.
7. Verify health:

```bash
curl --max-time 10 -sS http://127.0.0.1:8000/docs >/dev/null
curl --max-time 10 -sS http://127.0.0.1:3000/ >/dev/null
```

8. Report the local Next URL and the Cloudflare URL printed by `run_next_stable.sh`.

## Notes

- Use `PUBLIC_TUNNEL=1`; do not fall back to `run_with_cloudflare.sh` unless the user explicitly wants the old Streamlit route.
- Use `API_HOST=127.0.0.1` and `NEXT_HOST=127.0.0.1` for local-only binding unless the user asks for LAN binding.
- Local sandbox `curl` can occasionally fail even while `next-server` is listening. Check `lsof -nP -iTCP:3000 -sTCP:LISTEN` or use an approved external `curl --max-time` before forcing a full restart.
- If `cloudflared` is missing, say so and install only after user approval.
- Do not kill unrelated processes. Only rely on `run_next_stable.sh` cleanup or ask before destructive cleanup.
