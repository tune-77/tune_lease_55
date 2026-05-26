---
name: restart-next-cloudflare
description: Restart the tune_lease_55 Next/FastAPI production-style launcher with Cloudflare Tunnel. Use when the user says "再起動して", "Cloudflareも同時に", "public tunnel", "外部URL", "NextをCloudflare付きで起動", "run_next_stable", or asks to restart the lease app and expose it publicly.
---

# Restart Next With Cloudflare

Use this skill to restart the current app stack and publish the Next UI through Cloudflare Tunnel.

## Command

Run from the repository root:

```bash
FORCE_RESTART=1 PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

This does all of the following:

- Builds `frontend/`.
- Stops stale launcher, FastAPI, Next, and matching Cloudflare quick-tunnel processes.
- Starts FastAPI on `http://127.0.0.1:8000`.
- Starts Next on `http://127.0.0.1:3000`.
- Starts `cloudflared tunnel --url http://127.0.0.1:3000`.

## Workflow

1. Run the command above with escalation if sandboxing blocks process inspection, port cleanup, or tunnel startup.
2. Keep the launcher session running; it supervises FastAPI and Next.
3. Read the newest tunnel log:

```bash
ls -t logs/next/tunnel_*.log | head -1
```

Then extract the public URL:

```bash
rg -o "https://[a-zA-Z0-9-]+\\.trycloudflare\\.com" logs/next/tunnel_*.log
```

4. Verify local health:

```bash
curl -sS http://127.0.0.1:8000/docs >/dev/null
curl -sS http://127.0.0.1:3000/ >/dev/null
```

5. Report the local Next URL and the Cloudflare URL. If the tunnel URL has not appeared yet, poll the newest tunnel log a few times before reporting.

## Notes

- Use `PUBLIC_TUNNEL=1`; do not fall back to `run_with_cloudflare.sh` unless the user explicitly wants the old Streamlit route.
- Use `API_HOST=127.0.0.1` and `NEXT_HOST=127.0.0.1` for local-only binding unless the user asks for LAN binding.
- If `cloudflared` is missing, say so and install only after user approval.
- Do not kill unrelated processes. Only rely on `run_next_stable.sh` cleanup or ask before destructive cleanup.
