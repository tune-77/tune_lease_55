---
name: restart-next-cloudflare
description: Restart the tune_lease_55 Next/FastAPI production-style launcher with Cloudflare Tunnel. Use when the user says "再起動して", "Cloudflareも同時に", "public tunnel", "外部URL", "NextをCloudflare付きで起動", "run_next_stable", or asks to restart the lease app and expose it publicly.
---

# Restart Next With Cloudflare

Use this skill to keep the current app stack available and publish the Next UI through Cloudflare Tunnel. Prefer the persistent LaunchAgent so the app does not stop when the Codex or terminal session ends.

## Fast Path

Run from the repository root:

```bash
launchctl print gui/$(id -u)/com.tunelease.next
```

If the service exists, restart it persistently:

```bash
launchctl kickstart -k gui/$(id -u)/com.tunelease.next
```

If the service is not installed, install and start it:

```bash
bash scripts/install_next_launchagent.sh
```

Then check application status:

```bash
PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH RESTART_SCOPE=status PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

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

Use this only when LaunchAgent operation is unavailable or the user explicitly requests a foreground launcher:

```bash
FORCE_RESTART=1 PATH=/usr/local/bin:/opt/homebrew/bin:/Applications/Codex.app/Contents/Resources:$PATH PUBLIC_TUNNEL=1 API_HOST=127.0.0.1 NEXT_HOST=127.0.0.1 bash run_next_stable.sh
```

## Workflow

1. Run status first.
2. Confirm `com.tunelease.next` is registered with `launchctl`.
3. Use `launchctl kickstart -k` for a persistent full restart.
4. If only API is down, run `RESTART_SCOPE=api`.
5. If only Next is down, run `RESTART_SCOPE=next`.
6. If the public URL is dead but local Next is OK, run `RESTART_SCOPE=tunnel`.
7. Use the foreground full restart only if launchd is unavailable.
8. Verify health:

```bash
curl --max-time 10 -sS http://127.0.0.1:8000/docs >/dev/null
curl --max-time 10 -sS http://127.0.0.1:3000/ >/dev/null
```

9. Report the local Next URL and the newest Cloudflare URL.

## Notes

- Use `PUBLIC_TUNNEL=1`; do not fall back to `run_with_cloudflare.sh` unless the user explicitly wants the old Streamlit route.
- Use `API_HOST=127.0.0.1` and `NEXT_HOST=127.0.0.1` for local-only binding unless the user asks for LAN binding.
- Local sandbox `curl` can occasionally fail even while `next-server` is listening. Check `lsof -nP -iTCP:3000 -sTCP:LISTEN` or use an approved external `curl --max-time` before forcing a full restart.
- If `cloudflared` is missing, say so and install only after user approval.
- Do not kill unrelated processes. Only rely on `run_next_stable.sh` cleanup or ask before destructive cleanup.
- Default mode is Cloudflare's "quick tunnel" (`cloudflared tunnel --url ...`): no auth, ephemeral URL, not officially supported for production, and prone to dropping. If `CLOUDFLARE_TUNNEL_CONFIG` (path to a named-tunnel config.yml) and `CLOUDFLARE_TUNNEL_HOSTNAME` are set, the launcher switches to a Named Tunnel with a stable, authenticated hostname instead — see `CLOUD_RUN.md` / the top of `run_next_stable.sh` for setup.
