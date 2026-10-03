# Vertex credit daily LaunchAgent

The Obsidian-to-Vertex sync and nightly quality evaluation run in the dedicated
`com.tunelease.vertex-credit-daily` LaunchAgent at 05:30. Pulling the repository
does not register a new LaunchAgent automatically. After deploying a change to
`launchd/com.tunelease.vertex-credit-daily.plist` or introducing the agent on a
Mac, run:

```bash
bash scripts/install_vertex_credit_launchagent.sh
```

The installer resolves the current repository, Python, Vault, and log paths,
validates the installed plist, and reloads the service. It deliberately does
not start the job immediately; the next run follows the 05:30 schedule.

Verify installation without running the sync:

```bash
launchctl print "gui/$(id -u)/com.tunelease.vertex-credit-daily"
python scripts/audit_launchagents.py
```

The daily runner performs a destructive FULL reconciliation. It aborts before
any remote write when the local export is empty or has unexpectedly dropped by
more than 30%. For a reviewed, intentional corpus deletion, run the sync
manually with `--allow-large-delete`.
