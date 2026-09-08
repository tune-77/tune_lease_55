# Security Best Practices Report

Date: 2026-09-06
Project: `tune_lease_55`
Scope: FastAPI backend and Next.js/React frontend, with emphasis on authentication, public deployment, state-changing routes, browser storage, and response security headers.

## Executive summary

The paid Codex Security CLI did not complete a reportable scan before the account usage limit was reached. Its partial runs returned zero **confirmed** findings, but their status was `failed`, so that result must not be interpreted as a clean bill of health.

A subsequent no-cost local review found one high-severity deployment/authentication issue, three medium-severity hardening/privacy issues, and one low-severity configuration gap. At discovery time, the highest-priority risk was that Cloud Run demo deployments were public and did not require the API key by default. The API exposes many state-changing routes, so an unauthenticated visitor could have mutated demo data, promoted knowledge assets, altered settings, triggered pipelines, or consumed external-model capacity.

## Remediation status

Local fixes were completed on 2026-09-06. They have been tested but have **not** been deployed to Cloud Run.

- SEC-001: Fixed locally. Cloud Run Web/combined services require IAM in every data mode; public tunnels require a separate user-supplied Basic-auth password; backend API access requires an API key and fails closed when unavailable.
- SEC-002: Fixed locally. OpenAPI, Swagger UI, and ReDoc are disabled whenever API-key enforcement is active.
- SEC-003: Partially fixed. Production CSP no longer permits `unsafe-eval`; removing `unsafe-inline` safely requires a nonce/hash rollout across Next.js rendering.
- SEC-004: Mitigated locally. Sensitive chat and screening handoff data now uses tab-scoped `sessionStorage`, and legacy persistent entries are removed.
- SEC-005: Fixed locally. FastAPI now applies an environment-aware trusted-host allowlist.

## High severity

### SEC-001 — Public demo deployment permits unauthenticated state-changing API calls

Status: Fixed locally; deployment pending.

- Rule ID: FASTAPI-AUTH-001 / FASTAPI-AUTHZ-001
- Severity: High
- Locations:
  - `scripts/deploy_cloud_run_api.sh:23-29`
  - `scripts/deploy_cloud_run_api.sh:35-42`
  - `scripts/deploy_cloud_run_api.sh:109-115`
  - `scripts/deploy_cloud_run_api.sh:139`
  - `api/api_key_auth.py:30-42`
  - `api/api_key_auth.py:45-62`
  - Examples of affected routes: `api/routers/settings.py:32`, `api/routers/improvement.py:316`, `api/routers/feedback_loop.py:2989`, `api/routers/vault_hub.py:285`, `api/routers/demo.py:46`
- Evidence at discovery:
  - Demo deployments default `REQUIRE_API_ACCESS_KEY` to `0`.
  - Demo deployments default `DEMO_READONLY` to `0`.
  - A missing `API_ACCESS_KEY` is explicitly accepted in demo mode.
  - Cloud Run is deployed with `--allow-unauthenticated`.
  - The authentication middleware passes `/api/*` requests through when no key is configured and key enforcement is disabled.
- Impact: Any internet visitor who can reach the demo API may invoke state-changing endpoints. Depending on the route and deployed data, this can alter settings or stored judgments, start jobs, promote or delete logical records through POST-style endpoints, and consume Gemini/OpenAI capacity.
- Fix: Require `API_ACCESS_KEY` for every internet-facing deployment, including demo. If anonymous demo interaction is required, expose an explicit allowlist of safe demo routes and deny all other mutating routes by default. Use a separate disposable demo datastore and rate limits for cost-bearing endpoints.
- Deployment note: Keep the current service non-public until the fixed revision is deployed with the `API_ACCESS_KEY` secret. `DEMO_READONLY=1` remains optional defense in depth and currently blocks only DELETE.
- False-positive notes: Verify the live Cloud Run revision and any upstream gateway/IAM policy. If an authenticated proxy blocks all direct API access, practical exposure is reduced; the deployment script itself still defaults to public unauthenticated access.

## Medium severity

### SEC-002 — API documentation remains public even when API-key authentication is enabled

Status: Fixed locally; deployment pending.

- Rule ID: FASTAPI-OPENAPI-001 / FASTAPI-AUTH-001
- Severity: Medium
- Locations:
  - `api/api_key_auth.py:21-22`
  - `api/api_key_auth.py:49-54`
  - `api/main.py:504-509`
- Evidence at discovery: `/docs`, `/redoc`, and `/openapi.json` were explicitly exempt from authentication, while `FastAPI(...)` used the default enabled documentation URLs.
- Impact: An unauthenticated visitor can enumerate the full API surface, request schemas, and sensitive administrative or state-changing operations. This lowers the cost of exploiting SEC-001 and future authorization mistakes.
- Fix: Set `docs_url=None`, `redoc_url=None`, and `openapi_url=None` in production, or move documentation behind the same authentication boundary.
- Mitigation: Block these paths at Cloud Run/load-balancer/Cloudflare routing.
- False-positive notes: Public documentation may be intentional for a public API, but this project exposes internal operational routes and shared-secret authentication rather than a public developer API.

### SEC-003 — Production CSP allowed `unsafe-inline` and `unsafe-eval`

Status: Partially fixed locally; production `unsafe-eval` removed, `unsafe-inline` remains.

- Rule ID: NEXT-CSP / JS-CSP-001 / JS-CSP-002
- Severity: Medium
- Location: `frontend/next.config.ts:22-31`
- Evidence at discovery: `script-src` contained both `'unsafe-inline'` and `'unsafe-eval'`; `style-src` also permitted `'unsafe-inline'`.
- Impact: If an HTML/script injection bug is introduced elsewhere, these directives materially weaken CSP's ability to prevent script execution. `unsafe-eval` is especially unnecessary in most production Next.js builds.
- Fix: Use environment-specific CSP. Remove `unsafe-eval` from production first, then migrate inline scripts to nonces or hashes. Keep development relaxations out of production headers.
- Mitigation: Preserve React escaping, DOMPurify at HTML sinks, strict input validation, and the existing `frame-ancestors`, `nosniff`, and referrer protections.
- False-positive notes: Some development tooling may require eval-like behavior. Verify the production build before removal; do not weaken production policy solely to support local development.

### SEC-004 — Lease, company, and chat content is persisted in browser `localStorage`

Status: Mitigated locally; deployment pending.

- Rule ID: JS-STORAGE-001 / REACT-STORAGE-001
- Severity: Medium
- Locations:
  - `frontend/src/lib/chatLocalHistory.ts:12-19`
  - `frontend/src/lib/chatLocalHistory.ts:46-64`
  - `frontend/src/components/analysis/GunshiAdvice.tsx:372-384`
  - `frontend/src/app/screening/page.tsx:2246-2259`
  - `frontend/src/app/screening/page.tsx:2281-2295`
- Evidence at discovery: Up to 120 chat messages were stored persistently; company names, locations, case identifiers, financial ratios, acquisition cost, screening results, and generated advice were also serialized to `localStorage`.
- Impact: Any successful same-origin XSS can read this data. It also remains accessible to later users of the same browser profile and may persist longer than intended for credit-screening information.
- Fix: Minimize the stored fields and retention period. Prefer `sessionStorage` for navigation handoff, or server-side storage keyed by an opaque session identifier. Avoid persisting company-identifying and financial data unless the user explicitly opts in.
- Mitigation: Clear records after handoff/logout, namespace them by an authenticated opaque user ID, add expiry timestamps, and strengthen CSP as described in SEC-003.
- False-positive notes: `localStorage` is not being used for an authentication token here. Severity is driven by the sensitivity and persistence of lease-screening content.

## Low severity

### SEC-005 — Host-header validation is not visible in the FastAPI application

Status: Fixed locally; deployment pending.

- Rule ID: FASTAPI-HOST-001
- Severity: Low
- Location: `api/main.py:504-538`
- Evidence at discovery: The middleware stack included security headers, demo protection, API-key authentication, and CORS, but no `TrustedHostMiddleware` or equivalent application-level allowlist was found.
- Impact: If user-controlled Host or forwarded-host values are used by URL generation, redirects, password-reset links, or cache keys, host-header poisoning may become possible.
- Fix: Add environment-specific `TrustedHostMiddleware` allowed hosts for production services, or document and test the equivalent enforcement at the edge.
- Mitigation: Ensure Cloud Run/Cloudflare rejects unexpected hosts and never derive security-sensitive absolute URLs from untrusted headers.
- False-positive notes: Managed ingress may already enforce acceptable hosts. Runtime/edge configuration was not validated in this local review.

## Positive controls observed

- API keys are compared with `hmac.compare_digest`.
- Non-demo Cloud Run deployments fail closed when the API key secret is missing.
- CORS uses an explicit origin list rather than a hard-coded wildcard.
- FastAPI responses receive `nosniff`, frame, referrer, and permissions-policy headers.
- The reviewed OCR endpoint enforces a 20 MB size cap and a content-type allowlist.
- Reviewed React raw-HTML sinks use DOMPurify or explicit text escaping.
- Subprocess calls reviewed use argument arrays or quote the only interpolated filename; no direct request-controlled shell command was confirmed.

## Scan limitations and cost record

Codex Security CLI version `0.1.25` was run without patching or PR creation. Three attempts were made before the user requested that paid work stop:

1. Whole repository: stopped manually at estimated `$6.655771`; status `failed`; confirmed findings `0`.
2. `api` + `frontend/src`: stopped automatically at the `$4` cap (`$4.039032` estimated); status `failed`; confirmed findings `0`.
3. Eight critical files: stopped by the account usage limit at estimated `$1.693391`; no final report was produced.

The CLI-displayed cumulative model-cost estimate is `$12.388194`. This is **not evidence of a separate USD charge**. The scans authenticated with stored ChatGPT credentials rather than an API key, and the account usage response showed no purchased credit balance. The runs therefore consumed the plan's included Codex usage allowance and stopped at a usage limit. No Codex Security process remains running, and no scan was restarted after the stop request.
