#!/usr/bin/env python3
"""Compare AI chat answer quality between two deployed environments.

Example:
    python scripts/compare_chat_quality_between_envs.py \
        --cloudflare-url https://example.trycloudflare.com \
        --cloud-run-url https://service.run.app
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_answer_quality import DEFAULT_EVAL_SET, score_answer

DEFAULT_OUTPUT = REPO_ROOT / "reports" / "chat_quality_env_compare_latest.json"


def _base_url(raw: str) -> str:
    return str(raw or "").strip().rstrip("/")


def _post_chat(base_url: str, message: str, user_id: str, timeout: int) -> dict[str, Any]:
    url = f"{_base_url(base_url)}/api/chat"
    payload = json.dumps(
        {"message": message, "user_id": user_id, "debug_memory": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            data = json.loads(body)
            return {
                "ok": True,
                "status": resp.status,
                "elapsed_ms": elapsed_ms,
                "reply": str(data.get("reply") or ""),
                "raw": data,
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        try:
            error_body = exc.read().decode("utf-8")[:2000]
        except Exception:
            error_body = ""
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_ms": elapsed_ms,
            "reply": "",
            "error": error_body or str(exc),
        }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "status": None,
            "elapsed_ms": elapsed_ms,
            "reply": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _answer_stats(answer: str) -> dict[str, Any]:
    text = str(answer or "")
    markers = [
        "Obsidian",
        "参照",
        "過去事例",
        "判断",
        "要確認",
        "資金繰り",
        "支払原資",
        "撤去費",
        "保守",
        "薬機法",
        "条件",
    ]
    return {
        "chars": len(text),
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "marker_hits": [marker for marker in markers if marker in text],
    }


def _score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    scored = score_answer(case, response.get("reply") or "")
    scored["transport_ok"] = bool(response.get("ok"))
    scored["http_status"] = response.get("status")
    scored["elapsed_ms"] = response.get("elapsed_ms")
    scored["answer_stats"] = _answer_stats(response.get("reply") or "")
    raw = response.get("raw") if isinstance(response.get("raw"), dict) else {}
    if raw.get("memory_debug"):
        scored["memory_debug"] = raw.get("memory_debug")
    if not response.get("ok"):
        scored["error"] = response.get("error", "")
    return scored


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.get("transport_ok"))
    passed = sum(1 for r in results if r.get("passed"))
    avg_score = sum(float(r.get("score") or 0) for r in results) / total if total else 0
    avg_chars = sum(int((r.get("answer_stats") or {}).get("chars") or 0) for r in results) / total if total else 0
    avg_elapsed = sum(int(r.get("elapsed_ms") or 0) for r in results) / total if total else 0
    concept_hits = sum(int(r.get("concept_hits") or 0) for r in results)
    concept_total = sum(int(r.get("concept_total") or 0) for r in results)
    return {
        "total": total,
        "transport_ok": ok,
        "passed": passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "average_score": round(avg_score, 1),
        "concept_coverage": round(concept_hits / concept_total * 100, 1) if concept_total else 0.0,
        "average_chars": round(avg_chars, 1),
        "average_elapsed_ms": round(avg_elapsed, 1),
    }


def _memory_refs_overlap(cf_score: dict[str, Any], cr_score: dict[str, Any]) -> dict[str, Any] | None:
    """両環境の想起記憶ID（memory_debug.refs）の一致率を返す。debug情報が無ければ None。"""

    def _refs(scored: dict[str, Any]) -> set[str]:
        debug = scored.get("memory_debug")
        if not isinstance(debug, dict):
            return set()
        refs = debug.get("refs")
        return {str(r) for r in refs if r} if isinstance(refs, list) else set()

    cf_refs, cr_refs = _refs(cf_score), _refs(cr_score)
    if not cf_refs and not cr_refs:
        return None
    union = cf_refs | cr_refs
    shared = cf_refs & cr_refs
    return {
        "shared": len(shared),
        "cloudflare_only": len(cf_refs - cr_refs),
        "cloud_run_only": len(cr_refs - cf_refs),
        "jaccard": round(len(shared) / len(union), 3) if union else 0.0,
    }


def compare_environments(
    *,
    cloudflare_url: str,
    cloud_run_url: str,
    cases: list[dict[str, Any]],
    timeout: int = 60,
    pause_seconds: float = 0.0,
) -> dict[str, Any]:
    cf_results: list[dict[str, Any]] = []
    cr_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    run_id = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        print(f"[{index}/{len(cases)}] cloudflare: {case['id']}", file=sys.stderr, flush=True)
        cf_resp = _post_chat(
            cloudflare_url,
            query,
            user_id=f"quality-cf-{run_id}-{index}",
            timeout=timeout,
        )
        if pause_seconds:
            time.sleep(pause_seconds)
        print(f"[{index}/{len(cases)}] cloud_run: {case['id']}", file=sys.stderr, flush=True)
        cr_resp = _post_chat(
            cloud_run_url,
            query,
            user_id=f"quality-cr-{run_id}-{index}",
            timeout=timeout,
        )
        if pause_seconds:
            time.sleep(pause_seconds)

        cf_score = _score_case(case, cf_resp)
        cr_score = _score_case(case, cr_resp)
        cf_results.append(cf_score)
        cr_results.append(cr_score)

        comparisons.append(
            {
                "id": case["id"],
                "query": query,
                "winner": (
                    "cloudflare"
                    if float(cf_score["score"]) > float(cr_score["score"])
                    else "cloud_run"
                    if float(cr_score["score"]) > float(cf_score["score"])
                    else "tie"
                ),
                "score_delta_cloudflare_minus_cloud_run": round(
                    float(cf_score["score"]) - float(cr_score["score"]),
                    1,
                ),
                "char_delta_cloudflare_minus_cloud_run": (
                    int(cf_score["answer_stats"]["chars"]) - int(cr_score["answer_stats"]["chars"])
                ),
                # 両環境の紫苑が「同じ記憶を思い出しているか」（同一人物感の定点観測）
                "memory_refs_overlap": _memory_refs_overlap(cf_score, cr_score),
                "cloudflare": cf_score,
                "cloud_run": cr_score,
            }
        )

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cloudflare_url": _base_url(cloudflare_url),
        "cloud_run_url": _base_url(cloud_run_url),
        "cloudflare_summary": _summarize(cf_results),
        "cloud_run_summary": _summarize(cr_results),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloudflare-url", required=True)
    parser.add_argument("--cloud-run-url", required=True)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N eval cases.")
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    if args.limit > 0:
        cases = cases[: args.limit]
    report = compare_environments(
        cloudflare_url=args.cloudflare_url,
        cloud_run_url=args.cloud_run_url,
        cases=cases,
        timeout=args.timeout,
        pause_seconds=args.pause_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cf = report["cloudflare_summary"]
    cr = report["cloud_run_summary"]
    print(
        "[chat-quality-env-compare] "
        f"cloudflare pass={cf['passed']}/{cf['total']} avg={cf['average_score']} chars={cf['average_chars']} "
        f"cloud_run pass={cr['passed']}/{cr['total']} avg={cr['average_score']} chars={cr['average_chars']} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
