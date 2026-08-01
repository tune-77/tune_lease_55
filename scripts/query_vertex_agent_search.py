#!/usr/bin/env python3
"""Query the tune_lease_55 Vertex AI Search pilot app."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request


PROJECT_ID = "gen-lang-client-0420497423"
ENGINE_ID = "tune-lease-55-obsidian-search"
LOCATION = "global"
COLLECTION = "default_collection"


def access_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"],
        text=True,
    ).strip()


def search(query: str, page_size: int, include_summary: bool) -> dict:
    url = (
        "https://discoveryengine.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION}/"
        f"engines/{ENGINE_ID}/servingConfigs/default_search:search"
    )
    body: dict = {
        "query": query,
        "pageSize": page_size,
        "contentSearchSpec": {
            "extractiveContentSpec": {
                "maxExtractiveAnswerCount": 1,
            },
        },
    }
    if include_summary:
        body["contentSearchSpec"]["summarySpec"] = {
            "summaryResultCount": min(page_size, 5),
            "includeCitations": True,
        }

    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token()}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": PROJECT_ID,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    response = search(args.query, args.page_size, not args.no_summary)
    summary = response.get("summary", {}).get("summaryText")
    if summary:
        print("SUMMARY")
        print(summary)
        print()

    print("RESULTS")
    for i, result in enumerate(response.get("results", []), start=1):
        doc = result.get("document", {})
        data = doc.get("structData", {})
        derived = doc.get("derivedStructData", {})
        print(f"{i}. {data.get('title') or derived.get('title') or doc.get('id')}")
        if data.get("source_path"):
            print(f"   source: {data['source_path']}")
        if derived.get("link"):
            print(f"   link: {derived['link']}")


if __name__ == "__main__":
    main()
