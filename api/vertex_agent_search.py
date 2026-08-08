"""Vertex AI Search supplementary retrieval for lease-domain chat RAG.

This module is intentionally optional. If credentials, network, or config are
missing, callers get a skipped/error payload and the local Obsidian RAG remains
the only source.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_PROJECT_ID = "gen-lang-client-0420497423"
DEFAULT_ENGINE_ID = "tune-lease-55-obsidian-search"
DEFAULT_LOCATION = "global"
DEFAULT_COLLECTION = "default_collection"
DEFAULT_ANSWER_PREAMBLE = (
    "あなたはリース審査の補助AIです。Obsidian由来の社内ナレッジを根拠に、"
    "補助金、物件、耐用年数、残価、返済余力、業界リスクの確認事項を日本語で簡潔に整理してください。"
    "根拠が弱い場合は断定せず、追加確認事項として扱ってください。"
)
GROUNDING_DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class VertexSearchConfig:
    enabled: bool
    project_id: str
    engine_id: str
    location: str
    collection: str
    page_size: int
    timeout_seconds: float
    cache_ttl_seconds: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def get_config() -> VertexSearchConfig:
    return VertexSearchConfig(
        enabled=_env_bool("VERTEX_AGENT_SEARCH_ENABLED", True),
        project_id=(
            os.environ.get("VERTEX_AI_SEARCH_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
            or DEFAULT_PROJECT_ID
        ).strip(),
        engine_id=(os.environ.get("VERTEX_AI_SEARCH_ENGINE_ID") or DEFAULT_ENGINE_ID).strip(),
        location=(os.environ.get("VERTEX_AI_SEARCH_LOCATION") or DEFAULT_LOCATION).strip(),
        collection=(os.environ.get("VERTEX_AI_SEARCH_COLLECTION") or DEFAULT_COLLECTION).strip(),
        page_size=_env_int("VERTEX_AI_SEARCH_PAGE_SIZE", 3, minimum=1, maximum=5),
        timeout_seconds=float(os.environ.get("VERTEX_AI_SEARCH_TIMEOUT_SECONDS", "8") or 8),
        cache_ttl_seconds=float(os.environ.get("VERTEX_SEARCH_CACHE_TTL_SECONDS", "600") or 0),
    )


# Same query repeated within the TTL skips the paid Search/Answer call and
# reuses the last result (Enterprise tier + LLM add-on charges per call).
_SEARCH_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_ANSWER_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}


def _cache_lookup(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]],
    key: tuple[Any, ...],
    ttl_seconds: float,
) -> dict[str, Any] | None:
    if ttl_seconds <= 0:
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    cached_at, value = entry
    if time.time() - cached_at >= ttl_seconds:
        return None
    return dict(value)


def _cache_store(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]],
    key: tuple[Any, ...],
    value: dict[str, Any],
) -> None:
    cache[key] = (time.time(), value)


def _access_token() -> str:
    explicit = os.environ.get("VERTEX_AI_SEARCH_ACCESS_TOKEN")
    if explicit:
        return explicit.strip()

    try:
        import google.auth  # type: ignore[import-not-found]
        from google.auth.transport.requests import Request  # type: ignore[import-not-found]

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = str(getattr(credentials, "token", "") or "").strip()
        if token:
            return token
    except Exception:
        pass

    if _env_bool("VERTEX_AI_SEARCH_ALLOW_GCLOUD_TOKEN", True):
        return subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
            timeout=10,
        ).strip()
    return ""


def _compact_text(value: Any, limit: int) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _canonical_key(item: dict[str, Any]) -> str:
    source = str(item.get("source_path") or "")
    title = str(item.get("title") or "")
    topic = re.sub(r"/20\d{2}-\d{2}-\d{2}_[^/]+$", "", source)
    topic = re.sub(r"20\d{2}-\d{2}-\d{2}", "", topic)
    topic = re.sub(r"[_-]?[0-9a-f]{8,}$", "", topic)
    return (topic or title or source).strip().lower()


def _parse_results(response: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in response.get("results", []):
        doc = result.get("document", {}) or {}
        data = doc.get("structData", {}) or {}
        derived = doc.get("derivedStructData", {}) or {}
        answers = derived.get("extractive_answers") or derived.get("extractiveAnswers") or []
        snippets = derived.get("snippets") or []
        excerpt = ""
        if answers:
            excerpt = answers[0].get("content") or answers[0].get("pageContent") or ""
        elif snippets:
            excerpt = snippets[0].get("snippet") or snippets[0].get("htmlSnippet") or ""
        item = {
            "doc_id": str(doc.get("id") or ""),
            "title": str(data.get("title") or derived.get("title") or doc.get("id") or ""),
            "source_path": str(data.get("source_path") or ""),
            "link": str(derived.get("link") or ""),
            "excerpt": _compact_text(excerpt, 420),
        }
        key = _canonical_key(item)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(item)
        if len(parsed) >= limit:
            break
    return parsed


def _serving_config_url(config: VertexSearchConfig, method: str, *, api_version: str = "v1") -> str:
    return (
        f"https://discoveryengine.googleapis.com/{api_version}/"
        f"projects/{config.project_id}/locations/{config.location}/collections/{config.collection}/"
        f"engines/{config.engine_id}/servingConfigs/default_search:{method}"
    )


def _post_json(url: str, body: dict[str, Any], config: VertexSearchConfig, *, timeout_seconds: float | None = None) -> dict[str, Any]:
    token = _access_token()
    if not token:
        raise RuntimeError("auth_unavailable")
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": config.project_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds or config.timeout_seconds) as res:
        return json.loads(res.read().decode("utf-8"))


def build_lease_search_controls(query: str) -> dict[str, Any]:
    """Local search-control hints used for debug/reranking without mutating the data store."""
    q = query.lower()
    preferred: list[str] = []
    if any(term in query for term in ("補助金", "助成金", "ものづくり")):
        preferred.extend(["Research/補助金", "補助金", "対象要件"])
    if any(term in query for term in ("再リース", "残価", "中古", "耐用年数")):
        preferred.extend(["再リース", "残価", "中古流動性", "耐用年数"])
    if any(term in query for term in ("判断", "審査", "稟議", "条件付き承認")):
        preferred.extend(["Judgment Assets", "判断資産", "審査"])
    if any(term in q for term in ("transport", "logistics")) or any(term in query for term in ("運送", "物流")):
        preferred.extend(["運送", "物流", "燃料", "ドライバー"])
    return {
        "preferred_terms": list(dict.fromkeys(preferred)),
        "bury_terms": ["AI Chat", "Improvement Log", "Cloud Run", "debug", "雑談"],
        "mode": "local_rerank",
    }


def _control_score(item: dict[str, Any], controls: dict[str, Any]) -> int:
    haystack = " ".join(str(item.get(key) or "") for key in ("title", "source_path", "excerpt"))
    score = 0
    for term in controls.get("preferred_terms") or []:
        if str(term) and str(term) in haystack:
            score += 3
    for term in controls.get("bury_terms") or []:
        if str(term) and str(term) in haystack:
            score -= 4
    return score


def _apply_local_controls(results: list[dict[str, Any]], query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controls = build_lease_search_controls(query)
    ranked = sorted(enumerate(results), key=lambda pair: (_control_score(pair[1], controls), -pair[0]), reverse=True)
    return [item for _, item in ranked], controls


def search_vertex_agent(
    query: str,
    *,
    page_size: int | None = None,
    filter_expression: str | None = None,
    boost_spec: dict[str, Any] | None = None,
    apply_controls: bool = True,
) -> dict[str, Any]:
    config = get_config()
    if not config.enabled:
        return {"used": False, "status": "disabled", "refs": [], "prompt_context": ""}
    if not (config.project_id and config.engine_id):
        return {"used": False, "status": "not_configured", "refs": [], "prompt_context": ""}

    size = page_size or config.page_size
    boost_key = json.dumps(boost_spec, sort_keys=True) if boost_spec else None
    cache_key = (query, size, filter_expression, boost_key, apply_controls)
    cached = _cache_lookup(_SEARCH_CACHE, cache_key, config.cache_ttl_seconds)
    if cached is not None:
        return cached

    url = _serving_config_url(config, "search")
    body = {
        "query": query,
        "pageSize": size,
        "contentSearchSpec": {
            "extractiveContentSpec": {"maxExtractiveAnswerCount": 1},
            "summarySpec": {
                "summaryResultCount": min(size, 5),
                "includeCitations": True,
            },
        },
    }
    if filter_expression:
        body["filter"] = filter_expression
    if boost_spec:
        body["boostSpec"] = boost_spec
    try:
        response = _post_json(url, body, config)
    except RuntimeError as exc:
        if str(exc) == "auth_unavailable":
            return {"used": False, "status": "auth_unavailable", "refs": [], "prompt_context": ""}
        raise
    except (OSError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "used": False,
            "status": "error",
            "error": _compact_text(exc, 240),
            "refs": [],
            "prompt_context": "",
        }

    results = _parse_results(response, size)
    controls = build_lease_search_controls(query)
    if apply_controls and results:
        results, controls = _apply_local_controls(results, query)
    summary = _compact_text(response.get("summary", {}).get("summaryText"), 1000)
    refs = [item["source_path"] or item["title"] or item["doc_id"] for item in results if item]
    context_lines = []
    if summary or results:
        context_lines.append("【Vertex AI Search補助ナレッジ】")
        context_lines.append("既存Obsidian RAGを主根拠とし、不足する横断論点の補助として扱う。")
    if summary:
        context_lines.extend(["", "要約:", summary])
    if results:
        context_lines.extend(["", "検索ヒット:"])
        for item in results:
            label = item["source_path"] or item["title"] or item["doc_id"]
            excerpt = f": {item['excerpt']}" if item["excerpt"] else ""
            context_lines.append(f"- {label}{excerpt}")

    result = {
        "used": bool(summary or results),
        "status": "ok",
        "summary": summary,
        "refs": refs[:8],
        "results": results,
        "controls": controls,
        "prompt_context": "\n".join(context_lines).strip(),
    }
    _cache_store(_SEARCH_CACHE, cache_key, result)
    return dict(result)


def build_answer_request_body(
    query: str,
    *,
    page_size: int,
    preamble: str | None = None,
    include_related_questions: bool = True,
    include_grounding_supports: bool = True,
    grounding_filtering_level: str | None = None,
    filter_expression: str | None = None,
    boost_spec: dict[str, Any] | None = None,
    max_rephrase_steps: int = 3,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": {"text": query},
        "answerGenerationSpec": {
            "includeCitations": True,
            "answerLanguageCode": "ja",
            "ignoreAdversarialQuery": True,
            "ignoreJailBreakingQuery": True,
            "ignoreLowRelevantContent": False,
            "promptSpec": {"preamble": preamble or DEFAULT_ANSWER_PREAMBLE},
        },
        "queryUnderstandingSpec": {
            "queryRephraserSpec": {
                "maxRephraseSteps": max(1, min(5, int(max_rephrase_steps or 3))),
            }
        },
        "relatedQuestionsSpec": {"enable": bool(include_related_questions)},
        "groundingSpec": {
            "includeGroundingSupports": bool(include_grounding_supports),
        },
        "searchSpec": {
            "searchParams": {
                "maxReturnResults": max(1, min(10, int(page_size or 5))),
            }
        },
    }
    if grounding_filtering_level:
        body["groundingSpec"]["filteringLevel"] = grounding_filtering_level
    search_params = body["searchSpec"]["searchParams"]
    if filter_expression:
        search_params["filter"] = filter_expression
    if boost_spec:
        search_params["boostSpec"] = boost_spec
    return body


def _collect_answer_search_results(value: Any, limit: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if len(collected) >= limit:
            return
        if isinstance(node, dict):
            if "searchResults" in node and isinstance(node["searchResults"], list):
                for result in node["searchResults"]:
                    if isinstance(result, dict):
                        collected.append(
                            {
                                "title": _compact_text(result.get("title") or result.get("documentTitle"), 180),
                                "uri": str(result.get("uri") or result.get("link") or ""),
                                "snippet": _compact_text(
                                    (result.get("snippetInfo") or {}).get("snippet")
                                    if isinstance(result.get("snippetInfo") or {}, dict)
                                    else result.get("snippet"),
                                    420,
                                ),
                            }
                        )
                        if len(collected) >= limit:
                            return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in collected:
        key = (item.get("uri") or item.get("title") or item.get("snippet") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_answer_grounding(answer: dict[str, Any]) -> dict[str, Any]:
    supports = answer.get("groundingSupports") or answer.get("grounding_supports") or []
    grounding_score = answer.get("groundingScore")
    if grounding_score is None:
        grounding_score = answer.get("grounding_score")
    parsed_supports = []
    iterable_supports = supports if isinstance(supports, list) else []
    for support in iterable_supports:
        if not isinstance(support, dict):
            continue
        claim_text = (
            support.get("claimText")
            or support.get("claim_text")
            or support.get("claim")
            or support.get("text")
            or ""
        )
        score = support.get("supportScore")
        if score is None:
            score = support.get("support_score")
        parsed_supports.append(
            {
                "claim_text": _compact_text(claim_text, 500),
                "support_score": score,
                "citation_indices": support.get("citationIndices") or support.get("citation_indices") or [],
                "raw": support,
            }
        )
    low_claim_count = 0
    numeric_scores = []
    for item in parsed_supports:
        try:
            score = float(item.get("support_score"))
            numeric_scores.append(score)
            if score < 0.55:
                low_claim_count += 1
        except (TypeError, ValueError):
            pass
    if grounding_score is None and numeric_scores:
        grounding_score = round(sum(numeric_scores) / len(numeric_scores), 3)
    return {
        "grounding_score": grounding_score,
        "grounding_score_source": "official" if grounding_score is not None else "",
        "grounding_supports": parsed_supports,
        "low_support_claim_count": low_claim_count,
        "support_count": len(parsed_supports),
    }


def _citation_coverage_score(answer_text: str, citations: list[dict[str, Any]]) -> float | None:
    if not answer_text or not citations:
        return None
    length = len(answer_text)
    covered: set[int] = set()
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        try:
            start = int(citation.get("startIndex"))
            end = int(citation.get("endIndex"))
        except (TypeError, ValueError):
            continue
        start = max(0, min(length, start))
        end = max(start, min(length, end))
        covered.update(range(start, end))
    if not covered:
        return None
    return round(len(covered) / max(1, length), 3)


def answer_prompt_context(answer_result: dict[str, Any], *, limit: int = 1600) -> str:
    if not answer_result.get("used"):
        return ""
    answer_text = _compact_text(answer_result.get("answer_text"), limit)
    if not answer_text:
        return ""
    grounding_score = answer_result.get("grounding_score")
    low_count = int(answer_result.get("low_support_claim_count") or 0)
    lines = [
        "【Vertex Answer API根拠付き回答】",
        "既存Obsidian RAGとSearch補助の裏取りとして扱う。grounding score が低い主張は断定せず確認論点に留める。",
    ]
    if grounding_score is not None:
        source = answer_result.get("grounding_score_source") or "unknown"
        lines.append(f"Grounding score: {grounding_score} ({source})")
    if low_count:
        lines.append(f"Low support claims: {low_count}")
    lines.extend(["", answer_text])
    return "\n".join(lines).strip()


def answer_vertex_agent(
    query: str,
    *,
    page_size: int | None = None,
    preamble: str | None = None,
    include_related_questions: bool = True,
    include_grounding_supports: bool = True,
    grounding_filtering_level: str | None = None,
    filter_expression: str | None = None,
    boost_spec: dict[str, Any] | None = None,
    max_rephrase_steps: int = 3,
) -> dict[str, Any]:
    config = get_config()
    if not config.enabled:
        return {"used": False, "status": "disabled", "answer_text": "", "refs": []}
    if not (config.project_id and config.engine_id):
        return {"used": False, "status": "not_configured", "answer_text": "", "refs": []}

    size = page_size or max(config.page_size, 5)
    boost_key = json.dumps(boost_spec, sort_keys=True) if boost_spec else None
    cache_key = (
        query,
        size,
        preamble,
        include_related_questions,
        include_grounding_supports,
        grounding_filtering_level,
        filter_expression,
        boost_key,
        max_rephrase_steps,
    )
    cached = _cache_lookup(_ANSWER_CACHE, cache_key, config.cache_ttl_seconds)
    if cached is not None:
        return cached

    body = build_answer_request_body(
        query,
        page_size=size,
        preamble=preamble,
        include_related_questions=include_related_questions,
        include_grounding_supports=include_grounding_supports,
        grounding_filtering_level=grounding_filtering_level,
        filter_expression=filter_expression,
        boost_spec=boost_spec,
        max_rephrase_steps=max_rephrase_steps,
    )
    try:
        response = _post_json(
            _serving_config_url(config, "answer", api_version="v1beta"),
            body,
            config,
            timeout_seconds=float(os.environ.get("VERTEX_ANSWER_API_TIMEOUT_SECONDS", "30") or 30),
        )
    except RuntimeError as exc:
        if str(exc) == "auth_unavailable":
            return {"used": False, "status": "auth_unavailable", "answer_text": "", "refs": []}
        raise
    except (OSError, subprocess.SubprocessError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "used": False,
            "status": "error",
            "error": _compact_text(exc, 240),
            "answer_text": "",
            "refs": [],
        }

    answer = response.get("answer", {}) or {}
    answer_text = _compact_text(answer.get("answerText"), 3000)
    search_results = _collect_answer_search_results(response, size)
    grounding = _extract_answer_grounding(answer)
    citations = answer.get("citations") or []
    if grounding.get("grounding_score") is None:
        citation_score = _citation_coverage_score(answer_text, citations)
        if citation_score is not None:
            grounding["grounding_score"] = citation_score
            grounding["grounding_score_source"] = "citation_coverage_fallback"
    if not grounding.get("support_count") and citations:
        grounding["support_count"] = len(citations)
    refs = [
        item.get("uri") or item.get("title") or item.get("snippet")
        for item in search_results
        if item.get("uri") or item.get("title") or item.get("snippet")
    ]
    result = {
        "used": bool(answer_text or search_results),
        "status": "ok",
        "answer_text": answer_text,
        "answer_state": answer.get("state") or response.get("state"),
        "citations": citations,
        **grounding,
        "related_questions": response.get("relatedQuestions") or answer.get("relatedQuestions") or [],
        "refs": refs[:8],
        "search_results": search_results,
        "raw_answer_name": answer.get("name") or response.get("answerName"),
    }
    _cache_store(_ANSWER_CACHE, cache_key, result)
    return dict(result)


def google_search_grounding(query: str, *, model: str | None = None) -> dict[str, Any]:
    if not _env_bool("VERTEX_GOOGLE_SEARCH_GROUNDING_ENABLED", True):
        return {"used": False, "status": "disabled", "text": "", "sources": []}
    config = get_config()
    model_id = model or os.environ.get("VERTEX_GOOGLE_SEARCH_GROUNDING_MODEL") or GROUNDING_DEFAULT_MODEL
    location = os.environ.get("VERTEX_AI_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
    prompt = (
        "最新の公的制度・補助金・規制確認が必要なリース審査補助として、"
        "日本語で短く要点と確認先を整理してください。質問: "
        f"{query}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"googleSearch": {}}],
        "model": f"projects/{config.project_id}/locations/{location}/publishers/google/models/{model_id}",
    }
    try:
        url = (
            f"https://aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/global/"
            f"publishers/google/models/{model_id}:generateContent"
            if location == "global"
            else f"https://{location}-aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/{location}/"
            f"publishers/google/models/{model_id}:generateContent"
        )
        response = _post_json(url, body, config)
        text_parts = []
        candidate = (response.get("candidates") or [{}])[0]
        for part in candidate.get("content", {}).get("parts", []) or []:
            if part.get("text"):
                text_parts.append(str(part.get("text")))
        metadata = candidate.get("groundingMetadata", {}) or {}
        sources = []
        for chunk in metadata.get("groundingChunks", []) or []:
            web = chunk.get("web") or {}
            uri = web.get("uri")
            title = web.get("title")
            if uri or title:
                sources.append({"title": title or uri, "uri": uri or ""})
        return {
            "used": bool(text_parts or sources),
            "status": "ok",
            "text": _compact_text("\n".join(text_parts), 3000),
            "sources": sources[:10],
            "search_suggestions": metadata.get("searchEntryPoint", {}),
            "web_search_queries": metadata.get("webSearchQueries") or [],
        }
    except Exception as rest_exc:
        rest_error = _compact_text(rest_exc, 180)

    try:
        os.environ.setdefault("GOOGLE_GENAI_USE_ENTERPRISE", "True")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", config.project_id)
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
        from google import genai  # type: ignore[import-not-found]
        from google.genai.types import GenerateContentConfig, GoogleSearch, HttpOptions, Tool  # type: ignore[import-not-found]

        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(tools=[Tool(google_search=GoogleSearch())]),
        )
        raw = response.to_json_dict() if hasattr(response, "to_json_dict") else {}
        sources = []
        metadata = raw.get("candidates", [{}])[0].get("groundingMetadata", {}) if raw.get("candidates") else {}
        for chunk in metadata.get("groundingChunks", []) or []:
            web = chunk.get("web") or {}
            uri = web.get("uri")
            title = web.get("title")
            if uri or title:
                sources.append({"title": title or uri, "uri": uri or ""})
        return {
            "used": True,
            "status": "ok",
            "text": _compact_text(getattr(response, "text", "") or "", 3000),
            "sources": sources[:10],
            "search_suggestions": metadata.get("searchEntryPoint", {}),
        }
    except Exception as exc:
        return {
            "used": False,
            "status": "error",
            "error": f"REST: {rest_error}; SDK: {_compact_text(exc, 180)}",
            "text": "",
            "sources": [],
        }


def external_grounding_results(query: str, *, page_size: int | None = None) -> list[dict[str, str]]:
    """Return the custom-search-API shape accepted by Gemini grounding."""
    result = search_vertex_agent(query, page_size=page_size, apply_controls=True)
    rows = []
    for item in result.get("results") or []:
        snippet = item.get("excerpt") or item.get("title") or item.get("source_path") or ""
        uri = item.get("link") or item.get("source_path") or item.get("doc_id") or ""
        if snippet and uri:
            rows.append({"snippet": str(snippet), "uri": str(uri)})
    return rows
