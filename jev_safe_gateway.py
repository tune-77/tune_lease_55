"""Build a least-privilege projection for Jev / TypeSafe judgments.

The gateway never sends data by itself.  It converts local candidates into a
small outbound payload, keeps the source-id mapping local, and records a
content-free audit event.  Callers may pass only ``result["outbound"]`` to
Jev after ``result["status"] == "allowed"``.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_AUDIT_PATH = Path(__file__).resolve().parent / "data" / "jev_safe_gateway_audit.jsonl"
DEFAULT_PUBLIC_HOSTS = frozenset({"github.com", "raw.githubusercontent.com"})
MAX_ITEMS = 50
MAX_PURPOSE_CHARS = 240
MAX_ABSTRACT_CHARS = 320
MAX_PUBLIC_ITEM_CHARS = 4_000
MAX_PUBLIC_TOTAL_CHARS = 8_000

ABSTRACT_FIELDS = (
    "artifact_kind",
    "change_kind",
    "relation",
    "effect",
    "risk_category",
    "verification_status",
)
AGGREGATE_FIELDS = (
    "industry_bucket",
    "region_bucket",
    "size_bucket",
    "risk_flags",
    "boolean_signals",
)
LOCAL_ONLY_FIELDS = frozenset({"source_id", "local_id"})
AGGREGATE_PURPOSES = frozenset(
    {
        "check_policy_fit",
        "classify_review_route",
        "classify_risk_level",
        "prioritize_manual_review",
    }
)
AGGREGATE_ALLOWED_VALUES = {
    "industry_bucket": frozenset(
        {
            "agriculture",
            "construction",
            "finance_insurance",
            "healthcare_welfare",
            "hospitality",
            "information_communications",
            "manufacturing",
            "other",
            "professional_services",
            "public_sector",
            "real_estate",
            "transport",
            "utilities",
            "wholesale_retail",
        }
    ),
    "region_bucket": frozenset({"domestic", "mixed", "overseas", "unknown"}),
    "size_bucket": frozenset({"enterprise", "large", "medium", "micro", "small", "unknown"}),
    "risk_flags": frozenset(
        {
            "adverse_news",
            "data_incomplete",
            "governance_gap",
            "high_concentration",
            "high_leverage",
            "industry_headwind",
            "limited_collateral",
            "short_history",
            "volatile_earnings",
            "weak_liquidity",
        }
    ),
    "boolean_signals": frozenset(
        {
            "adverse_news_found",
            "data_complete",
            "delinquency_history",
            "financials_verified",
            "has_collateral",
            "has_guarantor",
            "positive_operating_cashflow",
        }
    ),
}

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|rk)-(?:live|test|proj)?[_-]?[A-Za-z0-9_-]{12,}\b", re.I),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(
        r"\b(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"]?[^\s,'\"]{8,}",
        re.I,
    ),
)
_PII_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)(?:0\d{1,4}[-ー]\d{1,4}[-ー]\d{3,4})(?!\d)"),
    re.compile(r"〒?\d{3}[-ー]\d{4}"),
    re.compile(r"(?:氏名|住所|電話番号|顧客名|会社名|申込者)\s*[:：]"),
)
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s'\"])/(?!/)[^\s'\"]+"),
    re.compile(r"\b[A-Za-z]:\\[^\s'\"]+", re.I),
)
_RAW_CODE_MARKERS = (
    re.compile(r"```"),
    re.compile(r"`[^`]+`"),
    re.compile(r"(?:^|\n)\s*(?:def |class |function |import |from .+ import |@@ |diff --git )"),
    re.compile(r"(?:^|\n)\s*[+-]{3}\s+[ab]/"),
    re.compile(r"(?:^|[\s'\"])(?:\.{1,2}/)?[\w.-]+(?:/[\w.-]+)+\b"),
    re.compile(
        r"(?:^|\s)(?:[\w.-]+/)+[\w.-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|md|sql|sh|toml)\b",
        re.I,
    ),
    re.compile(r"\b[\w.-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|sql|sh|toml)\b", re.I),
    re.compile(
        r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.[A-Za-z][A-Za-z0-9]{0,15}"
        r"(?![A-Za-z0-9_])"
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])(?:Dockerfile|Makefile|Procfile|README)(?![A-Za-z0-9_])",
        re.I,
    ),
)
_SAFE_TOKEN = re.compile(r"^[\w .:/+,-]+$", re.UNICODE)
_RAW_NUMERIC_VALUE = re.compile(r"\d")


class GatewayInputError(ValueError):
    """The local gateway request does not match the supported schema."""


def _text(value: Any, field: str, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise GatewayInputError(f"{field} must be a string")
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        raise GatewayInputError(f"{field} must not be empty")
    if len(cleaned) > max_chars:
        raise GatewayInputError(f"{field} exceeds {max_chars} characters")
    return cleaned


def _verbatim_text(value: Any, field: str, *, max_chars: int) -> str:
    if not isinstance(value, str):
        raise GatewayInputError(f"{field} must be a string")
    if not value.strip():
        raise GatewayInputError(f"{field} must not be empty")
    if len(value) > max_chars:
        raise GatewayInputError(f"{field} exceeds {max_chars} characters")
    return value


def _find_sensitive(text: str, *, reject_code: bool = False) -> set[str]:
    findings: set[str] = set()
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        findings.add("secret_like_content")
    if any(pattern.search(text) for pattern in _PII_PATTERNS):
        findings.add("pii_like_content")
    if any(pattern.search(text) for pattern in _PRIVATE_PATH_PATTERNS):
        findings.add("private_path")
    if reject_code and any(pattern.search(text) for pattern in _RAW_CODE_MARKERS):
        findings.add("raw_code_or_diff")
    return findings


def _safe_scalar(value: Any, field: str, *, max_chars: int = 120) -> str | bool:
    if isinstance(value, bool):
        return value
    text = _text(value, field, max_chars=max_chars)
    if not _SAFE_TOKEN.fullmatch(text):
        raise GatewayInputError(f"{field} contains unsupported characters")
    return text


def _source_id(item: Mapping[str, Any], index: int) -> str:
    value = item.get("source_id", item.get("local_id", f"source_{index}"))
    if not isinstance(value, (str, int)) or not str(value).strip():
        raise GatewayInputError(f"items[{index}].source_id must be a non-empty string or integer")
    return str(value)


def _reject_unknown(item: Mapping[str, Any], allowed: set[str], index: int) -> None:
    unknown = set(item) - allowed - LOCAL_ONLY_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise GatewayInputError(f"items[{index}] contains unsupported fields: {names}")


def _abstract_item(item: Mapping[str, Any], index: int) -> tuple[dict[str, Any], set[str]]:
    _reject_unknown(item, set(ABSTRACT_FIELDS), index)
    projected: dict[str, Any] = {"id": f"item_{index + 1}"}
    findings: set[str] = set()
    for field in ABSTRACT_FIELDS:
        if field not in item:
            continue
        raw_value = item[field]
        if isinstance(raw_value, bool):
            value: str | bool = raw_value
        else:
            value = _text(raw_value, f"items[{index}].{field}", max_chars=MAX_ABSTRACT_CHARS)
            value_findings = _find_sensitive(value, reject_code=True)
            findings |= value_findings
            if not value_findings and not _SAFE_TOKEN.fullmatch(value):
                raise GatewayInputError(
                    f"items[{index}].{field} contains unsupported characters"
                )
        projected[field] = value
    if len(projected) == 1:
        raise GatewayInputError(f"items[{index}] has no reviewable fields")
    return projected, findings


def _aggregate_item(item: Mapping[str, Any], index: int) -> tuple[dict[str, Any], set[str]]:
    _reject_unknown(item, set(AGGREGATE_FIELDS), index)
    projected: dict[str, Any] = {"id": f"item_{index + 1}"}
    findings: set[str] = set()
    for field in AGGREGATE_FIELDS:
        if field not in item:
            continue
        value = item[field]
        allowed_values = AGGREGATE_ALLOWED_VALUES[field]
        if field == "risk_flags":
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise GatewayInputError(f"items[{index}].risk_flags must be an array")
            if len(value) > 20:
                raise GatewayInputError(f"items[{index}].{field} has too many values")
            safe_value = []
            for entry in value:
                safe_entry = _safe_scalar(entry, f"items[{index}].{field}", max_chars=80)
                if isinstance(safe_entry, bool) or safe_entry not in allowed_values:
                    raise GatewayInputError(f"items[{index}].{field} contains an unknown value")
                safe_value.append(safe_entry)
        elif field == "boolean_signals":
            if not isinstance(value, Mapping):
                raise GatewayInputError(f"items[{index}].boolean_signals must be an object")
            if len(value) > 20:
                raise GatewayInputError(f"items[{index}].{field} has too many values")
            safe_value = {}
            for key, entry in value.items():
                safe_key = _safe_scalar(key, f"items[{index}].{field} key", max_chars=80)
                if isinstance(safe_key, bool) or safe_key not in allowed_values:
                    raise GatewayInputError(f"items[{index}].{field} contains an unknown key")
                if not isinstance(entry, bool):
                    raise GatewayInputError(f"items[{index}].{field} values must be booleans")
                safe_value[safe_key] = entry
        else:
            safe_value = _safe_scalar(value, f"items[{index}].{field}", max_chars=80)
            if isinstance(safe_value, bool) or safe_value not in allowed_values:
                raise GatewayInputError(f"items[{index}].{field} contains an unknown value")
        serialized = json.dumps(safe_value, ensure_ascii=False, sort_keys=True)
        findings |= _find_sensitive(serialized, reject_code=True)
        if isinstance(safe_value, Mapping):
            scalar_values = [*safe_value.keys(), *safe_value.values()]
        elif isinstance(safe_value, list):
            scalar_values = safe_value
        else:
            scalar_values = [safe_value]
        if any(
            isinstance(entry, str) and _RAW_NUMERIC_VALUE.search(entry)
            for entry in scalar_values
        ):
            findings.add("raw_numeric_value")
        projected[field] = safe_value
    if len(projected) == 1:
        raise GatewayInputError(f"items[{index}] has no reviewable fields")
    return projected, findings


def _public_item(
    item: Mapping[str, Any], index: int, public_hosts: frozenset[str]
) -> tuple[dict[str, Any], set[str]]:
    allowed = {"content", "visibility", "public_source_url", "artifact_kind"}
    _reject_unknown(item, allowed, index)
    if item.get("visibility") != "public":
        raise GatewayInputError(f"items[{index}].visibility must be 'public'")
    url = _text(item.get("public_source_url"), f"items[{index}].public_source_url", max_chars=500)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in public_hosts:
        raise GatewayInputError(f"items[{index}].public_source_url is not on an approved HTTPS host")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise GatewayInputError(
            f"items[{index}].public_source_url must not contain credentials, a query, or a fragment"
        )
    content = _verbatim_text(
        item.get("content"), f"items[{index}].content", max_chars=MAX_PUBLIC_ITEM_CHARS
    )
    findings = _find_sensitive(content)
    projected: dict[str, Any] = {
        "id": f"item_{index + 1}",
        "public_source_url": url,
        "content": content,
    }
    if "artifact_kind" in item:
        projected["artifact_kind"] = _safe_scalar(
            item["artifact_kind"], f"items[{index}].artifact_kind"
        )
    return projected, findings


def _audit_record(
    *, mode: str, outbound: Mapping[str, Any], decision: str, reasons: Sequence[str], item_count: int
) -> dict[str, Any]:
    serialized = json.dumps(outbound, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "decision": decision,
        "reason_codes": sorted(set(reasons)),
        "item_count": item_count,
        "outbound_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "content_logged": False,
    }


def append_audit(path: Path, record: Mapping[str, Any]) -> None:
    """Append a content-free audit event to a local JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def prepare_gateway_request(
    payload: Mapping[str, Any],
    *,
    audit_path: Path | None = None,
    public_hosts: frozenset[str] = DEFAULT_PUBLIC_HOSTS,
) -> dict[str, Any]:
    """Validate and project a local request into an outbound Jev-safe payload."""
    if not isinstance(payload, Mapping):
        raise GatewayInputError("request must be a JSON object")
    mode = payload.get("mode")
    if not isinstance(mode, str) or mode not in {"abstract", "aggregate", "public_excerpt"}:
        raise GatewayInputError("mode must be abstract, aggregate, or public_excerpt")
    purpose = _text(payload.get("purpose"), "purpose", max_chars=MAX_PURPOSE_CHARS)
    purpose_findings = _find_sensitive(purpose, reject_code=True)
    if mode == "aggregate":
        if purpose not in AGGREGATE_PURPOSES:
            raise GatewayInputError("aggregate purpose must use a predefined value")
        if _RAW_NUMERIC_VALUE.search(purpose):
            purpose_findings.add("raw_numeric_value")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise GatewayInputError("items must be a non-empty array")
    if len(raw_items) > MAX_ITEMS:
        raise GatewayInputError(f"items exceeds the limit of {MAX_ITEMS}")

    projected_items: list[dict[str, Any]] = []
    local_mapping: dict[str, str] = {}
    findings = set(purpose_findings)
    total_public_chars = 0
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise GatewayInputError(f"items[{index}] must be an object")
        local_mapping[f"item_{index + 1}"] = _source_id(raw_item, index)
        if mode == "abstract":
            projected, item_findings = _abstract_item(raw_item, index)
        elif mode == "aggregate":
            projected, item_findings = _aggregate_item(raw_item, index)
        else:
            projected, item_findings = _public_item(raw_item, index, public_hosts)
            total_public_chars += len(str(projected["content"]))
        projected_items.append(projected)
        findings |= item_findings

    if total_public_chars > MAX_PUBLIC_TOTAL_CHARS:
        findings.add("public_excerpt_total_too_large")

    outbound = {"purpose": purpose, "items": projected_items}
    decision = "blocked" if findings else "allowed"
    audit = _audit_record(
        mode=mode,
        outbound=outbound,
        decision=decision,
        reasons=sorted(findings),
        item_count=len(projected_items),
    )
    if audit_path is not None:
        append_audit(audit_path, audit)
    return {
        "status": decision,
        "mode": mode,
        "outbound": outbound if decision == "allowed" else None,
        "local_mapping": local_mapping,
        "audit": audit,
    }
