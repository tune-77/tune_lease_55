from __future__ import annotations

import hashlib

from api.main import app


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
EXPECTED_ROUTE_COUNT = 275
EXPECTED_ROUTE_SHA256 = "5b9d2c9cdeba21bd31402c55fd5caceccf11defba8cb366bd2082bb79ff77995"


def _route_signatures() -> list[str]:
    schema = app.openapi()
    return sorted(
        f"{method.upper()} {path}"
        for path, operations in schema["paths"].items()
        for method in operations
        if method in HTTP_METHODS
    )


def test_public_api_route_contract_is_stable() -> None:
    """Router moves must not silently add, remove, or rename public endpoints."""

    signatures = _route_signatures()
    digest = hashlib.sha256("\n".join(signatures).encode("utf-8")).hexdigest()

    assert len(signatures) == EXPECTED_ROUTE_COUNT
    assert digest == EXPECTED_ROUTE_SHA256
