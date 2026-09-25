import os
import math
import time

import httpx
from .errors import TheoryError
from .models import LiteratureHit

BASE = "https://api.openalex.org/works"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAlexError(TheoryError):
    pass


def _abstract_from_inverted(index: dict | None) -> str | None:
    if not index:
        return None
    positions = []
    for word, poss in index.items():
        for pos in poss:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def search_works(
    query: str, per_page: int = 8, *, client: httpx.Client | None = None
) -> list[LiteratureHit]:
    query = query.strip()
    if not query:
        raise ValueError("OpenAlex query cannot be empty")
    if not 1 <= per_page <= 50:
        raise ValueError("per_page must be between 1 and 50")
    params = {
        "search": query,
        "per-page": per_page,
        "select": (
            "id,display_name,publication_year,doi,primary_location,"
            "cited_by_count,abstract_inverted_index"
        ),
    }
    email = os.getenv("OPENALEX_EMAIL")
    if email:
        params["mailto"] = email
    headers = {"User-Agent": f"theory-research/0.1{f' (mailto:{email})' if email else ''}"}
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True, headers=headers)
    try:
        response = None
        for attempt in range(3):
            response = client.get(BASE, params=params, headers=headers)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                break
            if attempt < 2:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    parsed_delay = float(retry_after)
                    delay = min(parsed_delay, 5.0) if math.isfinite(parsed_delay) else 2**attempt
                except ValueError:
                    delay = float(2**attempt)
                time.sleep(max(delay, 0.0))
        assert response is not None
        if response.status_code == 429:
            key_hint = (
                "Check the key's daily credit budget."
                if api_key
                else "Set a free OPENALEX_API_KEY to avoid the tiny shared anonymous budget."
            )
            raise OpenAlexError(f"OpenAlex rate limit was reached after 3 attempts. {key_hint}")
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            client.close()
    out = []
    for item in payload.get("results", []):
        openalex_id = item.get("id")
        title = item.get("display_name")
        if not openalex_id or not title:
            continue
        primary = item.get("primary_location") or {}
        doi = item.get("doi")
        out.append(LiteratureHit(
            openalex_id=openalex_id,
            title=title,
            year=item.get("publication_year"),
            doi=doi,
            url=primary.get("landing_page_url") or doi or item.get("id"),
            cited_by_count=item.get("cited_by_count", 0) or 0,
            abstract=_abstract_from_inverted(item.get("abstract_inverted_index")),
        ))
    return out


def dedupe(hits: list[LiteratureHit], limit: int = 16) -> list[LiteratureHit]:
    if limit < 0:
        raise ValueError("limit cannot be negative")
    seen, out = set(), []
    for hit in hits:
        key = (hit.openalex_id or hit.doi or hit.title).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out
