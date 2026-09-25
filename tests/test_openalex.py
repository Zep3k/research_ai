import httpx

from theory.openalex import search_works


def test_search_works_preserves_source_metadata_and_rebuilds_abstract():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search"] == "distributed lower bound"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "A Lower Bound",
                        "publication_year": 2025,
                        "doi": "https://doi.org/10.1/example",
                        "cited_by_count": 7,
                        "primary_location": {"landing_page_url": "https://example.test/paper"},
                        "abstract_inverted_index": {"world": [1], "Hello": [0]},
                    },
                    {"id": None, "display_name": "Malformed"},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        hits = search_works("distributed lower bound", client=client)

    assert len(hits) == 1
    assert hits[0].openalex_id == "https://openalex.org/W1"
    assert hits[0].abstract == "Hello world"
    assert hits[0].url == "https://example.test/paper"


def test_search_works_uses_api_key_and_retries_rate_limit(monkeypatch):
    attempts = 0
    delays = []
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key")
    monkeypatch.setattr("theory.openalex.time.sleep", delays.append)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert "secret-key" not in str(request.url)
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"results": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert search_works("query", client=client) == []

    assert attempts == 2
    assert delays == [0.0]
