"""Provider model lists and key checks against mocked HTTP."""

from __future__ import annotations

import httpx2

from localflow.services.llm.catalog import check_key, list_models


def _client(handler) -> httpx2.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_openai_models_filtered_and_newest_first() -> None:
    ids = [
        "gpt-4o",
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-6-luna",
        "o3",
        "text-embedding-3-large",
        "gpt-5.1-codex",
        "gpt-5.5-pro",
        "gpt-realtime",
        "dall-e-3",
    ]

    def handler(req: httpx2.Request) -> httpx2.Response:
        assert req.headers["authorization"] == "Bearer sk-x"
        return httpx2.Response(200, json={"data": [{"id": i} for i in ids]})

    models = await list_models("openai", "sk-x", _client(handler))
    assert [m.id for m in models] == ["gpt-6-luna", "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17", "gpt-4o", "o3"]


async def test_openrouter_models_parse_pricing_and_capabilities() -> None:
    data = [
        {
            "id": "a/text",
            "name": "Text",
            "context_length": 1000,
            "supported_parameters": ["structured_outputs"],
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        },
        {"id": "b/image-only", "architecture": {"output_modalities": ["image"]}},
    ]
    models = await list_models(
        "openrouter", "k", _client(lambda req: httpx2.Response(200, json={"data": data}))
    )
    assert len(models) == 1
    m = models[0]
    assert m.supports_structured_output and m.supports_vision
    assert round(m.input_cost_per_mtok or 0, 6) == 2.0 and round(m.output_cost_per_mtok or 0, 6) == 10.0


async def test_key_checks() -> None:
    def handler(req: httpx2.Request) -> httpx2.Response:
        if req.url.host == "openrouter.ai":
            return httpx2.Response(200, json={"data": {"label": "my-key", "usage": 5, "limit": 5}})
        if req.url.host == "huggingface.co":
            return httpx2.Response(
                200, json={"name": "olen", "avatarUrl": "/a.png", "auth": {"accessToken": {"role": "write"}}}
            )
        if req.url.host == "civitai.com":
            return httpx2.Response(404)
        return httpx2.Response(401)

    c = _client(handler)
    orr = await check_key("openrouter_api_key", "k", c)
    assert orr.ok and orr.account == {"username": "my-key"} and "credit limit" in orr.warnings[0]
    hf = await check_key("hf_token", "hf_x", c)
    assert hf.ok and hf.account["username"] == "olen" and hf.token_type == "write" and hf.warnings  # type: ignore[index]
    civ = await check_key("civitai_api_key", "k", c)
    assert civ.ok and "not verified" in civ.warnings[0]
    bad = await check_key("openai_api_key", "k", c)
    assert not bad.ok and "rejected" in (bad.error or "")
