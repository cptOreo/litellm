"""Live e2e: a caller's own Anthropic key never follows a fallback onto a Bedrock deployment (#45309).

An admin turns on `forward_llm_provider_auth_headers` through /update/ui_settings, so a caller may bring
its own Anthropic key in `x-api-key`. The caller sends an invalid one, real Anthropic rejects it with a
401, and the per-request fallback reroutes to a Bedrock deployment that signs with the proxy's IAM
credentials. The caller must get a 200 from Bedrock. Before the fix the proxy handed Bedrock the
caller's Anthropic key as a bearer token, which Bedrock rejects.

The Anthropic deployment carries the proxy's real key, so until the setting reaches the worker that
serves the call, Anthropic answers without a fallback. The test resends until a call either falls back
or fails. The setting is proxy-wide, so the test is marked `proxy_wide_config`.
"""

from __future__ import annotations

import time
from typing import Final

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import Headers, NoBody, StreamingResponse, unwrap
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from models import ChatMessage, LiteLLMParamsBody, ReliabilityChatBody, RouterSettingsOverride
from pydantic import BaseModel, Field
from reliability_support import content_of

pytestmark = [pytest.mark.e2e, pytest.mark.proxy_wide_config]

ANTHROPIC_MODEL: Final = "anthropic/claude-haiku-5-5"
BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-haiku-5-5"


class ForwardingSettings(BaseModel):
    forward_llm_provider_auth_headers: bool


class CallerProviderKeyHeaders(Headers):
    x_litellm_api_key: str = Field(serialization_alias="x-litellm-api-key", repr=False)
    x_api_key: str = Field(serialization_alias="x-api-key", repr=False)


def _set_forwarding(client: ComplexityRouterClient, enabled: bool) -> None:
    unwrap(
        client.proxy.transport.patch(
            "/update/ui_settings",
            headers=client.proxy.transport.master,
            json=ForwardingSettings(forward_llm_provider_auth_headers=enabled),
            response_type=NoBody,
        )
    )


def _chat(client: ComplexityRouterClient, key: str, caller_key: str, primary: str, fallback: str) -> StreamingResponse:
    return client.proxy.transport.send(
        "/chat/completions",
        headers=CallerProviderKeyHeaders(x_litellm_api_key=key, x_api_key=caller_key),
        json=ReliabilityChatBody(
            model=primary,
            messages=[ChatMessage(role="user", content=f"Say hi. {unique_marker()}")],
            max_tokens=64,
            router_settings_override=RouterSettingsOverride(fallbacks=[{primary: [fallback]}]),
            cache={"no-cache": True},
        ),
    )


def _fell_back(response: StreamingResponse) -> bool:
    return int(response.headers.get("x-litellm-attempted-fallbacks", "0")) >= 1


def _first_call_after_forwarding_applies(
    client: ComplexityRouterClient, key: str, caller_key: str, primary: str, fallback: str
) -> StreamingResponse:
    deadline: Final = time.monotonic() + POLL_TIMEOUT
    while True:
        response = _chat(client, key, caller_key, primary, fallback)  # rebind-ok: resend until the setting lands
        if not response.ok or _fell_back(response) or time.monotonic() > deadline:
            return response
        time.sleep(POLL_INTERVAL)


class TestFallbackCallerCredentialScope:
    @meta(
        Subject(
            domain=Domain.ROUTING,
            providers=(Provider.ANTHROPIC, Provider.BEDROCK),
            models=(ANTHROPIC_MODEL, BEDROCK_MODEL),
            mode=Mode.NONSTREAM,
        )
    )
    def test_rejected_caller_anthropic_key_falls_back_to_bedrock_iam(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        primary: Final = f"e2e-caller-key-anthropic-{unique_marker()}"
        fallback: Final = f"e2e-caller-key-bedrock-{unique_marker()}"
        anthropic_id: Final = client.proxy.create_model(
            primary,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
            provider_live=True,
        )
        resources.defer(lambda: client.proxy.delete_model(anthropic_id))
        bedrock_id: Final = client.proxy.create_model(
            fallback,
            LiteLLMParamsBody(
                model=BEDROCK_MODEL,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
            provider_live=True,
        )
        resources.defer(lambda: client.proxy.delete_model(bedrock_id))
        resources.defer(lambda: _set_forwarding(client, False))
        _set_forwarding(client, True)

        response: Final = _first_call_after_forwarding_applies(
            client, scoped_key, f"sk-ant-api03-e2e-invalid-{unique_marker()}", primary, fallback
        )
        assert response.status_code == 200, (
            f"the Bedrock fallback should answer with the proxy's IAM credentials, got {response.status_code}: "
            f"{response.body[:300]}"
        )
        assert _fell_back(response), (
            f"the caller's invalid Anthropic key never made Anthropic fail within {POLL_TIMEOUT}s, so the "
            f"forwarding setting did not reach the proxy: {response.headers}"
        )
        assert content_of(response), f"the Bedrock fallback returned no content: {response.body[:300]}"
