import sys
import types

import pytest

from cf_reasoning import llm_client


class _RetryableOpenAIError(Exception):
    def __init__(self, message="temporary"):
        super().__init__(message)
        self.status_code = 500


class _RetryableAnthropicError(Exception):
    pass


class _FakeAnthropicResponse:
    stop_reason = "end_turn"
    content = [types.SimpleNamespace(type="text", text="ok")]


class _FakeAnthropicMessages:
    def __init__(self, calls, error):
        self.calls = calls
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) < 2:
            raise self.error
        return _FakeAnthropicResponse()


class _FakeAnthropicClient:
    def __init__(self, calls, error):
        self.messages = _FakeAnthropicMessages(calls, error)


class _FakeOpenAICompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) < 3:
            raise _RetryableOpenAIError()
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])


class _FakeOpenAIChat:
    def __init__(self, calls):
        self.completions = _FakeOpenAICompletions(calls)


class _FakeOpenAIClient:
    def __init__(self, calls, *args, **kwargs):
        self.chat = _FakeOpenAIChat(calls)


class _FakeAnthropicModule(types.SimpleNamespace):
    pass


def test_call_llm_supports_openai_compatible_provider(monkeypatch):
    calls = []

    def fake_openai_compatible(prompt, model, max_tokens, api_key, base_url=None, json_output=False, schema=None):
        calls.append((prompt, model, max_tokens, api_key, base_url, json_output, schema))
        return '{"answer":"true"}'

    monkeypatch.setattr(llm_client, "_call_openai_compatible", fake_openai_compatible)
    monkeypatch.setenv("CF_REASONING_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "provider-model")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://example.test/v1")

    assert llm_client._call_llm("prompt", 123, json_output=True, schema={"type": "object"}) == '{"answer":"true"}'
    assert calls == [("prompt", "provider-model", 123, "key", "https://example.test/v1", True, {"type": "object"})]


def test_call_llm_deepseek_uses_openai_compatible_transport(monkeypatch):
    calls = []

    def fake_openai_compatible(prompt, model, max_tokens, api_key, base_url=None, json_output=False, schema=None):
        calls.append((model, api_key, base_url))
        return "ok"

    monkeypatch.setattr(llm_client, "_call_openai_compatible", fake_openai_compatible)
    monkeypatch.setenv("CF_REASONING_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    assert llm_client._call_llm("prompt", 123) == "ok"
    assert calls == [("deepseek-chat", "deepseek-key", "https://api.deepseek.com")]




def test_openai_compatible_retries_retryable_errors(monkeypatch):
    calls = []
    fake_openai = types.SimpleNamespace(OpenAI=lambda *args, **kwargs: _FakeOpenAIClient(calls, *args, **kwargs))
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("CF_REASONING_LLM_RETRIES", "3")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    assert llm_client._call_openai_compatible("prompt", "model", 10, "key") == "ok"
    assert len(calls) == 3


def test_anthropic_retries_retryable_errors(monkeypatch):
    calls = []
    error = _RetryableAnthropicError("temporary")
    fake_anthropic = _FakeAnthropicModule(
        Anthropic=lambda: _FakeAnthropicClient(calls, error),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
        APIConnectionError=_RetryableAnthropicError,
        RateLimitError=type("RateLimitError", (_RetryableAnthropicError,), {}),
        APIStatusError=type("APIStatusError", (_RetryableAnthropicError,), {}),
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    monkeypatch.setenv("CF_REASONING_LLM_RETRIES", "2")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    assert llm_client._call_anthropic("prompt", "model", 10) == "ok"
    assert len(calls) == 2


def test_call_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("CF_REASONING_LLM_PROVIDER", "unknown-provider")
    with pytest.raises(RuntimeError, match="anthropic, deepseek, or openai-compatible"):
        llm_client._call_llm("prompt", 123)
