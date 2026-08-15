"""Tests for the one typed `Config` and its loader (CONVENTIONS.md §6, §9).

Isolation from the shell and from any real `.env` is provided by the autouse
`_isolated_env` fixture in `conftest.py`; these tests only add what they need.
"""

import pytest

from orchestra.config import DEFAULT_MODEL, load_config
from orchestra.core.errors import ConfigError, ExitCode

FAKE_KEY = "sk-ant-test-key"


def test_load_config_with_api_key_set_returns_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    config = load_config()

    assert config.anthropic_api_key.get_secret_value() == FAKE_KEY
    assert config.anthropic_model == DEFAULT_MODEL


def test_load_config_with_model_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env beats defaults — the `defaults < file < env < flags` precedence in §6."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-some-other-model")

    assert load_config().anthropic_model == "claude-some-other-model"


def test_load_config_missing_api_key_raises_config_error() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert ".env" in message  # §8: give the message, the cause, and the fix
    assert exc_info.value.exit_code == ExitCode.CONFIG


def test_load_config_empty_api_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ANTHROPIC_API_KEY=` is .env.example copied but not edited.

    Without the `min_length=1` guard it validates as "" and resurfaces minutes later
    as a provider auth error, which §9's fail-fast rule exists to prevent.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    assert exc_info.value.exit_code == ExitCode.CONFIG


def test_config_repr_redacts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """§9: no repr, log record, or `--debug` config dump may carry the secret."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    config = load_config()

    assert FAKE_KEY not in repr(config)
    assert FAKE_KEY not in str(config)
    assert FAKE_KEY not in config.model_dump_json()


def test_load_config_error_message_omits_pydantic_input_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message must be built from the field, never from `str(ValidationError)`.

    Pydantic renders `input_value=...` verbatim, so surfacing the raw error would
    print the rejected key (§9). Empty here only because that is the sole reachable
    failure — the assertion guards the whole class of them.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "input_value" not in message
    assert "errors.pydantic.dev" not in message
