"""Tests for the one typed `Config` and its loader (CONVENTIONS.md §6, §9).

Isolation from the shell and from any real `.env` is provided by the autouse
`_isolated_env` fixture in `conftest.py`; these tests only add what they need.
"""

import traceback
from pathlib import Path

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


def test_load_config_artifact_dir_defaults_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§9: user data lives under ~/.orchestra/, never the working directory."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    artifact_dir = load_config().artifact_dir

    # Asserted against $HOME, not against the module's own default, so the test can
    # actually fail if the default stops being home-relative.
    assert artifact_dir == tmp_path / "home" / ".orchestra" / "artifacts"


def test_load_config_artifact_dir_env_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "elsewhere"))

    assert load_config().artifact_dir == tmp_path / "elsewhere"


@pytest.mark.parametrize("raw", ["~/somewhere", "relative-artifacts"])
def test_load_config_artifact_dir_is_always_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: str
) -> None:
    """A literal `~` would create a directory called `~`; a relative path follows the shell.

    Both violate §9's "never the working directory", so the field resolves them.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ARTIFACT_DIR", raw)

    artifact_dir = load_config().artifact_dir

    assert artifact_dir.is_absolute()
    assert "~" not in str(artifact_dir)


def test_load_config_data_dir_defaults_to_the_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundled mock data is versioned with the code, so it is found from any cwd.

    Asserted by reading a file, not by comparing paths: what matters is that the
    default points at the dataset the agents actually query (#5).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    data_dir = load_config().data_dir

    assert data_dir.is_absolute()
    assert (data_dir / "quarterly_financials.csv").is_file()


def test_load_config_data_dir_env_var_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "fixtures"))

    assert load_config().data_dir == tmp_path / "fixtures"


def test_load_config_without_a_search_key_leaves_it_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset is the supported state, not a misconfiguration: it selects the offline corpus."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    assert load_config().tavily_api_key is None


def test_load_config_search_key_is_read_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """§9: a second secret is a second thing that must not reach a log or a config dump."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret")

    config = load_config()

    assert config.tavily_api_key is not None
    assert config.tavily_api_key.get_secret_value() == "tvly-secret"
    assert "tvly-secret" not in repr(config)
    assert "tvly-secret" not in config.model_dump_json()


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


def test_load_config_error_drops_the_pydantic_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: keeping the chain let `--debug` print what `_explain()` redacts.

    The boundary renders the traceback under `--debug`, and a chained
    `ValidationError` carries `input_value=<the key>` into it (§9).
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    exc = exc_info.value
    # `__context__` stays set — implicit chaining always records it. What both the
    # stdlib and Rich honour when rendering is `__suppress_context__`, so that is the
    # invariant, and the rendered traceback is the proof.
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    rendered = "".join(traceback.format_exception(exc))
    assert "input_value" not in rendered
    assert "errors.pydantic.dev" not in rendered
