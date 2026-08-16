"""Tests for the one typed `Config` and its loader (§6, §9).

`conftest._isolated_env` provides isolation from the shell and any real `.env`; these
tests only add what they need.
"""

import traceback
from pathlib import Path

import pytest

from orchestra.agents.engine import DEFAULT_MAX_CONCURRENCY, DEFAULT_SUBTASK_ATTEMPTS
from orchestra.agents.workers.tool_loop import DEFAULT_MAX_TURNS, DEFAULT_TOKEN_BUDGET
from orchestra.config import DEFAULT_MODEL, load_config
from orchestra.core.errors import ConfigError, ExitCode

# Imported here rather than in `config.py`, which must not pull the vendor SDK onto the
# startup path — this is what keeps the field default and the adapter's from drifting.
from orchestra.providers.anthropic import DEFAULT_MAX_TOKENS

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


def test_load_config_bounds_default_to_what_their_consumers_ship_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting nothing must run exactly as before these became settable, so the defaults are
    asserted against the consumers' constants rather than against repeated literals."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    config = load_config()

    assert config.anthropic_max_tokens == DEFAULT_MAX_TOKENS
    assert config.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert config.subtask_attempts == DEFAULT_SUBTASK_ATTEMPTS
    assert config.worker_token_budget == DEFAULT_TOKEN_BUDGET
    assert config.worker_max_turns == DEFAULT_MAX_TURNS


def test_load_config_bound_env_vars_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "2048")
    monkeypatch.setenv("MAX_CONCURRENCY", "2")
    monkeypatch.setenv("WORKER_TOKEN_BUDGET", "500")
    monkeypatch.setenv("WORKER_MAX_TURNS", "3")

    config = load_config()

    assert (config.anthropic_max_tokens, config.max_concurrency) == (2048, 2)
    assert (config.worker_token_budget, config.worker_max_turns) == (500, 3)


@pytest.mark.parametrize(
    "variable",
    [
        "ANTHROPIC_MAX_TOKENS",
        "MAX_CONCURRENCY",
        "SUBTASK_ATTEMPTS",
        "WORKER_TOKEN_BUDGET",
        "WORKER_MAX_TURNS",
    ],
)
def test_load_config_rejects_a_non_positive_bound(
    monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    """§9 fail-fast: `MAX_CONCURRENCY=0` would otherwise surface minutes later as the
    engine's `ValueError`, an exit-1 bug rather than a configuration error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv(variable, "0")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    assert variable in str(exc_info.value)
    assert exc_info.value.exit_code == ExitCode.CONFIG


def test_load_config_artifact_dir_defaults_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§9: user data lives under ~/.orchestra/, never the working directory."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    artifact_dir = load_config().artifact_dir

    # Against $HOME, not the module's own default, so it can fail if the default stops
    # being home-relative.
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
    """A literal `~` makes a directory named `~` and a relative path follows the shell —
    both violate §9's "never the working directory"."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ARTIFACT_DIR", raw)

    artifact_dir = load_config().artifact_dir

    assert artifact_dir.is_absolute()
    assert "~" not in str(artifact_dir)


def test_load_config_data_dir_defaults_to_the_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserted by reading a file, not comparing paths: the default must point at the
    dataset the agents actually query, from any cwd (#5)."""
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
    """§9: a second secret is a second thing that must not reach a log or config dump."""
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
    """`ANTHROPIC_API_KEY=` is .env.example copied but not edited. Without `min_length=1`
    it validates and resurfaces minutes later as a provider auth error (§9 fail-fast)."""
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
    """Pydantic renders `input_value=...` verbatim, so surfacing the raw error would print
    the rejected key (§9). The message must be built from the field."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "input_value" not in message
    assert "errors.pydantic.dev" not in message


def test_load_config_error_drops_the_pydantic_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: keeping the chain let `--debug` print what `_explain()` redacts — a
    chained `ValidationError` carries `input_value=<the key>` into the traceback (§9)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    exc = exc_info.value
    # `__context__` stays set (implicit chaining always records it); `__suppress_context__`
    # is what the stdlib and Rich honour when rendering, so that is the invariant.
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    rendered = "".join(traceback.format_exception(exc))
    assert "input_value" not in rendered
    assert "errors.pydantic.dev" not in rendered


@pytest.mark.parametrize("blank", ["", "   "])
def test_load_config_blank_search_key_is_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """`TAVILY_API_KEY=` uncommented but not filled in. Left as an empty secret it selects
    the live path with no credential: every search 401s and the run reports itself
    degraded throughout."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TAVILY_API_KEY", blank)

    assert load_config().tavily_api_key is None
