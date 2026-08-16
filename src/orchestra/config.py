"""One typed `Config`, loaded once at startup and injected from `app.py`.

Precedence: defaults < `.env` < environment. The only module that may read the
environment (§6); no global instance, so the dependency stays visible. Secrets are
`SecretStr` (§9) and validation failures become `ConfigError` (§8).
"""

from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from orchestra.agents.engine import DEFAULT_MAX_CONCURRENCY, DEFAULT_SUBTASK_ATTEMPTS
from orchestra.agents.workers.tool_loop import DEFAULT_MAX_TURNS, DEFAULT_TOKEN_BUDGET
from orchestra.core.errors import ConfigError

DEFAULT_MODEL = "claude-opus-5"


def default_artifact_dir() -> Path:
    """`~/.orchestra/artifacts` — §9's home for user data, never the working directory.

    A function, not a constant: `Path.home()` reads $HOME, which only config.py may (§6).
    """
    return Path.home() / ".orchestra" / "artifacts"


def default_data_dir() -> Path:
    """The repo's committed `data/` — read-only fixtures versioned with the code (#5).

    Resolved against this module, not the cwd. Installed as a wheel it points at
    nothing; `DATA_DIR` is the override and a tool without its dataset reports that as
    a `ToolResponse` (§6) rather than failing the run.
    """
    return Path(__file__).resolve().parents[2] / "data"


_FIX_HINT = (
    "Fix: copy .env.example to .env and set ANTHROPIC_API_KEY=<your-key>, "
    "or export it in your shell."
)


class Config(BaseSettings):
    """Runtime configuration. Build it with `load_config()`, never directly."""

    # No env_prefix: field names are the variable names documented in .env.example.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # an unrelated variable in .env is not a reason to refuse to start
    )

    # min_length: an unedited `ANTHROPIC_API_KEY=` would otherwise validate as "" and
    # fail later as a provider auth error.
    anthropic_api_key: SecretStr = Field(min_length=1)
    anthropic_model: str = DEFAULT_MODEL
    # The literal, not `providers.anthropic.DEFAULT_MAX_TOKENS`: importing it would pull the
    # vendor SDK onto every startup, which `providers/base.py` defers on purpose. The two are
    # pinned together by `test_config`.
    anthropic_max_tokens: int = Field(default=16_000, gt=0)
    artifact_dir: Path = Field(default_factory=default_artifact_dir)
    data_dir: Path = Field(default_factory=default_data_dir)
    # Unset means `search` reads the bundled corpus — the offline-safe path. `None`, not
    # "", so "no key" is a state the type carries rather than one callers test for.
    tavily_api_key: SecretStr | None = None
    # The run's bounds (§10), defaulting to the constants their consumers ship with, so
    # setting nothing runs exactly as before. `app.py` is what carries them to the engine
    # and the workers.
    max_concurrency: int = Field(default=DEFAULT_MAX_CONCURRENCY, ge=1)
    subtask_attempts: int = Field(default=DEFAULT_SUBTASK_ATTEMPTS, ge=1)
    worker_token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=1)
    worker_max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1)

    @field_validator("tavily_api_key")
    @classmethod
    def _blank_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat `TAVILY_API_KEY=` as no key at all.

        An empty credential would otherwise select the live path, 401 on every search
        and report the run degraded throughout. Optional, so it folds to the default
        rather than failing like `ANTHROPIC_API_KEY`'s `min_length=1`.
        """
        if value is None or not value.get_secret_value().strip():
            return None
        return value

    @field_validator("artifact_dir", "data_dir")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        """`ARTIFACT_DIR=~/x` would make a directory named `~`, and a relative path
        follows the shell — both against §9."""
        return value.expanduser().resolve()


def load_config() -> Config:
    """Load and validate configuration. Call once, at startup.

    Raises:
        ConfigError: a required setting is missing or invalid (exit code 3).
    """
    try:
        return Config()  # fields come from env/.env, not from arguments
    except ValidationError as exc:
        # `from None`: pydantic's chained traceback echoes the rejected input, so under
        # `--debug` it would print the key `_explain()` exists to keep out (§9).
        raise ConfigError(_explain(exc)) from None


def _explain(exc: ValidationError) -> str:
    """Name the offending variable and the fix.

    Uses the field location and pydantic's message only — never `str(exc)`, which echoes
    the input and would print a secret (§9).
    """
    # To the user an empty value is "unset", not a length violation.
    unset = {"missing", "string_too_short", "too_short"}
    problems = [
        f"{'.'.join(str(part) for part in err['loc']).upper()} "
        + ("is not set." if err["type"] in unset else f"is invalid: {err['msg']}.")
        for err in exc.errors()
    ]
    return "\n".join([*problems, _FIX_HINT])
