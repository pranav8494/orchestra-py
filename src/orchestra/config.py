"""One typed `Config`, loaded once at startup and injected from `app.py`.

Precedence: defaults < `.env` file < environment. This is the only module allowed to
read the environment (CONVENTIONS.md §6), and no global instance lives here — a
singleton would hide the dependency from everything that uses it.

The API key is a `SecretStr` so it cannot leak through a repr, a log record, or a
`--debug` config dump (§9). Validation failures become `ConfigError`, never a raw
pydantic traceback (§8).
"""

from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from orchestra.core.errors import ConfigError

DEFAULT_MODEL = "claude-opus-5"


def default_artifact_dir() -> Path:
    """`~/.orchestra/artifacts` — §9's home for user data, never the working directory.

    Decided here because `Path.home()` reads $HOME, and only config.py may (§6). A
    function, not a constant, so it resolves at load rather than at import.
    """
    return Path.home() / ".orchestra" / "artifacts"


def default_data_dir() -> Path:
    """The repo's committed `data/` — the bundled mock dataset the agents read (#5).

    Read-only fixtures, so unlike `artifact_dir` this does not belong under `~`: the
    files are part of the checkout and versioned with the code that reads them.

    Located relative to this module rather than to the working directory, so a run from
    any directory finds it. That resolution assumes the source layout — installed as a
    wheel, `data/` sits outside the package and this points at nothing. Deliberate:
    `DATA_DIR` is the override, and a tool that cannot find its dataset reports that as
    a `ToolResponse` the model can read (§6) rather than failing the run.
    """
    return Path(__file__).resolve().parents[2] / "data"


_FIX_HINT = (
    "Fix: copy .env.example to .env and set ANTHROPIC_API_KEY=<your-key>, "
    "or export it in your shell."
)


class Config(BaseSettings):
    """Runtime configuration. Build it with `load_config()`, never directly."""

    # No env_prefix: each field maps to its own upper-cased name, so the field
    # names below are the documented variable names in .env.example.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # an unrelated variable in .env is not a reason to refuse to start
    )

    # min_length catches the copied-but-unedited `ANTHROPIC_API_KEY=` in .env.example,
    # which would otherwise validate as "" and fail later as a provider auth error.
    anthropic_api_key: SecretStr = Field(min_length=1)
    anthropic_model: str = DEFAULT_MODEL
    artifact_dir: Path = Field(default_factory=default_artifact_dir)
    data_dir: Path = Field(default_factory=default_data_dir)
    # Optional by design: unset means the `search` tool reads the bundled corpus, which
    # is the offline-safe path the demo depends on. `None` rather than "" so "no key" is
    # a state the type carries, not a value every caller has to test for emptiness.
    tavily_api_key: SecretStr | None = None

    @field_validator("tavily_api_key")
    @classmethod
    def _blank_is_unset(cls, value: SecretStr | None) -> SecretStr | None:
        """Treat `TAVILY_API_KEY=` as no key at all.

        The line copied from `.env.example` and uncommented but not filled in would
        otherwise select the live path with an empty credential: every search would 401
        and fall back, so the run still works but reports itself degraded the whole way
        through. `ANTHROPIC_API_KEY` guards the same mistake with `min_length=1`; here
        the value is optional, so the empty case folds to the default instead of failing.
        """
        if value is None or not value.get_secret_value().strip():
            return None
        return value

    @field_validator("artifact_dir", "data_dir")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        """Taken as given, `ARTIFACT_DIR=~/x` makes a directory named `~` and a relative
        path follows the shell — both against §9."""
        return value.expanduser().resolve()


def load_config() -> Config:
    """Load and validate configuration. Call once, at startup.

    Returns:
        The validated `Config`.

    Raises:
        ConfigError: a required setting is missing or invalid (exit code 3).
    """
    try:
        return Config()  # fields come from env/.env, not from arguments
    except ValidationError as exc:
        # `from None`, not `from exc`: pydantic's rendering echoes the rejected input,
        # so keeping the chain would let `--debug`'s traceback print the key that
        # `_explain()` exists to keep out of the message (§9).
        raise ConfigError(_explain(exc)) from None


def _explain(exc: ValidationError) -> str:
    """Turn a `ValidationError` into a message naming the variable and the fix.

    Uses only the field location and pydantic's message — never `str(exc)`, which
    echoes the offending input and would print a secret (§9).
    """
    # An empty value is "unset" as far as the user is concerned, not a length violation.
    unset = {"missing", "string_too_short", "too_short"}
    problems = [
        f"{'.'.join(str(part) for part in err['loc']).upper()} "
        + ("is not set." if err["type"] in unset else f"is invalid: {err['msg']}.")
        for err in exc.errors()
    ]
    return "\n".join([*problems, _FIX_HINT])
