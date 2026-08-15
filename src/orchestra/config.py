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
    """`~/.orchestra/artifacts` — where §9 puts user data, never the working directory.

    Decided here because `Path.home()` reads $HOME and config.py is the only module
    allowed to read the environment (§6); leaving the default to the store's caller is
    how that rule gets broken. A function, not a constant, so it resolves when config is
    loaded rather than when this module is first imported.
    """
    return Path.home() / ".orchestra" / "artifacts"


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

    @field_validator("artifact_dir")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        """Expand `~` and resolve, so §9's "never the working directory" holds for any input.

        `ARTIFACT_DIR=~/somewhere` arrives as a literal tilde and `ARTIFACT_DIR=out` as a
        relative path; taken as given, the first creates a directory named `~` and the
        second scatters artifacts wherever the user happened to `cd`.
        """
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
