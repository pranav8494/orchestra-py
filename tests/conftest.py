"""Shared test fixtures.

`FakeProvider` belongs here once `orchestra.providers.base.Provider` exists: it is
the substitute that lets the whole suite run without touching the network, which
is the payoff for keeping vendor SDKs behind the provider port.

See CONVENTIONS.md §12.
"""

from pathlib import Path

import pytest

# Every setting `Config` reads. Listed once so a new field cannot silently start
# leaking the developer's shell into the suite.
_SETTING_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cut every test off from the ambient environment and any real `.env` (§9).

    `Config` reads the environment and a `.env` resolved against the *current working
    directory*, so without both halves of this a developer with an exported key gets a
    different result from CI. Autouse because that hazard is suite-wide, not per-test.
    """
    for var in _SETTING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
