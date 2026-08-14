"""Shared test fixtures.

`FakeProvider` belongs here once `orchestra.providers.base.Provider` exists: it is
the substitute that lets the whole suite run without touching the network, which
is the payoff for keeping vendor SDKs behind the provider port.

See CONVENTIONS.md §12.
"""
