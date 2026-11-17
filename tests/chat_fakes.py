"""Test doubles for the chat provider contract (C9.3)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from app.rag.chat_provider import ProviderResult


class ScriptedProvider:
    """Replays scripted results (or raises scripted exceptions), recording calls."""

    def __init__(
        self,
        script: Iterable[ProviderResult | BaseException],
        *,
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self._script = list(script)
        self._on_call = on_call
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_message: str, user_message: str) -> ProviderResult:
        self.calls.append((system_message, user_message))
        if self._on_call is not None:
            self._on_call(len(self.calls))
        if not self._script:
            raise AssertionError("ScriptedProvider script exhausted")
        step = self._script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
