"""Short-term rolling conversation window for the harness."""

from __future__ import annotations

from dataclasses import dataclass

from vektori.models.base import ChatModelProvider


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate based on character count."""
    return sum(len(m.get("content", "")) for m in messages) // 4


@dataclass
class WindowState:
    recent_messages: list[dict[str, str]]
    rolling_summary: str
    estimated_tokens: int
    compaction_count: int


class MessageWindow:
    """Local conversation buffer with optional summarizing compaction."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 12000,
        compaction_trigger_ratio: float = 0.8,
        keep_last_n_turns: int = 6,
        summary_max_tokens: int = 400,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.compaction_trigger_ratio = compaction_trigger_ratio
        self.keep_last_n_turns = keep_last_n_turns
        self.summary_max_tokens = summary_max_tokens
        self._recent_messages: list[dict[str, str]] = []
        self._rolling_summary = ""
        self._compaction_count = 0

    def add(self, role: str, content: str) -> None:
        self._recent_messages.append({"role": role, "content": content})

    def snapshot(self) -> WindowState:
        return WindowState(
            recent_messages=list(self._recent_messages),
            rolling_summary=self._rolling_summary,
            estimated_tokens=self.estimated_tokens(),
            compaction_count=self._compaction_count,
        )

    def estimated_tokens(self) -> int:
        summary_tokens = len(self._rolling_summary) // 4
        return estimate_tokens(self._recent_messages) + summary_tokens

    async def compact(self, summarizer: ChatModelProvider) -> bool:
        if self.estimated_tokens() < self.max_context_tokens * self.compaction_trigger_ratio:
            return False

        keep_messages = self.keep_last_n_turns * 2
        if len(self._recent_messages) <= keep_messages:
            return False

        old_messages = self._recent_messages[:-keep_messages]
        kept_messages = self._recent_messages[-keep_messages:]
        old_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in old_messages
        )

        prompt = (
            "Summarize the conversation state in this format:\n"
            "Conversation Summary\n"
            "- Active goals:\n"
            "- Confirmed preferences:\n"
            "- Open questions:\n"
            "- Constraints:\n"
            "- Recent commitments:\n\n"
            f"Conversation:\n{old_text}"
        )
        result = await summarizer.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
            temperature=0.0,
        )
        if result.content:
            self._rolling_summary = result.content.strip()
        self._recent_messages = kept_messages
        self._compaction_count += 1
        return True

    def reset(self) -> None:
        self._recent_messages = []
        self._rolling_summary = ""
        self._compaction_count = 0
