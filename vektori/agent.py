"""Native conversational harness built on top of the Vektori memory engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from vektori.client import Vektori
from vektori.context import AgentContextLoader
from vektori.memory.profile import InMemoryProfileStore, ProfilePatch, ProfileStore, SQLiteProfileStore
from vektori.memory.window import MessageWindow
from vektori.models.base import ChatModelProvider
from vektori.prompts import build_messages
from vektori.retrieval.gate import should_retrieve


@dataclass
class AgentConfig:
    max_context_tokens: int = 12000
    reserve_response_tokens: int = 1500
    retrieval_depth: str = "l1"
    retrieval_top_k: int = 8
    retrieve_on_every_turn: bool = False
    enable_retrieval_gate: bool = True
    enable_profile_learning: bool = True
    enable_tool_calling: bool = False
    max_tool_round_trips: int = 3
    compaction_trigger_ratio: float = 0.8
    keep_last_n_turns: int = 6
    summary_max_tokens: int = 400
    persist_assistant_messages: bool = True
    background_add: bool = True
    runtime_overrides: dict[str, Any] = field(default_factory=dict)
    profile_store_path: str | None = None


@dataclass
class AgentTurnResult:
    content: str
    messages: list[dict[str, str]]
    memories_used: dict[str, list[dict[str, Any]]]
    summary_updated: bool
    profile_updates: list[ProfilePatch]
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int] | None = None


class VektoriAgent:
    """Stateful chat harness that composes retrieval, prompt assembly, and storage."""

    def __init__(
        self,
        memory: Vektori,
        model: ChatModelProvider,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        config: AgentConfig | None = None,
        context_path: str | None = None,
        profile_store: ProfileStore | None = None,
    ) -> None:
        self.memory = memory
        self.model = model
        self.user_id = user_id
        self.agent_id = agent_id
        self.session_id = session_id or f"agent-session-{uuid4()}"
        self.config = config or AgentConfig()
        self.context_loader = AgentContextLoader(context_path)
        self.profile_store = profile_store or self._build_profile_store()
        self.window = MessageWindow(
            max_context_tokens=self.config.max_context_tokens,
            compaction_trigger_ratio=self.config.compaction_trigger_ratio,
            keep_last_n_turns=self.config.keep_last_n_turns,
            summary_max_tokens=self.config.summary_max_tokens,
        )
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def chat(
        self,
        user_message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTurnResult:
        self.window.add("user", user_message)

        memories: dict[str, list[dict[str, Any]]] = {"facts": [], "episodes": [], "sentences": []}
        if await self._should_retrieve(user_message):
            memories = await self.memory.search(
                query=user_message,
                user_id=self.user_id,
                agent_id=self.agent_id,
                depth=self.config.retrieval_depth,
                top_k=self.config.retrieval_top_k,
            )

        context = self.context_loader.load()
        profile_patches = await self.profile_store.list_active(
            observer_id=self.agent_id or "default-agent",
            observed_id=self.user_id,
        )
        messages = build_messages(
            context=context,
            profile_patches=profile_patches,
            memories=memories,
            window_state=self.window.snapshot(),
            runtime_overrides=self.config.runtime_overrides,
        )

        completion = await self.model.complete(
            messages,
            max_tokens=self.config.reserve_response_tokens,
            temperature=0.2,
        )
        assistant_content = completion.content or ""
        self.window.add("assistant", assistant_content)

        summary_updated = await self.window.compact(self.model)
        profile_updates = await self._learn_profile_patches(user_message)

        exchange = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_content},
        ]
        if self.config.background_add:
            self._schedule_background(
                self.memory.add(
                    messages=exchange,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    metadata=metadata,
                )
            )
        else:
            await self.memory.add(
                messages=exchange,
                session_id=self.session_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                metadata=metadata,
            )

        return AgentTurnResult(
            content=assistant_content,
            messages=self.window.snapshot().recent_messages,
            memories_used={
                "facts": memories.get("facts", []),
                "episodes": memories.get("episodes", []),
                "sentences": memories.get("sentences", []),
            },
            summary_updated=summary_updated,
            profile_updates=profile_updates,
            tool_calls=completion.tool_calls,
            usage=completion.usage,
        )

    async def add_messages(
        self,
        messages: list[dict[str, str]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for message in messages:
            self.window.add(message["role"], message["content"])
        await self.memory.add(
            messages=messages,
            session_id=self.session_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            metadata=metadata,
        )

    def reset_window(self) -> None:
        self.window.reset()

    async def close(self) -> None:
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.profile_store.close()

    async def _should_retrieve(self, user_message: str) -> bool:
        if self.config.retrieve_on_every_turn:
            return True
        if not self.config.enable_retrieval_gate:
            return True
        return should_retrieve(user_message)

    def _schedule_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _build_profile_store(self) -> ProfileStore:
        if self.config.profile_store_path:
            return SQLiteProfileStore(Path(self.config.profile_store_path))
        return InMemoryProfileStore()

    async def _learn_profile_patches(self, user_message: str) -> list[ProfilePatch]:
        if not self.config.enable_profile_learning:
            return []

        patch = self._extract_explicit_patch(user_message)
        if patch is None:
            return []

        await self.profile_store.save(patch)
        return [patch]

    def _extract_explicit_patch(self, user_message: str) -> ProfilePatch | None:
        observer_id = self.agent_id or "default-agent"
        text = user_message.strip()

        name_match = re.match(r"(?i)call me ([A-Za-z][A-Za-z0-9 _-]{0,49})\.?$", text)
        if name_match:
            return ProfilePatch(
                key="preferences.name",
                value=name_match.group(1).strip(),
                reason="User explicitly stated preferred form of address.",
                source="explicit_user_request",
                observer_id=observer_id,
                observed_id=self.user_id,
                confidence=0.98,
            )

        verbosity_patterns = [
            (r"(?i)^keep (your )?answers short\.?$", "short"),
            (r"(?i)^be concise\.?$", "short"),
            (r"(?i)^give detailed answers\.?$", "detailed"),
            (r"(?i)^be detailed\.?$", "detailed"),
        ]
        for pattern, value in verbosity_patterns:
            if re.match(pattern, text):
                return ProfilePatch(
                    key="response_style.verbosity",
                    value=value,
                    reason="User explicitly stated answer verbosity preference.",
                    source="explicit_user_request",
                    observer_id=observer_id,
                    observed_id=self.user_id,
                    confidence=0.95,
                )

        units_match = re.match(r"(?i)^remember that i prefer (metric|imperial) units\.?$", text)
        if units_match:
            return ProfilePatch(
                key="preferences.units",
                value=units_match.group(1).lower(),
                reason="User explicitly stated unit preference.",
                source="explicit_user_request",
                observer_id=observer_id,
                observed_id=self.user_id,
                confidence=0.96,
            )

        return None
