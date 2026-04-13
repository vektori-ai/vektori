"""
Pipecat FrameProcessors for Vektori memory integration.

Two processors work together in a voice pipeline:

  VektoriMemoryProcessor  — placed BEFORE the LLM
      • Intercepts OpenAILLMContextFrame each turn
      • Searches Vektori for relevant facts / episodes from the user's utterance
      • Injects a [Memory context] block into the LLM system prompt

  VektoriStorageProcessor — placed AFTER the LLM
      • Accumulates the user utterance (TranscriptionFrame) and
        the assistant reply (TextFrame chunks until LLMFullResponseEndFrame)
      • Stores the complete turn in Vektori for future retrieval

Minimal pipeline wiring
-----------------------
    context = OpenAILLMContext([{"role": "system", "content": SYSTEM}])
    ctx_agg = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx_agg.user(),
        VektoriMemoryProcessor(vektori, user_id, base_system_prompt=SYSTEM),
        llm,
        VektoriStorageProcessor(vektori, user_id, session_id, context),
        tts,
        transport.output(),
        ctx_agg.assistant(),
    ])
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from vektori import Vektori

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_memory(memory: dict) -> str:
    """Convert Vektori search results into a compact, voice-friendly string."""
    parts: list[str] = []

    facts = memory.get("facts", [])
    if facts:
        bullets = "\n".join(f"- {f['text']}" for f in facts)
        parts.append(f"Facts about this user:\n{bullets}")

    episodes = memory.get("episodes", [])
    if episodes:
        bullets = "\n".join(f"- {ep['text']}" for ep in episodes)
        parts.append(f"Past episodes:\n{bullets}")

    return "\n\n".join(parts)


def _set_system_message(context: OpenAILLMContext, content: str) -> None:
    """Upsert the system message inside an OpenAILLMContext."""
    msgs = list(context.messages)
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            msgs[i] = {"role": "system", "content": content}
            break
    else:
        msgs.insert(0, {"role": "system", "content": content})

    # Support both mutable-list and set_messages() API variants
    if hasattr(context, "set_messages"):
        context.set_messages(msgs)
    else:
        context.messages = msgs


def _last_message(context: OpenAILLMContext, role: str) -> str | None:
    """Return the content of the most recent message with the given role."""
    for m in reversed(list(context.messages)):
        if m.get("role") == role:
            content = m.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content (e.g. OpenAI vision format)
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content or None
    return None


# ---------------------------------------------------------------------------
# VektoriMemoryProcessor
# ---------------------------------------------------------------------------


class VektoriMemoryProcessor(FrameProcessor):
    """
    Injects Vektori long-term memory into the LLM system prompt before each turn.

    Place this processor **between the user context aggregator and the LLM**
    in your Pipecat pipeline.

    Parameters
    ----------
    vektori:
        An initialised (or un-initialised) ``Vektori`` instance.
    user_id:
        Stable identifier for the current user; used as the Vektori user key.
    base_system_prompt:
        The static part of your system prompt.  Memory is appended beneath it
        as a ``[Memory context]`` block.  If empty, only the memory block is
        used as the system message.
    depth:
        Vektori retrieval depth — ``"l0"`` (facts only), ``"l1"`` (facts +
        episodes + source sentences, default), or ``"l2"`` (full context window).
        Use ``"l1"`` for voice; ``"l2"`` is too verbose for spoken responses.
    top_k:
        Maximum number of facts to retrieve per turn (default 5).  Keep low for
        voice to avoid bloating the context injected before TTS.
    session_id:
        Optional Vektori session identifier.  If omitted, defaults to
        ``"pipecat-{user_id}"``.
    """

    def __init__(
        self,
        vektori: "Vektori",
        user_id: str,
        *,
        base_system_prompt: str = "",
        depth: str = "l1",
        top_k: int = 5,
        session_id: str = "",
    ) -> None:
        super().__init__()
        self._vektori = vektori
        self._user_id = user_id
        self._base_system = base_system_prompt
        self._depth = depth
        self._top_k = top_k
        self._session_id = session_id or f"pipecat-{user_id}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _inject_memory(self, context: OpenAILLMContext) -> None:
        """Search Vektori and update the system message in *context* in-place."""
        query = _last_message(context, "user")
        if not query:
            return

        try:
            memory = await self._vektori.search(
                query=query,
                user_id=self._user_id,
                depth=self._depth,
                top_k=self._top_k,
            )
        except Exception:
            logger.exception("VektoriMemoryProcessor: search failed — skipping injection")
            return

        memory_text = _format_memory(memory)
        if not memory_text:
            return

        new_system = self._base_system
        if new_system:
            new_system += f"\n\n[Memory context]\n{memory_text}"
        else:
            new_system = f"[Memory context]\n{memory_text}"

        _set_system_message(context, new_system)
        logger.debug(
            "VektoriMemoryProcessor: injected %d facts, %d episodes",
            len(memory.get("facts", [])),
            len(memory.get("episodes", [])),
        )

    # ------------------------------------------------------------------
    # FrameProcessor interface
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, OpenAILLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            await self._inject_memory(frame.context)

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# VektoriStorageProcessor
# ---------------------------------------------------------------------------


class VektoriStorageProcessor(FrameProcessor):
    """
    Stores each completed voice turn (user utterance + assistant reply) in Vektori.

    Place this processor **after the LLM** in your Pipecat pipeline.

    The processor buffers:

    * ``TranscriptionFrame`` — the user's transcribed speech
    * ``TextFrame`` — streamed LLM reply chunks

    When ``LLMFullResponseEndFrame`` arrives (end of the LLM generation), it
    writes the buffered turn to Vektori in the background and resets state.

    Parameters
    ----------
    vektori:
        The same ``Vektori`` instance used by ``VektoriMemoryProcessor``.
    user_id:
        Stable user identifier (matches the one passed to the memory processor).
    session_id:
        Vektori session identifier for grouping turns (e.g. one per WebSocket
        connection).
    context:
        The shared ``OpenAILLMContext`` object.  When provided, the processor
        falls back to reading the last user / assistant messages from the context
        if the frame-based buffers are empty.  Pass the same object you give to
        ``VektoriMemoryProcessor``.
    """

    def __init__(
        self,
        vektori: "Vektori",
        user_id: str,
        session_id: str,
        context: OpenAILLMContext | None = None,
    ) -> None:
        super().__init__()
        self._vektori = vektori
        self._user_id = user_id
        self._session_id = session_id
        self._context = context

        self._pending_user: str = ""
        self._pending_assistant_parts: list[str] = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._pending_user = ""
        self._pending_assistant_parts = []

    def _assistant_text(self) -> str:
        return "".join(self._pending_assistant_parts).strip()

    async def _store_turn(self) -> None:
        user_text = self._pending_user.strip()
        assistant_text = self._assistant_text()

        # Fall back to reading from context if frame buffers are empty
        if not user_text and self._context:
            user_text = _last_message(self._context, "user") or ""
        if not assistant_text and self._context:
            assistant_text = _last_message(self._context, "assistant") or ""

        if not user_text or not assistant_text:
            self._reset()
            return

        try:
            await self._vektori.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                session_id=self._session_id,
                user_id=self._user_id,
            )
            logger.debug(
                "VektoriStorageProcessor: stored turn for user=%s session=%s",
                self._user_id,
                self._session_id,
            )
        except Exception:
            logger.exception("VektoriStorageProcessor: failed to store turn — continuing")
        finally:
            self._reset()

    # ------------------------------------------------------------------
    # FrameProcessor interface
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TranscriptionFrame):
                # User finished speaking — capture transcribed text
                self._pending_user = frame.text or ""

            elif isinstance(frame, TextFrame):
                # LLM is streaming its reply — accumulate
                if frame.text:
                    self._pending_assistant_parts.append(frame.text)

            elif isinstance(frame, LLMFullResponseEndFrame):
                # LLM finished — store the completed turn asynchronously
                asyncio.ensure_future(self._store_turn())

        await self.push_frame(frame, direction)
