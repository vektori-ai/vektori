"""
Vektori + Pipecat — Voice Agent with Persistent Memory
=======================================================

A production-ready FastAPI WebSocket server that wires a real-time voice
pipeline with Vektori long-term memory.

Each spoken turn:
  1. Deepgram transcribes the audio stream to text.
  2. VektoriMemoryProcessor searches the user's memory and injects relevant
     facts / episodes into the LLM system prompt (before inference).
  3. GPT-4o-mini replies.
  4. VektoriStorageProcessor stores the exchange for future recall.
  5. ElevenLabs synthesises the reply to speech.

Quick-start
-----------
    pip install "vektori[pipecat]"
    pip install "pipecat-ai[deepgram,elevenlabs,openai,silero]"

    export OPENAI_API_KEY=...
    export DEEPGRAM_API_KEY=...
    export ELEVENLABS_API_KEY=...
    export ELEVENLABS_VOICE_ID=...   # e.g. 21m00Tcm4TlvDq8ikWAM (Rachel)

    uvicorn examples.pipecat_voice_agent:app --host 0.0.0.0 --port 8765

Connect from any Pipecat-compatible WebSocket client (browser, iOS, Android)
pointing at ws://localhost:8765/ws/{user_id}.

Alternative STT / TTS
----------------------
Swap ``DeepgramSTTService`` for ``WhisperSTTService`` (local) or
``AssemblyAISTTService``.

Swap ``ElevenLabsTTSService`` for ``CartesiaTTSService``,
``PlayHTTTSService``, or ``OpenAITTSService``.

Everything else — Vektori wiring, pipeline shape, session management —
stays the same.

Architecture notes
------------------
* ``VektoriMemoryProcessor`` sits between the user context aggregator and the
  LLM.  It intercepts ``OpenAILLMContextFrame``, calls ``vektori.search()``,
  and rewrites the system message in-place before the LLM sees it.

* ``VektoriStorageProcessor`` sits after the LLM.  It buffers the user
  transcription and the streamed LLM reply, then calls ``vektori.add()``
  once per completed turn (non-blocking via ``asyncio.ensure_future``).

* One ``Vektori`` instance per WebSocket connection — each connection gets its
  own async resources and is closed when the client disconnects.

* ``session_id`` encodes the connection timestamp so consecutive sessions for
  the same user are kept distinct in Vektori's L2 sentence graph.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from vektori import Vektori
from vektori.integrations.pipecat import VektoriMemoryProcessor, VektoriStorageProcessor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Vektori — default uses local SQLite + OpenAI embeddings
VEKTORI_EMBEDDING_MODEL = os.getenv("VEKTORI_EMBEDDING_MODEL", "openai:text-embedding-3-small")
VEKTORI_EXTRACTION_MODEL = os.getenv("VEKTORI_EXTRACTION_MODEL", f"openai:{OPENAI_MODEL}")

# Memory retrieval tuning for voice
# l1 = facts + episodes + source sentences — the right balance for voice
# top_k = 5 keeps the injected context short so TTS stays snappy
VEKTORI_DEPTH = os.getenv("VEKTORI_DEPTH", "l1")
VEKTORI_TOP_K = int(os.getenv("VEKTORI_TOP_K", "5"))

SYSTEM_PROMPT = """\
You are a helpful, friendly voice assistant. Keep your answers concise and
conversational — you are speaking out loud, not writing. Avoid bullet lists,
code blocks, or markdown; plain prose only.

If the memory context below contains relevant facts about the user, use them
naturally in your reply without explicitly quoting them.\
"""

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice agent server starting up")
    yield
    logger.info("Voice agent server shut down")


app = FastAPI(title="Vektori Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# WebSocket endpoint — one Pipecat pipeline per connection
# ---------------------------------------------------------------------------


@app.websocket("/ws/{user_id}")
async def voice_endpoint(websocket: WebSocket, user_id: str):
    """
    Connect a Pipecat-compatible WebSocket client.

    URL: ws://host:port/ws/{user_id}

    Each connection spawns an independent pipeline.  The ``user_id`` path
    parameter maps directly to Vektori's user namespace — all memories stored
    in this session are searchable in future sessions for the same user.
    """
    session_id = f"pipecat-{user_id}-{int(time.time())}"
    logger.info("New voice session: user=%s session=%s", user_id, session_id)

    # ------------------------------------------------------------------
    # Transport (WebSocket + VAD)
    # ------------------------------------------------------------------
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    # ------------------------------------------------------------------
    # AI services
    # ------------------------------------------------------------------
    stt = DeepgramSTTService(api_key=DEEPGRAM_API_KEY)
    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
    )

    # ------------------------------------------------------------------
    # LLM context (shared object — both processors hold a reference)
    # ------------------------------------------------------------------
    context = OpenAILLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    ctx_agg = llm.create_context_aggregator(context)

    # ------------------------------------------------------------------
    # Vektori memory
    # ------------------------------------------------------------------
    vektori = Vektori(
        embedding_model=VEKTORI_EMBEDDING_MODEL,
        extraction_model=VEKTORI_EXTRACTION_MODEL,
    )

    vektori_memory = VektoriMemoryProcessor(
        vektori=vektori,
        user_id=user_id,
        base_system_prompt=SYSTEM_PROMPT,
        depth=VEKTORI_DEPTH,
        top_k=VEKTORI_TOP_K,
        session_id=session_id,
    )

    vektori_storage = VektoriStorageProcessor(
        vektori=vektori,
        user_id=user_id,
        session_id=session_id,
        context=context,  # shared reference for fallback reads
    )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        [
            transport.input(),      # audio in (WebSocket frames → AudioRawFrame)
            stt,                    # Deepgram → TranscriptionFrame
            ctx_agg.user(),         # TranscriptionFrame → OpenAILLMContextFrame
            vektori_memory,         # inject memory into system prompt
            llm,                    # OpenAILLMContextFrame → TextFrame stream
            vektori_storage,        # buffer user + assistant text, store on end
            tts,                    # TextFrame → AudioRawFrame
            transport.output(),     # audio out (AudioRawFrame → WebSocket frames)
            ctx_agg.assistant(),    # accumulate assistant reply into context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    @transport.event_handler("on_client_connected")
    async def on_connected(t, client):
        logger.info("Client connected: user=%s", user_id)

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(t, client):
        logger.info("Client disconnected: user=%s — closing session", user_id)
        await task.queue_frames([EndFrame()])
        await vektori.close()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    runner = PipelineRunner()
    await runner.run(task)


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry-point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
