from types import SimpleNamespace
from unittest.mock import AsyncMock

from vektori.utils.async_worker import ExtractionRequest, ExtractionWorker


def test_schedule_reports_dropped_without_running_loop():
    worker = ExtractionWorker(extractor=AsyncMock())
    result = worker.schedule(
        ExtractionRequest(
            messages=[{"role": "user", "content": "hello"}],
            session_id="sess-no-loop",
            user_id="user-1",
        )
    )

    assert result.accepted is False
    assert result.status == "dropped_no_loop"


async def test_worker_status_tracks_failures():
    extractor = SimpleNamespace(extract=AsyncMock(side_effect=RuntimeError("boom")))
    worker = ExtractionWorker(extractor=extractor, token_threshold=1, debounce_seconds=0.01)

    result = worker.schedule(
        ExtractionRequest(
            messages=[{"role": "user", "content": "this should run immediately"}],
            session_id="sess-fail",
            user_id="user-2",
        )
    )
    assert result.accepted is True
    assert result.status == "queued"

    await worker.wait_for_idle("user-2")
    status = worker.get_status("user-2")
    assert status["queued_count"] == 0
    assert status["last_failure_time"] is not None
    assert status["last_error"] == "boom"
