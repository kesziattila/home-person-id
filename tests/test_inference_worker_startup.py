"""Inference worker start-up race.

Live 2026-10-16 20:30/20:38: after restarts no tracks were created at all.
py-spy showed no InferenceWorker thread: start() launched the worker BEFORE
setting `self._running = True`; the worker's `while self._running:` loop saw
False and the thread exited silently (timing-dependent — earlier restarts
happened to win the race).

Expected: `_running` is True when the worker starts, and the main loop's
watchdog restarts a dead worker.
"""

from unittest.mock import MagicMock, patch

from src.main import PersonIDSystem


def _bare_system():
    system = PersonIDSystem.__new__(PersonIDSystem)
    system._running = False
    system._inference_thread = None
    return system


def test_start_sets_running_before_launching_worker():
    system = _bare_system()
    for name in ("api_server", "config", "global_tracker", "identity_linker", "person_detector",
                 "repository", "stream_manager", "unidentified_face_manager"):
        setattr(system, name, MagicMock())
    system._mem_profile = False
    system._debug = False
    seen = {}

    class RecordingThread:
        def __init__(self, target, name, daemon):
            pass

        def start(self):
            seen["running_at_start"] = system._running

    with patch("src.main.Thread", RecordingThread):
        system.start()

    assert seen == {"running_at_start": True}


def test_watchdog_restarts_dead_worker():
    system = _bare_system()
    system._running = True
    system._inference_thread = MagicMock(is_alive=MagicMock(return_value=False))
    system._start_inference_worker = MagicMock()

    system._ensure_inference_worker()

    system._start_inference_worker.assert_called_once()


def test_watchdog_leaves_live_worker_alone():
    system = _bare_system()
    system._running = True
    system._inference_thread = MagicMock(is_alive=MagicMock(return_value=True))
    system._start_inference_worker = MagicMock()

    system._ensure_inference_worker()

    system._start_inference_worker.assert_not_called()
