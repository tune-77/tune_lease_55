"""Shared bounded executor for fire-and-forget API work."""

from concurrent.futures import Future, ThreadPoolExecutor


def _record_background_failure(fn, future: Future) -> None:
    # 投げっぱなしの Future は誰も result() を呼ばないので、例外がどこにも出ずに消えていた。
    if future.cancelled() or future.exception() is None:
        return
    from silent_failure_log import record_silent_failure

    module = str(getattr(fn, "__module__", "") or "unknown")
    name = str(getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "") or "callable").replace("<", "").replace(">", "")
    record_silent_failure(f"background.{module}.{name}", "swallowed", future.exception(), detail="背景処理の例外")


class _RecordingExecutor(ThreadPoolExecutor):
    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(lambda done, fn=fn: _record_background_failure(fn, done))
        return future


# Keep one bounded pool across routers. Creating a thread per request can exhaust
# native-library resources under load (notably OpenMP/MPS users in this process).
background_executor = _RecordingExecutor(max_workers=8, thread_name_prefix="bg-task")
