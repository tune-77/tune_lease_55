"""Shared bounded executor for fire-and-forget API work."""

import threading
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


def _record_thread_failure(args: threading.ExceptHookArgs) -> None:
    # threading.Thread 直の背景処理は executor を通らないので、未処理例外はここで拾う。
    if args.exc_type is not SystemExit:
        from silent_failure_log import record_silent_failure

        # 例外の時点で Thread._target は消えているので、traceback で threading の外に出た最初の段（target 本体）を名前にする。
        tb = args.exc_traceback
        while tb is not None and tb.tb_frame.f_globals.get("__name__") == "threading":
            tb = tb.tb_next
        frame = tb.tb_frame if tb is not None else None
        module = str(frame.f_globals.get("__name__") or "unknown") if frame else "unknown"
        name = (frame.f_code.co_name if frame else "") or "thread"
        record_silent_failure(f"background.{module}.{name}", "swallowed", args.exc_value, detail="背景スレッドの未処理例外")
    _previous_thread_excepthook(args)


_previous_thread_excepthook = threading.excepthook


def install_thread_failure_hook() -> None:
    """プロセス内の全スレッドの未処理例外を silent_failures に残す（標準の stderr 出力も維持）。"""
    if threading.excepthook is not _record_thread_failure:
        threading.excepthook = _record_thread_failure
