"""Shared bounded executor for fire-and-forget API work."""

from concurrent.futures import ThreadPoolExecutor


# Keep one bounded pool across routers. Creating a thread per request can exhaust
# native-library resources under load (notably OpenMP/MPS users in this process).
background_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bg-task")
