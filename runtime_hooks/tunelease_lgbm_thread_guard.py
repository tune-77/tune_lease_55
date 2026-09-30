"""LightGBM のスレッド上限を全 Python プロセスで 1 に固定するガード。

torch 同梱の libiomp5 と LightGBM/sklearn の libomp が同一プロセスに同居すると、
LightGBM の並列予測で OpenMP ワーカーが別ランタイムの関数を呼んで SIGSEGV する
（PR #1192）。pickle 済みモデルは n_jobs=-1 を保持し OMP_NUM_THREADS を無視するため、
lightgbm.basic の読込直後に LGBM_SetMaxThreads(1) を呼ぶ。

.venv の site-packages に置く .pth（scripts/install_lgbm_thread_guard.py が生成）から
インタープリタ起動時に import される。lightgbm 自体は読込まず、import フックだけを登録する。
"""
import importlib.abc
import sys

_TARGET = "lightgbm.basic"


def _apply(module) -> None:
    try:
        import ctypes

        module._LIB.LGBM_SetMaxThreads(ctypes.c_int(1))
    except Exception:
        pass


class _LightGBMThreadGuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            _apply(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    if _TARGET in sys.modules:
        _apply(sys.modules[_TARGET])
        return
    if not any(isinstance(f, _LightGBMThreadGuardFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LightGBMThreadGuardFinder())


install()
