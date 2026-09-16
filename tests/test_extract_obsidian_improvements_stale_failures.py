from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_obsidian_improvements_for_test",
        ROOT / "scripts" / "extract_obsidian_improvements.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_old_raw_failure_is_not_active_candidate() -> None:
    module = _load_module()

    assert module._is_stale_raw_failure(
        "蘭丸の機能で通信エラー障害が発生し、小説の作成ができない状態。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )
    assert module._is_stale_raw_failure(
        "リース審査AIがユーザーのぼやきを検知する機能が停止している、または機能していない。",
        "2026-06-23",
        today=dt.date(2026, 7, 15),
    )


def test_recent_or_recurring_failure_stays_active() -> None:
    module = _load_module()

    assert not module._is_stale_raw_failure(
        "蘭丸の機能で通信エラー障害が発生し、小説の作成ができない状態。",
        "2026-07-14",
        today=dt.date(2026, 7, 15),
    )
    assert not module._is_stale_raw_failure(
        "蘭丸の通信エラーがまだ再発している。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )


def test_non_failure_improvement_is_not_stale() -> None:
    module = _load_module()

    assert not module._is_stale_raw_failure(
        "ホーム画面に改善ログの要点を表示したい。",
        "2026-06-13",
        today=dt.date(2026, 7, 15),
    )


def _prepare_module_for_main(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_get_vault_path", lambda: tmp_path)
    monkeypatch.setattr(module, "OUTPUT_FILE", tmp_path / "obsidian_improvements_export.txt")
    # AI統合は外部API依存なので、テストでは常に未使用(deduped素通し)にする
    monkeypatch.setattr(module, "_load_consolidator", lambda: None)


def test_main_exits_nonzero_when_index_has_content_but_extracts_nothing(tmp_path, monkeypatch, capsys) -> None:
    # 見出し（未解決課題/Phase等）や絵文字マーカーに依存した抽出ロジックがドリフトすると、
    # インデックスに実体があっても抽出結果が静かに0件になりうる。それを検知できること。
    module = _load_module()
    _prepare_module_for_main(module, tmp_path, monkeypatch)

    index_file = tmp_path / "改善.md"
    index_file.write_text(
        "# 改善メモ\n\n"
        + ("見出し形式が変わってしまい、抽出ロジックが拾えなくなった普通の説明文。" * 30)
        + "\n",
        encoding="utf-8",
    )

    exit_code = module.main()

    assert exit_code == 1
    assert "警告" in capsys.readouterr().err


def test_main_exits_zero_when_index_is_genuinely_empty(tmp_path, monkeypatch) -> None:
    # インデックスファイル自体がほぼ空（新規/未使用）なら、抽出0件は誤検知ではなく正常。
    module = _load_module()
    _prepare_module_for_main(module, tmp_path, monkeypatch)

    index_file = tmp_path / "改善.md"
    index_file.write_text("# 改善策インデックス\n", encoding="utf-8")

    exit_code = module.main()

    assert exit_code == 0
