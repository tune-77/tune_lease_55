from __future__ import annotations

import sys
from types import ModuleType

from scripts import auto_trigger_optimizer as trigger


def test_main_exits_nonzero_when_dependency_import_fails(monkeypatch, capsys):
    # data_cases / auto_optimizer が壊れて import に失敗しても、これまでは
    # 「対象0件」と区別できずに無条件で exit 0 になっていた。
    monkeypatch.setitem(sys.modules, "data_cases", None)
    monkeypatch.setitem(sys.modules, "auto_optimizer", None)

    exit_code = trigger.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "インポート失敗" in captured.err


def test_main_exits_zero_when_no_old_cases_to_complete(monkeypatch):
    # 依存モジュールは正常にimportできるが、対象ケースが本当に0件のときは
    # 誤検知せず exit 0 のままであること。
    fake_data_cases = ModuleType("data_cases")
    fake_data_cases.load_all_cases = lambda: []
    fake_data_cases.update_case = lambda case_id, patch: True
    monkeypatch.setitem(sys.modules, "data_cases", fake_data_cases)

    fake_auto_optimizer = ModuleType("auto_optimizer")
    fake_auto_optimizer.run_auto_optimization = lambda: None
    monkeypatch.setitem(sys.modules, "auto_optimizer", fake_auto_optimizer)

    exit_code = trigger.main()

    assert exit_code == 0
