from __future__ import annotations

import joblib

from scripts import check_model_pickle_compat as compat


def test_build_report_flags_broken_pickle(monkeypatch, tmp_path):
    monkeypatch.setattr(compat, "PROJECT_ROOT", tmp_path)
    good = tmp_path / "good.pkl"
    joblib.dump({"a": 1}, good)
    broken = tmp_path / "broken.pkl"
    broken.write_bytes(b"not a real pickle")

    report = compat.build_report([tmp_path])

    assert report["scanned_count"] == 2
    assert report["failed_count"] == 1
    assert report["status"] == "warn"


def test_main_fails_when_a_model_fails_to_load(monkeypatch, capsys, tmp_path):
    """壊れたpickleを検知しても常にexit 0では、このチェックの存在理由（docstring）が果たせない。"""
    monkeypatch.setattr(compat, "PROJECT_ROOT", tmp_path)
    broken = tmp_path / "broken.pkl"
    broken.write_bytes(b"not a real pickle")
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_model_pickle_compat.py",
            "--json", str(output_json),
            "--report", str(output_md),
            "--scan-dir", str(tmp_path),
        ],
    )

    exit_code = compat.main()

    assert exit_code == 1
    assert "読み込みに失敗したモデルファイル" in capsys.readouterr().err


def test_main_fails_when_scan_dirs_are_missing(monkeypatch, capsys, tmp_path):
    """スキャン対象ディレクトリ自体が見つからない場合も区別して検知する。"""
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_model_pickle_compat.py",
            "--json", str(output_json),
            "--report", str(output_md),
            "--scan-dir", str(tmp_path / "does-not-exist"),
        ],
    )

    exit_code = compat.main()

    assert exit_code == 1
    assert "1件もスキャンできませんでした" in capsys.readouterr().err


def test_main_is_ok_when_all_models_load_cleanly(monkeypatch, capsys, tmp_path):
    """すべて正常に読み込める（本来の良いケース）は誤検知しない。"""
    monkeypatch.setattr(compat, "PROJECT_ROOT", tmp_path)
    good = tmp_path / "good.pkl"
    joblib.dump({"a": 1}, good)
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_model_pickle_compat.py",
            "--json", str(output_json),
            "--report", str(output_md),
            "--scan-dir", str(tmp_path),
        ],
    )

    exit_code = compat.main()

    assert exit_code == 0
