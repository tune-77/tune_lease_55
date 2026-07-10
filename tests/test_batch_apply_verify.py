"""batch_apply.py の適用後検証・ロールバック機能のテスト。"""

from api.rule_engine import batch_apply


def test_verify_file_accepts_valid_json(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    ok, _ = batch_apply._verify_file(path)
    assert ok


def test_verify_file_rejects_broken_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"a": 1,,}', encoding="utf-8")
    ok, msg = batch_apply._verify_file(path)
    assert not ok
    assert "JSON検証エラー" in msg


def test_verify_file_accepts_valid_python(tmp_path):
    path = tmp_path / "ok.py"
    path.write_text("x = 1\n", encoding="utf-8")
    ok, _ = batch_apply._verify_file(path)
    assert ok


def test_verify_file_rejects_broken_python(tmp_path):
    path = tmp_path / "broken.py"
    path.write_text("def f(:\n", encoding="utf-8")
    ok, msg = batch_apply._verify_file(path)
    assert not ok
    assert "Python構文エラー" in msg


def test_verify_file_skips_missing_and_other_extensions(tmp_path):
    ok, _ = batch_apply._verify_file(tmp_path / "nothing.json")
    assert ok
    other = tmp_path / "note.md"
    other.write_text("# hello", encoding="utf-8")
    ok, _ = batch_apply._verify_file(other)
    assert ok


def test_snapshot_and_restore_existing_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"before": true}', encoding="utf-8")
    snapshot = batch_apply._snapshot_paths([path])
    path.write_text("BROKEN", encoding="utf-8")
    batch_apply._restore_snapshot(snapshot)
    assert path.read_text(encoding="utf-8") == '{"before": true}'


def test_restore_deletes_file_created_during_apply(tmp_path):
    path = tmp_path / "created.json"
    snapshot = batch_apply._snapshot_paths([path])
    path.write_text("{}", encoding="utf-8")
    batch_apply._restore_snapshot(snapshot)
    assert not path.exists()


def test_candidate_paths_covers_target_and_type_specific_files():
    rule = {"rev_id": "R1", "type": "patch_json", "target": "static_data/x.json"}
    paths = batch_apply._candidate_paths(rule)
    assert any(str(p).endswith("static_data/x.json") for p in paths)

    rule_sw = {"rev_id": "R2", "type": "scoring_weight"}
    paths_sw = batch_apply._candidate_paths(rule_sw)
    assert any(str(p).endswith("api/scoring_weights.json") for p in paths_sw)

    rule_ui = {"rev_id": "R3", "type": "ui_text"}
    paths_ui = batch_apply._candidate_paths(rule_ui)
    assert any(str(p).endswith("ui_labels.json") for p in paths_ui)
