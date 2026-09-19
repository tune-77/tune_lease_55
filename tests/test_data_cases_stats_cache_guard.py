"""統計キャッシュの「接続先ごとのパス分離」と「退化更新の拒否」の回帰テスト。

背景: DATABASE_URL 設定時、読みは Cloud SQL なのに書き先がローカル _DATA_DIR
固定だったため、クラウド側の数件だけを見た集計がローカルキャッシュを
closed_count=None で上書きし固着した（2026-09-19 実障害）。
"""

import json
import os

import pytest

import data_cases


# --- パス分離 -----------------------------------------------------------


def test_local_context_uses_base_path(monkeypatch):
    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: False)
    assert data_cases._stats_cache_path("/tmp/dashboard_stats_cache.json") == (
        "/tmp/dashboard_stats_cache.json"
    )


def test_cloud_context_uses_separate_file(monkeypatch):
    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: True)
    assert data_cases._stats_cache_path("/tmp/dashboard_stats_cache.json") == (
        "/tmp/dashboard_stats_cache.cloud.json"
    )


def test_cloud_cache_does_not_leak_into_local_load(monkeypatch, tmp_path):
    """クラウド文脈のキャッシュがあってもローカル文脈からは見えない。"""
    local = tmp_path / "dashboard_stats_cache.json"
    local.write_text(json.dumps({"analysis": {"closed_count": 1174}}), encoding="utf-8")
    monkeypatch.setattr(data_cases, "DASHBOARD_STATS_CACHE_FILE", str(local))

    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: False)
    assert data_cases.load_dashboard_stats_cache()["analysis"]["closed_count"] == 1174

    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: True)
    assert data_cases.load_dashboard_stats_cache() is None


# --- 件数の取り出し -----------------------------------------------------


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"analysis": {"closed_count": 1174}}, 1174),
        ({"analysis": {"closed_count": None}}, None),
        ({"analysis": {}}, None),
        ({}, None),
        (None, None),
        ({"analysis": {"closed_count": "1174"}}, None),
        ({"analysis": {"closed_count": True}}, None),  # bool は int だが件数ではない
    ],
)
def test_extract_stats_count(payload, expected):
    assert data_cases._extract_stats_count(payload, ("analysis", "closed_count")) == expected


# --- 退化判定 -----------------------------------------------------------


@pytest.mark.parametrize(
    "previous,new,expected,why",
    [
        (None, None, False, "初回起動: 拒否するとキャッシュが永久に作られない"),
        (None, 1174, False, "初回起動の正常な書き込み"),
        (0, None, False, "守るべき値が無い"),
        (1174, None, True, "実障害の経路: 集計不能で上書き"),
        (2157, 0, True, "全件消失"),
        (1174, 1174, False, "変化なし"),
        (1174, 1180, False, "増加"),
        (1174, 1100, False, "通常の微減"),
        (1174, 587, True, "ちょうど半減は崩壊扱い"),
        (1174, 588, False, "半減をわずかに上回れば通す"),
        (4, 1, False, "母数が小さい時は比率判定しない"),
        (4, None, True, "母数が小さくても集計不能は拒否"),
        (10, 5, True, "しきい値ちょうどの母数では比率判定が効く"),
    ],
)
def test_is_degenerate_stats_update(monkeypatch, previous, new, expected, why):
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    assert data_cases._is_degenerate_stats_update(previous, new) is expected, why


def test_escape_hatch_allows_legitimate_mass_deletion(monkeypatch):
    """正当な一括削除は環境変数で明示的に通す。"""
    monkeypatch.setenv("STATS_CACHE_ALLOW_SHRINK", "1")
    assert data_cases._is_degenerate_stats_update(1174, None) is False
    assert data_cases._is_degenerate_stats_update(2157, 0) is False


def test_operation_scoped_bypass_allows_legitimate_mass_deletion(monkeypatch):
    """確定済み削除処理は環境全体を緩めず、その更新だけ縮小を許可できる。"""
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)

    assert (
        data_cases._is_degenerate_stats_update(2157, 0, allow_shrink=True)
        is False
    )
    assert (
        data_cases._is_degenerate_stats_update(2157, None, allow_shrink=True)
        is True
    )


def test_refresh_stats_caches_forwards_operation_scoped_bypass(monkeypatch):
    dashboard_calls = []
    department_calls = []
    monkeypatch.setattr(
        data_cases,
        "refresh_dashboard_stats_cache",
        lambda **kwargs: dashboard_calls.append(kwargs),
    )
    monkeypatch.setattr(
        data_cases,
        "refresh_department_stats_cache",
        lambda **kwargs: department_calls.append(kwargs),
    )

    data_cases.refresh_stats_caches(allow_shrink=True)

    assert dashboard_calls == [{"allow_shrink": True}]
    assert department_calls == [{"allow_shrink": True}]


# --- 門番の通し動作 -----------------------------------------------------


def _write(path, count):
    path.write_text(json.dumps({"analysis": {"closed_count": count}}), encoding="utf-8")


def test_reject_degenerate_write_blocks_collapse(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    path = tmp_path / "dashboard_stats_cache.json"
    _write(path, 1174)

    rejected = data_cases._reject_degenerate_write(
        str(path), {"analysis": {"closed_count": None}}, ("analysis", "closed_count"), "dashboard"
    )

    assert rejected is True
    assert "退化更新を拒否" in capsys.readouterr().err
    # 既存キャッシュは温存されている
    assert json.loads(path.read_text(encoding="utf-8"))["analysis"]["closed_count"] == 1174


def test_reject_degenerate_write_allows_normal_update(monkeypatch, tmp_path):
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    path = tmp_path / "dashboard_stats_cache.json"
    _write(path, 1174)

    assert (
        data_cases._reject_degenerate_write(
            str(path),
            {"analysis": {"closed_count": 1180}},
            ("analysis", "closed_count"),
            "dashboard",
        )
        is False
    )


def test_reject_degenerate_write_allows_first_write(monkeypatch, tmp_path):
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    path = tmp_path / "does_not_exist.json"

    assert (
        data_cases._reject_degenerate_write(
            str(path),
            {"analysis": {"closed_count": 1174}},
            ("analysis", "closed_count"),
            "dashboard",
        )
        is False
    )


def test_corrupted_existing_cache_does_not_block_repair(monkeypatch, tmp_path):
    """既存が壊れたJSONなら previous_count=None 扱いで、修復の書き込みを妨げない。"""
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    path = tmp_path / "dashboard_stats_cache.json"
    path.write_text("{ broken", encoding="utf-8")

    assert (
        data_cases._reject_degenerate_write(
            str(path),
            {"analysis": {"closed_count": 1174}},
            ("analysis", "closed_count"),
            "dashboard",
        )
        is False
    )


def test_department_count_path_is_honored(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("STATS_CACHE_ALLOW_SHRINK", raising=False)
    path = tmp_path / "department_stats_cache.json"
    path.write_text(json.dumps({"overall": {"total_count": 2157}}), encoding="utf-8")

    rejected = data_cases._reject_degenerate_write(
        str(path), {"overall": {"total_count": 0}}, ("overall", "total_count"), "department"
    )

    assert rejected is True
    assert "department" in capsys.readouterr().err


# --- 取得不能 vs 確定的に0件 ---------------------------------------------


class _BoomConnection:
    def __enter__(self):
        raise RuntimeError("could not connect to server")

    def __exit__(self, *exc):
        return False


def _make_db_unreadable(monkeypatch):
    """DBファイルは存在扱いにしつつ、接続で必ず失敗させる。"""
    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: True)
    monkeypatch.setattr(data_cases, "_case_db_connection", lambda: _BoomConnection())


def test_non_strict_load_keeps_swallowing_for_compatibility(monkeypatch, capsys):
    """既定は従来通り。find_similar_past_cases 等の5経路は無変更で動く。"""
    _make_db_unreadable(monkeypatch)

    assert data_cases.load_all_cases() == []
    assert "[Error in load_all_cases]" in capsys.readouterr().err


def test_strict_load_raises_when_db_unreadable(monkeypatch):
    _make_db_unreadable(monkeypatch)

    with pytest.raises(data_cases.CaseDataUnavailable):
        data_cases.load_all_cases(strict=True)


def test_strict_load_returns_empty_for_missing_db(monkeypatch, tmp_path):
    """DBファイル不在は「確定的に0件」。新規環境で例外にしてはいけない。"""
    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: False)
    monkeypatch.setattr(data_cases, "DB_PATH", str(tmp_path / "does_not_exist.db"))

    assert data_cases.load_all_cases(strict=True) == []


def test_write_preserves_cache_when_db_unreadable(monkeypatch, tmp_path, capsys):
    """DBが読めない時に 0件集計を焼き付けず、既存キャッシュを返す。"""
    cache = tmp_path / "dashboard_stats_cache.json"
    _write(cache, 1174)
    monkeypatch.setattr(data_cases, "DASHBOARD_STATS_CACHE_FILE", str(cache))
    monkeypatch.setattr(data_cases, "_cloud_db_enabled", lambda: False)

    def _boom(*_a, **_kw):
        raise data_cases.CaseDataUnavailable("案件DBの読み込みに失敗しました")

    monkeypatch.setattr(data_cases, "build_dashboard_stats_cache", _boom)

    result = data_cases._write_dashboard_stats_cache_locked()

    assert result["analysis"]["closed_count"] == 1174
    assert "更新を見送り" in capsys.readouterr().err
    assert json.loads(cache.read_text(encoding="utf-8"))["analysis"]["closed_count"] == 1174


def test_guardrail_reports_unavailable_instead_of_crashing(monkeypatch):
    from system_guardrails import _audit_stats_cache_consistency

    monkeypatch.setattr(
        data_cases, "load_dashboard_stats_cache", lambda: {"analysis": {"closed_count": 1174}}
    )

    def _boom():
        raise data_cases.CaseDataUnavailable("案件DBの読み込みに失敗しました")

    monkeypatch.setattr(data_cases, "build_dashboard_stats_cache", _boom)
    monkeypatch.setattr(data_cases, "load_department_stats_cache", lambda: None)

    issues = _audit_stats_cache_consistency()

    assert len(issues) == 1
    assert issues[0]["kind"] == "stats_cache_unavailable"
    assert issues[0]["severity"] == "warn"
