from __future__ import annotations

from scripts import fetch_estat_industry as fetcher


def test_main_exits_nonzero_when_api_call_fails(monkeypatch, capsys):
    # ESTAT_APP_ID はあるのに e-Stat API 呼び出し自体が失敗した場合は、
    # 既存データを黙って維持するだけでなく異常終了として報告すること。
    monkeypatch.setenv("ESTAT_APP_ID", "dummy-app-id")
    monkeypatch.setattr(fetcher, "fetch_corporate_enterprise_stats", lambda app_id: None)

    exit_code = fetcher.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "警告" in captured.err


def test_main_exits_zero_when_app_id_unset_and_no_cache(monkeypatch, tmp_path):
    # ESTAT_APP_ID 未設定はドキュメント化された正常スキップ経路であり、
    # キャッシュも無ければ本当に「今回は対象なし」なので exit 0 のままであること。
    monkeypatch.delenv("ESTAT_APP_ID", raising=False)
    monkeypatch.setattr(fetcher, "CACHE_PATH", tmp_path / "industry_estat_cache.json")

    exit_code = fetcher.main()

    assert exit_code == 0
