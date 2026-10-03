"""scripts/ 直下のモジュールがリポジトリ直下のモジュール名を隠さないこと。

API は実行中に scripts/ を sys.path 先頭へ足すため、同名ファイルがあると `import X` が
scripts 側に化ける（2026-10 に jev_safe_gateway で循環 import になり Jev shadow が全滅した）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scripts_do_not_shadow_root_modules():
    root_modules = {p.stem for p in ROOT.glob("*.py")}
    clashes = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if p.stem in root_modules)
    assert clashes == []


def test_chat_teaching_mask_imports_with_scripts_first_on_path(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "path", [str(ROOT / "scripts"), *sys.path])
    monkeypatch.delitem(sys.modules, "jev_safe_gateway", raising=False)
    from api.chat_judgment_asset_capture import mask_for_jev

    assert mask_for_jev("株式会社テスト は 運送業の リース では 車齢を 必ず 確認する").startswith("〈企業〉")
