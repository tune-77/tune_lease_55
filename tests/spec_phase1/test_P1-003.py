"""
P1-003 テスト — index.html リースルール警告バナーUI

ブラウザ操作が不要なロジックを Python で検証する:
- escapeHtml 相当のサニタイズロジック
- severity ホワイトリスト検証ロジック
- renderWarnings のデータ変換ロジック
- HTML 構造（BeautifulSoup による静的解析）
"""
import re
import html
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

INDEX_HTML = Path(__file__).parents[2] / "mobile_app" / "index.html"


@pytest.fixture(scope="module")
def soup():
    return BeautifulSoup(INDEX_HTML.read_text(encoding="utf-8"), "html.parser")


# ─── escapeHtml 相当ロジック（Python 実装）─────────────────────────────────

def escape_html(s: str) -> str:
    """index.html の escapeHtml() と同等の変換"""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ─── severity ホワイトリスト検証ロジック ─────────────────────────────────

VALID_SEVERITIES = {"high", "medium", "low"}


def sanitize_severity(sev: str) -> str:
    return sev if sev in VALID_SEVERITIES else "low"


# ─── AC-309: XSS 対策テスト ───────────────────────────────────────────────

class TestEscapeHtml:
    def test_script_tag_escaped(self):
        result = escape_html("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_ampersand_escaped(self):
        assert escape_html("a&b") == "a&amp;b"

    def test_double_quote_escaped(self):
        assert escape_html('say "hello"') == "say &quot;hello&quot;"

    def test_greater_less_than_escaped(self):
        result = escape_html("<b>bold</b>")
        assert "&lt;b&gt;" in result
        assert "&lt;/b&gt;" in result

    def test_plain_text_unchanged(self):
        assert escape_html("法定耐用年数を超えています") == "法定耐用年数を超えています"

    def test_non_string_coerced(self):
        assert escape_html(None) == "None"
        assert escape_html(42) == "42"


# ─── AC-310 / BR-124: severity ホワイトリストテスト ─────────────────────────

class TestSeverityValidation:
    def test_high_is_valid(self):
        assert sanitize_severity("high") == "high"

    def test_medium_is_valid(self):
        assert sanitize_severity("medium") == "medium"

    def test_low_is_valid(self):
        assert sanitize_severity("low") == "low"

    def test_unknown_falls_back_to_low(self):
        assert sanitize_severity("critical") == "low"

    def test_empty_string_falls_back_to_low(self):
        assert sanitize_severity("") == "low"

    def test_injection_attempt_falls_back_to_low(self):
        assert sanitize_severity("high; background:red") == "low"

    def test_uppercase_falls_back_to_low(self):
        assert sanitize_severity("HIGH") == "low"


# ─── AC-305: asset_type select の HTML 構造テスト ───────────────────────────

class TestAssetTypeSelect:
    def test_asset_type_select_exists(self, soup):
        sel = soup.find("select", id="asset_type")
        assert sel is not None, "#asset_type select が存在しない"

    def test_asset_type_has_enough_options(self, soup):
        sel = soup.find("select", id="asset_type")
        options = sel.find_all("option")
        assert len(options) >= 10, f"オプション数が不足: {len(options)}"

    def test_asset_type_has_empty_value_option(self, soup):
        sel = soup.find("select", id="asset_type")
        values = [o.get("value", "") for o in sel.find_all("option")]
        assert "" in values, "「その他・不明」（value=''）オプションが存在しない"

    def test_asset_type_options_include_key_items(self, soup):
        sel = soup.find("select", id="asset_type")
        values = {o.get("value", "") for o in sel.find_all("option")}
        required = {"電子計算機", "工作機械", "医療機器", "自動車（普通）"}
        missing = required - values
        assert not missing, f"必須オプションが欠けている: {missing}"


# ─── AC-306: re_lease_insurance が初期状態で disabled ───────────────────────

class TestReleaseInsuranceDisabled:
    def test_re_lease_insurance_initially_disabled(self, soup):
        sel = soup.find("select", id="re_lease_insurance")
        assert sel is not None, "#re_lease_insurance select が存在しない"
        assert sel.has_attr("disabled"), "初期状態で disabled 属性がない"


# ─── BR-121 / AC-301: warnings-section が初期非表示 ─────────────────────────

class TestWarningSectionInitiallyHidden:
    def test_warnings_section_exists(self, soup):
        section = soup.find(id="warnings-section")
        assert section is not None, "#warnings-section が存在しない"

    def test_warnings_section_has_display_none(self, soup):
        section = soup.find(id="warnings-section")
        style = section.get("style", "")
        assert "display: none" in style or "display:none" in style, \
            "#warnings-section の初期スタイルが display:none でない"

    def test_warnings_list_exists(self, soup):
        lst = soup.find(id="warnings-list")
        assert lst is not None, "#warnings-list が存在しない"


# ─── アクセシビリティ: role="alert" ─────────────────────────────────────────

class TestAccessibility:
    def test_warnings_section_has_role_alert(self, soup):
        section = soup.find(id="warnings-section")
        assert section.get("role") == "alert", \
            '#warnings-section に role="alert" がない'


# ─── AC-303 / BR-126: clearAll に新規フィールドリセットが含まれる ────────────

class TestClearAllScript:
    def test_clearall_resets_asset_type(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("asset_type").value = ""' in src

    def test_clearall_resets_is_re_lease(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("is_re_lease").checked = false' in src

    def test_clearall_resets_insurance_applicable(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("insurance_applicable").value = "不明"' in src

    def test_clearall_resets_re_lease_insurance(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("re_lease_insurance").value = "不明"' in src

    def test_clearall_disables_re_lease_insurance(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("re_lease_insurance").disabled = true' in src

    def test_clearall_hides_warnings_section(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("warnings-section").style.display = "none"' in src

    def test_clearall_clears_warnings_list(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("warnings-list").innerHTML = ""' in src


# ─── AC-308: runPredict リクエストボディに新規フィールドが含まれる ────────────

class TestRunPredictFields:
    def test_request_includes_asset_type(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("asset_type").value' in src

    def test_request_includes_is_re_lease(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("is_re_lease").checked' in src

    def test_request_includes_insurance_applicable(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("insurance_applicable").value' in src

    def test_request_includes_re_lease_insurance(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert 'getElementById("re_lease_insurance").value' in src


# ─── BR-124: _VALID_SEVERITIES ホワイトリストが JS 内に定義されている ─────────

class TestValidSeveritiesInScript:
    def test_valid_severities_defined(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert "_VALID_SEVERITIES" in src

    def test_valid_severities_includes_all_three(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        match = re.search(r'_VALID_SEVERITIES\s*=\s*\[([^\]]+)\]', src)
        assert match, "_VALID_SEVERITIES の定義が見つからない"
        content = match.group(1)
        for sev in ("high", "medium", "low"):
            assert sev in content, f"_VALID_SEVERITIES に '{sev}' がない"


# ─── CSS: warning クラスが定義されている ─────────────────────────────────────

class TestWarningCssClasses:
    def test_warning_high_medium_css_exists(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert ".warning-high" in src
        assert ".warning-medium" in src

    def test_warning_low_css_exists(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        assert ".warning-low" in src

    def test_warning_low_has_blue_background(self):
        src = INDEX_HTML.read_text(encoding="utf-8")
        # warning-low セクションに #eff6ff が含まれること（AC-310）
        low_section = re.search(r'\.warning-low\s*\{([^}]+)\}', src)
        assert low_section, ".warning-low CSS ブロックが見つからない"
        assert "#eff6ff" in low_section.group(1), \
            ".warning-low の背景色 #eff6ff が設定されていない"
