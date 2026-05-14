"""
P3-003 テスト — index.html ステルス競合推定パネル追加

ブラウザ操作が不要なロジックを Python で検証する:
- HTML 構造（BeautifulSoup による静的解析）
- JavaScript ロジックの静的検証（文字列・パターン検索）
- AC-901〜AC-910 全件カバー
"""
import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup, Tag

INDEX_HTML = Path(__file__).parents[2] / "mobile_app" / "index.html"

SPEC_ID = "P3-003"
PHASE = 3


@pytest.fixture(scope="module")
def soup():
    return BeautifulSoup(INDEX_HTML.read_text(encoding="utf-8"), "html.parser")


@pytest.fixture(scope="module")
def src():
    return INDEX_HTML.read_text(encoding="utf-8")


def _extract_fn(src: str, fn_name: str) -> str:
    """JS 関数の先頭から最大 4000 文字を返す（次の function 定義で切る）"""
    idx = src.find(f"function {fn_name}(")
    if idx < 0:
        return ""
    chunk = src[idx: idx + 4000]
    # 次の function 定義が現れたらそこで切る（自分自身を除く）
    next_fn = re.search(r'\bfunction \w+\(', chunk[20:])
    if next_fn:
        return chunk[: 20 + next_fn.start()]
    return chunk


# ─── AC-901: patterns が空のとき競合圧力パネルが非表示 ──────────────────────────

class TestAC901PanelHiddenWhenEmpty:
    def test_panel_exists_in_dom(self, soup):
        panel = soup.find(id="competitor-pressure-panel")
        assert panel is not None, "#competitor-pressure-panel が DOM に存在しない"

    def test_panel_initially_hidden(self, soup):
        panel = soup.find(id="competitor-pressure-panel")
        style = panel.get("style", "").replace(" ", "")
        assert "display:none" in style, \
            "#competitor-pressure-panel の初期スタイルが display:none でない"

    def test_empty_patterns_check_exists(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "patterns.length === 0" in fn or "patterns.length == 0" in fn, \
            "renderCompetitorPressure() が patterns.length === 0 チェックをしていない"

    def test_panel_hidden_when_no_cp(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert 'panel.style.display = "none"' in fn, \
            "renderCompetitorPressure() がパネルを非表示にするコードがない"


# ─── AC-902: caution 時にパネルが表示され黄色配色になる ─────────────────────────

class TestAC902CautionYellow:
    def test_caution_background_color(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "#fffbeb" in fn, \
            "caution 用背景色 #fffbeb が renderCompetitorPressure() に含まれていない"

    def test_caution_border_color(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "#f59e0b" in fn, \
            "caution 用ボーダー色 #f59e0b が renderCompetitorPressure() に含まれていない"

    def test_panel_display_block_on_patterns(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "display:block" in fn or 'display: block' in fn, \
            "renderCompetitorPressure() でパネルを display:block にするコードがない"


# ─── AC-903: high_risk 時にパネルが表示され赤色配色になる ────────────────────────

class TestAC903HighRiskRed:
    def test_high_risk_background_color(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "#fef2f2" in fn, \
            "high_risk 用背景色 #fef2f2 が renderCompetitorPressure() に含まれていない"

    def test_high_risk_border_color(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "#ef4444" in fn, \
            "high_risk 用ボーダー色 #ef4444 が renderCompetitorPressure() に含まれていない"

    def test_high_risk_key_in_color_map(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "high_risk" in fn, \
            "renderCompetitorPressure() に high_risk キーが定義されていない"


# ─── AC-904: スコア値が #competitor-pressure-score に表示される ─────────────────

class TestAC904ScoreDisplayed:
    def test_score_element_id_in_template(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "competitor-pressure-score" in fn, \
            "renderCompetitorPressure() 内に #competitor-pressure-score が含まれていない"

    def test_score_value_interpolated(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "${score}" in fn, \
            "renderCompetitorPressure() でスコア値が template literal で挿入されていない"

    def test_score_is_numeric_checked(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert 'typeof cp.score === "number"' in fn or "typeof cp.score ==" in fn, \
            "renderCompetitorPressure() でスコアの型チェックがない"


# ─── AC-905: high severity → ⚠️ アイコン + escapeHtml 適用 ─────────────────────

class TestAC905HighSeverityIconAndEscape:
    def test_high_severity_icon_present(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "⚠️" in fn, \
            "renderCompetitorPressure() に ⚠️ アイコンが含まれていない"

    def test_patterns_list_id_in_template(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "competitor-pressure-patterns" in fn, \
            "renderCompetitorPressure() に #competitor-pressure-patterns が含まれていない"

    def test_escape_html_on_message(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "escapeHtml(p.message" in fn, \
            "renderCompetitorPressure() で p.message に escapeHtml が適用されていない"

    def test_escape_html_on_code(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "escapeHtml(p.code" in fn, \
            "renderCompetitorPressure() で p.code に escapeHtml が適用されていない"

    def test_disclaimer_text_present(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "審査スコアには影響しません" in fn, \
            "renderCompetitorPressure() に免責表示「審査スコアには影響しません」がない"


# ─── AC-906: medium severity → 🔶 アイコン ──────────────────────────────────────

class TestAC906MediumSeverityIcon:
    def test_medium_icon_present(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "🔶" in fn, \
            "renderCompetitorPressure() に 🔶 アイコンが含まれていない"

    def test_severity_map_has_medium_key(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "medium" in fn, \
            "renderCompetitorPressure() の severity マップに medium キーがない"


# ─── AC-907: values が存在する場合は値が表示される ───────────────────────────────

class TestAC907ValuesDisplayed:
    def test_object_entries_used(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "Object.entries(p.values)" in fn, \
            "renderCompetitorPressure() で Object.entries(p.values) が使われていない"

    def test_values_key_escaped(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "escapeHtml(String(k))" in fn, \
            "renderCompetitorPressure() で values のキーに escapeHtml が適用されていない"

    def test_values_value_escaped(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "escapeHtml(String(v))" in fn, \
            "renderCompetitorPressure() で values の値に escapeHtml が適用されていない"


# ─── AC-908: undefined 入力でも例外が発生しない ──────────────────────────────────

class TestAC908UndefinedNoException:
    def test_null_guard_for_cp(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "!cp" in fn, \
            "renderCompetitorPressure() に cp の null/undefined ガードがない"

    def test_try_catch_present(self, src):
        fn = _extract_fn(src, "renderCompetitorPressure")
        assert "try" in fn and "catch" in fn, \
            "renderCompetitorPressure() に try/catch がない（例外非伝播が保証されていない）"


# ─── AC-909: renderQRisk() の既存動作が変化しない（回帰確認） ────────────────────

class TestAC909QRiskRegression:
    def test_q_risk_panel_still_exists(self, soup):
        panel = soup.find(id="q-risk-panel")
        assert panel is not None, "#q-risk-panel が存在しない（P2-003 回帰）"

    def test_render_q_risk_function_exists(self, src):
        assert "function renderQRisk(" in src, "renderQRisk() 関数が存在しない"

    def test_render_q_risk_called_with_aurion_q_risk(self, src):
        assert "renderQRisk(aurion?.q_risk)" in src, \
            "renderQRisk(aurion?.q_risk) の呼び出しが存在しない"

    def test_render_cp_called_after_render_q_risk(self, src):
        q_idx = src.find("renderQRisk(aurion?.q_risk)")
        cp_idx = src.find("renderCompetitorPressure(aurion?.competitor_pressure)")
        assert q_idx >= 0, "renderQRisk(aurion?.q_risk) の呼び出しが見つからない"
        assert cp_idx >= 0, "renderCompetitorPressure(aurion?.competitor_pressure) の呼び出しが見つからない"
        assert cp_idx > q_idx, \
            "renderCompetitorPressure が renderQRisk の前に呼ばれている"

    def test_optional_chaining_aurion_q_risk(self, src):
        assert "aurion?.q_risk" in src, \
            "aurion?.q_risk オプショナルチェーンが変更されている"


# ─── AC-910: ステルス競合圧力パネルは Q_risk パネルの直下に配置 ─────────────────

class TestAC910PanelDomOrder:
    def test_both_panels_in_result_card(self, soup):
        result_card = soup.find(id="result")
        assert result_card is not None, "#result カードが存在しない"
        assert result_card.find(id="q-risk-panel") is not None, \
            "#q-risk-panel が #result 内に存在しない"
        assert result_card.find(id="competitor-pressure-panel") is not None, \
            "#competitor-pressure-panel が #result 内に存在しない"

    def test_competitor_panel_after_q_risk_in_dom(self, soup):
        result_card = soup.find(id="result")
        children_ids = [el.get("id") for el in result_card.find_all(id=True)]
        assert "q-risk-panel" in children_ids, "#q-risk-panel が #result 内に見つからない"
        assert "competitor-pressure-panel" in children_ids, \
            "#competitor-pressure-panel が #result 内に見つからない"
        qi = children_ids.index("q-risk-panel")
        ci = children_ids.index("competitor-pressure-panel")
        assert qi < ci, \
            "#competitor-pressure-panel が #q-risk-panel の後に配置されていない"

    def test_competitor_panel_is_direct_next_sibling(self, soup):
        q_panel = soup.find(id="q-risk-panel")
        assert q_panel is not None, "#q-risk-panel が存在しない"
        next_tag = q_panel.find_next_sibling(True)
        assert next_tag is not None, "#q-risk-panel の次の兄弟タグが存在しない"
        assert next_tag.get("id") == "competitor-pressure-panel", \
            f"#q-risk-panel の直後のタグが '{next_tag.get('id')}' であり #competitor-pressure-panel でない"
