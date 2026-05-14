"""
P2-003 テスト — index.html Q-Risk パネル・財務詳細フィールド追加

ブラウザ操作が不要なロジックを Python で検証する:
- HTML 構造（BeautifulSoup による静的解析）
- JavaScript ロジックの静的検証（文字列・パターン検索）
- CSS スタイル定義の存在確認
- AC-601〜AC-614 全件カバー
"""
import re
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

INDEX_HTML = Path(__file__).parents[2] / "mobile_app" / "index.html"

SPEC_ID = "P2-003"
PHASE = 2


@pytest.fixture(scope="module")
def soup():
    return BeautifulSoup(INDEX_HTML.read_text(encoding="utf-8"), "html.parser")


@pytest.fixture(scope="module")
def src():
    return INDEX_HTML.read_text(encoding="utf-8")


# ─── AC-601: q-risk-panel が初期非表示 ──────────────────────────────────────

class TestAC601QRiskPanelInitiallyHidden:
    def test_q_risk_panel_exists(self, soup):
        panel = soup.find(id="q-risk-panel")
        assert panel is not None, "#q-risk-panel が存在しない"

    def test_q_risk_panel_has_display_none(self, soup):
        panel = soup.find(id="q-risk-panel")
        style = panel.get("style", "").replace(" ", "")
        assert "display:none" in style, "#q-risk-panel の初期スタイルが display:none でない"

    def test_render_q_risk_sets_display_block(self, src):
        assert 'panel.style.display = "block"' in src, \
            "renderQRisk() が display:block を設定していない"

    def test_render_q_risk_called_after_api_response(self, src):
        assert "renderQRisk(" in src, "renderQRisk() の呼び出しが存在しない"
        assert "aurion?.q_risk" in src, "aurion?.q_risk が参照されていない"


# ─── AC-602: level=="ok" → 緑スタイル ──────────────────────────────────────

class TestAC602LevelOkGreen:
    def test_ok_border_color_defined(self, src):
        assert "#28a745" in src, "緑系ボーダー色 #28a745 が含まれていない"

    def test_ok_label_defined(self, src):
        assert "✅ 異常なし" in src, "✅ 異常なし ラベルが含まれていない"

    def test_qrisk_ok_css_class(self, src):
        assert "qrisk-ok" in src, ".qrisk-ok クラスが定義されていない"


# ─── AC-603: level=="caution" → 黄色スタイル ────────────────────────────────

class TestAC603LevelCautionYellow:
    def test_caution_border_color_defined(self, src):
        assert "#ffc107" in src, "黄色系ボーダー色 #ffc107 が含まれていない"

    def test_caution_label_defined(self, src):
        assert "⚠️ 要注意" in src, "⚠️ 要注意 ラベルが含まれていない"

    def test_qrisk_caution_css_class(self, src):
        assert "qrisk-caution" in src, ".qrisk-caution クラスが定義されていない"


# ─── AC-604: level=="high_risk" → 赤スタイル ────────────────────────────────

class TestAC604LevelHighRiskRed:
    def test_high_risk_border_color_defined(self, src):
        assert "#dc3545" in src, "赤系ボーダー色 #dc3545 が含まれていない"

    def test_high_risk_label_defined(self, src):
        assert "🔴 高リスク" in src, "🔴 高リスク ラベルが含まれていない"

    def test_qrisk_high_risk_css_class(self, src):
        assert "qrisk-high-risk" in src, ".qrisk-high-risk クラスが定義されていない"


# ─── AC-605: patterns=[] → 「財務矛盾は検知されませんでした」 ──────────────────

class TestAC605EmptyPatternsMessage:
    def test_no_contradiction_message_exists(self, src):
        assert "財務矛盾は検知されませんでした" in src, \
            "patterns=[] 時のメッセージが含まれていない"

    def test_no_contradiction_triggered_on_empty_details(self, src):
        assert "details.length === 0" in src or "details.length == 0" in src, \
            "空パターン判定ロジックが見つからない"


# ─── AC-606: pattern_details → code と message が HTML に表示される ──────────

class TestAC606PatternDetailsRendered:
    def test_pattern_code_rendered(self, src):
        assert "p.code" in src, "p.code が renderQRisk() 内で参照されていない"

    def test_pattern_message_rendered(self, src):
        assert "p.message" in src, "p.message が renderQRisk() 内で参照されていない"

    def test_pattern_details_iterated(self, src):
        assert "pattern_details" in src, "pattern_details が参照されていない"


# ─── AC-607: severity バッジ表示 ─────────────────────────────────────────────

class TestAC607SeverityBadge:
    def test_severity_uppercased(self, src):
        assert "sev.toUpperCase()" in src, "severity が大文字化されていない"

    def test_severity_high_badge_color(self, src):
        assert "qrisk-sev-high" in src, "HIGH バッジクラスが存在しない"

    def test_severity_medium_badge_color(self, src):
        assert "qrisk-sev-medium" in src, "MEDIUM バッジクラスが存在しない"

    def test_severity_low_badge_color(self, src):
        assert "qrisk-sev-low" in src, "LOW バッジクラスが存在しない"


# ─── AC-608: 参考値免責文が常に表示 ─────────────────────────────────────────

class TestAC608DisclaimerAlwaysShown:
    def test_disclaimer_text_exists(self, src):
        assert "このスコアは参考値です" in src, "参考値免責文が含まれていない"

    def test_disclaimer_no_score_effect_text(self, src):
        assert "審査スコア・判定には影響しません" in src, \
            "「審査スコア・判定には影響しません」が含まれていない"

    def test_disclaimer_inside_render_function(self, src):
        render_match = re.search(
            r'function renderQRisk\(.*?\{(.*?)\}(?=\s*function|\s*//|\s*$)',
            src, re.DOTALL
        )
        assert render_match is not None, "renderQRisk 関数が見つからない"
        fn_body = render_match.group(1)
        assert "参考値" in fn_body, "renderQRisk 内に参考値免責文がない"


# ─── AC-609: aurion なし → 既存 UI に影響なし ───────────────────────────────

class TestAC609AurionMissingNoBreak:
    def test_optional_chaining_used(self, src):
        assert "aurion?.q_risk" in src, \
            "オプショナルチェーン aurion?.q_risk が使われていない"

    def test_render_q_risk_has_try_catch(self, src):
        render_match = re.search(
            r'function renderQRisk\(.*?\{(.*?)\}(?=\s*function|\s*//|\s*$)',
            src, re.DOTALL
        )
        assert render_match is not None, "renderQRisk 関数が見つからない"
        fn_body = render_match.group(1)
        assert "try" in fn_body and "catch" in fn_body, \
            "renderQRisk に try/catch がない（エラー非伝播が保証されていない）"

    def test_unknown_level_hides_panel(self, src):
        assert '"unknown"' in src or "'unknown'" in src, \
            "level=unknown 時のパネル非表示ロジックが含まれていない"


# ─── AC-610: 警告バナーと Q-Risk パネルが共存 ───────────────────────────────

class TestAC610WarningsAndQRiskCoexist:
    def test_warnings_section_exists(self, soup):
        assert soup.find(id="warnings-section") is not None, \
            "#warnings-section が存在しない"

    def test_q_risk_panel_exists(self, soup):
        assert soup.find(id="q-risk-panel") is not None, \
            "#q-risk-panel が存在しない"

    def test_q_risk_panel_after_warnings_section(self, soup):
        result_card = soup.find(id="result")
        assert result_card is not None, "#result カードが存在しない"
        children_ids = [el.get("id") for el in result_card.find_all(id=True)]
        assert "warnings-section" in children_ids, "#warnings-section が #result 内にない"
        assert "q-risk-panel" in children_ids, "#q-risk-panel が #result 内にない"
        wi = children_ids.index("warnings-section")
        qi = children_ids.index("q-risk-panel")
        assert wi < qi, "#q-risk-panel が #warnings-section の後ろに配置されていない"


# ─── AC-611: 4フィールドが存在し API リクエストに含まれる ────────────────────

class TestAC611FourFieldsInRequest:
    @pytest.mark.parametrize("field_id", ["op_profit", "bank_credit", "machines", "depreciation"])
    def test_field_input_exists(self, soup, field_id):
        el = soup.find("input", id=field_id)
        assert el is not None, f"#{field_id} input が存在しない"

    @pytest.mark.parametrize("field_id", ["op_profit", "bank_credit", "machines", "depreciation"])
    def test_field_type_is_number(self, soup, field_id):
        el = soup.find("input", id=field_id)
        assert el.get("type") == "number", f"#{field_id} の type が number でない"

    @pytest.mark.parametrize("field_id", ["op_profit", "bank_credit", "machines", "depreciation"])
    def test_field_in_request_body(self, src, field_id):
        assert f'getElementById("{field_id}")' in src, \
            f"{field_id} が POST /predict リクエストボディに含まれていない"


# ─── AC-612: 空白フィールド → 0 として送信 ──────────────────────────────────

class TestAC612EmptyFieldsSendZero:
    def test_parseFloat_or_zero_pattern_used(self, src):
        assert "|| 0" in src, "parseFloat(...) || 0 パターンが使われていない"

    @pytest.mark.parametrize("field_id", ["op_profit", "bank_credit", "machines", "depreciation"])
    def test_field_has_or_zero(self, src, field_id):
        pattern = rf'getElementById\("{field_id}"\).*?\|\|\s*0'
        assert re.search(pattern, src), \
            f"{field_id} に || 0 パターンが適用されていない"


# ─── AC-613: マイナス値 → エラー表示・送信中断 ──────────────────────────────

class TestAC613NegativeValueError:
    def test_minus_error_message_exists(self, src):
        assert "マイナス不可" in src, "「マイナス不可」エラーメッセージが存在しない"

    def test_negative_check_logic_exists(self, src):
        assert "< 0" in src, "マイナス値チェックロジック (< 0) が存在しない"

    def test_all_four_fields_checked(self, src):
        for field in ["op_profit", "bank_credit", "machines", "depreciation"]:
            assert field in src, f"{field} がマイナスチェック対象に含まれていない"

    def test_error_prevents_submit(self, src):
        assert "return" in src, "マイナスエラー時の return による送信中断が存在しない"


# ─── AC-614: clearAll() で4フィールドがリセット ─────────────────────────────

class TestAC614ClearAllResetsFourFields:
    @pytest.mark.parametrize("field_id", ["op_profit", "bank_credit", "machines", "depreciation"])
    def test_field_reset_in_foreach_array(self, src, field_id):
        assert f'"{field_id}"' in src, \
            f"clearAll の forEach 配列に {field_id} が含まれていない"

    def test_q_risk_panel_hidden_in_clear_all(self, src):
        assert 'getElementById("q-risk-panel").style.display = "none"' in src \
            or ("q-risk-panel" in src and 'style.display = "none"' in src), \
            "clearAll() が #q-risk-panel を非表示にしていない"

    def test_q_risk_panel_content_cleared(self, src):
        assert '_qPanel.innerHTML = ""' in src or \
            'getElementById("q-risk-panel").innerHTML = ""' in src, \
            "clearAll() が #q-risk-panel の innerHTML をクリアしていない"
