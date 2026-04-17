"""
期待使用期間UI改善版のテスト用Streamlitアプリ
streamlit run test_usage_period_ui.py で実行
"""

import streamlit as st
import sys
import os

# パス設定
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

st.set_page_config(
    page_title="期待使用期間UI テスト",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📋 期待使用期間による最適性評価 - UI TEST")
st.markdown("#### 法定耐用年数とリース期間の管理システム")

# 説明
st.markdown("""
このテストアプリは、**法定耐用年数ベースのリース期間管理UI**の動作確認用です。

#### 🎯 テスト内容
- 法定耐用年数とリース期間の税務ルール表示
- 残り期間と再リース機会スコアの自動計算
- リスク警告の表示
""")

st.divider()

# ウィジェットをインポート＆レンダリング
try:
    from components.usage_period_widget import render_usage_period_widget
    
    # ウィジェットの実行
    result = render_usage_period_widget(key_prefix="test_upw")
    
    st.divider()
    
    # 結果の表示（デバッグ用）
    if st.checkbox("🔍 JSON結果を表示"):
        st.json(result)
except Exception as e:
    st.error(f"❌ ウィジェット読み込みエラー: {e}")
    import traceback
    st.error(traceback.format_exc())

# フッター情報
st.divider()
st.markdown("""
#### 📚 参考情報

**法定耐用年数別のリース期間下限**
| 法定耐用年数 | ルール | リース期間下限（これ以上ならOK） |
|-------------|--------|:---------------------------:|
| 4年（IT機器） | ×70% | 2.8年 |
| 5年（通信機器） | ×70% | 3.5年 |
| 6年（産業機械） | ×70% | 4.2年 |
| 15年（建築物） | ×60% | 9.0年 |
| 20年以上 | ×60% | 可変 |

**再リース機会スコア判定**
- 🟢 85-100: 残り期間4年以上（充分）
- 🟡 70-84: 残り期間2-4年（良好）
- 🟠 50-69: 残り期間1-2年（限定的）
- 🔴 30-49: 残り期間0-1年（困難）
- 🔴 0-29: 再リース不可能

**リース期間の税務チェック（法定下限ベース）**
- 🚨 リース期間 < 法定下限：短すぎる（NG）→ 実質的な購入と見なされる可能性
- ⚠️ リース期間 ≈ 法定下限：下限に接近（要注意）→ より長い期間を推奨
- ✅ リース期間 ≥ 法定下限：安全圏（OK）→ 適格リース判定の可能性高い
""")

st.caption("テスト用アプリ | lease_logic_sumaho12")
