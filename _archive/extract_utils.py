import re

file_path_main = "lease_logic_sumaho12.py"
file_path_utils = "utils.py"

with open(file_path_main, "r", encoding="utf-8") as f:
    main_content = f.read()

# ----------------- _reset_shinsa_inputs の抽出 -----------------
# def _reset_shinsa_inputs() から始まるブロックを探す
# _reset_shinsa_inputsの終わりは def process_auto_judgment などの次の関数定義、またはわかりやすい区切りまで等。
reset_match = re.search(r'(def _reset_shinsa_inputs\(\):.*?)(?=def |st\.cache_data)', main_content, re.DOTALL)
reset_code = reset_match.group(1) if reset_match else ""

# ----------------- _slider_and_number の抽出 -----------------
slider_match = re.search(r'(def _slider_and_number\(.*?:.*?)(?=def |st\.cache_data)', main_content, re.DOTALL)
slider_code = slider_match.group(1) if slider_match else ""

if reset_code or slider_code:
    utils_content = "import streamlit as st\n\n" + reset_code + "\n\n" + slider_code
    with open(file_path_utils, "w", encoding="utf-8") as f:
        f.write(utils_content)
    
    # mainから削除
    if reset_code:
        main_content = main_content.replace(reset_code, "")
    if slider_code:
        main_content = main_content.replace(slider_code, "")
        
    # mainに import 追加
    import_stmt = "from utils import _reset_shinsa_inputs, _slider_and_number\n"
    main_content = main_content.replace("import streamlit as st\n", "import streamlit as st\n" + import_stmt, 1)

    with open(file_path_main, "w", encoding="utf-8") as f:
        f.write(main_content)
    print("utils.py 作成・抽出成功")
else:
    print("抽出対象の関数が見つかりません。")

