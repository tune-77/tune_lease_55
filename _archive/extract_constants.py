import re

file_path_main = "lease_logic_sumaho12.py"
file_path_const = "constants.py"

with open(file_path_main, "r", encoding="utf-8") as f:
    main_content = f.read()

with open(file_path_const, "r", encoding="utf-8") as f:
    const_content = f.read()

# LEASE_ASSETS_LIST を探す
match_assets = re.search(r'LEASE_ASSETS_LIST\s*=\s*\[(.*?)\]', main_content, re.DOTALL)
if match_assets:
    block = match_assets.group(0)
    print("Found LEASE_ASSETS_LIST")
    
    # mainから削除
    main_content = main_content.replace(block, "")
    
    # constに追加
    const_content = const_content + "\n\n# リース物件一覧\n" + block + "\n"

# QUALITATIVE_SCORING_LEVELS, QUALITATIVE_SCORING_CORRECTION_ITEMS は既に constants.py にあるため
# ここでは LEASE_ASSETS_LIST の抽出のみ

with open(file_path_main, "w", encoding="utf-8") as f:
    f.write(main_content)

with open(file_path_const, "w", encoding="utf-8") as f:
    f.write(const_content)

print("Extraction complete.")
