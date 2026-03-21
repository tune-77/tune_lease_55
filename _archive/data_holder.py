# -*- coding: utf-8 -*-
"""
メインスクリプトが JSON 等を読み込んだあと、ここに格納する。
他モジュール (scoring, web_benchmarks 等) が参照する。
"""
# メインで load_json_data 後に代入される
jsic_data = None
benchmarks_data = None
hints_data = None
jgb_rates = None
avg_data = None
knowhow_data = None
bankruptcy_data = None
subsidy_schedule_data = None
useful_life_data = None
lease_classification_data = None
LEASE_ASSETS_LIST = []
