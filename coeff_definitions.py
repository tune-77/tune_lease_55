
COEFFS = {
    # --------------------------------------------------------------------------
    # 1. 全体モデル (既存先) -> 業種に関係なく全てのデータ（全業種）
    # --------------------------------------------------------------------------
    "全体_既存先": {
        "intercept": 0.36071,
        "ind_service": 1.605345,       
        "ind_medical": 0.469126879,    
        "ind_transport": -0.920492947, 
        "ind_construction": -0.56425,  
        "ind_manufacturing": -0.250500974, 
        "sales_log": -0.105997591,     
        "op_profit": -0.0310891,       
        "ord_profit": 0.0341508,       
        "net_income": -0.005823,       
        "machines": 0.0002824,         
        "other_assets": -0.001173,     
        "rent": 0.0038969,             
        "gross_profit": 0.00021184,    
        "depreciation": -0.00289,      
        "dep_expense": -0.016091,      
        "rent_expense": 0.007046,      
        "grade_4_6": 0.696099992,      
        "grade_watch": 0.2222855,      
        "grade_none": -0.822671899,    
        "bank_credit_log": -0.045431,  
        "lease_credit_log": 1.140132,  
        "contracts": 0.28131           
    },
    
    # --------------------------------------------------------------------------
    # 2. 全体モデル (新規先) -> 業種に関係なく全てのデータ（全業種）
    # --------------------------------------------------------------------------
    "全体_新規先": {
        "intercept": -0.00005,
        "ind_service": 0.000121,
        "ind_medical": 0.0000165,
        "ind_transport": -0.000106,
        "ind_construction": -0.00000953,
        "ind_manufacturing": -0.00000719,
        "sales_log": 0.001120852,
        "op_profit": -0.0075414,
        "ord_profit": 0.005226,
        "net_income": 0.0029689,
        "machines": -0.000327,
        "other_assets": -0.000304,
        "rent": -0.007116,
        "gross_profit": 0.00347583,
        "depreciation": -0.00711,
        "dep_expense": 0.004581,
        "rent_expense": 0.001533,
        "grade_4_6": -0.0000799,
        "grade_watch": 0.0000176,
        "grade_none": -0.0000155,
        "bank_credit_log": 0.0010763,
        "lease_credit_log": 0,
        "contracts": 0
    },

    # --------------------------------------------------------------------------
    # 2b. 医療・福祉モデル (既存先) -> 適用対象: P（後から事前係数を個別入力可）
    # --------------------------------------------------------------------------
    "医療_既存先": {
        "intercept": 0.36,
        "ind_service": 0,
        "ind_medical": 1.0,
        "ind_transport": 0,
        "ind_construction": 0,
        "ind_manufacturing": 0,
        "sales_log": -0.106,
        "op_profit": -0.031,
        "ord_profit": 0.034,
        "net_income": -0.006,
        "machines": 0.0003,
        "other_assets": -0.001,
        "rent": 0.004,
        "gross_profit": 0.0002,
        "depreciation": -0.003,
        "dep_expense": -0.016,
        "rent_expense": 0.007,
        "grade_4_6": 0.70,
        "grade_watch": 0.22,
        "grade_none": -0.82,
        "bank_credit_log": -0.045,
        "lease_credit_log": 1.14,
        "contracts": 0.28
    },

    # --------------------------------------------------------------------------
    # 2c. 医療・福祉モデル (新規先) -> 適用対象: P
    # --------------------------------------------------------------------------
    "医療_新規先": {
        "intercept": -0.00005,
        "ind_service": 0,
        "ind_medical": 1.0,
        "ind_transport": 0,
        "ind_construction": 0,
        "ind_manufacturing": 0,
        "sales_log": 0.001,
        "op_profit": -0.008,
        "ord_profit": 0.005,
        "net_income": 0.003,
        "machines": -0.0003,
        "other_assets": -0.0003,
        "rent": -0.007,
        "gross_profit": 0.003,
        "depreciation": -0.007,
        "dep_expense": 0.005,
        "rent_expense": 0.002,
        "grade_4_6": -0.00008,
        "grade_watch": 0.00002,
        "grade_none": -0.00002,
        "bank_credit_log": 0.001,
        "lease_credit_log": 0,
        "contracts": 0
    },

    # --------------------------------------------------------------------------
    # 3. 運送業モデル (既存) -> 適用対象: H（後から事前係数を個別入力可）
    # --------------------------------------------------------------------------
    "運送業_既存先": {
        "intercept": -0.31034,
        "sales_log": -0.050714421,
        "op_profit": -0.0580448,
        "ord_profit": 0.0479385,
        "net_income": 0.0118724,
        "machines": 0.0025427,
        "other_assets": 0.01797,
        "rent": 0.00815,
        "gross_profit": 0.001277,
        "depreciation": -0.01364,
        "dep_expense": 0.169766,
        "rent_expense": -0.02354,
        "grade_4_6": -0.4948218,
        "grade_watch": 0.255426,
        "grade_none": -0.069184496,
        "bank_credit_log": -0.177162,
        "lease_credit_log": 1.47963,
        "contracts": 0.167972
    },

    # --------------------------------------------------------------------------
    # 4. 運送業モデル (新規) -> 適用対象: H
    # --------------------------------------------------------------------------
    "運送業_新規先": {
        "intercept": 0.004699,
        "sales_log": -0.020375579,
        "op_profit": -0.0191223,
        "ord_profit": 0.0525122,
        "net_income": -0.044084,
        "machines": 0.0101606,
        "other_assets": 0.0049948,
        "rent": -0.017811,
        "gross_profit": 0.00839064,
        "depreciation": -0.03884,
        "dep_expense": 0.046943,
        "rent_expense": 0.011385,
        "grade_4_6": -0.358236225,
        "grade_watch": -0.0465,
        "grade_none": 0.33509,
        "bank_credit_log": -0.010998,
        "lease_credit_log": 0,
        "contracts": 0
    },

    # --------------------------------------------------------------------------
    # 5. サービス業モデル (既存) -> 適用対象: I, K, M, R
    # --------------------------------------------------------------------------
    "サービス業_既存先": {
        "intercept": 0.0000633,
        "sales_log": 0.002876787,
        "op_profit": -0.0737395,
        "ord_profit": -0.003886,
        "net_income": -0.006448,
        "machines": -0.028891,
        "other_assets": 0.0004489,
        "rent": -0.000268,
        "gross_profit": 0.04625454,
        "depreciation": -0.0073,
        "dep_expense": -0.001238,
        "rent_expense": 0.002962,
        "grade_4_6": -0.000033,
        "grade_watch": 0.00000636,
        "grade_none": 0.0000755,
        "bank_credit_log": 0.0022899,
        "lease_credit_log": 0.003845,
        "contracts": 0.000525
    },

    # --------------------------------------------------------------------------
    # 6. サービス業モデル (新規) -> 適用対象: I, K, M, R
    # --------------------------------------------------------------------------
    "サービス業_新規先": {
        "intercept": -0.00038,
        "sales_log": 0.033478,
        "op_profit": -0.095386,
        "ord_profit": 0.0249398,
        "net_income": 0.0125393,
        "machines": -0.019529,
        "other_assets": -0.002425,
        "rent": -0.005374,
        "gross_profit": 0.03022986,
        "depreciation": -0.00043,
        "dep_expense": -0.004116,
        "rent_expense": 0.0542,
        "grade_4_6": -0.0022357,
        "grade_watch": 0.0000799,
        "grade_none": 0.0012169,
        "bank_credit_log": 0.02405,
        "lease_credit_log": 0,
        "contracts": 0
    },

    # --------------------------------------------------------------------------
    # 7. 製造業モデル (既存) -> 適用対象: E
    # --------------------------------------------------------------------------
    "製造業_既存先": {
        "intercept": 0.013,
        "sales_log": 0.013, 
        "op_profit": -0.064,
        "ord_profit": -0.119,
        "net_income": 0.032,
        "machines": 0.050,
        "other_assets": 0.022,
        "rent": -0.005,
        "gross_profit": -0.125,
        "depreciation": 0.032,
        "dep_expense": -0.007,
        "rent_expense": 0.033,
        "grade_4_6": 0.023,
        "grade_watch": 0.006,
        "grade_none": 0,
        "bank_credit_log": 0.005,
        "lease_credit_log": 0.009,
        "contracts": 0.144
    },
    
    # --------------------------------------------------------------------------
    # 8. 製造業モデル (新規) -> 適用対象: E
    # --------------------------------------------------------------------------
    "製造業_新規先": {
        "intercept": 0.013,
        "sales_log": 0.013, 
        "op_profit": -0.062,
        "ord_profit": -0.117,
        "net_income": 0.031,
        "machines": 0.045,
        "other_assets": 0.021,
        "rent": -0.005,
        "gross_profit": -0.124,
        "depreciation": 0.035,
        "dep_expense": -0.022,
        "rent_expense": 0.034,
        "grade_4_6": 0.023,
        "grade_watch": 0.006,
        "grade_none": 0,
        "bank_credit_log": 0.005,
        "lease_credit_log": 0.009,
        "contracts": 0
    },

    # --------------------------------------------------------------------------
    # 9. 指標モデル (全体) -> 適用対象: D, P
    # --------------------------------------------------------------------------
    "全体_指標": {
        "intercept": -1.40173,
        "ind_service": 0.420609,
        "ind_medical": -0.74361,
        "ind_transport": -1.140695738,
        "ind_construction": -0.07587544,
        "ind_manufacturing": 0.336239736,
        "ratio_op_margin": -4.702301046,     
        "ratio_gross_margin": 5.70061655,    
        "ratio_ord_margin": 2.2610414,       
        "ratio_net_margin": 3.171582,        
        "ratio_fixed_assets": 0.5527767,     
        "ratio_rent": 1.4039157,             
        "ratio_depreciation": 4.2631677,     
        "ratio_machines": -2.9214377,        
        "grade_4_6": 0.836596,
        "grade_watch": 1.5674,
        "grade_none": 0.778885
    },

    # --------------------------------------------------------------------------
    # 9b. 指標モデル (医療・福祉) -> 適用対象: P（後から事前係数を個別入力可）
    # --------------------------------------------------------------------------
    "医療_指標": {
        "intercept": -1.4,
        "ind_service": 0,
        "ind_medical": 1.0,
        "ind_transport": 0,
        "ind_construction": 0,
        "ind_manufacturing": 0,
        "ratio_op_margin": -4.7,
        "ratio_gross_margin": 5.7,
        "ratio_ord_margin": 2.26,
        "ratio_net_margin": 3.17,
        "ratio_fixed_assets": 0.55,
        "ratio_rent": 1.4,
        "ratio_depreciation": 4.26,
        "ratio_machines": -2.92,
        "grade_4_6": 0.84,
        "grade_watch": 1.57,
        "grade_none": 0.78
    },

    # --------------------------------------------------------------------------
    # 9c. 指標モデル (運送業) -> 適用対象: H（後から事前係数を個別入力可）
    # --------------------------------------------------------------------------
    "運送業_指標": {
        "intercept": -1.2,
        "ind_service": 0,
        "ind_medical": 0,
        "ind_transport": 1.0,
        "ind_construction": 0,
        "ind_manufacturing": 0,
        "ratio_op_margin": -4.0,
        "ratio_gross_margin": 5.0,
        "ratio_ord_margin": 2.0,
        "ratio_net_margin": 3.0,
        "ratio_fixed_assets": 0.5,
        "ratio_rent": 1.4,
        "ratio_depreciation": 4.0,
        "ratio_machines": -2.5,
        "grade_4_6": 0.8,
        "grade_watch": 1.5,
        "grade_none": 0.7
    },

    # --------------------------------------------------------------------------
    # 10. 指標モデル (サービス業) -> 適用対象: I, K, M, R
    # --------------------------------------------------------------------------
    "サービス業_指標": {
        "intercept": 0.00201,
        "ind_service": 0.00201, 
        "ratio_op_margin": -0.245264,
        "ratio_gross_margin": 0.079622,
        "ratio_ord_margin": -0.018412,
        "ratio_net_margin": 0.0796224,
        "ratio_fixed_assets": -0.008077,
        "ratio_rent": -0.029334,
        "ratio_depreciation": 0.03202,
        "ratio_machines": -0.0098938,
        "grade_4_6": 0.000224,
        "grade_watch": -0.000296,
        "grade_none": 0.00212
    },

    # --------------------------------------------------------------------------
    # 11. 指標モデル (製造業) -> 適用対象: E
    # --------------------------------------------------------------------------
    "製造業_指標": {
        "intercept": -0.01,
        "ratio_op_margin": -0.01,
        "ratio_gross_margin": -0.960,
        "ratio_ord_margin": 0.060,
        "ratio_net_margin": 0.952,
        "ratio_fixed_assets": 0.082,
        "ratio_rent": -0.150,
        "ratio_depreciation": -0.094,
        "ratio_machines": -0.026,
        "grade_4_6": 0.085,
        "grade_watch": 0,
        "grade_none": 0
    }
}

# --------------------------------------------------------------------------
# 完全版ベイズ初期モデル: 不足項目の補完係数（AI知見に基づく初期値）
# 継承: 上記COEFFSはそのまま「事前分布の核」。以下は追加補正（%ポイント換算）。
# 標準化: z_scaled に対して係数をかけた値を%で加算する想定。
# --------------------------------------------------------------------------
BAYESIAN_PRIOR_EXTRA = {
    # 競合他社の存在: いる=1 → 成約率を下げる（負の係数）
    "competitor_present": -5.0,   # 1あたり -5%pt
    # 金利差（自社が競合より低い%pt）: 標準化 z = rate_diff_pt / 5 のときの係数
    "rate_diff_per_z": 2.5,      # z=1 (5%pt有利) → +2.5%pt
    # 業界景気動向: ポジティブ=1, ネガティブ=-1, 不明=0 の z に対する係数
    "industry_sentiment_per_z": 3.0,  # ポジティブで+3%pt
    # 定性スコア: 強みタグ数（0〜8を4で割ってz）の係数 + 熱意テキスト有無
    "qualitative_tag_per_z": 2.0,     # タグ1つあたり約+2%pt（z=0.25）
    "qualitative_passion_bonus": 2.5,  # 熱意記述ありで+2.5%pt（旧5.0、定性上乗せを抑える）
}

# 定性タグの標準的な加点重み（強みタグ1つあたりの%ポイント寄与目安）
STRENGTH_TAG_WEIGHTS = {
    "技術力": 2.0,
    "業界人脈": 2.0,
    "特許": 2.5,
    "立地": 1.5,
    "後継者あり": 2.0,
    "関係者資産あり": 2.0,
    "取引行と付き合い長い": 2.0,
    "既存返済懸念ない": 2.0,
}
DEFAULT_STRENGTH_WEIGHT = 2.0
