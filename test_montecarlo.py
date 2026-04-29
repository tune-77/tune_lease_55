from montecarlo_pricing import simulate_optimal_yield

# 実績金利がない（PD 33%）場合
res1 = simulate_optimal_yield(pd_percent=33.0, lease_term_months=60)
print("--- PD 33% (実績金利なし) ---")
print(f"推奨金利: {res1['recommended_yield']}% / 基準金利: {res1['base_yield']}% / 成約確率: {res1['success_prob']}% / ステータス: {res1['status']}")

# 実績金利がある（PD 33%, 実績金利が 4.5%）場合
res2 = simulate_optimal_yield(pd_percent=33.0, lease_term_months=60, historical_winning_rate=4.5)
print("\n--- PD 33% (実績金利 4.5%) ---")
print(f"推奨金利: {res2['recommended_yield']}% / 基準金利: {res2['base_yield']}% / 成約確率: {res2['success_prob']}% / ステータス: {res2['status']}")

# PD 5% (実績金利 3.8%) の場合
res3 = simulate_optimal_yield(pd_percent=5.0, lease_term_months=60, historical_winning_rate=3.8)
print("\n--- PD 5% (実績金利 3.8%) ---")
print(f"推奨金利: {res3['recommended_yield']}% / 基準金利: {res3['base_yield']}% / 成約確率: {res3['success_prob']}% / ステータス: {res3['status']}")
