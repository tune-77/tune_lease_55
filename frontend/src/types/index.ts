export interface ScoringFormData {
  // 属性情報
  company_no: string;    // 企業番号
  company_name: string;  // 企業名
  industry_major: string;
  industry_sub: string;
  industry_detail: string; // 詳細キーワード
  grade: string;
  customer_type: string;
  main_bank: string;
  competitor: string;
  competitor_rate?: number; // 競合提示金利
  num_competitors: string; // 競合社数
  deal_occurrence: string; // 発生経緯
  deal_source: string;
  
  // P/L (損益計算書)
  nenshu: number;
  gross_profit: number; // item9_gross
  op_profit: number;    // rieki
  ord_profit: number;   // item4_ord_profit
  net_income: number;   // item5_net_income
  depreciation: number; // item10_dep
  dep_expense: number;  // item11_dep_exp
  rent: number;         // item8_rent
  rent_expense: number; // item12_rent_exp

  // B/S (貸借対照表)
  machines: number;     // item6_machine
  other_assets: number; // item7_other
  net_assets: number;
  total_assets: number;
  
  // 審査・借入状況
  bank_credit: number;
  lease_credit: number;
  contracts: number;
  contract_type: string;
  
  // 物件情報
  lease_term: number;
  acceptance_year: number;
  acquisition_cost: number;
  selected_asset_id: string;
  asset_name: string;
  asset_score: number;
  
  // 定性評価 (6大項目)
  qual_corr_company_history: string;
  qual_corr_customer_stability: string;
  qual_corr_repayment_history: string;
  qual_corr_business_future: string;
  qual_corr_equipment_purpose: string;
  qual_corr_main_bank: string;

  // 定性評価フリーテキスト
  passion_text: string;
  strength_tags: string[];

  // 直感スコア
  intuition: number;
}

export const defaultFormData: ScoringFormData = {
  company_no: "",
  company_name: "",
  industry_major: "D 建設業",
  industry_sub: "06 総合工事業",
  industry_detail: "",
  grade: "②4-6 (標準)",
  customer_type: "既存先",
  main_bank: "メイン先",
  competitor: "競合なし",
  competitor_rate: undefined,
  num_competitors: "未入力",
  deal_occurrence: "不明",
  deal_source: "銀行紹介",
  nenshu: 200000,
  gross_profit: 50000,
  op_profit: 15000,
  ord_profit: 14000,
  net_income: 10000,
  depreciation: 10000,
  dep_expense: 10000,
  rent: 0,
  rent_expense: 0,
  machines: 5000,
  other_assets: 2000,
  net_assets: 50000,
  total_assets: 150000,
  bank_credit: 10000,
  lease_credit: 5000,
  contracts: 2,
  contract_type: "一般",
  lease_term: 60,
  acceptance_year: new Date().getFullYear(),
  acquisition_cost: 3000,
  selected_asset_id: "",
  asset_name: "",
  asset_score: 50.0,
  
  qual_corr_company_history: "未選択",
  qual_corr_customer_stability: "未選択",
  qual_corr_repayment_history: "未選択",
  qual_corr_business_future: "未選択",
  qual_corr_equipment_purpose: "未選択",
  qual_corr_main_bank: "未選択",

  passion_text: "",
  strength_tags: [],
  intuition: 3
};
