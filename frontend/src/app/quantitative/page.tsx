"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { LineChart, BarChart3, Zap, Info, BrainCircuit, Sigma } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

const FEATURE_LABELS: Record<string, string> = {
  intercept: "定数項",
  ind_medical: "業種: 医療・福祉",
  ind_transport: "業種: 運輸",
  ind_construction: "業種: 建設",
  ind_manufacturing: "業種: 製造",
  ind_service: "業種: サービス",
  sales_log: "売上高(対数)",
  bank_credit_log: "銀行与信(対数)",
  lease_credit_log: "リース与信(対数)",
  op_profit: "営業利益",
  ord_profit: "経常利益",
  net_income: "当期純利益",
  machines: "機械装置",
  other_assets: "その他資産",
  rent: "賃借料",
  gross_profit: "売上総利益",
  depreciation: "減価償却",
  dep_expense: "減価償却費",
  rent_expense: "賃借料等",
  grade_4_6: "格付4〜6",
  grade_watch: "要注意",
  grade_none: "無格付",
  contracts: "契約数",
  main_bank: "メイン取引先",
  competitor_present: "競合あり",
  competitor_none: "競合なし",
  rate_diff_z: "金利差(有利)",
  industry_sentiment_z: "業界景気動向",
  qualitative_tag_score: "定性スコア(強みタグ)",
  qualitative_passion: "熱意・裏事情",
  equity_ratio: "自己資本比率",
  qualitative_combined: "定性スコアリング合計(0-1)",
  bn_approval_prob: "BN承認確率",
  bn_fc: "BN財務信用度",
  bn_hc: "BNヘッジ条件",
  bn_av: "BN物件価値",
  qual_weighted: "定性加重スコア(0-1)",
  qual_rank_good: "定性優良ランク(A/B)",
  qual_repayment: "返済履歴スコア(0-1)",
  quantum_risk: "量子矛盾リスク",
  new_customer_main_bank: "新規先×メイン先",
  new_customer_competitor_present: "新規先×競合あり",
  new_customer_competitor_count: "新規先×競合社数",
  new_customer_competitor_rate: "新規先×競合提示金利",
  new_customer_deal_source_bank: "新規先×銀行紹介",
  new_customer_deal_occurrence_nomination: "新規先×指名案件",
  new_customer_deal_occurrence_comp: "新規先×相見積もり",
  new_customer_contract_auto: "新規先×自動車契約",
  customer_new: "新規先フラグ",
  deal_source_bank: "銀行紹介",
  dscr_approx: "返済余力(DSCR近似)",
  interest_coverage: "支払利息カバー率",
  ratio_op_margin: "営業利益率",
  ratio_gross_margin: "売上総利益率",
  ratio_ord_margin: "経常利益率",
  ratio_net_margin: "当期純利益率",
  ratio_fixed_assets: "固定資産比率",
  ratio_rent: "賃借料比率",
  ratio_depreciation: "減価償却比率",
  ratio_machines: "機械装置比率",
};

export default function QuantitativePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '定量的な成約要因の分析画面です。\\nロジスティック回帰・RandomForest・LGBMを組み合わせて、成約に効いている項目を確認します。');
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`/api/analysis/quantitative`);
      setData(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '分析データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const toPercent = (value?: number) => (
    typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '-'
  );

  const toScore = (value?: number) => (
    typeof value === 'number' ? value.toFixed(3) : '-'
  );

  const getImportanceData = (key: 'lgb_importance' | 'rf_importance' | 'lr_coef', absolute = false) => {
    if (!data?.[key]) return [];
    return data[key]
      .map(([name, value]: [string, number]) => ({ name: FEATURE_LABELS[name] ?? name, value }))
      .sort((a: any, b: any) => (absolute ? Math.abs(b.value) - Math.abs(a.value) : b.value - a.value))
      .slice(0, 15);
  };

  const shortAxisLabel = (value: unknown) => {
    const text = String(value ?? "");
    return text.length > 18 ? `${text.slice(0, 17)}...` : text;
  };

  const modelCards = [
    { label: 'ロジスティック回帰', key: 'lr', color: 'text-emerald-600' },
    { label: 'ランダムフォレスト', key: 'rf', color: 'text-amber-600' },
    { label: 'LGBM', key: 'lgb', color: 'text-rose-600' },
    { label: 'アンサンブル', key: 'ensemble', color: 'text-indigo-600' },
  ];

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center gap-4">
        <Zap className="w-12 h-12 text-rose-500 animate-pulse" />
        <p className="text-slate-500 font-bold">複数モデルで定量要因を分析中...</p>
      </div>
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
            <LineChart className="w-8 h-8 text-rose-500" />
            定量要因・ML分析
          </h1>
          <p className="text-slate-500 font-bold mt-2">
            全 {data?.n_cases} 件のデータから、ロジスティック回帰・RandomForest・LGBMで「成約の決め手」を複合分析。
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8">
        <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
          <div className="flex items-start gap-4">
            <BrainCircuit className="w-6 h-6 text-indigo-500 mt-1" />
            <div>
              <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">
                Gemini統合コメント
              </div>
              <p className="text-slate-700 font-bold whitespace-pre-line leading-relaxed">
                {data?.gemini_comment?.text || 'Gemini所見を取得中です。'}
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {modelCards.map((model) => (
            <div key={model.key} className="bg-white border border-slate-200 px-6 py-4 rounded-2xl shadow-sm">
              <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{model.label}</div>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <div>
                  <div className="text-[10px] font-bold text-slate-400">正解率</div>
                  <div className={`text-2xl font-black ${model.color}`}>{toPercent(data?.[`accuracy_${model.key}`])}</div>
                </div>
                <div>
                  <div className="text-[10px] font-bold text-slate-400">AUC</div>
                  <div className="text-2xl font-black text-slate-800">{toScore(data?.[`auc_${model.key}`])}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {data?.best_auc_model && (
          <div className="bg-emerald-50 border border-emerald-200 p-4 rounded-2xl text-sm font-bold text-emerald-800">
            現在のベストAUCは {data.best_auc_model} です。AUC {typeof data?.best_auc_value === 'number' ? data.best_auc_value.toFixed(3) : '-'}。
            {data.best_auc_model === 'Ensemble' && typeof data?.ensemble_alpha === 'number' ? ` LR と LGB の比率は ${(data.ensemble_alpha * 100).toFixed(0)}% / ${(100 - data.ensemble_alpha * 100).toFixed(0)}% です。` : ''}
          </div>
        )}

        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-xl font-black text-slate-700 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-rose-500" />
              LGBM 特徴量重要度 Top 15
            </h3>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
              <Info className="w-4 h-4" />
              値が大きいほど、AIが成約判断に利用した度合いが高いことを示します
            </div>
          </div>

          <div className="h-[500px] overflow-hidden">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={getImportanceData('lgb_importance')} layout="vertical">
                 <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                 <XAxis type="number" />
                 <YAxis dataKey="name" type="category" width={170} interval={0} tickFormatter={shortAxisLabel} tick={{fontSize: 10, fontWeight: 'bold'}} />
                 <Tooltip
                   cursor={{fill: 'rgba(244, 63, 94, 0.05)'}}
                   contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}
                 />
                 <Bar dataKey="value" fill="#f43f5e" radius={[0, 4, 4, 0]}>
                   {getImportanceData('lgb_importance').map((entry: any, index: number) => (
                     <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.05} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
            <h3 className="text-xl font-black text-slate-700 flex items-center gap-2 mb-8">
              <BarChart3 className="w-5 h-5 text-amber-500" />
              ランダムフォレスト 特徴量重要度 Top 15
            </h3>
            <div className="h-[420px] overflow-hidden">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getImportanceData('rf_importance')} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={170} interval={0} tickFormatter={shortAxisLabel} tick={{fontSize: 10, fontWeight: 'bold'}} />
                  <Tooltip cursor={{fill: 'rgba(245, 158, 11, 0.06)'}} />
                  <Bar dataKey="value" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
            <h3 className="text-xl font-black text-slate-700 flex items-center gap-2 mb-8">
              <Sigma className="w-5 h-5 text-emerald-500" />
              ロジスティック回帰 係数影響度 Top 15
            </h3>
            <div className="h-[420px] overflow-hidden">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getImportanceData('lr_coef', true)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={170} interval={0} tickFormatter={shortAxisLabel} tick={{fontSize: 10, fontWeight: 'bold'}} />
                  <Tooltip cursor={{fill: 'rgba(16, 185, 129, 0.06)'}} />
                  <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="bg-slate-50 border border-slate-200 p-8 rounded-2xl mt-4">
           <h4 className="text-sm font-black text-slate-500 uppercase tracking-widest mb-4">AIモデルの構成とハイパーパラメータ</h4>
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">アンサンブル比率(α)</div>
                 <div className="text-slate-800 font-bold break-all">{data?.ensemble_alpha}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">成約件数</div>
                 <div className="text-slate-800 font-bold break-all">{data?.n_positive}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">失注件数</div>
                 <div className="text-slate-800 font-bold break-all">{data?.n_negative}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">乱数シード</div>
                 <div className="text-slate-800 font-bold">42</div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}
