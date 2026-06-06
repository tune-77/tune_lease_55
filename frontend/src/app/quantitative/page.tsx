"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { LineChart, BarChart3, Zap, Info, BrainCircuit, Sigma } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

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
      .map(([name, value]: [string, number]) => ({ name, value }))
      .sort((a: any, b: any) => (absolute ? Math.abs(b.value) - Math.abs(a.value) : b.value - a.value))
      .slice(0, 15);
  };

  const shortAxisLabel = (value: unknown) => {
    const text = String(value ?? "");
    return text.length > 18 ? `${text.slice(0, 17)}...` : text;
  };

  const modelCards = [
    { label: 'Logistic Regression', key: 'lr', color: 'text-emerald-600' },
    { label: 'RandomForest', key: 'rf', color: 'text-amber-600' },
    { label: 'LGBM', key: 'lgb', color: 'text-rose-600' },
    { label: 'Ensemble', key: 'ensemble', color: 'text-indigo-600' },
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
                Gemini Integrated Comment
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
                  <div className="text-[10px] font-bold text-slate-400">Accuracy</div>
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
              LGBM 特徴量重要度 (Top 15)
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
              RandomForest 特徴量重要度 (Top 15)
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
              Logistic Regression 係数影響度 (Top 15)
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
                 <div className="text-slate-400 mb-1">Ensemble Alpha</div>
                 <div className="text-slate-800 font-bold break-all">{data?.ensemble_alpha}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">N Positive</div>
                 <div className="text-slate-800 font-bold break-all">{data?.n_positive}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">N Negative</div>
                 <div className="text-slate-800 font-bold break-all">{data?.n_negative}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">Random Seed</div>
                 <div className="text-slate-800 font-bold">42</div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}
