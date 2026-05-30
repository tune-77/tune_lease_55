import React, { useState, useEffect } from 'react';
import { BarChart2, Activity, Network, Box } from 'lucide-react';

interface NetworkRiskResult {
  network_risk_pct: number;
  base_risk_pct: number;
  sim_mean_pct: number;
  sim_var95_pct: number;
  target_industry: string;
  impacted_by: { from: string; impact: number }[];
  error?: string;
}

interface McResult {
  gbm_paths: number[][];
  gbm_median: number[];
  revenues: number[];
  timesfm_available: boolean;
}

interface TfmResult {
  months_history?: string[];
  avg_score_hist?: number[];
  months_forecast?: string[];
  avg_score_fore?: number[];
  risk_signal?: string;
  method?: string;
  error?: string;
}

interface Props {
  industrySub?: string;
  companyName?: string;
  score?: number;
}

function normalizePaths(paths: number[][], h: number): string[] {
  if (!paths.length || !paths[0].length) return [];
  const all = paths.flat();
  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const n = paths[0].length;
  return paths.map(p =>
    p.map((v, i) => `${(i / (n - 1)) * 100},${h - ((v - min) / range) * h}`).join(' ')
  );
}

function normalizeArray(arr: number[], h: number): string {
  if (!arr.length) return '';
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;
  const n = arr.length;
  return arr.map((v, i) => `${(i / (n - 1)) * 100},${h - ((v - min) / range) * h}`).join(' ');
}

export default function AdvancedAnalysis({ industrySub = "", companyName = "", score = 50 }: Props) {
  const [networkLoading, setNetworkLoading] = useState(false);
  const [networkResult, setNetworkResult] = useState<NetworkRiskResult | null>(null);

  const [mcLoading, setMcLoading] = useState(false);
  const [mcResult, setMcResult] = useState<McResult | null>(null);

  const [tfmLoading, setTfmLoading] = useState(false);
  const [tfmResult, setTfmResult] = useState<TfmResult | null>(null);

  // TimesFMは業種があれば自動取得
  useEffect(() => {
    if (!industrySub) return;
    setTfmLoading(true);
    fetch('/api/timesfm/industry_trend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ industry: industrySub, horizon_months: 12 }),
    })
      .then(r => r.json())
      .then((d: TfmResult) => setTfmResult(d))
      .catch(() => setTfmResult({ error: 'fetch failed' }))
      .finally(() => setTfmLoading(false));
  }, [industrySub]);

  const analyzeNetworkRisk = async () => {
    setNetworkLoading(true);
    try {
      const res = await fetch(`/api/analysis/network_risk?industry=${encodeURIComponent(industrySub)}`);
      setNetworkResult(await res.json());
    } catch {
      setNetworkResult(null);
    } finally {
      setNetworkLoading(false);
    }
  };

  const runMC = async () => {
    setMcLoading(true);
    setMcResult(null);
    try {
      const res = await fetch('/api/timesfm/financial_paths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: companyName || '（未入力）', n_periods: 60 }),
      });
      setMcResult(await res.json());
    } catch {
      setMcResult(null);
    } finally {
      setMcLoading(false);
    }
  };

  // スコアから倒産確率を推定（score 100→0%, score 0→80%）
  const bankruptcyPct = Math.max(0, Math.min(80, Math.round((100 - score) * 0.8)));

  const svgH = 40;
  const polylines = mcResult ? normalizePaths(mcResult.gbm_paths.slice(0, 20), svgH) : [];
  const medianLine = mcResult ? normalizeArray(mcResult.gbm_median, svgH) : '';

  const riskColor: Record<string, string> = { positive: '#10b981', neutral: '#f59e0b', negative: '#ef4444' };
  const riskLabel: Record<string, string> = { positive: '改善傾向', neutral: '横ばい', negative: '悪化傾向' };
  const tfmSignal = tfmResult?.risk_signal || 'neutral';

  // TFM スコア折れ線
  const histLine = tfmResult?.avg_score_hist ? normalizeArray(tfmResult.avg_score_hist, svgH) : '';
  const foreLine = tfmResult?.avg_score_fore ? normalizeArray(tfmResult.avg_score_fore, svgH) : '';

  return (
    <div className="mt-12 space-y-6">
      <h3 className="text-xl font-black text-slate-800 border-b border-slate-200 pb-3 flex items-center gap-2">
        <Box className="w-5 h-5 text-indigo-500" />
        Advanced Simulation (高度分析)
      </h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* モンテカルロ シミュレーション */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h4 className="font-bold text-slate-800 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-emerald-500" />
                モンテカルロ リース審査シミュレーション
              </h4>
              <p className="text-xs text-slate-500 mt-1">GBM 200パスから5年後の売上分布を可視化</p>
            </div>
            <button
              onClick={runMC}
              disabled={mcLoading}
              className="px-4 py-2 bg-slate-900 text-white text-xs font-bold rounded-lg hover:bg-slate-800 transition disabled:opacity-50"
            >
              {mcLoading ? '演算中...' : '実行する'}
            </button>
          </div>

          <div className="h-48 bg-slate-50 rounded-2xl border border-slate-100 flex items-center justify-center relative overflow-hidden">
            {mcLoading && (
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs font-bold text-emerald-600 animate-pulse">GBM 200パス演算中...</span>
              </div>
            )}
            {!mcLoading && !mcResult && (
              <span className="text-sm font-medium text-slate-400">実行待機中</span>
            )}
            {!mcLoading && mcResult && (
              <div className="w-full h-full p-2 relative">
                <svg className="w-full h-full" viewBox={`0 0 100 ${svgH}`} preserveAspectRatio="none">
                  {polylines.map((pts, i) => (
                    <polyline key={i} points={pts} fill="none" stroke="#a7f3d0" strokeWidth="0.4" opacity="0.6" />
                  ))}
                  {medianLine && (
                    <polyline points={medianLine} fill="none" stroke="#10b981" strokeWidth="1.5" />
                  )}
                </svg>
                <div className={`absolute top-2 left-2 text-[10px] font-bold px-2 py-1 rounded shadow border text-white ${
                  bankruptcyPct < 3 ? 'bg-emerald-500 border-emerald-600' :
                  bankruptcyPct < 8 ? 'bg-amber-500 border-amber-600' :
                  'bg-red-500 border-red-600'
                }`}>
                  PD（デフォルト確率）: {bankruptcyPct}%
                  <span className="ml-1 opacity-80">{bankruptcyPct < 3 ? '▼低リスク' : bankruptcyPct < 8 ? '▲要注意' : '⚠要警戒'}</span>
                </div>
                <div className="absolute top-2 right-2 bg-white/90 text-[9px] text-slate-400 px-1.5 py-0.5 rounded border border-slate-100">
                  スコア推定・5年後参考値
                </div>
                <div className="absolute bottom-2 right-2 text-[9px] text-slate-400">
                  {mcResult.timesfm_available ? 'TimesFM使用' : 'GBM'}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* サプライチェーン・ネットワーク波及 */}
        <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h4 className="font-bold text-slate-800 flex items-center gap-2">
                <Network className="w-4 h-4 text-orange-500" />
                サプライチェーン不調 波及リスク
              </h4>
              <p className="text-xs text-slate-500 mt-1">
                Leontief 逆行列による依存企業からの連鎖リスク
              </p>
            </div>
            <button
              onClick={analyzeNetworkRisk}
              disabled={networkLoading}
              className="px-4 py-2 bg-orange-100 text-orange-700 text-xs font-bold rounded-lg hover:bg-orange-200 transition disabled:opacity-50"
            >
              {networkLoading ? '計算中...' : '分析'}
            </button>
          </div>

          <div className="h-48 bg-slate-900 rounded-2xl flex items-center justify-center relative overflow-hidden">
            <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 100" preserveAspectRatio="none">
              <circle cx="20" cy="30" r="3" fill="#fb923c" />
              <circle cx="50" cy="50" r="5" fill="#f97316" />
              <circle cx="80" cy="40" r="4" fill="#fb923c" />
              <circle cx="30" cy="80" r="3" fill="#fb923c" />
              <line x1="20" y1="30" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
              <line x1="80" y1="40" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
              <line x1="30" y1="80" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
            </svg>
            {networkLoading ? (
              <div className="flex flex-col items-center gap-2 z-10">
                <div className="w-7 h-7 border-4 border-orange-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-orange-300 font-bold animate-pulse">Leontief 逆行列演算中...</span>
              </div>
            ) : networkResult ? (
              <div className="z-10 p-4 bg-slate-950/80 backdrop-blur rounded-xl border border-slate-700 shadow-xl w-full mx-4">
                <div className="text-[10px] text-slate-400 font-bold mb-1 text-center">
                  {networkResult.target_industry || industrySub} — 波及リスク
                </div>
                <div className="flex justify-around items-end">
                  <div className="text-center">
                    <div className="text-[9px] text-slate-500">ベースリスク</div>
                    <div className="text-lg font-black text-yellow-400">{networkResult.base_risk_pct}%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[9px] text-slate-500">連鎖波及</div>
                    <div className="text-2xl font-black text-orange-400">{networkResult.network_risk_pct}%</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[9px] text-slate-500">VaR95%</div>
                    <div className="text-lg font-black text-red-400">{networkResult.sim_var95_pct}%</div>
                  </div>
                </div>
                {networkResult.impacted_by.length > 0 && (
                  <div className="mt-2 text-[9px] text-slate-400 text-center">
                    主な影響元: {networkResult.impacted_by.slice(0, 2).map(i => i.from).join(' / ')}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center z-10 p-4 bg-slate-950/80 backdrop-blur rounded-xl border border-slate-700 shadow-xl">
                <div className="text-[10px] text-slate-400 font-bold mb-1">「分析」を押すと計算します</div>
                <div className="text-xs text-slate-500">{industrySub || '業種を入力後に実行'}</div>
              </div>
            )}
          </div>
        </div>

      </div>

      {/* TimesFM 予測表示エリア */}
      <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-200">
        <div className="flex justify-between items-start mb-4">
          <h4 className="font-bold text-slate-800 flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-500" />
            TimesFM AI 時系列指標予測
          </h4>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 bg-blue-50 text-blue-600 text-[10px] font-bold rounded-md">Google Research</span>
            {tfmResult?.risk_signal && (
              <span
                className="px-3 py-1 text-[10px] font-bold rounded-md text-white"
                style={{ backgroundColor: riskColor[tfmSignal] || '#94a3b8' }}
              >
                {riskLabel[tfmSignal] || tfmSignal}
              </span>
            )}
          </div>
        </div>
        <div className="h-32 bg-slate-50 border border-slate-100 rounded-2xl flex flex-col items-center justify-center relative overflow-hidden">
          {tfmLoading && (
            <div className="flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-4 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-blue-500 font-bold animate-pulse">業種トレンド予測中...</span>
            </div>
          )}
          {!tfmLoading && tfmResult?.error && (
            <p className="text-xs text-slate-400">{tfmResult.error}</p>
          )}
          {!tfmLoading && !tfmResult && !industrySub && (
            <p className="text-xs text-slate-400">業種コードを入力後にスコア計算すると自動表示されます</p>
          )}
          {!tfmLoading && tfmResult && !tfmResult.error && (
            <>
              <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 100 ${svgH}`} preserveAspectRatio="none">
                {histLine && (
                  <polyline points={histLine} fill="none" stroke="#3b82f6" strokeWidth="1.2" />
                )}
                {foreLine && (
                  <polyline points={foreLine} fill="none" stroke={riskColor[tfmSignal] || '#94a3b8'} strokeWidth="1.2" strokeDasharray="2 1.5" />
                )}
              </svg>
              <div className="relative z-10 text-center">
                <p className="font-black text-slate-800 text-sm">
                  {industrySub} — {riskLabel[tfmSignal] || '予測中'}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  {(tfmResult.months_history?.length || 0)}ヶ月実績 → {(tfmResult.months_forecast?.length || 0)}ヶ月予測
                  （{tfmResult.method || 'GBM'}）
                </p>
              </div>
            </>
          )}
        </div>
      </div>

    </div>
  );
}
