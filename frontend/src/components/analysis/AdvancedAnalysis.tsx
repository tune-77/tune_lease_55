import React, { useState } from 'react';
import { LineChart, BarChart2, Activity, Network, Box } from 'lucide-react';

export default function AdvancedAnalysis() {
  const [runningMC, setRunningMC] = useState(false);
  const [mcDone, setMcDone] = useState(false);

  const simulateMC = () => {
    setRunningMC(true);
    setTimeout(() => {
      setRunningMC(false);
      setMcDone(true);
    }, 2500); // モンテカルロの計算時間をフェイク
  };

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
              <p className="text-xs text-slate-500 mt-1">
                10,000回の確率パスから将来リスクを可視化します
              </p>
            </div>
            {!runningMC && !mcDone && (
              <button 
                onClick={simulateMC}
                className="px-4 py-2 bg-slate-900 text-white text-xs font-bold rounded-lg hover:bg-slate-800 transition"
              >
                実行する
              </button>
            )}
          </div>

          <div className="h-48 bg-slate-50 rounded-2xl border border-slate-100 flex items-center justify-center relative overflow-hidden">
            {!runningMC && !mcDone && (
              <span className="text-sm font-medium text-slate-400">実行待機中</span>
            )}
            
            {runningMC && (
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-xs font-bold text-emerald-600 animate-pulse">演算中 (n=10,000)...</span>
              </div>
            )}

            {mcDone && (
              <div className="w-full h-full p-4 flex flex-col justify-end relative">
                {/* グラフのモックSVG */}
                <svg className="w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
                  {/* 中央の線 */}
                  <path d="M0,20 Q25,18 50,22 T100,15" fill="none" stroke="#10b981" strokeWidth="1.5" />
                  {/* 分散の帯 */}
                  <path d="M0,15 Q25,10 50,15 T100,2 M100,38 Q75,35 50,30 T0,25" fill="none" stroke="#a7f3d0" strokeWidth="1" strokeDasharray="2 2" />
                  {/* 最悪ケースの線 */}
                  <path d="M0,25 Q25,28 50,35 T100,38" fill="none" stroke="#ef4444" strokeWidth="1" />
                  
                  {/* 塗りつぶし領域 */}
                  <path d="M0,15 Q25,10 50,15 T100,2 L100,38 Q75,35 50,30 T0,25 Z" fill="#ecfdf5" opacity="0.5" />
                </svg>
                <div className="absolute top-2 left-2 bg-white/80 backdrop-blur text-[10px] font-bold px-2 py-1 rounded text-slate-600 shadow-sm border border-slate-100">
                  倒産確率: 2.4% (5年後)
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
            <button className="px-4 py-2 bg-orange-100 text-orange-700 text-xs font-bold rounded-lg hover:bg-orange-200 transition">
              分析
            </button>
          </div>

          <div className="h-48 bg-slate-900 rounded-2xl flex items-center justify-center relative overflow-hidden">
            {/* ネットワークのモックSVG装飾 */}
            <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 100" preserveAspectRatio="none">
              <circle cx="20" cy="30" r="3" fill="#fb923c" />
              <circle cx="50" cy="50" r="5" fill="#f97316" />
              <circle cx="80" cy="40" r="4" fill="#fb923c" />
              <circle cx="30" cy="80" r="3" fill="#fb923c" />
              
              <line x1="20" y1="30" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
              <line x1="80" y1="40" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
              <line x1="30" y1="80" x2="50" y2="50" stroke="#fdba74" strokeWidth="0.5" />
            </svg>
            <div className="text-center z-10 p-4 bg-slate-950/80 backdrop-blur rounded-xl border border-slate-700 shadow-xl">
              <div className="text-[10px] text-slate-400 font-bold mb-1">連鎖倒産波及確率</div>
              <div className="text-2xl font-black text-orange-400">14.2%</div>
            </div>
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
          <span className="px-3 py-1 bg-blue-50 text-blue-600 text-[10px] font-bold rounded-md">Google Research</span>
        </div>
        <div className="h-32 bg-slate-50 border border-slate-100 rounded-2xl flex flex-col items-center justify-center relative overflow-hidden">
           <svg className="absolute inset-0 w-full h-full opacity-40" viewBox="0 0 100 40" preserveAspectRatio="none">
             <path d="M0,30 Q10,35 20,25 T40,15 T60,20 T80,10 T100,5" fill="none" stroke="#3b82f6" strokeWidth="1" />
           </svg>
           <h5 className="font-black text-slate-800 text-lg relative z-10">予測レンジ: 改善傾向</h5>
           <p className="text-xs text-slate-500 mt-1 relative z-10">ゼロショット予測モデルにより、次期決算での回復が見込まれます</p>
        </div>
      </div>
      
    </div>
  );
}
