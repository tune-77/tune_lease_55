import React from 'react';
import { Network, ArrowRight } from 'lucide-react';

interface ScoreDAGProps {
  data: any;
}

export default function ScoreDAG({ data }: ScoreDAGProps) {
  if (!data) return null;

  // バックエンド(API)からの簡易データに基づくノード構築
  const w_main = 0.5; // Dummy blend weights for simplified API
  const score_borrower = data.score_borrower || 0;
  const score_base = data.score_base || data.score || 0;
  const total_score = data.score || 0;
  const hantei = data.hantei || '未判定';

  // AI 補正因子（フル版API連携用。簡易版では空）
  const factors = data.ai_completed_factors || [];

  return (
    <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 p-6">
      <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 mb-6">
        <Network className="w-5 h-5 text-indigo-500" />
        スコアリング因果グラフ (DAG)
      </h3>
      
      <div className="overflow-x-auto pb-4">
        <div className="min-w-[700px] flex justify-between items-center px-4 relative">
          
          {/* 背景の連結線 (簡易) */}
          <div className="absolute top-1/2 left-10 right-10 h-0.5 bg-slate-200 -z-10 transform -translate-y-1/2 rounded-full border-t border-slate-200 border-dashed"></div>

          {/* Col 0: 回帰モデル */}
          <div className="flex flex-col items-center gap-2 relative z-10 w-32">
            <span className="text-xs font-bold text-slate-400 mb-2">予測モデル群</span>
            <div className="bg-sky-50 border-2 border-sky-300 rounded-xl p-3 shadow-sm w-full text-center">
              <div className="text-[10px] text-sky-600 font-semibold mb-1">全体回帰モデル</div>
              <div className="text-xl font-black text-sky-700">{score_borrower.toFixed(1)}%</div>
            </div>
            <div className="text-[10px] text-slate-400 bg-white px-2 mt-1">weight: {w_main*100}%</div>
          </div>

          <ArrowRight className="w-5 h-5 text-slate-400 mx-2" />

          {/* Col 1: 補正因子 */}
          <div className="flex flex-col items-center gap-2 relative z-10 w-32">
            <span className="text-xs font-bold text-slate-400 mb-2">補正因子 (AI)</span>
            {factors.length > 0 ? (
              factors.map((f: any, i: number) => (
                <div key={i} className={`border-2 rounded-xl p-2 w-full text-center ${f.effect_percent >= 0 ? 'bg-emerald-50 border-emerald-300 text-emerald-700' : 'bg-rose-50 border-rose-300 text-rose-700'}`}>
                  <div className="text-[10px] font-semibold truncate">{f.factor.substring(0, 8)}</div>
                  <div className="text-sm font-black">{f.effect_percent > 0 ? '+' : ''}{f.effect_percent}%</div>
                </div>
              ))
            ) : (
              <div className="bg-slate-50 border-2 border-slate-200 rounded-xl p-3 shadow-sm w-full text-center">
                <div className="text-[10px] text-slate-500 font-semibold mb-1">補正なし</div>
                <div className="text-sm font-black text-slate-400">±0%</div>
              </div>
            )}
          </div>

          <ArrowRight className="w-5 h-5 text-slate-400 mx-2" />

          {/* Col 2: スコア成分 */}
          <div className="flex flex-col items-center gap-2 relative z-10 w-32">
            <span className="text-xs font-bold text-slate-400 mb-2">スコア成分</span>
            <div className="bg-blue-50 border-2 border-blue-400 rounded-xl p-3 shadow-sm w-full text-center">
              <div className="text-[10px] text-blue-600 font-semibold mb-1">借手スコア</div>
              <div className="text-xl font-black text-blue-800">{score_borrower.toFixed(1)}</div>
            </div>
          </div>

          <ArrowRight className="w-5 h-5 text-slate-400 mx-2" />

          {/* Col 3: 総合スコア */}
          <div className="flex flex-col items-center gap-2 relative z-10 w-32">
            <span className="text-xs font-bold text-blue-500 mb-2">総合スコア</span>
            <div className="bg-indigo-600 border-2 border-indigo-700 rounded-2xl p-4 shadow-lg shadow-indigo-200 w-full text-center transform scale-110">
              <div className="text-[10px] text-indigo-200 font-semibold mb-1">Total Score</div>
              <div className="text-2xl font-black text-white">{total_score.toFixed(1)}</div>
            </div>
          </div>

          <ArrowRight className="w-5 h-5 text-slate-400 mx-2" />

          {/* Col 4: 判定 */}
          <div className="flex flex-col items-center gap-2 relative z-10 w-32">
            <span className="text-xs font-bold text-slate-400 mb-2">最終判定</span>
            <div className={`border-2 rounded-2xl p-4 shadow-md w-full text-center ${
              score_base >= 71 ? 'bg-emerald-500 border-emerald-600 shadow-emerald-200' : 'bg-amber-500 border-amber-600 shadow-amber-200'
            }`}>
              <div className="text-[10px] text-white/80 font-semibold mb-1">Decision</div>
              <div className="text-lg font-black text-white">{hantei}</div>
            </div>
          </div>

        </div>
      </div>
      
      <div className="mt-4 text-xs text-slate-400 bg-slate-50 p-3 rounded-lg border border-slate-100">
        📌 <strong>因果の連鎖:</strong> 左から右へ計算の流れを表しています。AIによる補正ペナルティや加点は「補正因子」の列に赤・緑で表示されます。（現在は簡易APIのためベース項目のみ表示）
      </div>
    </div>
  );
}
