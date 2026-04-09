import React from 'react';
import { Sparkles, MessageCircleWarning } from 'lucide-react';

interface AIAnalysisProps {
  comparisonText: string;
}

export default function AIAnalysis({ comparisonText }: AIAnalysisProps) {
  if (!comparisonText) return null;

  // Streamlitのマークダウン出力を簡易的にパースして見やすくする
  const lines = comparisonText.split('\n').filter(l => l.trim().length > 0);

  return (
    <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200 relative overflow-hidden">
      {/* 背景装飾 */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-full blur-3xl -z-10 translate-x-1/2 -translate-y-1/4 opacity-70"></div>
      
      <h4 className="text-xl font-black text-slate-800 mb-6 flex items-center gap-2">
        <span className="bg-gradient-to-r from-indigo-500 to-purple-600 text-transparent bg-clip-text">AI 審査アドバイス</span>
        <Sparkles className="w-5 h-5 text-indigo-400" />
      </h4>

      <div className="space-y-4 relative">
        <div className="absolute left-6 top-6 bottom-6 w-0.5 bg-indigo-100 rounded-full"></div>
        
        {lines.map((line, idx) => {
          // 強調部分を太字にする簡易処理
          const isWarning = line.includes('注意') || line.includes('低い') || line.includes('※');
          const isHighlight = line.includes('高い') || line.includes('改善');
          
          return (
            <div key={idx} className="relative pl-12">
              {/* ドット */}
              <div className={`absolute left-5 top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full border-2 border-white shadow-sm z-10 ${
                isWarning ? 'bg-amber-400' : isHighlight ? 'bg-emerald-400' : 'bg-indigo-400'
              }`}></div>
              
              <div className={`p-4 rounded-2xl ${
                isWarning ? 'bg-amber-50/50 border border-amber-100' : 'bg-slate-50 border border-slate-100'
              }`}>
                <p className="text-sm font-semibold text-slate-700 leading-relaxed">
                  {line.replace(/[-*]/g, '').trim()}
                </p>
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="mt-8 flex bg-blue-50/50 p-4 rounded-xl border border-blue-100 items-start gap-4">
        <div className="p-2 bg-blue-100 text-blue-600 rounded-full shrink-0">
          <MessageCircleWarning className="w-5 h-5" />
        </div>
        <p className="text-xs text-slate-600 font-medium leading-relaxed">
          八奈見BOTからの補足：<br/>
          上記のアドバイスは、業界平均（ベンチマーク）と当社の数値を論理的に比較したものです。「リースくん」メニューから対話形式でさらに深堀り質問することも可能です。
        </p>
      </div>
    </div>
  );
}
