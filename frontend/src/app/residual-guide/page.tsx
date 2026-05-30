"use client";

import React from "react";
import { Calculator } from "lucide-react";

type ResidualRow = {
  years: number;
  category: string;
  examples: string;
  rate: string;
  note: string;
};

const RESIDUAL_TABLE: ResidualRow[] = [
  { years: 2, category: "PC・タブレット", examples: "ノートPC、iPad", rate: "15〜25%", note: "陳腐化が早い。再販市場あり" },
  { years: 3, category: "サーバー・通信機器", examples: "ラックサーバー、NAS", rate: "10〜20%", note: "3〜4年で性能陳腐化" },
  { years: 4, category: "情報機器全般", examples: "複合機、POS端末", rate: "8〜15%", note: "耐用年数内ならほぼ担保価値あり" },
  { years: 5, category: "一般車両・トラック", examples: "普通車、小型トラック", rate: "20〜35%", note: "走行距離・整備履歴で変動大" },
  { years: 6, category: "建設機械", examples: "ユンボ、フォークリフト", rate: "25〜40%", note: "稼働時間・メーカー品で安定" },
  { years: 8, category: "製造機械", examples: "CNC旋盤、マシニング", rate: "15〜30%", note: "汎用性の高い機種は残価安定" },
  { years: 10, category: "大型設備", examples: "食品加工機、印刷機", rate: "10〜20%", note: "特殊用途は残価低め" },
  { years: 15, category: "空調・電気設備", examples: "エアコン、変圧器", rate: "5〜10%", note: "撤去コストが発生する場合あり" },
];

const CALC_EXAMPLES = [
  { asset: "ノートPC (30万円, 4年リース)", residual: "30万×15%=4.5万円", monthly: "(30-4.5)万÷48回=約5,312円/月" },
  { asset: "フォークリフト (300万円, 5年リース)", residual: "300万×30%=90万円", monthly: "(300-90)万÷60回=約35,000円/月" },
  { asset: "CNC旋盤 (1,000万円, 7年リース)", residual: "1,000万×20%=200万円", monthly: "(1,000-200)万÷84回=約95,238円/月" },
];

export default function ResidualGuidePage() {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <Calculator className="text-purple-600 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">残価設定ガイドライン</h1>
          <p className="text-sm text-slate-500">物件カテゴリ別の推奨残価率と月額計算例。審査・見積作成時の参考にご活用ください。</p>
        </div>
      </div>

      {/* 残価率テーブル */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">物件カテゴリ別 推奨残価率</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-500 bg-slate-50 border-b border-slate-100">
                <th className="text-left px-4 py-2">耐用年数目安</th>
                <th className="text-left px-4 py-2">カテゴリ</th>
                <th className="text-left px-3 py-2">代表例</th>
                <th className="text-center px-3 py-2 font-bold">推奨残価率</th>
                <th className="text-left px-3 py-2">備考</th>
              </tr>
            </thead>
            <tbody>
              {RESIDUAL_TABLE.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                  <td className="px-4 py-2 text-slate-500 text-xs">{row.years}年</td>
                  <td className="px-4 py-2 font-medium text-slate-700">{row.category}</td>
                  <td className="px-3 py-2 text-xs text-slate-500">{row.examples}</td>
                  <td className="px-3 py-2 text-center">
                    <span className="inline-block px-2 py-0.5 bg-purple-50 text-purple-700 border border-purple-200 rounded-full text-xs font-bold">
                      {row.rate}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-xs text-slate-400">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 計算例 */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">💡 月額計算例</p>
          <p className="text-xs text-slate-400 mt-0.5">月額 = (取得価格 − 残価) ÷ リース月数 ※金利・手数料は除く概算</p>
        </div>
        <div className="divide-y divide-slate-100">
          {CALC_EXAMPLES.map((ex, i) => (
            <div key={i} className="px-4 py-3">
              <p className="text-sm font-medium text-slate-700 mb-1">{ex.asset}</p>
              <div className="flex gap-6 text-xs text-slate-500">
                <span>残価: <strong className="text-purple-600">{ex.residual}</strong></span>
                <span>月額目安: <strong className="text-blue-600">{ex.monthly}</strong></span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ガイドライン */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-green-50 border border-green-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-green-800 mb-2">✅ 残価を高く設定できるケース</p>
          <ul className="text-xs text-green-700 space-y-1 list-disc list-inside">
            <li>メーカー保証・認定品で再販市場が確立</li>
            <li>汎用性が高く多業種で使用可能な設備</li>
            <li>リース期間が耐用年数の50%以内</li>
            <li>稼働記録・整備履歴が明確</li>
          </ul>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-red-800 mb-2">⚠️ 残価を低く設定すべきケース</p>
          <ul className="text-xs text-red-700 space-y-1 list-disc list-inside">
            <li>特定業種・特定用途専用の設備</li>
            <li>技術革新が早い分野（IT機器等）</li>
            <li>撤去・廃棄コストが発生する設備</li>
            <li>リース期間が耐用年数の80%超</li>
          </ul>
        </div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500">
        残価率は市況・物件コンディション・業者査定により変動します。最終的な残価設定は審査部門と協議の上で決定してください。
      </div>
    </div>
  );
}
