"use client";

import React, { useState, useMemo } from "react";
import { Receipt } from "lucide-react";

const DEPRECIATION_RATES: Record<number, number> = {
  2: 0.500, 3: 0.334, 4: 0.250, 5: 0.200, 6: 0.167, 7: 0.143,
  8: 0.125, 9: 0.112, 10: 0.100, 12: 0.084, 15: 0.067, 20: 0.050,
};

const TAX_RATE_BY_YEAR: Record<number, number> = {
  1: 1.000, 2: 0.800, 3: 0.640, 4: 0.512, 5: 0.410,
  6: 0.328, 7: 0.262, 8: 0.210, 9: 0.168, 10: 0.134,
  11: 0.107, 12: 0.086, 13: 0.069, 14: 0.055, 15: 0.044,
};

function calcDepreciation(cost: number, usefulLife: number): { year: number; book: number; tax: number }[] {
  const rate = DEPRECIATION_RATES[usefulLife] ?? (1 / usefulLife);
  const rows = [];
  let book = cost;
  for (let y = 1; y <= usefulLife; y++) {
    const dep = y < usefulLife ? Math.round(book * rate) : Math.round(book - 1);
    book = Math.max(1, book - dep);
    const taxBase = cost * (TAX_RATE_BY_YEAR[y] ?? Math.pow(0.8, y));
    const taxAmount = Math.round(taxBase * 0.014);
    rows.push({ year: y, book, tax: taxAmount });
  }
  return rows;
}

export default function TaxCalcPage() {
  const [cost, setCost] = useState(1000000);
  const [usefulLife, setUsefulLife] = useState(5);

  const rows = useMemo(() => calcDepreciation(cost, usefulLife), [cost, usefulLife]);
  const totalTax = rows.reduce((s, r) => s + r.tax, 0);

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <Receipt className="text-orange-500 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">固定資産税 計算シミュレーター</h1>
          <p className="text-sm text-slate-500">取得価格と法定耐用年数から固定資産税の年次推移を試算します。</p>
        </div>
      </div>

      <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 text-xs text-orange-700">
        <p className="font-semibold mb-1">📌 計算の前提</p>
        <ul className="list-disc list-inside space-y-0.5">
          <li>償却資産税率: <strong>1.4%</strong>（標準税率）</li>
          <li>評価額の計算: 前年評価額 × 0.8（減価残存率）で逓減</li>
          <li>最低評価額: 取得価格の5%（下限あり）</li>
          <li>リース物件はリース会社が申告主体（ユーザーは非課税が多い）</li>
        </ul>
      </div>

      {/* 入力 */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">取得価格（円）</label>
            <input
              type="number"
              value={cost}
              onChange={(e) => setCost(Math.max(0, Number(e.target.value)))}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-orange-400"
              step={100000}
            />
            <p className="text-xs text-slate-400 mt-1">{cost.toLocaleString()} 円</p>
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">法定耐用年数</label>
            <select
              value={usefulLife}
              onChange={(e) => setUsefulLife(Number(e.target.value))}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-orange-400"
            >
              {[2,3,4,5,6,7,8,9,10,12,15,20].map((y) => (
                <option key={y} value={y}>{y}年</option>
              ))}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm pt-2 border-t border-slate-100">
          <div className="flex-1">
            <p className="text-xs text-slate-400">全期間合計固定資産税（目安）</p>
            <p className="text-xl font-bold text-orange-600">{totalTax.toLocaleString()} 円</p>
          </div>
          <div className="text-xs text-slate-400">
            ※リース会社が所有する場合、<br/>ユーザー負担はリース料に含まれます
          </div>
        </div>
      </div>

      {/* 年次推移テーブル */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">年次推移</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-500 bg-slate-50 border-b border-slate-100">
              <th className="text-center px-4 py-2">年次</th>
              <th className="text-right px-4 py-2">帳簿価額（円）</th>
              <th className="text-right px-4 py-2">固定資産税額（目安）</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.year} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                <td className="px-4 py-2 text-center text-slate-600">{r.year}年目</td>
                <td className="px-4 py-2 text-right text-slate-700">{r.book.toLocaleString()}</td>
                <td className="px-4 py-2 text-right font-medium text-orange-600">{r.tax.toLocaleString()}</td>
              </tr>
            ))}
            <tr className="bg-orange-50 border-t border-orange-200">
              <td className="px-4 py-2 text-center text-xs font-bold text-slate-600">合計</td>
              <td className="px-4 py-2 text-right text-xs text-slate-400">—</td>
              <td className="px-4 py-2 text-right font-bold text-orange-700">{totalTax.toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p className="text-xs text-slate-400">
        ※ この計算は概算です。実際の税額は各市区町村の固定資産税課に確認してください。リース物件はリース会社が申告するため、ユーザーへの直接課税は通常発生しません。
      </p>
    </div>
  );
}
