"use client";

import React, { useState, useMemo } from "react";
import { Calculator } from "lucide-react";

// リース料率テーブル (年数 → 月額リース料率%)
const LEASE_RATE_FACTORS: Record<number, number> = {
  2: 4.45, 3: 3.05, 4: 2.35, 5: 1.92, 6: 1.63, 7: 1.43,
};

// 元利均等返済 月額計算
function calcPMT(principal: number, annualRate: number, months: number): number {
  if (annualRate === 0) return principal / months;
  const r = annualRate / 100 / 12;
  return principal * r * Math.pow(1 + r, months) / (Math.pow(1 + r, months) - 1);
}

// 固定資産税 累計概算（評価額 × 0.8 逓減、税率1.4%）
function calcTotalPropertyTax(price: number, years: number): number {
  let tax = 0;
  let value = price;
  for (let i = 0; i < years; i++) {
    tax += Math.round(value * 0.014);
    value = Math.round(value * 0.8);
  }
  return tax;
}

type ComparisonResult = {
  monthlyPayment: number;
  totalPayment: number;
  taxSaving: number;
  propertyTax: number;
  netBurden: number;
};

function calcLease(price: number, years: number, taxRate: number): ComparisonResult {
  const factor = LEASE_RATE_FACTORS[years] ?? 1.92;
  const monthlyPayment = Math.round(price * factor / 100);
  const totalPayment = monthlyPayment * years * 12;
  const taxSaving = Math.round(totalPayment * (taxRate / 100));
  return { monthlyPayment, totalPayment, taxSaving, propertyTax: 0, netBurden: totalPayment - taxSaving };
}

function calcLoan(price: number, years: number, annualRate: number, taxRate: number): ComparisonResult {
  const months = years * 12;
  const monthlyPayment = Math.round(calcPMT(price, annualRate, months));
  const totalPayment = monthlyPayment * months;
  const totalInterest = totalPayment - price;
  // 節税: 支払利息の損金 + 減価償却の損金
  const taxSaving = Math.round((totalInterest + price) * (taxRate / 100));
  const propertyTax = calcTotalPropertyTax(price, years);
  return { monthlyPayment, totalPayment, taxSaving, propertyTax, netBurden: totalPayment - taxSaving + propertyTax };
}

function calcCash(price: number, years: number, taxRate: number): ComparisonResult {
  // 節税: 減価償却の損金（期間内全額）
  const taxSaving = Math.round(price * (taxRate / 100));
  const propertyTax = calcTotalPropertyTax(price, years);
  return { monthlyPayment: 0, totalPayment: price, taxSaving, propertyTax, netBurden: price - taxSaving + propertyTax };
}

const fmt = (n: number) => n.toLocaleString() + " 円";

type RowData = {
  label: string;
  lease: number;
  loan: number;
  cash: number;
  isHighlight: boolean;
  negate?: boolean;
};

export default function SimulatorPage() {
  const [priceMillion, setPriceMillion] = useState(10);
  const [years, setYears] = useState(5);
  const [loanRate, setLoanRate] = useState(2.5);
  const [taxRate, setTaxRate] = useState(30);

  const price = Math.max(0.1, priceMillion) * 1_000_000;

  const lease = useMemo(() => calcLease(price, years, taxRate), [price, years, taxRate]);
  const loan = useMemo(() => calcLoan(price, years, loanRate, taxRate), [price, years, loanRate, taxRate]);
  const cash = useMemo(() => calcCash(price, years, taxRate), [price, years, taxRate]);

  const rows: RowData[] = [
    { label: "月額支払い", lease: lease.monthlyPayment, loan: loan.monthlyPayment, cash: cash.monthlyPayment, isHighlight: false },
    { label: "総支払額", lease: lease.totalPayment, loan: loan.totalPayment, cash: cash.totalPayment, isHighlight: false },
    { label: "節税効果", lease: lease.taxSaving, loan: loan.taxSaving, cash: cash.taxSaving, isHighlight: false, negate: true },
    { label: "固定資産税（累計概算）", lease: lease.propertyTax, loan: loan.propertyTax, cash: cash.propertyTax, isHighlight: false },
    { label: "実質負担（税効果考慮後）", lease: lease.netBurden, loan: loan.netBurden, cash: cash.netBurden, isHighlight: true },
  ];

  const options = [
    { label: "リース", value: lease.netBurden },
    { label: "融資購入", value: loan.netBurden },
    { label: "現金購入", value: cash.netBurden },
  ];
  const bestOption = [...options].sort((a, b) => a.value - b.value)[0];

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <Calculator className="text-blue-500 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">リース / 融資 / 現金購入 比較シミュレーター</h1>
          <p className="text-sm text-slate-500">物件取得方法別の実質負担を比較します（節税効果・固定資産税を含む）</p>
        </div>
      </div>

      {/* 入力 */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">入力条件</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">物件価格（百万円）</label>
            <input
              type="number"
              value={priceMillion}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setPriceMillion(isNaN(v) ? 0 : v);
              }}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-400"
              step={0.1}
              min={0.1}
            />
            <p className="text-xs text-slate-400 mt-1">{price.toLocaleString()} 円</p>
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">期間（2〜7年）</label>
            <select
              value={years}
              onChange={(e) => setYears(Number(e.target.value))}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-400"
            >
              {[2, 3, 4, 5, 6, 7].map((y) => (
                <option key={y} value={y}>{y}年</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">融資金利（年率 %）</label>
            <input
              type="number"
              value={loanRate}
              onChange={(e) => setLoanRate(Math.max(0, Number(e.target.value)))}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-400"
              step={0.1}
              min={0}
            />
          </div>
          <div>
            <label className="block text-xs font-bold text-slate-600 mb-1">実効税率（%）</label>
            <input
              type="number"
              value={taxRate}
              onChange={(e) => setTaxRate(Math.min(60, Math.max(0, Number(e.target.value))))}
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-400"
              step={1}
              min={0}
              max={60}
            />
          </div>
        </div>
      </div>

      {/* 最適解バナー */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-center gap-4">
        <span className="text-3xl">🏆</span>
        <div>
          <p className="text-xs text-blue-500 font-bold uppercase tracking-widest">実質負担が最も低い選択肢</p>
          <p className="text-2xl font-black text-blue-700">{bestOption.label}</p>
          <p className="text-sm text-blue-600 font-medium">{fmt(bestOption.value)}</p>
        </div>
        <div className="ml-auto hidden md:flex gap-6">
          {options.map((opt) => (
            <div key={opt.label} className={`text-center px-4 py-2 rounded-lg ${opt.label === bestOption.label ? "bg-blue-100 border border-blue-300" : "bg-white border border-slate-200"}`}>
              <p className="text-xs font-bold text-slate-500">{opt.label}</p>
              <p className={`text-sm font-black ${opt.label === bestOption.label ? "text-blue-700" : "text-slate-700"}`}>{fmt(opt.value)}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 比較テーブル */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <p className="text-sm font-semibold text-slate-700">3者比較詳細</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-500 bg-slate-50 border-b border-slate-100">
                <th className="text-left px-4 py-3 w-52">項目</th>
                <th className="text-right px-4 py-3">リース</th>
                <th className="text-right px-4 py-3">融資購入</th>
                <th className="text-right px-4 py-3">現金購入</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => {
                const vals = [row.lease, row.loan, row.cash];
                // 節税効果は大きい方が良い（negate=true）、それ以外は小さい方が良い
                const bestVal = row.negate ? Math.max(...vals) : Math.min(...vals.filter((v) => v > 0));
                return (
                  <tr
                    key={row.label}
                    className={row.isHighlight ? "bg-blue-50 border-t-2 border-blue-200" : i % 2 === 0 ? "bg-white" : "bg-slate-50"}
                  >
                    <td className={`px-4 py-3 font-medium ${row.isHighlight ? "text-blue-800 font-bold text-base" : "text-slate-600"}`}>
                      {row.label}
                    </td>
                    {[
                      { val: row.lease, key: "lease" },
                      { val: row.loan, key: "loan" },
                      { val: row.cash, key: "cash" },
                    ].map(({ val, key }) => {
                      const isBest = val > 0 && val === bestVal;
                      return (
                        <td
                          key={key}
                          className={`px-4 py-3 text-right font-mono ${
                            row.isHighlight
                              ? isBest
                                ? "font-black text-blue-700 text-base"
                                : "text-slate-600 font-bold"
                              : isBest && !row.isHighlight
                              ? "text-emerald-600 font-bold"
                              : "text-slate-700"
                          }`}
                        >
                          {val === 0 ? <span className="text-slate-300">—</span> : fmt(val)}
                          {isBest && <span className="ml-1 text-xs text-blue-400">★</span>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 前提条件 */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500 space-y-1">
        <p className="font-bold text-slate-600 mb-2">📌 計算の前提</p>
        <ul className="list-disc list-inside space-y-0.5">
          <li>リース料率: 標準料率テーブル使用（{years}年 = {LEASE_RATE_FACTORS[years]}%/月）</li>
          <li>融資: 元利均等返済、金利 {loanRate}%/年</li>
          <li>節税効果: リース＝リース料全額損金、融資＝支払利息+減価償却相当、現金＝減価償却相当のみ</li>
          <li>固定資産税: 評価額 × 0.8 逓減 × 1.4%（{years}年累計概算）。リース物件はリース会社負担のため0円</li>
          <li>この計算は概算です。実際の節税効果・税額は税理士・各市区町村にご確認ください。</li>
        </ul>
      </div>
    </div>
  );
}
