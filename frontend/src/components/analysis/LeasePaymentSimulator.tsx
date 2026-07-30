"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Calculator, ChevronDown, Trophy } from "lucide-react";
import { apiClient } from "@/lib/api";

const LEASE_RATE_FACTORS: Record<number, number> = {
  2: 4.45,
  3: 3.05,
  4: 2.35,
  5: 1.92,
  6: 1.63,
  7: 1.43,
};

type ComparisonResult = {
  monthlyPayment: number;
  totalPayment: number;
  taxSaving: number;
  propertyTax: number;
  netBurden: number;
};

type RowData = {
  label: string;
  lease: number;
  loan: number;
  cash: number;
  isHighlight: boolean;
  negate?: boolean;
};

type LeasePaymentSimulatorProps = {
  source: "screening" | "lease-intelligence" | "simulator";
  initialPriceMillion?: number;
  initialYears?: number;
  compact?: boolean;
  defaultOpen?: boolean;
  title?: string;
  description?: string;
};

function calcPMT(principal: number, annualRate: number, months: number): number {
  if (annualRate === 0) return principal / months;
  const r = annualRate / 100 / 12;
  return (principal * r * Math.pow(1 + r, months)) / (Math.pow(1 + r, months) - 1);
}

function calcTotalPropertyTax(price: number, years: number): number {
  let tax = 0;
  let value = price;
  for (let i = 0; i < years; i += 1) {
    tax += Math.round(value * 0.014);
    value = Math.round(value * 0.8);
  }
  return tax;
}

function calcLease(price: number, years: number, taxRate: number): ComparisonResult {
  const factor = LEASE_RATE_FACTORS[years] ?? 1.92;
  const monthlyPayment = Math.round((price * factor) / 100);
  const totalPayment = monthlyPayment * years * 12;
  const taxSaving = Math.round(totalPayment * (taxRate / 100));
  return { monthlyPayment, totalPayment, taxSaving, propertyTax: 0, netBurden: totalPayment - taxSaving };
}

function calcLoan(price: number, years: number, annualRate: number, taxRate: number): ComparisonResult {
  const months = years * 12;
  const monthlyPayment = Math.round(calcPMT(price, annualRate, months));
  const totalPayment = monthlyPayment * months;
  const totalInterest = totalPayment - price;
  const taxSaving = Math.round((totalInterest + price) * (taxRate / 100));
  const propertyTax = calcTotalPropertyTax(price, years);
  return { monthlyPayment, totalPayment, taxSaving, propertyTax, netBurden: totalPayment - taxSaving + propertyTax };
}

function calcCash(price: number, years: number, taxRate: number): ComparisonResult {
  const taxSaving = Math.round(price * (taxRate / 100));
  const propertyTax = calcTotalPropertyTax(price, years);
  return { monthlyPayment: 0, totalPayment: price, taxSaving, propertyTax, netBurden: price - taxSaving + propertyTax };
}

const fmt = (n: number) => `${n.toLocaleString("ja-JP")} 円`;

function recordSimulatorActivity(source: LeasePaymentSimulatorProps["source"], action: string) {
  if (typeof window === "undefined") return;
  const date = new Date().toLocaleDateString("sv-SE");
  const eventId = `lease-simulator:${source}:${action}:${date}`;
  if (window.sessionStorage.getItem(eventId)) return;
  apiClient
    .post("/api/lease-intelligence/activity", {
      surface: `simulator:${source}`,
      action,
      event_id: eventId,
    })
    .then(() => window.sessionStorage.setItem(eventId, "1"))
    .catch(() => {});
}

export default function LeasePaymentSimulator({
  source,
  initialPriceMillion = 10,
  initialYears = 5,
  compact = false,
  defaultOpen = true,
  title = "取得方法シミュレーション",
  description = "リース / 融資 / 現金購入の実質負担を比較します。",
}: LeasePaymentSimulatorProps) {
  const [priceMillion, setPriceMillion] = useState(initialPriceMillion || 10);
  const [years, setYears] = useState(initialYears || 5);
  const [loanRate, setLoanRate] = useState(2.5);
  const [taxRate, setTaxRate] = useState(30);
  const interactionRecordedRef = useRef(false);

  useEffect(() => {
    recordSimulatorActivity(source, "simulator_view");
  }, [source]);

  useEffect(() => {
    if (initialPriceMillion && initialPriceMillion > 0) {
      setPriceMillion(initialPriceMillion);
    }
  }, [initialPriceMillion]);

  useEffect(() => {
    if (initialYears && LEASE_RATE_FACTORS[initialYears]) {
      setYears(initialYears);
    }
  }, [initialYears]);

  const recordInteraction = () => {
    if (interactionRecordedRef.current) return;
    interactionRecordedRef.current = true;
    recordSimulatorActivity(source, "simulator_input_changed");
  };

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
    <details
      id="lease-payment-simulator"
      open={defaultOpen}
      className="group overflow-hidden rounded-2xl border border-sky-200 bg-white shadow-sm"
    >
      <summary className="flex cursor-pointer list-none items-center gap-3 bg-sky-50 px-4 py-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-white text-sky-700 shadow-sm">
          <Calculator className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <h3 className="text-sm font-black text-slate-900">{title}</h3>
          <p className="mt-0.5 text-xs font-bold leading-relaxed text-slate-500">{description}</p>
        </div>
        <ChevronDown className="ml-auto h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" />
      </summary>

      <div className="space-y-4 border-t border-sky-100 p-4">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <div>
            <label className="block text-xs font-black text-slate-600 mb-1">物件価格（百万円）</label>
            <input
              type="number"
              value={priceMillion}
              onChange={(e) => {
                recordInteraction();
                const v = Number.parseFloat(e.target.value);
                setPriceMillion(Number.isNaN(v) ? 0 : v);
              }}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-sky-400"
              step={0.1}
              min={0.1}
            />
            {!compact && <p className="mt-1 text-xs text-slate-400">{price.toLocaleString("ja-JP")} 円</p>}
          </div>
          <div>
            <label className="block text-xs font-black text-slate-600 mb-1">期間</label>
            <select
              value={years}
              onChange={(e) => {
                recordInteraction();
                setYears(Number(e.target.value));
              }}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-sky-400"
            >
              {[2, 3, 4, 5, 6, 7].map((year) => (
                <option key={year} value={year}>{year}年</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs font-black text-slate-600 mb-1">融資金利（年率 %）</label>
            <input
              type="number"
              value={loanRate}
              onChange={(e) => {
                recordInteraction();
                setLoanRate(Math.max(0, Number(e.target.value)));
              }}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-sky-400"
              step={0.1}
              min={0}
            />
          </div>
          <div>
            <label className="block text-xs font-black text-slate-600 mb-1">実効税率（%）</label>
            <input
              type="number"
              value={taxRate}
              onChange={(e) => {
                recordInteraction();
                setTaxRate(Math.min(60, Math.max(0, Number(e.target.value))));
              }}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-sky-400"
              step={1}
              min={0}
              max={60}
            />
          </div>
        </div>

        <div className="rounded-xl border border-sky-200 bg-sky-50 p-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <Trophy className="h-5 w-5 text-sky-700" />
              <div>
                <p className="text-[10px] font-black uppercase tracking-wider text-sky-600">実質負担が最も低い選択肢</p>
                <p className="text-xl font-black text-sky-800">{bestOption.label}</p>
              </div>
            </div>
            <div className="text-left sm:text-right">
              <p className="text-xs font-bold text-sky-600">税効果考慮後</p>
              <p className="text-lg font-black text-slate-900">{fmt(bestOption.value)}</p>
            </div>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {options.map((option) => (
              <div
                key={option.label}
                className={`rounded-lg border px-3 py-2 ${
                  option.label === bestOption.label ? "border-sky-300 bg-white" : "border-slate-200 bg-white/70"
                }`}
              >
                <p className="text-[10px] font-black text-slate-400">{option.label}</p>
                <p className={`mt-0.5 text-sm font-black ${option.label === bestOption.label ? "text-sky-700" : "text-slate-700"}`}>
                  {fmt(option.value)}
                </p>
              </div>
            ))}
          </div>
        </div>

        {!compact && (
          <div className="overflow-x-auto rounded-xl border border-slate-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50 text-xs text-slate-500">
                  <th className="w-52 px-4 py-3 text-left">項目</th>
                  <th className="px-4 py-3 text-right">リース</th>
                  <th className="px-4 py-3 text-right">融資購入</th>
                  <th className="px-4 py-3 text-right">現金購入</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => {
                  const values = [row.lease, row.loan, row.cash];
                  const bestValue = row.negate ? Math.max(...values) : Math.min(...values.filter((value) => value > 0));
                  return (
                    <tr key={row.label} className={row.isHighlight ? "border-t-2 border-sky-200 bg-sky-50" : index % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                      <td className={`px-4 py-3 font-medium ${row.isHighlight ? "text-sky-800 font-black" : "text-slate-600"}`}>
                        {row.label}
                      </td>
                      {[
                        { value: row.lease, key: "lease" },
                        { value: row.loan, key: "loan" },
                        { value: row.cash, key: "cash" },
                      ].map(({ value, key }) => {
                        const isBest = value > 0 && value === bestValue;
                        return (
                          <td key={key} className={`px-4 py-3 text-right font-mono ${isBest ? "font-black text-sky-700" : "text-slate-700"}`}>
                            {value === 0 ? <span className="text-slate-300">-</span> : fmt(value)}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-500">
          リース料率は標準料率テーブル（{years}年 = {LEASE_RATE_FACTORS[years]}%/月）。融資は元利均等、固定資産税は評価額逓減の概算です。実際の税務判断は税理士・自治体確認が前提です。
        </div>
      </div>
    </details>
  );
}
