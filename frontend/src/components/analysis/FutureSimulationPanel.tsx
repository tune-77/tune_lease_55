"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { TrendingUp } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { apiClient } from "@/lib/api";

type Percentiles = Record<string, number[]>;

type FutureSimulationResponse = {
  available: boolean;
  reason?: string;
  years?: number[];
  sales_percentiles?: Percentiles;
  op_percentiles?: Percentiles;
  deficit_prob?: number;
  final_op_median?: number;
  final_op_worst10?: number;
  method?: string;
  unit?: string;
};

type Props = {
  /** 現在の売上高（百万円。フォーム入力そのままの単位） */
  salesMillionYen: number;
  /** 現在の営業利益（百万円） */
  opProfitMillionYen: number;
  caseId?: string;
  years?: number;
};

type ChartRow = {
  year: string;
  band: [number, number];
  median: number;
};

const DRIFT = 0.01;
const VOLATILITY = 0.15;

/** 千円 → 百万円。API は千円で返すが、画面は百万円で揃える。 */
function toMillion(thousandYen: number): number {
  return thousandYen / 1000;
}

/** Recharts の Tooltip は帯（配列）と中央値（数値）の両方を渡してくる。 */
function formatTooltipValue(value: unknown): string {
  if (Array.isArray(value)) {
    const [low, high] = value as number[];
    return `${Number(low).toFixed(1)} 〜 ${Number(high).toFixed(1)} 百万円`;
  }
  return `${Number(value ?? 0).toFixed(1)} 百万円`;
}

function buildChartRows(years: number[], percentiles: Percentiles): ChartRow[] {
  const low = percentiles["10"] ?? [];
  const high = percentiles["90"] ?? [];
  const median = percentiles["50"] ?? [];
  return years.map((year, index) => ({
    year: `${year}年後`,
    band: [toMillion(low[index] ?? 0), toMillion(high[index] ?? 0)],
    median: toMillion(median[index] ?? 0),
  }));
}

export default function FutureSimulationPanel({
  salesMillionYen,
  opProfitMillionYen,
  caseId,
  years = 5,
}: Props) {
  const [data, setData] = useState<FutureSimulationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [actualYear, setActualYear] = useState("1");
  const [actualSales, setActualSales] = useState("");
  const [actualOpProfit, setActualOpProfit] = useState("");
  const [actualDate, setActualDate] = useState("");
  const [actualSaving, setActualSaving] = useState(false);
  const [actualMessage, setActualMessage] = useState("");

  const fetchSimulation = useCallback(async () => {
    if (!salesMillionYen || salesMillionYen <= 0) {
      setData(null);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await apiClient.post<FutureSimulationResponse>("/api/future-simulation", {
        // API は千円で受ける（CLAUDE.md の数値単位）
        sales: salesMillionYen * 1000,
        op_profit: opProfitMillionYen * 1000,
        drift: DRIFT,
        volatility: VOLATILITY,
        years,
        case_id: caseId || "",
      });
      setData(response.data);
    } catch {
      setError("将来シミュレーションを取得できませんでした。");
    } finally {
      setLoading(false);
    }
  }, [caseId, opProfitMillionYen, salesMillionYen, years]);

  useEffect(() => {
    fetchSimulation();
  }, [fetchSimulation]);

  const saveActual = useCallback(async () => {
    if (!caseId || (!actualSales.trim() && !actualOpProfit.trim())) {
      setActualMessage("売上高または営業利益を入力してください。");
      return;
    }
    const sales = actualSales.trim() ? Number(actualSales) : null;
    const opProfit = actualOpProfit.trim() ? Number(actualOpProfit) : null;
    if ((sales !== null && !Number.isFinite(sales)) || (opProfit !== null && !Number.isFinite(opProfit))) {
      setActualMessage("実績値を数値で入力してください。");
      return;
    }
    setActualSaving(true);
    setActualMessage("");
    try {
      await apiClient.post("/api/future-simulation/actuals", {
        case_id: caseId,
        observed_year: Number(actualYear),
        sales: sales === null ? null : sales * 1000,
        op_profit: opProfit === null ? null : opProfit * 1000,
        observed_at: actualDate,
      });
      setActualMessage("実績をshadow記録しました。審査スコアには影響しません。");
      setActualSales("");
      setActualOpProfit("");
    } catch {
      setActualMessage("実績を記録できませんでした。");
    } finally {
      setActualSaving(false);
    }
  }, [actualDate, actualOpProfit, actualSales, actualYear, caseId]);

  const salesRows = useMemo(
    () => (data?.years && data.sales_percentiles ? buildChartRows(data.years, data.sales_percentiles) : []),
    [data],
  );
  const opRows = useMemo(
    () => (data?.years && data.op_percentiles ? buildChartRows(data.years, data.op_percentiles) : []),
    [data],
  );

  if (!salesMillionYen || salesMillionYen <= 0) return null;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-emerald-600" />
        <h3 className="text-sm font-black text-slate-800">将来シミュレーション（{years}期）</h3>
        {data?.method ? (
          <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-600">
            {data.method.toUpperCase()}
          </span>
        ) : null}
      </div>
      <p className="mt-1 text-xs leading-5 text-slate-500">
        現在の財務を起点に、売上高と営業利益の推移を確率的に試算します。審査スコアには影響しません。
      </p>

      {loading ? <p className="mt-3 text-sm text-slate-500">計算中…</p> : null}
      {error ? <p className="mt-3 text-sm text-rose-600">{error}</p> : null}

      {data?.available ? (
        <>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <Stat label="5期後 営業利益（中央値）" value={`${toMillion(data.final_op_median ?? 0).toFixed(1)} 百万円`} />
            <Stat label="悲観シナリオ（下位10%）" value={`${toMillion(data.final_op_worst10 ?? 0).toFixed(1)} 百万円`} />
            <Stat
              label="最終期の赤字確率"
              value={`${((data.deficit_prob ?? 0) * 100).toFixed(1)} %`}
              emphasis={(data.deficit_prob ?? 0) >= 0.3}
            />
          </div>

          <FanChart title="売上高（百万円）" rows={salesRows} color="#059669" />
          <FanChart title="営業利益（百万円）" rows={opRows} color="#ea580c" />

          <p className="mt-2 text-[11px] leading-5 text-slate-400">
            帯は10〜90パーセンタイル。成長率{(DRIFT * 100).toFixed(0)}%・ボラティリティ
            {(VOLATILITY * 100).toFixed(0)}%の仮定値による試算です。
          </p>

          {caseId ? (
            <div className="mt-4 rounded-lg border border-sky-200 bg-sky-50 p-3">
              <p className="text-xs font-black text-sky-900">後日実績を戻して予測を採点</p>
              <p className="mt-1 text-[11px] leading-5 text-sky-700">
                確定した決算値を入力すると予測誤差をshadow集計します。審査スコア・承認判定は変更しません。
              </p>
              <div className="mt-2 grid gap-2 sm:grid-cols-4">
                <label className="text-[11px] font-bold text-slate-600">
                  予測から
                  <select
                    value={actualYear}
                    onChange={(event) => setActualYear(event.target.value)}
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm"
                  >
                    {Array.from({ length: years }, (_, index) => index + 1).map((year) => (
                      <option key={year} value={year}>{year}年後</option>
                    ))}
                  </select>
                </label>
                <ActualInput label="売上高（百万円）" value={actualSales} onChange={setActualSales} />
                <ActualInput label="営業利益（百万円）" value={actualOpProfit} onChange={setActualOpProfit} />
                <label className="text-[11px] font-bold text-slate-600">
                  実績日
                  <input
                    type="date"
                    value={actualDate}
                    onChange={(event) => setActualDate(event.target.value)}
                    className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm"
                  />
                </label>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={saveActual}
                  disabled={actualSaving}
                  className="rounded-md bg-sky-700 px-3 py-2 text-xs font-bold text-white disabled:opacity-50"
                >
                  {actualSaving ? "記録中…" : "実績を記録"}
                </button>
                {actualMessage ? <p className="text-xs text-sky-800">{actualMessage}</p> : null}
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function ActualInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="text-[11px] font-bold text-slate-600">
      {label}
      <input
        type="number"
        step="0.1"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-2 text-sm"
      />
    </label>
  );
}

function Stat({ label, value, emphasis }: { label: string; value: string; emphasis?: boolean }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <p className="text-[11px] font-bold text-slate-500">{label}</p>
      <p className={`mt-1 text-lg font-black ${emphasis ? "text-rose-600" : "text-slate-900"}`}>{value}</p>
    </div>
  );
}

function FanChart({ title, rows, color }: { title: string; rows: ChartRow[]; color: string }) {
  if (!rows.length) return null;
  return (
    <div className="mt-4">
      <p className="mb-1 text-xs font-bold text-slate-600">{title}</p>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="year" tick={{ fontSize: 11 }} stroke="#94a3b8" />
            <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" width={56} />
            <Tooltip formatter={formatTooltipValue} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Area
              dataKey="band"
              name="10〜90%レンジ"
              stroke="none"
              fill={color}
              fillOpacity={0.15}
              isAnimationActive={false}
            />
            <Line
              dataKey="median"
              name="中央値"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
