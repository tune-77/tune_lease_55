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
        </>
      ) : null}
    </div>
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
