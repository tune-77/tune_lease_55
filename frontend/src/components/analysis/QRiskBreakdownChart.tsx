"use client";

import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity } from "lucide-react";

import type { QRiskBreakdown, QRiskBreakdownItem } from "../../lib/shionReview";

// Q_risk の水準に合わせた配色。QRiskPanel の LEVEL_CONFIG と同じ判定（35/60）に揃える。
const levelColor = (total: number) => {
  if (total >= 60) return "#f43f5e"; // rose-500 / 強警戒
  if (total >= 35) return "#fbbf24"; // amber-400 / 要注意
  return "#34d399"; // emerald-400 / 低位
};

type ChartRow = {
  name: string;
  weighted: number;
  contribution: number;
  share: number;
  detail: string;
};

type TooltipPayload = {
  payload?: ChartRow;
};

const BreakdownTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) => {
  const row = active ? payload?.[0]?.payload : undefined;
  if (!row) return null;
  return (
    <div className="max-w-72 rounded-lg border border-violet-200 bg-white p-2.5 text-[11px] font-bold leading-5 text-slate-700 shadow-md">
      <div className="font-black text-violet-800">{row.name}</div>
      <div>
        寄与 {row.weighted.toFixed(1)}点 / 全体の {Math.round(row.share * 100)}%
      </div>
      {row.detail && <div className="mt-1 font-medium text-slate-600">{row.detail}</div>}
    </div>
  );
};

type Props = {
  breakdown?: QRiskBreakdown | null;
};

/**
 * Q_risk（0-100）を構成するルール別寄与を横棒で表示する。
 * 表示専用でスコアには影響しない。素点が 100 を超えて頭打ちした案件では、
 * 内訳合計と表示 Q_risk がずれないよう按分済みの weighted を描画する。
 */
export default function QRiskBreakdownChart({ breakdown }: Props) {
  if (!breakdown) return null;

  const items: QRiskBreakdownItem[] = breakdown.items ?? [];
  const total = Number(breakdown.total ?? 0);
  const color = levelColor(total);

  if (!items.length) {
    return (
      <div className="mt-3 rounded-xl border border-violet-100 bg-white p-3">
        <h4 className="flex items-center gap-1.5 text-[11px] font-black text-violet-700">
          <Activity className="h-3.5 w-3.5" />
          Q_riskの内訳（紫苑が見た違和感の成分）
        </h4>
        <p className="mt-1.5 text-[11px] font-bold text-slate-500">
          Q_risk 加点なし。財務矛盾ルールに該当する成分はありません。
        </p>
      </div>
    );
  }

  const rows: ChartRow[] = items.map((item) => ({
    name: `${item.code} ${item.label}`,
    weighted: Number(item.weighted ?? item.contribution ?? 0),
    contribution: Number(item.contribution ?? 0),
    share: Number(item.share ?? 0),
    detail: item.detail || "",
  }));

  return (
    <div className="mt-3 rounded-xl border border-violet-100 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="flex items-center gap-1.5 text-[11px] font-black text-violet-700">
          <Activity className="h-3.5 w-3.5" />
          Q_riskの内訳（紫苑が見た違和感の成分）
        </h4>
        <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-black text-violet-700">
          Q_risk {total.toFixed(1)}
        </span>
      </div>
      <div className="mt-2" style={{ height: Math.max(96, rows.length * 34 + 24) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 4, right: 16, bottom: 4, left: 4 }}
          >
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 10 }} stroke="#94a3b8" />
            <YAxis
              type="category"
              dataKey="name"
              width={148}
              tick={{ fontSize: 10 }}
              stroke="#94a3b8"
            />
            <Tooltip content={<BreakdownTooltip />} cursor={{ fill: "#f5f3ff" }} />
            <Bar dataKey="weighted" radius={[0, 4, 4, 0]} maxBarSize={18}>
              {rows.map((row) => (
                <Cell key={row.name} fill={color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      {breakdown.clipped && (
        <p className="mt-1 text-[10px] font-bold text-amber-700">
          素点 {Number(breakdown.raw_total ?? 0).toFixed(1)} が上限100で頭打ちのため、内訳は按分表示です。
        </p>
      )}
      <p className="mt-1 text-[10px] font-bold text-slate-400">
        表示専用の分解です。Q_risk はスコアを減点しません（35以上で要注意・60以上で強警戒）。
      </p>
    </div>
  );
}
