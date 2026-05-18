"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity,
  BarChart3,
  Building2,
  RefreshCw,
  Target,
  TrendingUp,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { triggerMebuki } from "@/components/layout/FloatingMebuki";

type DepartmentRow = {
  department: string;
  total_count: number;
  won_count: number;
  lost_count: number;
  pending_count: number;
  decided_count: number;
  contract_rate: number | null;
  avg_score: number | null;
  avg_rate: number | null;
  avg_spread: number | null;
  top_industry: string | null;
  contract_rate_diff: number | null;
  avg_score_diff: number | null;
  avg_rate_diff: number | null;
  contract_rate_rank: number | null;
  avg_score_rank: number | null;
  avg_rate_rank: number | null;
};

type SignificanceRow = {
  item?: string;
  test?: string;
  p_value?: number | null;
  effect_size?: number | null;
  significance?: string;
  note?: string;
};

type DepartmentStats = {
  generated_at?: string;
  overall?: {
    total_count?: number;
    won_count?: number;
    lost_count?: number;
    pending_count?: number;
    decided_count?: number;
    contract_rate?: number | null;
    avg_score?: number | null;
    avg_rate?: number | null;
    avg_spread?: number | null;
  };
  departments?: DepartmentRow[];
  industry_keys?: string[];
  industry_composition?: Array<Record<string, number | string>>;
  industry_metrics?: MetricRow[];
  monthly_metrics?: MetricRow[];
  significance_summary?: SignificanceRow[];
};

type MetricRow = {
  department: string;
  industry?: string;
  month?: string;
  count: number;
  won_count: number;
  avg_rate: number | null;
  avg_contract_amount?: number | null;
  avg_contract_amount_million: number | null;
  total_contract_amount: number | null;
  total_contract_amount_million: number | null;
};

const COLORS = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#0891b2", "#dc2626", "#4f46e5", "#64748b"];

const formatNum = (value: number | null | undefined, digits = 1, suffix = "") =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "-";

const formatRateLabel = (value: unknown) => (typeof value === "number" ? `${value.toFixed(1)}%` : "");
const formatScoreLabel = (value: unknown) => (typeof value === "number" ? value.toFixed(1) : "");
const formatAmountLabel = (value: unknown) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  return `${value.toFixed(value >= 10 ? 0 : 1)}M`;
};

const formatDateTime = (value?: string) => {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

export default function DepartmentDashboardPage() {
  const [stats, setStats] = useState<DepartmentStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDepartment, setSelectedDepartment] = useState<string>("");

  const fetchStats = useCallback(async (silent = false) => {
    if (silent) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    try {
      const res = await axios.get<DepartmentStats>("/api/department/stats", {
        headers: { "Cache-Control": "no-cache" },
      });
      setStats(res.data);
      setError(null);
    } catch (err) {
      console.error("Failed to load department stats", err);
      setError("営業部集計データの取得に失敗しました。");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    triggerMebuki("guide", "営業部別の成約率・スコア・業種構成を集計しました。");
    fetchStats(false);
    const timer = window.setInterval(() => fetchStats(true), 30000);
    return () => window.clearInterval(timer);
  }, [fetchStats]);

  const departments = useMemo(() => stats?.departments || [], [stats]);
  const industryKeys = stats?.industry_keys || [];
  const industryComposition = stats?.industry_composition || [];
  const topDepartment = departments.find((d) => d.contract_rate_rank === 1);
  const activeDepartment = selectedDepartment || departments[0]?.department || "";
  const industryMetricRows = useMemo(
    () => (stats?.industry_metrics || []).filter((row) => row.department === activeDepartment).slice(0, 10),
    [activeDepartment, stats?.industry_metrics],
  );
  const monthlyMetricRows = useMemo(
    () => (stats?.monthly_metrics || []).filter((row) => row.department === activeDepartment).slice(-18),
    [activeDepartment, stats?.monthly_metrics],
  );
  const monthlyAllDeptRows = useMemo(() => {
    const rows = stats?.monthly_metrics || [];
    const months = Array.from(new Set(rows.map((r) => r.month).filter(Boolean))) as string[];
    months.sort();
    const recentMonths = months.slice(-18);
    const deptList = departments.map((d) => d.department);
    return recentMonths.map((month) => {
      const entry: Record<string, number | string | null> = { month };
      for (const dept of deptList) {
        const row = rows.find((r) => r.month === month && r.department === dept);
        entry[`${dept}__amount`] = row?.total_contract_amount_million ?? null;
        entry[`${dept}__rate`] = row?.avg_rate ?? null;
      }
      return entry;
    });
  }, [departments, stats?.monthly_metrics]);

  if (loading) {
    return (
      <div className="flex min-h-[calc(100vh-2rem)] items-center justify-center p-6">
        <div className="flex flex-col items-center gap-4">
          <Activity className="h-12 w-12 animate-spin text-emerald-500" />
          <p className="font-bold text-slate-500">営業部別データを集計中...</p>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-[calc(100vh-2rem)] p-4 sm:p-6 lg:p-8">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-black text-emerald-700">
            <Building2 className="h-4 w-4" />
            SALES DEPARTMENT
          </div>
          <h1 className="flex items-center gap-3 text-2xl font-black text-slate-800 sm:text-3xl">
            <Users className="h-8 w-8 text-emerald-600" />
            営業部ダッシュボード
          </h1>
          <p className="mt-2 text-sm font-bold text-slate-500">
            成約率、平均スコア、平均金利、業種構成を営業部別に比較します。
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-500">
            最終更新 {formatDateTime(stats?.generated_at)}
          </span>
          <button
            type="button"
            onClick={() => fetchStats(true)}
            disabled={refreshing}
            className="inline-flex h-10 items-center gap-2 rounded-lg bg-slate-900 px-4 text-sm font-black text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            更新
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700">
          {error}
        </div>
      )}

      {departments.length === 0 ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-6 text-amber-800">
          <h2 className="text-lg font-black">営業部データが不足しています</h2>
          <p className="mt-2 text-sm font-bold">案件に営業担当部署が入ると、このページに営業部別の比較が表示されます。</p>
        </div>
      ) : (
        <div className="space-y-6">
          <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <KpiCard
              icon={<Target className="h-5 w-5" />}
              label="全体成約率"
              value={formatNum(stats?.overall?.contract_rate, 1, "%")}
              sub={`${stats?.overall?.won_count ?? 0} / ${stats?.overall?.decided_count ?? 0} 件`}
              tone="blue"
            />
            <KpiCard
              icon={<BarChart3 className="h-5 w-5" />}
              label="平均スコア"
              value={formatNum(stats?.overall?.avg_score, 1)}
              sub="営業部平均の比較基準"
              tone="emerald"
            />
            <KpiCard
              icon={<TrendingUp className="h-5 w-5" />}
              label="平均金利"
              value={formatNum(stats?.overall?.avg_rate, 2, "%")}
              sub={`平均スプレッド ${formatNum(stats?.overall?.avg_spread, 2, "%")}`}
              tone="orange"
            />
            <KpiCard
              icon={<Users className="h-5 w-5" />}
              label="首位営業部"
              value={topDepartment?.department || "-"}
              sub={`成約率 ${formatNum(topDepartment?.contract_rate, 1, "%")}`}
              tone="violet"
            />
          </section>

          <section className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-black text-slate-800">営業部別 詳細分析</h2>
                <p className="mt-1 text-xs font-bold text-slate-500">業種別・月別の平均金利と成約額合計を確認します。</p>
              </div>
              <select
                value={activeDepartment}
                onChange={(event) => setSelectedDepartment(event.target.value)}
                className="h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm font-bold text-slate-700 outline-none focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
              >
                {departments.map((dept) => (
                  <option key={dept.department} value={dept.department}>
                    {dept.department}
                  </option>
                ))}
              </select>
            </div>
          </section>

          <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <ChartPanel title="営業部別 成約率・件数" subtitle="棒=成約率、背景件数はTooltipで確認">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={departments} margin={{ top: 12, right: 8, left: 0, bottom: 18 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="department" tick={{ fontSize: 11, fontWeight: 700 }} interval={0} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip
                    formatter={(value, name) => [name === "contract_rate" ? `${value}%` : value, name === "contract_rate" ? "成約率" : name]}
                    labelFormatter={(label) => `営業部: ${label}`}
                    contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }}
                  />
                  <Bar dataKey="contract_rate" name="成約率" radius={[6, 6, 0, 0]}>
                    <LabelList dataKey="contract_rate" position="top" formatter={formatRateLabel} fontSize={11} fontWeight={800} fill="#334155" />
                    {departments.map((_, index) => (
                      <Cell key={`rate-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel title="営業部別 平均スコア / 平均金利" subtitle="信用スコアと実行金利の並列比較">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={departments} margin={{ top: 12, right: 8, left: 0, bottom: 18 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="department" tick={{ fontSize: 11, fontWeight: 700 }} interval={0} />
                  <YAxis yAxisId="score" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="rate" orientation="right" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                  <Legend />
                  <Bar yAxisId="score" dataKey="avg_score" name="平均スコア" fill="#2563eb" radius={[6, 6, 0, 0]}>
                    <LabelList dataKey="avg_score" position="top" formatter={formatScoreLabel} fontSize={11} fontWeight={800} fill="#334155" />
                  </Bar>
                  <Bar yAxisId="rate" dataKey="avg_rate" name="平均金利(%)" fill="#f97316" radius={[6, 6, 0, 0]}>
                    <LabelList dataKey="avg_rate" position="top" formatter={formatRateLabel} fontSize={11} fontWeight={800} fill="#334155" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>
          </section>

          <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <ChartPanel title={`${activeDepartment} 業種別 平均金利・成約額合計`} subtitle="金利0・成約額0は除外">
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={industryMetricRows} margin={{ top: 12, right: 8, left: 0, bottom: 42 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="industry" tick={{ fontSize: 10, fontWeight: 700 }} interval={0} angle={-18} textAnchor="end" height={70} />
                  <YAxis yAxisId="rate" tick={{ fontSize: 11 }} unit="%" />
                  <YAxis yAxisId="amount" orientation="right" tick={{ fontSize: 11 }} unit="M" />
                  <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                  <Legend />
                  <Bar yAxisId="amount" dataKey="total_contract_amount_million" name="成約額合計(M)" fill="#16a34a" radius={[6, 6, 0, 0]}>
                    <LabelList dataKey="total_contract_amount_million" position="top" formatter={formatAmountLabel} fontSize={10} fontWeight={800} fill="#334155" />
                  </Bar>
                  <Line yAxisId="rate" type="monotone" dataKey="avg_rate" name="平均金利(%)" stroke="#f97316" strokeWidth={3} dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartPanel>

            <ChartPanel title={`${activeDepartment} 月別 平均金利・成約額合計`} subtitle="直近18か月、金利0・成約額0は除外">
              <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={monthlyMetricRows} margin={{ top: 12, right: 8, left: 0, bottom: 18 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fontWeight: 700 }} interval={0} />
                  <YAxis yAxisId="rate" tick={{ fontSize: 11 }} unit="%" />
                  <YAxis yAxisId="amount" orientation="right" tick={{ fontSize: 11 }} unit="M" />
                  <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                  <Legend />
                  <Bar yAxisId="amount" dataKey="total_contract_amount_million" name="成約額合計(M)" fill="#2563eb" radius={[6, 6, 0, 0]}>
                    <LabelList dataKey="total_contract_amount_million" position="top" formatter={formatAmountLabel} fontSize={10} fontWeight={800} fill="#334155" />
                  </Bar>
                  <Line yAxisId="rate" type="monotone" dataKey="avg_rate" name="平均金利(%)" stroke="#dc2626" strokeWidth={3} dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartPanel>
          </section>

          <section>
            <ChartPanel title="全営業部 月別 平均金利・成約額合計（重ね表示）" subtitle="直近18か月、営業部ごとに色分け（棒=成約額M / 破線=金利%）">
              <ResponsiveContainer width="100%" height={420}>
                <ComposedChart data={monthlyAllDeptRows} margin={{ top: 12, right: 8, left: 0, bottom: 18 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fontWeight: 700 }} interval={0} />
                  <YAxis yAxisId="rate" tick={{ fontSize: 11 }} unit="%" />
                  <YAxis yAxisId="amount" orientation="right" tick={{ fontSize: 11 }} unit="M" />
                  <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {departments.map((dept, index) => (
                    <Bar
                      key={`bar-${dept.department}`}
                      yAxisId="amount"
                      dataKey={`${dept.department}__amount`}
                      name={`${dept.department} 成約額(M)`}
                      fill={COLORS[index % COLORS.length]}
                      radius={[4, 4, 0, 0]}
                    />
                  ))}
                  {departments.map((dept, index) => (
                    <Line
                      key={`line-${dept.department}`}
                      yAxisId="rate"
                      type="monotone"
                      dataKey={`${dept.department}__rate`}
                      name={`${dept.department} 金利(%)`}
                      stroke={COLORS[index % COLORS.length]}
                      strokeWidth={2}
                      strokeDasharray="4 2"
                      dot={{ r: 2 }}
                      connectNulls
                    />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </ChartPanel>
          </section>

          <section className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
            <ChartPanel title="営業部別 業種構成" subtitle="主要業種を件数ベースで積み上げ表示">
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={industryComposition} margin={{ top: 12, right: 8, left: 0, bottom: 18 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="department" tick={{ fontSize: 11, fontWeight: 700 }} interval={0} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #e2e8f0" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {industryKeys.map((key, index) => (
                    <Bar key={key} dataKey={key} stackId="industry" fill={COLORS[index % COLORS.length]} radius={index === industryKeys.length - 1 ? [6, 6, 0, 0] : undefined} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </ChartPanel>

            <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-black text-slate-800">有意性・特徴量サマリ</h2>
              <p className="mt-1 text-xs font-bold text-slate-500">営業部差が見える検定結果と平均との差です。</p>
              <div className="mt-5 space-y-3">
                {(stats?.significance_summary || []).length > 0 ? (
                  stats?.significance_summary?.map((row, index) => (
                    <div key={`${row.item}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-sm font-black text-slate-700">{row.item}</p>
                          <p className="mt-1 text-xs font-bold text-slate-500">{row.test}</p>
                        </div>
                        <span className={`rounded-full px-2 py-1 text-[11px] font-black ${row.significance === "有意" ? "bg-rose-100 text-rose-700" : "bg-emerald-100 text-emerald-700"}`}>
                          {row.significance || "判定外"}
                        </span>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-bold text-slate-600">
                        <span>p値 {formatNum(row.p_value, 4)}</span>
                        <span>効果量 {formatNum(row.effect_size, 3)}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm font-bold text-slate-500">
                    検定に必要な件数が不足しています。
                  </div>
                )}
              </div>
            </div>
          </section>

          <section className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-5 py-4">
              <h2 className="text-lg font-black text-slate-800">営業部別 平均との差・順位</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-[920px] w-full text-left text-sm">
                <thead className="bg-slate-50 text-xs font-black text-slate-500">
                  <tr>
                    <th className="px-4 py-3">営業部</th>
                    <th className="px-4 py-3 text-right">件数</th>
                    <th className="px-4 py-3 text-right">成約率</th>
                    <th className="px-4 py-3 text-right">平均との差</th>
                    <th className="px-4 py-3 text-right">平均スコア</th>
                    <th className="px-4 py-3 text-right">平均との差</th>
                    <th className="px-4 py-3 text-right">平均金利</th>
                    <th className="px-4 py-3 text-right">平均スプレッド</th>
                    <th className="px-4 py-3">最多業種</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {departments.map((dept) => (
                    <tr key={dept.department} className="hover:bg-slate-50">
                      <td className="px-4 py-3 font-black text-slate-800">{dept.department}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-600">{dept.total_count}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-700">{formatNum(dept.contract_rate, 1, "%")}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-600">{formatSigned(dept.contract_rate_diff, "%")}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-700">{formatNum(dept.avg_score, 1)}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-600">{formatSigned(dept.avg_score_diff, "")}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-700">{formatNum(dept.avg_rate, 2, "%")}</td>
                      <td className="px-4 py-3 text-right font-bold text-slate-700">{formatNum(dept.avg_spread, 2, "%")}</td>
                      <td className="px-4 py-3 font-bold text-slate-600">{dept.top_industry || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}

function formatSigned(value: number | null | undefined, suffix: string) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(suffix === "%" ? 1 : 1)}${suffix}`;
}

function KpiCard({
  icon,
  label,
  value,
  sub,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  tone: "blue" | "emerald" | "orange" | "violet";
}) {
  const toneClass = {
    blue: "bg-blue-50 text-blue-600",
    emerald: "bg-emerald-50 text-emerald-600",
    orange: "bg-orange-50 text-orange-600",
    violet: "bg-violet-50 text-violet-600",
  }[tone];

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs font-black uppercase tracking-wide text-slate-500">{label}</p>
          <p className="mt-2 truncate text-2xl font-black text-slate-800">{value}</p>
          <p className="mt-1 text-xs font-bold text-slate-500">{sub}</p>
        </div>
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${toneClass}`}>{icon}</div>
      </div>
    </div>
  );
}

function ChartPanel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <h2 className="text-lg font-black text-slate-800">{title}</h2>
        <p className="mt-1 text-xs font-bold text-slate-500">{subtitle}</p>
      </div>
      <div className="h-[320px] min-w-0 sm:h-[360px]">{children}</div>
    </div>
  );
}
