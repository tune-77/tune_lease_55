"use client";
import React, { useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import { AlertTriangle, RefreshCw } from "lucide-react";
import ShionGrowthBadge from "@/components/analysis/ShionGrowthBadge";

type LoopProof = {
  proposals?: number;
  applied?: number;
  applied_pct?: number;
  pr_traced?: number;
  distinct_rev?: number;
  period_start?: string;
  period_end?: string;
  weeks?: number;
  per_month?: Record<string, number>;
  gen_date?: string;
  growth_score?: number;
  coverage?: number;
  reuse?: number;
  judgment_change?: number;
  human_align?: number;
  field?: number;
  negative?: number;
  materials?: number;
  inbox?: number;
  active_rules?: number;
  risk_axes?: number;
  concepts?: number;
  user_evidence?: number;
  feedback_total?: number;
  feedback_pct?: number;
  fb_diffs?: number;
  fb_diff_pct?: number;
  fb_other?: number;
  needs_review?: number;
  scoring_status?: string;
  source?: string;
};

const n = (v: number | undefined, d = 0): number => (typeof v === "number" ? v : d);
const JP_MONTH: Record<string, string> = {
  "01": "1月", "02": "2月", "03": "3月", "04": "4月", "05": "5月", "06": "6月",
  "07": "7月", "08": "8月", "09": "9月", "10": "10月", "11": "11月", "12": "12月",
};

function Meter({ label, value, dormant }: { label: string; value: number; dormant?: boolean }) {
  return (
    <div className="grid grid-cols-[7rem_1fr_2.2rem] items-center gap-3 py-1.5 sm:grid-cols-[8rem_1fr_2.2rem]">
      <span className="text-[13px] text-slate-600 dark:text-slate-300">{label}</span>
      <span className="h-2.5 overflow-hidden rounded-full bg-violet-100 dark:bg-slate-700">
        <span
          className={`block h-full rounded-full ${dormant ? "bg-slate-400/60" : "bg-violet-600 dark:bg-violet-400"}`}
          style={{ width: `${Math.max(dormant ? 2 : value, 2)}%` }}
        />
      </span>
      <span className={`text-right text-[13px] font-bold tabular-nums ${dormant ? "text-slate-400" : "text-slate-900 dark:text-slate-100"}`}>
        {Math.round(value)}
      </span>
    </div>
  );
}

export default function LoopProofPage() {
  const [data, setData] = useState<LoopProof | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<LoopProof>("/api/loop-proof");
      setData(res.data);
    } catch {
      setError("集計値を取得できませんでした。API（/api/loop-proof）を確認してください。");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-slate-500">
        <RefreshCw className="mr-2 h-5 w-5 animate-spin" /> 集計中…
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="mx-auto max-w-lg p-10 text-center">
        <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-amber-500" />
        <p className="text-slate-600 dark:text-slate-300">{error ?? "データがありません。"}</p>
        <button onClick={() => void load()} className="mt-4 rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700">
          再取得
        </button>
      </div>
    );
  }

  const d = data;
  const fieldVal = Math.round(n(d.field));
  const months = Object.entries(d.per_month ?? {}).sort(([a], [b]) => a.localeCompare(b));
  const monthMax = months.reduce((m, [, v]) => Math.max(m, v), 1);
  const latestMonth = months.length ? months[months.length - 1][0] : "";

  const tiles: Array<{ k: string; v: number; sub: string; unit?: string; pill?: string }> = [
    { k: "AIの改善提案", v: n(d.proposals), sub: "台帳に記録された改善候補（REV）" },
    { k: "実際に適用", v: n(d.applied), pill: `適用率 ${n(d.applied_pct)}%`, sub: "提案 → コードへ反映まで到達" },
    { k: "PRに紐づく適用", v: n(d.pr_traced), sub: `${n(d.distinct_rev)} の独立REVがPR経路で追跡可能` },
    { k: "人間評価の反映", v: n(d.feedback_total), unit: "件", pill: `PDCA ${Math.round(n(d.feedback_pct))}%`, sub: "全件が次のプロンプトへ反映" },
  ];

  const stages = [
    { n: "01 提案", t: "AIが改善を起票", big: n(d.proposals), dsc: "改善候補を台帳へ", live: true },
    { n: "02 適用", t: "PRで反映", big: n(d.applied), dsc: `うち${n(d.pr_traced)}がPR追跡可`, live: true },
    { n: "03 人間評価", t: "効いた／微妙／外した", big: n(d.feedback_total), dsc: `${n(d.fb_diff_pct)}%が次回応答を変えた`, live: true },
    { n: "04 資産化", t: "判断資産が育つ", big: n(d.materials), dsc: `Materials／Active rules ${n(d.active_rules)}`, live: true },
    { n: "05 実戦検証", t: "実案件で効いたか", big: fieldVal, dsc: "Field validation はこれから", live: false },
  ];

  const meters: Array<[string, number, boolean]> = [
    ["網羅性 Coverage", n(d.coverage), false],
    ["判断変化 proxy", n(d.judgment_change), false],
    ["人間整合 proxy", n(d.human_align), false],
    ["再利用 proxy", n(d.reuse), false],
    ["負のシグナル", n(d.negative), false],
    ["実戦検証 Field", n(d.field), fieldVal === 0],
  ];

  const counts: Array<[number, string]> = [
    [n(d.materials), "判断材料 Materials"],
    [n(d.active_rules), "現役ルール Active"],
    [n(d.user_evidence), "ユーザー根拠"],
    [n(d.concepts), "概念 Concepts"],
    [n(d.risk_axes), "リスク軸"],
    [n(d.inbox), "Inbox候補"],
  ];

  const diffPct = n(d.fb_diff_pct);
  const otherPct = Math.round((100 - diffPct) * 10) / 10;

  return (
    <div className="mx-auto max-w-5xl px-5 py-8 tabular-nums sm:px-8 lg:py-12">
      {/* header */}
      <p className="mb-3 text-xs font-bold uppercase tracking-[0.16em] text-violet-700 dark:text-violet-300">
        紫苑 — 判断資産 DevOps ループ ／ 審査員向け証拠
      </p>
      <h1 className="text-balance text-3xl font-semibold leading-tight text-slate-900 dark:text-slate-50 sm:text-4xl">
        AIが賢くなった話ではない。
        <br />
        <span className="border-b-2 border-violet-400 text-violet-700 dark:text-violet-300">人間の判断が、次の案件へ戻ってきた</span>証拠だ。
      </h1>
      <p className="mt-4 max-w-2xl text-slate-600 dark:text-slate-300">
        「作って終わり」を防ぐ——現場の判断・修正・結果をAIが回収し、PR とプロンプト資産として実際に反映し続けた。以下はすべて実運用ログの実数値。
      </p>
      <div className="mt-5 flex flex-wrap gap-2 text-[12.5px] text-slate-500 dark:text-slate-400">
        <span className="rounded-full border border-slate-200 px-3 py-1 dark:border-slate-700">
          計測期間 {d.period_start} → {(d.period_end ?? "").slice(5)}（約{n(d.weeks)}週間）
        </span>
        <span className="rounded-full border border-slate-200 px-3 py-1 dark:border-slate-700">必須技術 Cloud Run ／ Gemini API ／ ADK</span>
        <span className="rounded-full border border-slate-200 px-3 py-1 dark:border-slate-700">盛らず、実ログ集計のみ</span>
      </div>

      {/* 紫苑の成長判定 */}
      <div className="mt-6">
        <ShionGrowthBadge />
      </div>

      {/* tiles */}
      <div className="mt-8 grid grid-cols-2 gap-3.5 lg:grid-cols-4">
        {tiles.map((t) => (
          <div key={t.k} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800/60">
            <p className="mb-2 text-xs font-semibold text-slate-500 dark:text-slate-400">{t.k}</p>
            <div className="text-4xl font-semibold leading-none text-slate-900 dark:text-slate-50">
              {t.v}
              {t.unit ? <span className="ml-0.5 text-base font-semibold text-slate-400">{t.unit}</span> : null}
            </div>
            {t.pill ? (
              <span className="mt-2 inline-block rounded-md bg-emerald-50 px-2 py-0.5 text-xs font-bold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                {t.pill}
              </span>
            ) : null}
            <p className="mt-2 text-[12.5px] text-slate-600 dark:text-slate-300">{t.sub}</p>
          </div>
        ))}
      </div>

      {/* loop strip */}
      <h2 className="mb-4 mt-10 text-xs font-bold uppercase tracking-[0.13em] text-slate-500 dark:text-slate-400">
        閉ループの各段が、実データで点灯している
      </h2>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5 md:gap-0">
        {stages.map((s, i) => (
          <div
            key={s.n}
            className={`relative flex flex-col gap-1.5 border p-4 md:rounded-none ${
              i === 0 ? "md:rounded-l-2xl" : ""
            } ${i === stages.length - 1 ? "md:rounded-r-2xl" : ""} rounded-xl ${
              s.live
                ? "border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800/60"
                : "border-dashed border-slate-300 bg-slate-50 dark:border-slate-600 dark:bg-slate-800/30"
            }`}
          >
            <span
              className={`absolute right-3 top-3 rounded px-1.5 py-0.5 text-[10.5px] font-bold ${
                s.live
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                  : "bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300"
              }`}
            >
              {s.live ? "稼働" : "次の点火点"}
            </span>
            <span className="text-[11px] font-bold tracking-wider text-slate-400">{s.n}</span>
            <span className="text-[13.5px] font-bold text-slate-900 dark:text-slate-100">{s.t}</span>
            <span className={`text-2xl font-semibold ${s.live ? "text-violet-700 dark:text-violet-300" : "text-slate-400"}`}>{s.big}</span>
            <span className="mt-auto text-xs text-slate-500 dark:text-slate-400">{s.dsc}</span>
          </div>
        ))}
      </div>

      {/* two columns */}
      <div className="mt-8 grid grid-cols-1 gap-5 lg:grid-cols-[1.15fr_0.85fr]">
        {/* growth */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-800/60">
          <h3 className="text-[15px] font-bold text-slate-900 dark:text-slate-100">
            判断資産の成長スコア <span className="font-semibold text-violet-700 dark:text-violet-300">{n(d.growth_score)}</span>
          </h3>
          <p className="mb-4 mt-1 text-[12.5px] text-slate-500 dark:text-slate-400">
            {d.gen_date} 時点・日次トラッキング。人間判断から蒸留した構成要素の充足度。
          </p>
          {meters.map(([label, value, dormant]) => (
            <Meter key={label} label={label} value={value} dormant={dormant} />
          ))}
          <div className="mt-4 grid grid-cols-3 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-700">
            {counts.map(([cn, cl], i) => (
              <div key={cl} className={`bg-white p-3 dark:bg-slate-800/60 ${i % 3 !== 2 ? "border-r" : ""} ${i < 3 ? "border-b" : ""} border-slate-200 dark:border-slate-700`}>
                <div className="text-2xl font-semibold leading-none text-slate-900 dark:text-slate-100">{cn}</div>
                <div className="mt-1.5 text-[11.5px] text-slate-500 dark:text-slate-400">{cl}</div>
              </div>
            ))}
          </div>
        </div>

        {/* feedback + monthly */}
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-800/60">
          <h3 className="text-[15px] font-bold text-slate-900 dark:text-slate-100">人間評価 → 応答の変化</h3>
          <p className="mb-3 mt-1 text-[12.5px] text-slate-500 dark:text-slate-400">
            {n(d.feedback_total)}件が全件PDCAに反映。うち実際に前回応答からの差分を生んだ割合。
          </p>
          <div className="flex h-6 overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700">
            <span className="h-full bg-violet-600 dark:bg-violet-500" style={{ width: `${diffPct}%` }} />
            <span className="h-full bg-violet-100 dark:bg-slate-700" style={{ width: `${otherPct}%` }} />
          </div>
          <div className="mt-3 flex flex-wrap gap-4 text-[12.5px] text-slate-600 dark:text-slate-300">
            <span className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-sm bg-violet-600" /> 応答が変化 {n(d.fb_diffs)}件 <b className="ml-0.5">{diffPct}%</b>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-3 w-3 rounded-sm bg-violet-100 dark:bg-slate-600" /> 反映・現状維持 {n(d.fb_other)}件
            </span>
          </div>

          <h3 className="mt-6 text-[15px] font-bold text-slate-900 dark:text-slate-100">適用REVの推移</h3>
          <p className="mb-3 mt-1 text-[12.5px] text-slate-500 dark:text-slate-400">月別の「適用済み」件数。最新月は途中集計。</p>
          <div className="flex flex-col gap-3.5">
            {months.map(([ym, cnt]) => {
              const mo = ym.split("-")[1];
              const partial = ym === latestMonth;
              return (
                <div key={ym} className="grid grid-cols-[3.5rem_1fr] items-center gap-3">
                  <span className="text-[13px] text-slate-600 dark:text-slate-300">
                    {JP_MONTH[mo] ?? ym}
                    {partial ? <span className="ml-1 text-[10px] text-slate-400">※途中</span> : null}
                  </span>
                  <div className="h-[30px]">
                    <div
                      className={`flex h-full items-center justify-end rounded px-2 text-[12.5px] font-bold text-white ${partial ? "bg-violet-400" : "bg-violet-600"}`}
                      style={{ width: `${Math.max((cnt / monthMax) * 100, 12)}%` }}
                    >
                      {cnt}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* honest gaps */}
      <div className="mt-8 rounded-xl border border-slate-200 border-l-4 border-l-amber-500 bg-white p-5 dark:border-slate-700 dark:bg-slate-800/60">
        <h3 className="mb-3 text-sm font-bold text-slate-900 dark:text-slate-100">正直な現在地 — ここが次の伸びしろ</h3>
        <ul className="flex flex-col gap-2.5 text-[13.5px] text-slate-600 dark:text-slate-300">
          <li className="flex gap-2.5">
            <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-amber-500" />
            <span>
              <b className="text-slate-900 dark:text-slate-100">実戦検証がまだ{fieldVal}件。</b> 蒸留した{n(d.active_rules)}ルールが「実案件で本当に効いたか」を回収する段（Field validation）は未点灯。ここが点けばループが一周する。
            </span>
          </li>
          <li className="flex gap-2.5">
            <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-amber-500" />
            <span>
              <b className="text-slate-900 dark:text-slate-100">レビュー圧が高い。</b> {n(d.needs_review)}件が needs-review で滞留。適用スピードに対し人間承認がボトルネック。
            </span>
          </li>
          {d.scoring_status && d.scoring_status !== "ok" ? (
            <li className="flex gap-2.5">
              <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-amber-500" />
              <span>
                <b className="text-slate-900 dark:text-slate-100">スコアリング健全性に要注意フラグ。</b> ローカルのRFモデル読込でヘルスチェックが {d.scoring_status}。デモ本番前に再学習/検証で解消予定。
              </span>
            </li>
          ) : null}
        </ul>
      </div>

      <p className="mt-6 text-[11.5px] text-slate-400">
        出典: <code className="rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">scripts/improvement_ledger.jsonl</code>{" "}
        <code className="rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">reports/*_latest.md</code> を{" "}
        <code className="rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">scripts/build_loop_proof.py</code> が機械集計（{d.source ?? "live"}）。誇張・推測なし。
      </p>
    </div>
  );
}
