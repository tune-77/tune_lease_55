"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiClient } from "@/lib/api";
import { Sprout, ArrowUpRight } from "lucide-react";

// 紫苑の期間成長判定バッジ。
// evaluate_shion_growth.py の判定（育った / 在庫は増えたが実戦検証不足 ほか）を
// /api/loop-proof の growth_judgment（static_data スナップショット経由で本番にも届く）
// から読み、審査画面に「この案件を判断した紫苑の育ち具合」として表示する。
type GrowthJudgment = { label: string; score: number; summary: string };

type Tone = { border: string; bg: string; text: string; bar: string };

const TONES: Record<string, Tone> = {
  "育った": {
    border: "border-emerald-200 dark:border-emerald-500/30",
    bg: "bg-emerald-50 dark:bg-emerald-500/10",
    text: "text-emerald-700 dark:text-emerald-300",
    bar: "bg-emerald-500",
  },
  "育っている途中": {
    border: "border-teal-200 dark:border-teal-500/30",
    bg: "bg-teal-50 dark:bg-teal-500/10",
    text: "text-teal-700 dark:text-teal-300",
    bar: "bg-teal-500",
  },
  "在庫は増えたが実戦検証不足": {
    border: "border-amber-200 dark:border-amber-500/30",
    bg: "bg-amber-50 dark:bg-amber-500/10",
    text: "text-amber-700 dark:text-amber-300",
    bar: "bg-amber-500",
  },
  "判定保留": {
    border: "border-slate-200 dark:border-slate-600",
    bg: "bg-slate-50 dark:bg-slate-800/60",
    text: "text-slate-600 dark:text-slate-300",
    bar: "bg-slate-400",
  },
  "後退・要点検": {
    border: "border-rose-200 dark:border-rose-500/30",
    bg: "bg-rose-50 dark:bg-rose-500/10",
    text: "text-rose-700 dark:text-rose-300",
    bar: "bg-rose-500",
  },
};

const DEFAULT_TONE: Tone = TONES["判定保留"];

export default function ShionGrowthBadge() {
  const [gj, setGj] = useState<GrowthJudgment | null>(null);

  useEffect(() => {
    let alive = true;
    apiClient
      .get("/api/loop-proof")
      .then((r) => {
        if (alive) setGj((r.data?.growth_judgment as GrowthJudgment) ?? null);
      })
      .catch(() => {
        if (alive) setGj(null);
      });
    return () => {
      alive = false;
    };
  }, []);

  if (!gj?.label) return null;

  const tone = TONES[gj.label] ?? DEFAULT_TONE;
  const score = Math.max(0, Math.min(100, Number(gj.score) || 0));

  return (
    <section className={`rounded-2xl border ${tone.border} ${tone.bg} p-3 shadow-sm`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Sprout className={`h-4 w-4 ${tone.text}`} />
          <span className="text-[11px] font-black uppercase tracking-widest text-slate-500 dark:text-slate-400">
            この案件を判断した紫苑の育ち
          </span>
        </div>
        <Link
          href="/judgment-asset-graph"
          className="inline-flex items-center gap-1 text-[11px] font-bold text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100"
        >
          判断資産グラフ
          <ArrowUpRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      <div className="mt-2 flex items-end gap-3">
        <span className={`text-lg font-black leading-none ${tone.text}`}>{gj.label}</span>
        <span className="text-sm font-black tabular-nums text-slate-500 dark:text-slate-400">
          {score.toFixed(1)}
        </span>
      </div>

      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${Math.max(score, 2)}%` }} />
      </div>

      {gj.summary && (
        <p className="mt-2 text-[12px] leading-relaxed text-slate-600 dark:text-slate-400">{gj.summary}</p>
      )}
    </section>
  );
}
