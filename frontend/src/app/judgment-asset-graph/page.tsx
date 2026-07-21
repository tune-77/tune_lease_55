"use client";

import React from "react";
import { Network, ExternalLink, Sparkles } from "lucide-react";

// 判断資産グラフ（紫苑の成長）
// scripts/build_judgment_asset_graph.py が夜間に生成する自己完結HTMLを
// frontend/public/judgment-asset-graph/index.html として配信し、iframe で表示する。
// このグラフは evaluate_shion_growth.py の期間成長判定も内包する。
const GRAPH_SRC = "/judgment-asset-graph/index.html";

export default function JudgmentAssetGraphPage() {
  return (
    <div className="flex min-h-[calc(100vh-4rem)] flex-col gap-4 p-4 sm:p-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/30">
            <Network className="h-5 w-5 text-white" />
          </span>
          <div>
            <h1 className="text-lg font-black tracking-tight text-slate-900 dark:text-slate-100">
              判断資産グラフ（紫苑の成長）
            </h1>
            <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
              紫苑が蓄えた判断ルールが、リスク軸・業種・根拠・実案件とどう繋がり、
              どれが実戦で
              <span className="font-bold text-emerald-600 dark:text-emerald-400"> 効いた</span>
              ／
              <span className="font-bold text-rose-600 dark:text-rose-400">覆された</span>
              かを示すネットワーク図。期間ごとの成長判定（育った／在庫は増えたが実戦検証不足 ほか）を内包します。
            </p>
          </div>
        </div>
        <a
          href={GRAPH_SRC}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-[13px] font-bold text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          <ExternalLink className="h-4 w-4" />
          別タブで全画面
        </a>
      </header>

      <p className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-widest text-slate-400">
        <Sparkles className="h-3.5 w-3.5" />
        夜間パイプラインが自動生成・毎日更新
      </p>

      <iframe
        title="判断資産グラフ"
        src={GRAPH_SRC}
        className="w-full flex-1 rounded-xl border border-slate-200 bg-white shadow-sm dark:border-slate-700"
        style={{ minHeight: 520 }}
      />
    </div>
  );
}
