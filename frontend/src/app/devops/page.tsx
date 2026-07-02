"use client";

import React from "react";
import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Cloud,
  Database,
  GitBranch,
  Lock,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const requiredTech = [
  {
    slot: "Google Cloud アプリケーション実行",
    tech: "Cloud Run（API / Web 分離）",
    detail: "cloudbuild.yaml / .api / .web, scripts/deploy_cloud_run*.sh",
  },
  {
    slot: "Google Cloud AI 技術",
    tech: "Gemini API + ADK (Agent Development Kit)",
    detail: "api/shion_agent.py — LlmAgent + Runner + ツール自律呼び出し",
  },
];

const cycleSteps = [
  {
    title: "企画・開発",
    detail: "ローカルで run_next_stable.sh を使い、Next.js + FastAPI を起動して開発する",
    icon: GitBranch,
    tone: "from-blue-500 to-indigo-600",
  },
  {
    title: "ビルド",
    detail: "Cloud Build（cloudbuild.yaml / .api / .web）でコンテナイメージを作成する",
    icon: RefreshCw,
    tone: "from-amber-500 to-orange-600",
  },
  {
    title: "デプロイ",
    detail: "Cloud Run へ API / Web を分離してデプロイする",
    icon: Cloud,
    tone: "from-emerald-500 to-green-600",
  },
  {
    title: "デモ/本番分離",
    detail: "CLOUDRUN_DATA_MODE=demo で demo.db のみを使い、本体DBには接続しない",
    icon: ShieldCheck,
    tone: "from-teal-500 to-emerald-700",
  },
  {
    title: "事前チェック",
    detail: "check_cloudrun_demo_readiness.py でデプロイ直前の状態を確認する",
    icon: CheckCircle2,
    tone: "from-cyan-500 to-sky-600",
  },
  {
    title: "検疫",
    detail: "Cloud Run上の入力はGCSイベント経由で隔離DB（cloudrun_experience_return.db）へ帰還させる",
    icon: Database,
    tone: "from-orange-600 to-red-700",
  },
  {
    title: "人間承認",
    detail: "/cloudrun-return-review で隔離DB内のデータを人が確認・承認する",
    icon: Sparkles,
    tone: "from-violet-500 to-purple-700",
  },
  {
    title: "昇格 → 次サイクルへ",
    detail: "承認済みデータだけを promote_cloudrun_return_data.py --apply で demo.db へ昇格し、次回開発の改善材料にする",
    icon: ArrowRight,
    tone: "from-fuchsia-500 to-pink-600",
  },
];

export default function DevOpsPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-10 md:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-black text-emerald-800">
                <Sparkles className="h-4 w-4" />
                DevOps × AI Agent Hackathon
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                DevOpsサイクルとしての紫苑
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
                「プロトタイプは作れるが実運用まで持っていけない」という課題に対し、本体データを守ったまま
                Cloud Run 上で実際に動かし続けるためのループを持っています。
              </p>
            </div>
            <div className="grid gap-3 text-sm font-bold text-slate-700 sm:grid-cols-3 lg:w-[520px]">
              <Link
                href="/demo"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                ハッカソンデモ
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/cloudrun-return-review"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                帰還データ検疫
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/shion-memory-system"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                記憶システム
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-5 py-8 md:px-8">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-black">必須技術の充足状況</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {requiredTech.map((row) => (
              <div key={row.slot} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="text-[11px] font-black uppercase tracking-widest text-slate-500">{row.slot}</div>
                <div className="mt-1 text-base font-black text-slate-950">{row.tech}</div>
                <div className="mt-1 text-xs font-bold leading-5 text-slate-500">{row.detail}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-5 pb-8 md:px-8">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <GitBranch className="h-6 w-6 text-emerald-600" />
            <h2 className="text-2xl font-black">開発 → デプロイ → 検疫 → 昇格</h2>
          </div>
          <p className="mt-3 text-sm leading-7 text-slate-600">
            Cloud Run 上で生まれたデータは、無条件に本体DBへ書き戻しません。デモ/本番を分離し、隔離DBでの人間承認を経てから初めて昇格します。
          </p>

          <div className="mt-6 space-y-3">
            {cycleSteps.map((step, index) => (
              <div key={step.title} className="flex items-start gap-4">
                <div
                  className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br ${step.tone} text-white shadow-sm`}
                >
                  <step.icon className="h-5 w-5" />
                </div>
                <div className="flex-1 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                  <div className="text-[11px] font-black uppercase tracking-widest text-slate-400">
                    Step {index + 1}
                  </div>
                  <div className="text-base font-black text-slate-950">{step.title}</div>
                  <p className="mt-1 text-xs font-bold leading-5 text-slate-600">{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-5xl px-5 pb-12 md:px-8">
        <div className="flex items-start gap-3 rounded-lg border border-violet-200 bg-violet-50 p-5">
          <Lock className="mt-0.5 h-5 w-5 flex-shrink-0 text-violet-700" />
          <p className="text-sm font-bold leading-7 text-violet-950">
            本体 <code className="rounded bg-white px-1 py-0.5">data/lease_data.db</code> を昇格先にする場合は、別途明示指定と安全確認が必要です。
            この分離のおかげで、ハッカソン期間中の入力で審査データベースが壊れる事故を防いでいます。
          </p>
        </div>
      </section>
    </main>
  );
}
