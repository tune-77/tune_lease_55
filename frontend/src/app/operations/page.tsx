"use client";

import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cloud,
  Database,
  GitBranch,
  Lock,
  Orbit,
  Settings2,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const operationCards = [
  {
    title: "構成を確認する",
    href: "/system-overview",
    detail: "紫苑を中心に、審査AI・Obsidian・判断資産・改善ループがどう接続されているかを見る。",
    icon: Orbit,
    tone: "from-indigo-500 to-violet-600",
  },
  {
    title: "運用サイクルを見る",
    href: "/devops",
    detail: "Cloud Run、Cloud Build、デモDB分離、検疫、昇格までの運用ループを確認する。",
    icon: GitBranch,
    tone: "from-emerald-500 to-teal-600",
  },
  {
    title: "記憶システムを点検する",
    href: "/shion-memory-system",
    detail: "判断資産候補、記憶レビュー、Memory Engineering の状態を確認する。",
    icon: Database,
    tone: "from-sky-500 to-cyan-600",
  },
];

const runtimeSteps = [
  {
    title: "Cloud Run / local runtime",
    detail: "API・Web・スコアリング・チャットを動かす実行環境。Cloud Run版はデモDBで本体DBを守る。",
    icon: Cloud,
  },
  {
    title: "Brain / judgment assets",
    detail: "Obsidian / Markdown Vault 側に判断資産・違和感・改善ログを保持し、正本は直接書き換えない。",
    icon: Database,
  },
  {
    title: "Review / quarantine / promote",
    detail: "改善候補や帰還データは人間レビューを通し、採用・修正採用・保留・却下を残してから昇格する。",
    icon: ShieldCheck,
  },
];

const successMetrics = [
  "変更後30日間の /operations 訪問数",
  "統合画面から /system-overview・/devops・/shion-memory-system へ進んだ回数",
  "改善ログ上の関連情報閲覧数と、success_metric の事後変化",
];

export default function OperationsPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-10 md:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-fuchsia-200 bg-fuchsia-50 px-3 py-1 text-xs font-black text-fuchsia-800">
                <Settings2 className="h-4 w-4" />
                システム管理 / 運用情報
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                低頻度の管理画面を、ここで一度見渡す
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
                システム概要とDevOpsサイクルを個別に探すのではなく、運用で見るべき情報を一画面にまとめます。
                詳細が必要な時だけ、下のカードから元画面へ進みます。
              </p>
            </div>
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm font-bold leading-7 text-emerald-950 lg:w-[360px]">
              <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-emerald-700">
                <CheckCircle2 className="h-4 w-4" />
                採用中の仮説
              </div>
              <p className="mt-2">
                個別アクセスが少ない管理系情報を統合し、到達性と情報閲覧数の増加を30日で確認します。
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-4 px-5 py-8 md:px-8 lg:grid-cols-3">
        {operationCards.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="group rounded-lg border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-fuchsia-200 hover:shadow-md"
          >
            <div className={`inline-flex h-11 w-11 items-center justify-center rounded-lg bg-gradient-to-br ${card.tone} text-white shadow-sm`}>
              <card.icon className="h-5 w-5" />
            </div>
            <div className="mt-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-black text-slate-950">{card.title}</h2>
              <ArrowRight className="h-4 w-4 text-slate-400 transition group-hover:text-fuchsia-600" />
            </div>
            <p className="mt-2 text-sm font-bold leading-7 text-slate-600">{card.detail}</p>
          </Link>
        ))}
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-8 md:px-8 lg:grid-cols-[1fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-fuchsia-600" />
            <h2 className="text-2xl font-black">運用で見るべき中核だけ</h2>
          </div>
          <div className="mt-5 space-y-3">
            {runtimeSteps.map((step) => (
              <div key={step.title} className="flex gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-slate-700 shadow-sm">
                  <step.icon className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-sm font-black text-slate-950">{step.title}</div>
                  <p className="mt-1 text-xs font-bold leading-6 text-slate-600">{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
          <Link
            href="/cloudrun-return-review"
            className="mt-5 inline-flex items-center gap-2 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3 text-sm font-black text-teal-800 hover:bg-teal-100"
          >
            帰還データ検疫を開く
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>

        <aside className="space-y-5">
          <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-5 w-5 text-emerald-600" />
              <h2 className="text-lg font-black">効き方の追跡</h2>
            </div>
            <ul className="mt-4 space-y-3">
              {successMetrics.map((metric) => (
                <li key={metric} className="flex gap-2 text-sm font-bold leading-7 text-slate-600">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                  <span>{metric}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="rounded-lg border border-violet-200 bg-violet-50 p-5">
            <div className="flex gap-3">
              <Lock className="mt-0.5 h-5 w-5 shrink-0 text-violet-700" />
              <p className="text-sm font-bold leading-7 text-violet-950">
                統合は入口を変えるだけです。詳細情報は削除せず、必要な人が深掘りできるよう元画面を残します。
              </p>
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}
