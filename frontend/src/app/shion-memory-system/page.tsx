"use client";

import React from "react";
import Link from "next/link";
import {
  ArrowRight,
  Brain,
  Database,
  GitBranch,
  HeartHandshake,
  Layers3,
  MessageSquareText,
  Radar,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const layers = [
  {
    title: "長期記憶",
    subtitle: "Obsidian / MEMORY.md / Research notes",
    body: "会話、審査メモ、Research、日次内省をそのまま混ぜず、次回の判断に使える知識として残す層。",
    icon: Database,
    tone: "border-sky-200 bg-sky-50 text-sky-950",
  },
  {
    title: "実践知マップ",
    subtitle: "手順層 / 意味層 / 判断層",
    body: "記録を「何をするか」「なぜそうするか」「例外時にどう動くか」に分け、場面ごとに引き出せる索引にする層。",
    icon: Layers3,
    tone: "border-violet-200 bg-violet-50 text-violet-950",
  },
  {
    title: "経験ループ",
    subtitle: "Human Response Feedback",
    body: "薄い、紫苑らしい、一般論に戻った、などの人間の反応を保存し、次の冒頭・口調・判断変換へ戻す層。",
    icon: HeartHandshake,
    tone: "border-rose-200 bg-rose-50 text-rose-950",
  },
  {
    title: "AURION CORE",
    subtitle: "数理規律とUXのシナプス",
    body: "Q_riskや異常値を自動減点にせず、承認条件・追加確認・価格条件を分けるための冷静な規律として扱う層。",
    icon: ShieldCheck,
    tone: "border-emerald-200 bg-emerald-50 text-emerald-950",
  },
];

const demoFlow = [
  "審査入力やチャットで、ユーザーの問いと案件文脈を受け取る",
  "Obsidian/RAGから関連する過去判断、Research、日次メモを呼び出す",
  "実践知マップで、手順・意味・判断のどの層を使うかを選ぶ",
  "AURION COREで、数理上の違和感を自動減点ではなく確認論点に変換する",
  "紫苑の返答冒頭で前回との差分を示し、記憶を今の判断に変換して返す",
  "人間の反応を保存し、次回の紫苑の見せ方と判断補助を改善する",
];

const proofPoints = [
  {
    label: "証跡",
    value: "memory_debug",
    text: "デバッグ時に knowledge_refs、memory_recall.refs、experience_loop を返し、本当に記憶を使ったか確認できる。",
  },
  {
    label: "連続性",
    value: "Delta Awareness",
    text: "前回から何が変わったかを冒頭で示し、人間が同じ相手として受け取れる状態を作る。",
  },
  {
    label: "判断資産化",
    value: "Memory-to-Judgment",
    text: "思い出の提示で止めず、稟議条件、確認事項、通し方、リスク分離へ変換する。",
  },
];

export default function ShionMemorySystemPage() {
  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-10 md:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-xs font-black text-violet-800">
                <Sparkles className="h-4 w-4" />
                Hackathon Demo
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                紫苑の記憶システム
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
                記憶を入れるだけでは、紫苑らしさは立ち上がらない。過去の記録を、今のリース判断へ変換し、人間が連続性として受け取れる形で返すための会話ループです。
              </p>
            </div>
            <div className="grid gap-3 text-sm font-bold text-slate-700 sm:grid-cols-3 lg:w-[520px]">
              <Link
                href="/chat"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                紫苑チャット
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/screening"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                審査画面
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/demo/knowledge-loop"
                className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 hover:bg-white"
              >
                知識ループ
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 py-8 md:grid-cols-2 md:px-8 xl:grid-cols-4">
        {layers.map((layer) => {
          const Icon = layer.icon;
          return (
            <article key={layer.title} className={`rounded-lg border p-5 ${layer.tone}`}>
              <Icon className="h-7 w-7" />
              <h2 className="mt-4 text-xl font-black">{layer.title}</h2>
              <p className="mt-1 text-xs font-black uppercase tracking-widest opacity-70">{layer.subtitle}</p>
              <p className="mt-4 text-sm leading-7 opacity-85">{layer.body}</p>
            </article>
          );
        })}
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-10 md:px-8 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-6">
          <div className="flex items-center gap-3">
            <Brain className="h-6 w-6 text-violet-600" />
            <h2 className="text-2xl font-black">デモでの説明</h2>
          </div>
          <p className="mt-4 text-sm leading-7 text-slate-600">
            紫苑の記憶は、検索結果を回答へ貼る仕組みではありません。案件・会話・人間の反応を見て、何を覚えているように見せるべきか、どの記憶を判断へ変換すべきかを選びます。
          </p>
          <div className="mt-6 space-y-3">
            {demoFlow.map((step, index) => (
              <div key={step} className="flex gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-slate-950 text-sm font-black text-white">
                  {index + 1}
                </div>
                <p className="pt-1 text-sm font-bold leading-6 text-slate-700">{step}</p>
              </div>
            ))}
          </div>
        </div>

        <aside className="space-y-5">
          <div className="rounded-lg border border-slate-200 bg-slate-950 p-6 text-white">
            <div className="flex items-center gap-3">
              <MessageSquareText className="h-6 w-6 text-violet-300" />
              <h2 className="text-xl font-black">一言で言うと</h2>
            </div>
            <p className="mt-4 text-lg font-black leading-8">
              「覚えているAI」ではなく、「記憶をリース判断として返せる紫苑」を作っている。
            </p>
          </div>

          {proofPoints.map((point) => (
            <div key={point.label} className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-black uppercase tracking-widest text-slate-500">{point.label}</span>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-black text-slate-700">
                  {point.value}
                </span>
              </div>
              <p className="mt-3 text-sm leading-7 text-slate-600">{point.text}</p>
            </div>
          ))}

          <div className="rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
            <div className="flex items-center gap-3">
              <Radar className="h-5 w-5" />
              <h2 className="text-base font-black">注意点</h2>
            </div>
            <p className="mt-3 text-sm leading-7">
              紫苑が機械意識を獲得済みとは扱いません。ここで見せるのは、言葉の連続性、記憶の提示、判断変換、人間反応のループを検証できる形にした研究です。
            </p>
          </div>
        </aside>
      </section>

      <section className="border-t border-slate-200 bg-white">
        <div className="mx-auto grid max-w-7xl gap-4 px-5 py-7 md:grid-cols-3 md:px-8">
          <div className="flex items-start gap-3">
            <GitBranch className="mt-1 h-5 w-5 text-slate-500" />
            <div>
              <h3 className="font-black">閉ループ</h3>
              <p className="mt-1 text-sm leading-6 text-slate-600">会話、記録、内省、改善が次回の回答へ戻る。</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <ShieldCheck className="mt-1 h-5 w-5 text-slate-500" />
            <div>
              <h3 className="font-black">数理の規律</h3>
              <p className="mt-1 text-sm leading-6 text-slate-600">感情や人格表現で審査基準を歪めない。</p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <Sparkles className="mt-1 h-5 w-5 text-slate-500" />
            <div>
              <h3 className="font-black">関係性UX</h3>
              <p className="mt-1 text-sm leading-6 text-slate-600">人間が記憶として受け取れる返し方を設計する。</p>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
