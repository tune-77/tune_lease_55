"use client";

import React from "react";
import Link from "next/link";
import {
  ArrowRight,
  Database,
  HeartHandshake,
  Layers3,
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

const pyramidLayers = [
  {
    title: "紫苑の返答",
    label: "人間が記憶として受け取る",
    body: "差分は内部で使い、判断・確認質問・言い切りの精度として返す",
    width: "w-[46%]",
    tone: "from-violet-600 to-fuchsia-600 text-white",
  },
  {
    title: "AURION CORE",
    label: "数理規律",
    body: "Q_riskや違和感を、減点ではなく確認論点へ変換する",
    width: "w-[62%]",
    tone: "from-emerald-500 to-teal-600 text-white",
  },
  {
    title: "経験ループ",
    label: "人間反応",
    body: "薄い、紫苑らしい、一般論に戻った、を次回へ戻す",
    width: "w-[76%]",
    tone: "from-rose-400 to-pink-500 text-white",
  },
  {
    title: "実践知マップ",
    label: "手順 / 意味 / 判断",
    body: "場面ごとに、何をするか・なぜか・例外時どうするかを索引化",
    width: "w-[90%]",
    tone: "from-indigo-500 to-violet-600 text-white",
  },
  {
    title: "長期記憶",
    label: "Obsidian / Research / Daily",
    body: "会話、審査メモ、調査結果、内省を判断資産として蓄積",
    width: "w-full",
    tone: "from-sky-500 to-cyan-600 text-white",
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

      <section className="mx-auto max-w-5xl px-5 py-8 md:px-8">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <Layers3 className="h-6 w-6 text-violet-600" />
            <h2 className="text-2xl font-black">記憶ピラミッド</h2>
          </div>
          <p className="mt-3 text-sm leading-7 text-slate-600">
            下にあるほど素材に近く、上に行くほど「紫苑の返答」として人間が受け取る形になります。
          </p>

          <div className="mt-6 flex flex-col items-center gap-2">
            {pyramidLayers.map((layer) => (
              <div
                key={layer.title}
                className={`${layer.width} rounded-lg bg-gradient-to-r ${layer.tone} px-4 py-3 text-center shadow-sm`}
              >
                <div className="text-xs font-black uppercase tracking-widest opacity-80">{layer.label}</div>
                <div className="mt-1 text-lg font-black">{layer.title}</div>
                <div className="mt-1 text-xs font-bold leading-5 opacity-90">{layer.body}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
