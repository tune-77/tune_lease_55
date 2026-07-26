"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  BarChart3,
  CheckCircle2,
  Database,
  RefreshCw,
  ShieldCheck,
  UsersRound,
} from "lucide-react";
import type { AudienceSummary } from "@/lib/audienceSurvey";

const emptySummary: AudienceSummary = {
  total: 0,
  storage: "jsonl",
  quiz_correct: 0,
  quiz_correct_rate: 0,
  counts: {
    quiz_answer: {},
    product_reaction: {},
    adoption_interest: {},
    poc_interest: {},
    macbook_vote: {},
  },
  leads: 0,
  trial_interest: 0,
  recent_insights: [],
};

const demoSummary: AudienceSummary = {
  total: 96,
  storage: "firestore",
  quiz_correct: 61,
  quiz_correct_rate: 64,
  counts: {
    quiz_answer: {
      条件付きなら通せる: 61,
      通せる: 18,
      要警戒: 14,
      否認寄り: 3,
    },
    product_reaction: {
      現場で欲しい: 48,
      実用的で面白い: 31,
      アイデアは良い: 14,
      ピンとこない: 3,
    },
    adoption_interest: {
      検証してみたい: 43,
      今すぐ使いたい: 22,
      興味はある: 27,
      不要: 4,
    },
    poc_interest: {
      自社データで試したい: 25,
      開発者と話したい: 34,
      情報収集のみ: 30,
      今は不要: 7,
    },
    macbook_vote: {
      買ってあげて: 68,
      賞金次第: 21,
      "7年前のiMacで頑張れ": 7,
    },
  },
  leads: 18,
  trial_interest: 65,
  recent_insights: [
    {
      created_at: "2026-07-26T10:00:00.000Z",
      text: "受注額だけでなく、正式発注書と入金時期が返済原資に変わるかを見るべき。",
      company: "会場コメント",
    },
    {
      created_at: "2026-07-26T10:01:00.000Z",
      text: "開発機械の代替利用と再リース可能性は、スタートアップ案件では特に重要。",
      company: "会場コメント",
    },
    {
      created_at: "2026-07-26T10:02:00.000Z",
      text: "Findy案件なら前向きだが、発注確度と検収条件の確認は残したい。",
      company: "会場コメント",
    },
  ],
};

function StatTile({
  label,
  value,
  note,
}: {
  label: string;
  value: string | number;
  note: string;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-3xl font-bold text-slate-950">{value}</p>
      <p className="mt-1 text-sm text-slate-600">{note}</p>
    </section>
  );
}

function CountBars({
  title,
  counts,
  accent = "bg-cyan-600",
}: {
  title: string;
  counts: Record<string, number>;
  accent?: string;
}) {
  const rows = useMemo(
    () => Object.entries(counts).sort((a, b) => b[1] - a[1]),
    [counts],
  );
  const max = Math.max(1, ...rows.map(([, count]) => count));

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-900">
        <BarChart3 className="h-5 w-5 text-cyan-700" />
        {title}
      </h2>
      <div className="space-y-3">
        {rows.length ? (
          rows.map(([label, count]) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-semibold text-slate-800">{label}</span>
                <span className="tabular-nums text-slate-600">{count}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div
                  className={`h-full rounded-full ${accent}`}
                  style={{ width: `${Math.max(8, (count / max) * 100)}%` }}
                />
              </div>
            </div>
          ))
        ) : (
          <p className="text-sm text-slate-500">まだ回答がありません。</p>
        )}
      </div>
    </section>
  );
}

function AudienceResultsContent() {
  const searchParams = useSearchParams();
  const demoMode = searchParams.get("demo") === "1";
  const [summary, setSummary] = useState<AudienceSummary>(demoMode ? demoSummary : emptySummary);
  const [updatedAt, setUpdatedAt] = useState("");
  const [error, setError] = useState("");

  const openFindySceneInChat = () => {
    window.localStorage.setItem("audience-findy-chat-scene", "1");
    window.location.href = "/chat";
  };

  useEffect(() => {
    if (demoMode) {
      setSummary(demoSummary);
      setUpdatedAt(new Date().toLocaleTimeString("ja-JP"));
      setError("");
      return;
    }

    let active = true;

    const load = async () => {
      try {
        const res = await fetch("/api/audience/summary", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "集計の取得に失敗しました");
        if (!active) return;
        setSummary(data);
        setUpdatedAt(new Date().toLocaleTimeString("ja-JP"));
        setError("");
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "集計の取得に失敗しました");
      }
    };

    load();
    const timer = window.setInterval(load, 1000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [demoMode]);

  const trialRate = summary.total ? Math.round((summary.trial_interest / summary.total) * 100) : 0;
  const leadRate = summary.total ? Math.round((summary.leads / summary.total) * 100) : 0;

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 sm:py-10">
      <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div
            className={[
              "mb-3 inline-flex items-center gap-2 rounded-md border px-3 py-1 text-xs font-bold",
              demoMode
                ? "border-amber-200 bg-amber-50 text-amber-800"
                : "border-cyan-200 bg-cyan-50 text-cyan-800",
            ].join(" ")}
          >
            <RefreshCw className="h-4 w-4" />
            {demoMode ? "DEMO BACKUP" : "LIVE"} / {updatedAt || "待機中"}
          </div>
          <h1 className="text-3xl font-bold text-slate-950 sm:text-4xl">
            会場アンケート リアルタイム集計
          </h1>
          <p className="mt-2 text-sm text-slate-600">
            {demoMode
              ? "通信不調時の説明用デモ集計です。実運用時はLIVE画面に切り替えます。"
              : `QR回答が入るたびに自動更新します。保存先: ${summary.storage}`}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link
            href="/audience"
            className="inline-flex h-10 items-center justify-center rounded-md border border-slate-200 bg-white px-4 text-sm font-bold text-slate-700"
          >
            回答画面へ
          </Link>
          <Link
            href={demoMode ? "/audience/results" : "/audience/results?demo=1"}
            className="inline-flex h-10 items-center justify-center rounded-md border border-slate-200 bg-white px-4 text-sm font-bold text-slate-700"
          >
            {demoMode ? "LIVEへ戻す" : "デモ表示"}
          </Link>
        </div>
      </header>

      {error ? (
        <div className="mb-5 rounded-md border border-rose-200 bg-rose-50 p-3 text-sm font-semibold text-rose-700">
          {error}
        </div>
      ) : null}

      <section className="mb-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="回答数" value={summary.total} note="会場からの送信件数" />
        <StatTile
          label="目利き一致率"
          value={`${summary.quiz_correct_rate}%`}
          note={`${summary.quiz_correct}人が紫苑と一致`}
        />
        <StatTile label="導入・検証意向" value={`${trialRate}%`} note={`${summary.trial_interest}人`} />
        <StatTile label="リード候補" value={`${leadRate}%`} note={`${summary.leads}件の任意連絡先`} />
      </section>

      <section className="mb-5 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-bold text-slate-900">
              <ShieldCheck className="h-5 w-5 text-emerald-700" />
              Findy案件の人間判断
            </h2>
            <p className="mt-2 text-sm leading-7 text-slate-600">
              紫苑の判定は条件付き承認。ここで人間が会場文脈を入れて判断を上書きします。
            </p>
          </div>
          <button
            type="button"
            onClick={openFindySceneInChat}
            className="inline-flex h-11 items-center justify-center rounded-md bg-slate-950 px-4 text-sm font-bold text-white transition hover:bg-slate-800"
          >
            紫苑チャットで確認要請を出す
          </button>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <CountBars title="AI vs 審査員の目利き" counts={summary.counts.quiz_answer} />
        <CountBars
          title="紫苑の着眼点"
          counts={summary.counts.product_reaction}
          accent="bg-emerald-600"
        />
        <CountBars
          title="職場・自社での利用意向"
          counts={summary.counts.adoption_interest}
          accent="bg-indigo-600"
        />
        <CountBars
          title="MacBook投票"
          counts={summary.counts.macbook_vote}
          accent="bg-amber-500"
        />
      </section>

      <section className="mt-5 rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="mb-4 flex items-center gap-2 text-sm font-bold text-slate-900">
          <UsersRound className="h-5 w-5 text-cyan-700" />
          1001本目の違和感
        </h2>
        {summary.recent_insights.length ? (
          <div className="grid gap-3">
            {summary.recent_insights.map((insight, index) => (
              <article key={`${insight.created_at}-${index}`} className="rounded-md bg-slate-50 p-3">
                <p className="text-sm leading-7 text-slate-800">{insight.text}</p>
                {insight.company || insight.contact ? (
                  <p className="mt-2 text-xs font-semibold text-slate-500">
                    {[insight.company, insight.contact].filter(Boolean).join(" / ")}
                  </p>
                ) : null}
              </article>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">自由記述が入るとここに表示されます。</p>
        )}
      </section>

      <section className="mt-5 flex flex-wrap items-center gap-3 text-xs text-slate-500">
        <span className="inline-flex items-center gap-1">
          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
          LLM API不使用
        </span>
        <span className="inline-flex items-center gap-1">
          <Database className="h-4 w-4 text-cyan-700" />
          Firestore保存対応
        </span>
      </section>
    </main>
  );
}

export default function AudienceResultsPage() {
  return (
    <Suspense fallback={<main className="mx-auto max-w-6xl px-4 py-10 text-sm text-slate-600">Loading...</main>}>
      <AudienceResultsContent />
    </Suspense>
  );
}
