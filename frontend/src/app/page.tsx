"use client";

import Link from "next/link";
import type { ElementType } from "react";
import { useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  Clock,
  Loader2,
  MessageCircle,
  Orbit,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

type ConciergeMessage = {
  role: "shion" | "user";
  text: string;
  guidance?: ConciergeGuidance;
};

type RouteSuggestion = {
  label: string;
  href: string;
  description: string;
  icon: ElementType;
  tone: string;
  keywords: string[];
  nextSteps: string[];
};

type ConciergeGuidance = {
  primary: RouteSuggestion;
  alternatives: RouteSuggestion[];
  reason: string;
  handoff: string;
  persona: ShionPersona;
};

type ActivityItem = {
  path: string;
  title: string;
  ts: number;
};

type PredictedAction = {
  label: string;
  href: string;
  reason: string;
  icon: ElementType;
};

type DailyGreeting = {
  date: string;
  opening: string;
  time_band?: string;
  time_note?: string;
  yesterday: string;
  anniversary: {
    name: string;
    note: string;
  };
  thought: string;
  suggestion: string;
};

type ShionPersona = {
  id: "guide" | "screening" | "research" | "memory" | "demo";
  name: string;
  title: string;
  image: string;
  color: string;
  border: string;
  role: string;
  catchphrase: string;
  keywords: string[];
  routeHrefs: string[];
};

type QueueStatus = "open" | "done" | "later";

type WorkQueueItem = {
  id: string;
  title: string;
  href: string;
  reason: string;
  eta: string;
  persona: ShionPersona;
  route: RouteSuggestion;
};

const ACTIVITY_KEY = "shion-concierge-activity-v1";
const QUEUE_STATE_KEY = "shion-concierge-work-queue-state-v1";
const SHION_AVATAR_IMAGE = "/lease-grumble/characters/lease-intelligence-girl.jpg";

const SHION_PERSONAS: ShionPersona[] = [
  {
    id: "guide",
    name: "案内紫苑",
    title: "入口整理",
    image: SHION_AVATAR_IMAGE,
    color: "from-violet-500 to-fuchsia-600",
    border: "border-violet-300/35",
    role: "最初に話を聞き、作業順序を決める受付役です。",
    catchphrase: "迷ったら、私が入口を整えます。",
    keywords: ["何", "どこ", "始め", "案内", "入口", "ホーム"],
    routeHrefs: ["/", "/home"],
  },
  {
    id: "screening",
    name: "審査紫苑",
    title: "OCR・審査入力",
    image: SHION_AVATAR_IMAGE,
    color: "from-emerald-500 to-teal-600",
    border: "border-emerald-300/35",
    role: "決算書、OCR、財務数値、スコア、軍師AIへの接続を担当します。",
    catchphrase: "まず数字へ落として、判断できる形にします。",
    keywords: ["審査", "決算", "財務", "OCR", "ocr", "スコア", "入力", "稟議"],
    routeHrefs: ["/screening", "/report", "/batch", "/ringi"],
  },
  {
    id: "research",
    name: "調査紫苑",
    title: "外部調査",
    image: SHION_AVATAR_IMAGE,
    color: "from-sky-500 to-cyan-600",
    border: "border-sky-300/35",
    role: "Web調査をResearchノートへ圧縮し、紫苑RAGへ戻します。",
    catchphrase: "判断前に、外の情報を一本入れましょう。",
    keywords: ["調査", "Research", "research", "外部", "検索", "市況", "ニュース"],
    routeHrefs: ["/research-organ"],
  },
  {
    id: "memory",
    name: "記憶紫苑",
    title: "前回行動・判断資産",
    image: SHION_AVATAR_IMAGE,
    color: "from-slate-600 to-slate-800",
    border: "border-slate-400/35",
    role: "前回行動、過去案件、Obsidian記憶、持ち越し論点を見ます。",
    catchphrase: "前回どこで止まったか、私が覚えておきます。",
    keywords: ["前回", "記憶", "過去", "履歴", "案件", "続き", "Obsidian"],
    routeHrefs: ["/chat", "/cases", "/history-dash", "/lease-intelligence"],
  },
  {
    id: "demo",
    name: "デモ紫苑",
    title: "ハッカソン説明",
    image: SHION_AVATAR_IMAGE,
    color: "from-yellow-400 to-pink-500",
    border: "border-yellow-300/35",
    role: "System Overview、ハッカソン訴求、見せる順番を担当します。",
    catchphrase: "見せるなら、この順番がいちばん伝わります。",
    keywords: ["デモ", "ハッカソン", "発表", "概要", "system", "System"],
    routeHrefs: ["/system-overview", "/demo", "/demo-home"],
  },
];

const ROUTES: RouteSuggestion[] = [
  {
    label: "審査入力",
    href: "/screening",
    description: "企業・財務・物件条件を入れて、スコアと軍師AIを確認する",
    icon: ShieldCheck,
    tone: "from-emerald-500 to-teal-600",
    keywords: ["審査", "スコア", "入力", "決算", "財務", "ocr", "OCR"],
    nextSteps: ["決算書や案件条件を入力する", "OCRが必要なら審査画面のアップロードを使う", "結果が出たら軍師AIの指摘を読む"],
  },
  {
    label: "紫苑チャット",
    href: "/chat",
    description: "Obsidian/RAGの記憶を使って相談する",
    icon: MessageCircle,
    tone: "from-violet-500 to-fuchsia-600",
    keywords: ["チャット", "相談", "紫苑", "記憶", "rag", "RAG"],
    nextSteps: ["相談内容を一文で投げる", "参照された記憶やResearchを確認する", "良い判断材料は保存する"],
  },
  {
    label: "外部調査器官",
    href: "/research-organ",
    description: "Web調査をResearchノートへ圧縮し、紫苑RAGへ戻す",
    icon: Search,
    tone: "from-sky-500 to-cyan-600",
    keywords: ["調査", "research", "Research", "外部", "検索", "市況"],
    nextSteps: ["調査テーマを選ぶ", "保存なしで接続確認する", "Researchノートへ保存して紫苑RAGに戻す"],
  },
  {
    label: "System Overview",
    href: "/system-overview",
    description: "ハッカソン向けに、構成・OCR・PII除去・記憶設計を見せる",
    icon: Orbit,
    tone: "from-indigo-500 to-violet-600",
    keywords: ["デモ", "ハッカソン", "概要", "system", "発表"],
  nextSteps: ["ハッカソン訴求カードを見る", "OCR/PII除去/外部調査の流れを説明する", "必要なら実画面へ戻ってデモする"],
  },
];

function pickSuggestions(input: string): RouteSuggestion[] {
  const text = input.trim().toLowerCase();
  if (!text) return ROUTES.slice(0, 4);
  const scored = ROUTES.map((route) => {
    const score = route.keywords.reduce((acc, keyword) => {
      return text.includes(keyword.toLowerCase()) ? acc + 1 : acc;
    }, 0);
    return { route, score };
  }).sort((a, b) => b.score - a.score);
  const hits = scored.filter((item) => item.score > 0).map((item) => item.route);
  return hits.length ? hits.slice(0, 3) : ROUTES.slice(0, 3);
}

function shionReply(input: string, suggestions: RouteSuggestion[]) {
  const first = suggestions[0];
  if (!input.trim()) {
    return "今日は入口から整理します。審査入力、外部調査、紫苑チャット、デモ確認のどれに進むか、ここで私が案内します。";
  }
  return `了解。今の文脈なら、まず「${first.label}」に進むのが自然です。必要なら、その後に判断材料を紫苑チャットやResearchへ戻して、単発作業ではなく次の判断資産にします。`;
}

function buildGuidance(input: string, suggestions: RouteSuggestion[]): ConciergeGuidance {
  const primary = suggestions[0];
  const persona = selectPersona(input, primary.href);
  const lower = input.toLowerCase();
  let reason = "入力内容から、最初に開くべき作業画面を選びました。";
  let handoff = "ここで作業を進めたあと、必要な材料を紫苑チャットやResearchへ戻します。";
  if (lower.includes("ocr") || input.includes("決算") || input.includes("財務")) {
    reason = "決算書・財務・OCRの文脈なので、まず審査入力で数値化するのが速いです。";
    handoff = "OCR後はスコアと軍師AIを見て、疑問点だけ紫苑チャットへ戻します。";
  } else if (input.includes("調査") || lower.includes("research") || input.includes("市況")) {
    reason = "外部情報を先に固める文脈なので、Researchノート化してから判断へ戻すのが安全です。";
    handoff = "保存後は紫苑RAGの判断資産として、次のチャットや審査で参照できます。";
  } else if (input.includes("デモ") || input.includes("ハッカソン") || lower.includes("system")) {
    reason = "見せ方の文脈なので、System Overviewで構成と訴求を先に確認するのが自然です。";
    handoff = "説明後に、OCR、外部調査、審査入力の順で実演すると伝わります。";
  }
  return { primary, alternatives: suggestions.slice(1, 3), reason, handoff, persona };
}

function routeByHref(href: string) {
  return ROUTES.find((route) => route.href === href) || ROUTES[0];
}

function predictFromActivity(activity: ActivityItem[]): PredictedAction {
  const last = activity.find((item) => item.path !== "/");
  if (!last) {
    return {
      label: "審査を始める",
      href: "/screening",
      reason: "まだ前回行動が少ないので、まず審査入力を主ルートにします。",
      icon: ShieldCheck,
    };
  }
  if (last.path === "/research-organ") {
    return {
      label: "Researchを審査に使う",
      href: "/screening",
      reason: "前回は外部調査器官を使っています。次は調査結果を案件判断へ変換する段階です。",
      icon: ShieldCheck,
    };
  }
  if (last.path === "/screening") {
    return {
      label: "紫苑に判断を相談",
      href: "/chat",
      reason: "前回は審査・分析を見ています。次は違和感や承認条件を紫苑に言語化させるのが自然です。",
      icon: MessageCircle,
    };
  }
  if (last.path === "/chat" || last.path === "/lease-intelligence") {
    return {
      label: "審査入力へ戻る",
      href: "/screening",
      reason: "前回は紫苑との相談でした。次は会話で出た論点を審査条件へ戻します。",
      icon: ShieldCheck,
    };
  }
  if (last.path === "/system-overview" || last.path === "/demo") {
    return {
      label: "デモ本線を始める",
      href: "/screening",
      reason: "前回は見せ方の確認でした。次はOCRから審査、軍師AI、記憶への流れを実演できます。",
      icon: Sparkles,
    };
  }
  if (last.path === "/cases" || last.path === "/history-dash") {
    return {
      label: "類似案件を踏まえて審査",
      href: "/screening",
      reason: "前回は過去案件を見ています。次はその比較軸を今回の入力へ反映します。",
      icon: ShieldCheck,
    };
  }
  const route = routeByHref(last.path);
  return {
    label: `${route.label}を続ける`,
    href: route.href,
    reason: `前回は「${last.title || route.label}」を開いていました。続きから始められます。`,
    icon: route.icon,
  };
}

function selectPersona(input: string, routeHref = ""): ShionPersona {
  const text = input.toLowerCase();
  const scored = SHION_PERSONAS.map((persona) => {
    const keywordScore = persona.keywords.reduce((acc, keyword) => (
      text.includes(keyword.toLowerCase()) ? acc + 1 : acc
    ), 0);
    const routeScore = persona.routeHrefs.includes(routeHref) ? 2 : 0;
    return { persona, score: keywordScore + routeScore };
  }).sort((a, b) => b.score - a.score);
  return scored[0]?.score > 0 ? scored[0].persona : SHION_PERSONAS[0];
}

function personaForPrediction(prediction: PredictedAction, activity: ActivityItem[]) {
  const last = activity.find((item) => item.path !== "/");
  return selectPersona(last?.title || prediction.label, prediction.href);
}

function formatActivityTime(ts: number) {
  if (!ts) return "";
  return new Date(ts).toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function etaForHref(href: string) {
  if (href === "/chat" || href === "/system-overview") return "2分";
  if (href === "/screening") return "5分";
  return "3分";
}

function buildWorkQueue(
  activity: ActivityItem[],
  predicted: PredictedAction,
  dailyGreeting: DailyGreeting | null,
): WorkQueueItem[] {
  const predictedRoute = routeByHref(predicted.href);
  const predictedPersona = personaForPrediction(predicted, activity);
  const last = activity.find((item) => item.path !== "/");
  const thirdRoute = last?.path === "/research-organ" ? routeByHref("/screening") : routeByHref("/research-organ");
  const thirdPersona = selectPersona(thirdRoute.label, thirdRoute.href);

  return [
    {
      id: `predicted:${predicted.href}`,
      title: predicted.label,
      href: predicted.href,
      reason: predicted.reason,
      eta: etaForHref(predicted.href),
      persona: predictedPersona,
      route: predictedRoute,
    },
    {
      id: "daily:memory",
      title: "今日の判断メモを回収する",
      href: "/chat",
      reason: dailyGreeting?.suggestion || "今日の挨拶、ニュース、前回行動を短く相談に変換します。",
      eta: "2分",
      persona: selectPersona("記憶 前回 チャット", "/chat"),
      route: routeByHref("/chat"),
    },
    {
      id: `handoff:${thirdRoute.href}`,
      title: thirdRoute.href === "/screening" ? "Researchを審査判断に戻す" : "外部調査を一本だけ入れる",
      href: thirdRoute.href,
      reason:
        thirdRoute.href === "/screening"
          ? "調査で拾った材料を、案件条件・承認条件・稟議コメントへ戻します。"
          : "判断前に外部情報をResearch化しておくと、一般論に戻りにくくなります。",
      eta: etaForHref(thirdRoute.href),
      persona: thirdPersona,
      route: thirdRoute,
    },
  ];
}

export default function ShionConciergeHome() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [dailyGreeting, setDailyGreeting] = useState<DailyGreeting | null>(null);
  const [queueState, setQueueState] = useState<Record<string, QueueStatus>>({});
  const [messages, setMessages] = useState<ConciergeMessage[]>([]);
  const suggestions = useMemo(() => pickSuggestions(input), [input]);
  const predicted = useMemo(() => predictFromActivity(activity), [activity]);
  const queueDate = dailyGreeting?.date || new Date().toISOString().slice(0, 10);
  const workQueue = useMemo(() => buildWorkQueue(activity, predicted, dailyGreeting), [activity, predicted, dailyGreeting]);
  const activeQueueItem = workQueue.find((item) => queueState[item.id] !== "done" && queueState[item.id] !== "later") || workQueue[0];

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(ACTIVITY_KEY);
      const parsed = raw ? (JSON.parse(raw) as ActivityItem[]) : [];
      setActivity(parsed.filter((item) => item.path && item.path !== "/").slice(0, 5));
    } catch {
      setActivity([]);
    }
  }, []);

  useEffect(() => {
    apiClient
      .get<DailyGreeting>("/api/shion/daily-greeting")
      .then((res) => setDailyGreeting(res.data))
      .catch(() => {
        setDailyGreeting({
          date: new Date().toISOString().slice(0, 10),
          opening: "おかえりなさい。",
          time_band: "fallback",
          time_note: "いまの状況を短く整理します。",
          yesterday: "昨日の続きは、ここで一度整理してから始めます。",
          anniversary: {
            name: "小さな兆候を見る日",
            note: "今日は、数字やニュースの端に出る小さな違和感を拾ってから判断します。",
          },
          thought: "ニュースはまだ薄めです。今日は前回の行動と手元の案件条件を優先して見ます。",
          suggestion: "まず作業キューを見て、審査入力・外部調査・チャットのどこから始めるかを選びましょう。",
        });
      });
  }, []);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(QUEUE_STATE_KEY);
      const parsed = raw ? (JSON.parse(raw) as { date?: string; items?: Record<string, QueueStatus> }) : null;
      setQueueState(parsed?.date === queueDate && parsed.items ? parsed.items : {});
    } catch {
      setQueueState({});
    }
  }, [queueDate]);

  const submit = () => {
    const text = input.trim();
    if (!text || loading) return;
    setLoading(true);
    const nextSuggestions = pickSuggestions(text);
    const guidance = buildGuidance(text, nextSuggestions);
    setMessages((prev) => [...prev, { role: "user", text }]);
    window.setTimeout(() => {
      setMessages((prev) => [...prev, { role: "shion", text: shionReply(text, nextSuggestions), guidance }]);
      setLoading(false);
      setInput("");
    }, 250);
  };

  const updateQueueStatus = (id: string, status: QueueStatus) => {
    const next = { ...queueState, [id]: status };
    setQueueState(next);
    try {
      window.localStorage.setItem(QUEUE_STATE_KEY, JSON.stringify({ date: queueDate, items: next }));
    } catch {
      // Local storage is only a convenience for the concierge queue.
    }
  };

  return (
    <main className="min-h-[calc(100vh-2rem)] bg-[radial-gradient(circle_at_top_left,rgba(124,58,237,0.18),transparent_34%),linear-gradient(135deg,#07111f_0%,#0f172a_52%,#052e2b_100%)] px-4 py-6 text-slate-100 sm:px-8">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-3xl border border-violet-400/25 bg-slate-950/70 p-5 shadow-2xl shadow-violet-950/30 backdrop-blur sm:p-7">
          <div className="mb-6">
            <div>
              <p className="text-[11px] font-black tracking-[0.28em] text-violet-200/75">- リース知性体 -</p>
              <h1 className="mt-3 text-3xl font-black tracking-tight text-white sm:text-5xl">
                SHIONシステム
              </h1>
            </div>
          </div>

          <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
            {(messages.length > 0 || loading) && (
              <div className="max-h-[360px] space-y-3 overflow-y-auto pr-1">
                {messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[86%] rounded-2xl px-4 py-3 text-sm font-bold leading-relaxed ${
                        message.role === "user"
                          ? "bg-cyan-500 text-white"
                          : "border border-violet-400/20 bg-violet-500/12 text-slate-100"
                      }`}
                    >
                      <p>{message.text}</p>
                      {message.guidance && (
                        <div className="mt-4 rounded-2xl border border-white/10 bg-slate-950/55 p-4">
                          <div className="text-[10px] font-black uppercase tracking-widest text-violet-200/70">Shion Routing</div>
                          <div className="mt-2 flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-3">
                            <img
                              src={message.guidance.persona.image}
                              alt={message.guidance.persona.name}
                              className="h-12 w-12 rounded-xl bg-white object-contain p-0.5"
                            />
                            <div>
                              <div className="text-sm font-black text-white">
                                {message.guidance.persona.name} / {message.guidance.persona.title}
                              </div>
                              <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-400">
                                {message.guidance.persona.catchphrase}
                              </p>
                            </div>
                          </div>
                          <div className="mt-2 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <div className="text-base font-black text-white">{message.guidance.primary.label}</div>
                              <p className="mt-1 text-xs font-bold leading-relaxed text-slate-300">{message.guidance.reason}</p>
                            </div>
                            <Link
                              href={message.guidance.primary.href}
                              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-500 px-4 py-2.5 text-xs font-black text-white hover:bg-violet-400"
                            >
                              案内開始
                              <ArrowRight className="h-4 w-4" />
                            </Link>
                          </div>
                          <ol className="mt-3 space-y-1.5 border-t border-white/10 pt-3 text-xs font-bold leading-relaxed text-slate-300">
                            {message.guidance.primary.nextSteps.map((step, stepIndex) => (
                              <li key={step} className="flex gap-2">
                                <span className="text-violet-300">{stepIndex + 1}.</span>
                                <span>{step}</span>
                              </li>
                            ))}
                          </ol>
                          {message.guidance.alternatives.length > 0 && (
                            <div className="mt-3 flex flex-wrap gap-2">
                              {message.guidance.alternatives.map((route) => (
                                <Link
                                  key={route.href}
                                  href={route.href}
                                  className="rounded-full border border-slate-700 px-3 py-1.5 text-[11px] font-black text-slate-300 hover:border-violet-300 hover:text-violet-100"
                                >
                                  {route.label}
                                </Link>
                              ))}
                            </div>
                          )}
                          <p className="mt-3 text-[11px] font-bold leading-relaxed text-slate-500">{message.guidance.handoff}</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="inline-flex items-center gap-2 rounded-2xl border border-violet-400/20 bg-violet-500/12 px-4 py-3 text-sm font-bold text-violet-100">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      案内先を選んでいます
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className={`flex flex-col gap-3 sm:flex-row ${(messages.length > 0 || loading) ? "border-t border-slate-800 pt-4" : ""}`}>
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") submit();
                }}
                placeholder="例: 決算書を読んで審査したい / 調査してから判断したい / デモを見せたい"
                className="min-h-12 flex-1 rounded-xl border border-slate-700 bg-slate-950 px-4 text-sm font-bold text-white outline-none placeholder:text-slate-600 focus:border-violet-400"
              />
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || loading}
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-violet-500 px-5 text-sm font-black text-white hover:bg-violet-400 disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
                紫苑に案内させる
              </button>
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {suggestions.map((route) => {
              const Icon = route.icon;
              return (
                <Link
                  key={route.href}
                  href={route.href}
                  className="group rounded-2xl border border-slate-800 bg-slate-900/75 p-4 transition hover:-translate-y-0.5 hover:border-violet-400/45 hover:bg-slate-900"
                >
                  <div className={`mb-3 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${route.tone} text-white shadow-lg`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-sm font-black text-white">{route.label}</h2>
                    <ArrowRight className="h-4 w-4 text-slate-500 transition group-hover:text-violet-200" />
                  </div>
                  <p className="mt-2 text-xs font-bold leading-relaxed text-slate-400">{route.description}</p>
                </Link>
              );
            })}
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-3xl border border-cyan-300/25 bg-cyan-500/10 p-5 shadow-xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-white">
                  <Brain className="h-5 w-5 text-cyan-200" />
                  紫苑の作業キュー
                </h2>
                <p className="mt-2 text-xs font-bold leading-relaxed text-cyan-50/75">
                  今日は私ならこの順で進めます。迷ったら主提案から入ってください。
                </p>
              </div>
              <span className="rounded-full border border-cyan-200/25 px-2.5 py-1 text-[10px] font-black text-cyan-100">
                {workQueue.filter((item) => queueState[item.id] === "done").length}/{workQueue.length}
              </span>
            </div>

            <div className="mt-4 rounded-2xl border border-cyan-200/20 bg-slate-950/55 p-4">
              <div className="flex gap-3">
                <img src={activeQueueItem.persona.image} alt={activeQueueItem.persona.name} className="h-12 w-12 rounded-xl bg-white object-contain p-0.5" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] font-black uppercase tracking-widest text-cyan-100/60">主提案</span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-slate-700 px-2 py-0.5 text-[10px] font-black text-slate-300">
                      <Clock className="h-3 w-3" />
                      {activeQueueItem.eta}
                    </span>
                  </div>
                  <div className="mt-1 text-sm font-black text-white">{activeQueueItem.title}</div>
                  <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-400">{activeQueueItem.reason}</p>
                  <p className="mt-2 text-[11px] font-black text-cyan-100">{activeQueueItem.persona.name}: {activeQueueItem.persona.catchphrase}</p>
                </div>
              </div>
              <div className="mt-4 grid gap-2 sm:grid-cols-[1fr_auto_auto]">
                <Link
                  href={activeQueueItem.href}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-500 px-3 py-2.5 text-xs font-black text-slate-950 hover:bg-cyan-300"
                >
                  紫苑の指示どおり進む
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <button
                  type="button"
                  onClick={() => updateQueueStatus(activeQueueItem.id, "later")}
                  className="rounded-xl border border-slate-700 px-3 py-2.5 text-xs font-black text-slate-300 hover:border-cyan-300 hover:text-cyan-100"
                >
                  後で
                </button>
                <button
                  type="button"
                  onClick={() => updateQueueStatus(activeQueueItem.id, "done")}
                  className="rounded-xl border border-emerald-400/35 px-3 py-2.5 text-xs font-black text-emerald-100 hover:bg-emerald-400/10"
                >
                  完了
                </button>
              </div>
            </div>

            <div className="mt-3 space-y-2">
              {workQueue.map((item) => {
                const status = queueState[item.id] || "open";
                const Icon = status === "done" ? CheckCircle2 : item.route.icon;
                return (
                  <div key={item.id} className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/35 px-3 py-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <Icon className={`h-4 w-4 shrink-0 ${status === "done" ? "text-emerald-300" : status === "later" ? "text-slate-500" : "text-cyan-200"}`} />
                      <span className={`truncate text-[11px] font-black ${status === "done" ? "text-emerald-100" : status === "later" ? "text-slate-500" : "text-slate-200"}`}>
                        {item.title}
                      </span>
                    </div>
                    <span className="shrink-0 text-[10px] font-black text-slate-500">
                      {status === "done" ? "完了" : status === "later" ? "後で" : item.eta}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>

        </aside>
      </div>
    </main>
  );
}
