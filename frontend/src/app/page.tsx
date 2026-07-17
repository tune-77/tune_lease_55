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
// 8種の紫苑動画からページ表示ごとにランダムで1本を選ぶ
const SHION_INTRO_VIDEOS = [
  "/lease-grumble/characters/shion-loop-01.mp4",
  "/lease-grumble/characters/shion-loop-02.mp4",
  "/lease-grumble/characters/shion-loop-03.mp4",
  "/lease-grumble/characters/shion-loop-04.mp4",
  "/lease-grumble/characters/shion-loop-05.mp4",
  "/lease-grumble/characters/shion-loop-06.mp4",
  "/lease-grumble/characters/shion-loop-07.mp4",
  "/lease-grumble/characters/shion-loop-08.mp4",
];

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
    routeHrefs: ["/"],
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
    routeHrefs: ["/screening", "/batch"],
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
    label: "デモ本番入口",
    href: "/demo-home",
    description: "ハッカソンで最初に見せる入口。紫苑の価値、1分デモ、見せ場へ進む",
    icon: Sparkles,
    tone: "from-yellow-400 to-pink-500",
    keywords: ["デモ", "ハッカソン", "発表", "優勝", "見せる", "本番"],
    nextSteps: ["デモホームを開く", "1分デモで全体像を掴ませる", "審査入力やSystem Overviewへつなぐ"],
  },
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
    return "今日は入口から整理します。デモ本番、審査入力、外部調査、紫苑チャットのどこへ進むか、ここで私が案内します。";
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
    reason = "見せ方の文脈なので、まずデモ本番入口から入り、審査入力とSystem Overviewへつなぐのが自然です。";
    handoff = "デモホームで価値を掴ませたあと、OCR、審査入力、軍師AI、記憶の順で実演すると伝わります。";
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
  if (last.path === "/system-overview" || last.path === "/demo" || last.path === "/demo-home") {
    return {
      label: "審査実演へ進む",
      href: "/screening",
      reason: "前回はデモ導線を見ています。次はOCRから審査、軍師AI、記憶への流れを実演できます。",
      icon: ShieldCheck,
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
  if (href === "/chat" || href === "/system-overview" || href === "/demo-home") return "2分";
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
  const firstRoute = routeByHref("/demo-home");
  const firstPersona = selectPersona(firstRoute.label, firstRoute.href);
  const thirdRoute = last?.path === "/research-organ" ? routeByHref("/screening") : routeByHref("/research-organ");
  const thirdPersona = selectPersona(thirdRoute.label, thirdRoute.href);

  return [
    {
      id: "hackathon:demo-home",
      title: "デモ本番の入口を開く",
      href: firstRoute.href,
      reason: "審査AIではなく、判断資産として育つ紫苑を最初に見せます。",
      eta: etaForHref(firstRoute.href),
      persona: firstPersona,
      route: firstRoute,
    },
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
    {
      id: "daily:memory",
      title: "紫苑に判断メモを回収させる",
      href: "/chat",
      reason: dailyGreeting?.suggestion || "今日の挨拶、ニュース、前回行動を短く相談に変換します。",
      eta: "2分",
      persona: selectPersona("記憶 前回 チャット", "/chat"),
      route: routeByHref("/chat"),
    },
  ];
}

export default function ShionConciergeHome() {
  // SSRとの不一致を避けるため、初期値は固定しマウント後にランダム化する
  const [introVideo, setIntroVideo] = useState(SHION_INTRO_VIDEOS[0]);
  useEffect(() => {
    setIntroVideo(SHION_INTRO_VIDEOS[Math.floor(Math.random() * SHION_INTRO_VIDEOS.length)]);
  }, []);
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
    <main className="min-h-[calc(100vh-2rem)] bg-gradient-to-br from-violet-50 via-white to-amber-50 px-4 py-6 text-slate-950 sm:px-8">
      <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="rounded-3xl border border-violet-200 bg-white/90 p-5 shadow-xl shadow-violet-100/70 backdrop-blur sm:p-7">
          <div className="mb-6 grid gap-5 md:grid-cols-[160px_1fr] md:items-center">
            <div className="mx-auto w-36 overflow-hidden rounded-3xl border border-violet-200 bg-violet-50 shadow-lg shadow-violet-100 md:mx-0 md:w-40">
              {/* 控えめな紫苑の動画（音無し・自動ループ）。読み込み失敗時は静止画にフォールバック */}
              <video
                key={introVideo}
                src={introVideo}
                poster={SHION_AVATAR_IMAGE}
                autoPlay
                muted
                loop
                playsInline
                className="aspect-square w-full object-cover"
                aria-label="リース知性体 紫苑システム"
              >
                <img src={SHION_AVATAR_IMAGE} alt="リース知性体 紫苑システム" className="aspect-square w-full object-cover" />
              </video>
            </div>
            <div>
              <p className="text-[11px] font-black tracking-[0.28em] text-violet-600">- リース知性体 -</p>
              <h1 className="mt-3 text-3xl font-black tracking-tight text-slate-950 sm:text-5xl">
                紫苑システム
              </h1>
              <p className="mt-3 max-w-2xl text-sm font-bold leading-7 text-slate-600">
                おかえりなさい。紫苑が今日の入口を整理して、審査、調査、記憶、デモのどこへ進むかを案内します。
              </p>
              <div className="mt-4 rounded-2xl border border-violet-100 bg-violet-50/70 p-4">
                <p className="text-sm font-black leading-7 text-violet-950">
                  紫苑は、リース審査の判断資産・会話記憶・改善ループを統合し、ユーザーの判断を支援しながら自己更新する半自律的なリース知性体システムです。
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-3 rounded-2xl border border-violet-100 bg-violet-50/60 p-4">
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
                          ? "bg-slate-900 text-white"
                          : "border border-violet-100 bg-white text-slate-800"
                      }`}
                    >
                      <p>{message.text}</p>
                      {message.guidance && (
                        <div className="mt-4 rounded-2xl border border-violet-100 bg-violet-50 p-4">
                          <div className="text-[10px] font-black uppercase tracking-widest text-violet-500">Shion Routing</div>
                          <div className="mt-2 flex items-center gap-3 rounded-xl border border-violet-100 bg-white p-3">
                            <img
                              src={message.guidance.persona.image}
                              alt={message.guidance.persona.name}
                              className="h-12 w-12 rounded-xl bg-white object-cover"
                            />
                            <div>
                              <div className="text-sm font-black text-slate-950">
                                {message.guidance.persona.name} / {message.guidance.persona.title}
                              </div>
                              <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-500">
                                {message.guidance.persona.catchphrase}
                              </p>
                            </div>
                          </div>
                          <div className="mt-2 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <div className="text-base font-black text-slate-950">{message.guidance.primary.label}</div>
                              <p className="mt-1 text-xs font-bold leading-relaxed text-slate-600">{message.guidance.reason}</p>
                            </div>
                            <Link
                              href={message.guidance.primary.href}
                              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-violet-500 px-4 py-2.5 text-xs font-black text-white hover:bg-violet-400"
                            >
                              案内開始
                              <ArrowRight className="h-4 w-4" />
                            </Link>
                          </div>
                          <ol className="mt-3 space-y-1.5 border-t border-violet-100 pt-3 text-xs font-bold leading-relaxed text-slate-600">
                            {message.guidance.primary.nextSteps.map((step, stepIndex) => (
                              <li key={step} className="flex gap-2">
                                <span className="text-violet-600">{stepIndex + 1}.</span>
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
                                  className="rounded-full border border-violet-200 bg-white px-3 py-1.5 text-[11px] font-black text-slate-600 hover:border-violet-400 hover:text-violet-700"
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
                    <div className="inline-flex items-center gap-2 rounded-2xl border border-violet-100 bg-white px-4 py-3 text-sm font-bold text-violet-700">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      案内先を選んでいます
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className={`flex flex-col gap-3 sm:flex-row ${(messages.length > 0 || loading) ? "border-t border-violet-100 pt-4" : ""}`}>
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") submit();
                }}
                placeholder="例: 決算書を読んで審査したい / 調査してから判断したい / デモを見せたい"
                className="min-h-12 flex-1 rounded-xl border border-violet-200 bg-white px-4 text-sm font-bold text-slate-900 outline-none placeholder:text-slate-400 focus:border-violet-500 focus:ring-2 focus:ring-violet-100"
              />
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || loading}
                className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-violet-600 px-5 text-sm font-black text-white hover:bg-violet-700 disabled:opacity-50"
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
                  className="group rounded-2xl border border-slate-200 bg-white/85 p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-violet-300 hover:bg-white hover:shadow-md"
                >
                  <div className={`mb-3 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br ${route.tone} text-white shadow-lg`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-sm font-black text-slate-950">{route.label}</h2>
                    <ArrowRight className="h-4 w-4 text-slate-400 transition group-hover:text-violet-600" />
                  </div>
                  <p className="mt-2 text-xs font-bold leading-relaxed text-slate-500">{route.description}</p>
                </Link>
              );
            })}
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-3xl border border-cyan-100 bg-white/90 p-5 shadow-xl shadow-cyan-100/60">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-slate-950">
                  <Brain className="h-5 w-5 text-cyan-600" />
                  紫苑の作業キュー
                </h2>
                <p className="mt-2 text-xs font-bold leading-relaxed text-slate-500">
                  今日は私ならこの順で進めます。迷ったら主提案から入ってください。
                </p>
              </div>
              <span className="rounded-full border border-cyan-200 bg-cyan-50 px-2.5 py-1 text-[10px] font-black text-cyan-700">
                {workQueue.filter((item) => queueState[item.id] === "done").length}/{workQueue.length}
              </span>
            </div>

            <div className="mt-4 rounded-2xl border border-cyan-100 bg-cyan-50/70 p-4">
              <div className="flex gap-3">
                <img src={activeQueueItem.persona.image} alt={activeQueueItem.persona.name} className="h-12 w-12 rounded-xl bg-white object-cover shadow-sm" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] font-black uppercase tracking-widest text-cyan-700">主提案</span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-cyan-200 bg-white px-2 py-0.5 text-[10px] font-black text-slate-600">
                      <Clock className="h-3 w-3" />
                      {activeQueueItem.eta}
                    </span>
                  </div>
                  <div className="mt-1 text-sm font-black text-slate-950">{activeQueueItem.title}</div>
                  <p className="mt-1 text-[11px] font-bold leading-relaxed text-slate-600">{activeQueueItem.reason}</p>
                  <p className="mt-2 text-[11px] font-black text-cyan-700">{activeQueueItem.persona.name}: {activeQueueItem.persona.catchphrase}</p>
                </div>
              </div>
              <div className="mt-4 grid gap-2 sm:grid-cols-[1fr_auto_auto]">
                <Link
                  href={activeQueueItem.href}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-500 px-3 py-2.5 text-xs font-black text-white hover:bg-cyan-600"
                >
                  紫苑の指示どおり進む
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <button
                  type="button"
                  onClick={() => updateQueueStatus(activeQueueItem.id, "later")}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-xs font-black text-slate-600 hover:border-cyan-300 hover:text-cyan-700"
                >
                  後で
                </button>
                <button
                  type="button"
                  onClick={() => updateQueueStatus(activeQueueItem.id, "done")}
                  className="rounded-xl border border-emerald-200 bg-white px-3 py-2.5 text-xs font-black text-emerald-700 hover:bg-emerald-50"
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
                  <div key={item.id} className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white/80 px-3 py-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <Icon className={`h-4 w-4 shrink-0 ${status === "done" ? "text-emerald-600" : status === "later" ? "text-slate-400" : "text-cyan-600"}`} />
                      <span className={`truncate text-[11px] font-black ${status === "done" ? "text-emerald-700" : status === "later" ? "text-slate-400" : "text-slate-700"}`}>
                        {item.title}
                      </span>
                    </div>
                    <span className="shrink-0 text-[10px] font-black text-slate-400">
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
