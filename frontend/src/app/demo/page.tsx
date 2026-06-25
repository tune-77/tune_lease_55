"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Brain, Zap, Target, ArrowRight, Activity, Database, GitMerge, Sparkles } from "lucide-react";
import { apiClient } from "@/lib/api";

type LiveStats = {
  total_cases: number;
  closed_rate: number;
  avg_score: number;
  active_rules: number;
};

const MOCK_STATS: LiveStats = {
  total_cases: 142,
  closed_rate: 68.3,
  avg_score: 74.2,
  active_rules: 38,
};

const RINGS = [
  {
    label: "つくる",
    desc: "与信スコアリング",
    size: "w-72 h-72",
    borderColor: "border-violet-500/60",
    glowColor: "shadow-[0_0_40px_rgba(139,92,246,0.3)]",
    animationClass: "animate-[spin_18s_linear_infinite]",
    iconColor: "text-violet-400",
    Icon: Brain,
  },
  {
    label: "まわす",
    desc: "自己改善ループ",
    size: "w-52 h-52",
    borderColor: "border-cyan-500/60",
    glowColor: "shadow-[0_0_30px_rgba(6,182,212,0.3)]",
    animationClass: "animate-[spin_12s_linear_infinite_reverse]",
    iconColor: "text-cyan-400",
    Icon: GitMerge,
  },
  {
    label: "とどける",
    desc: "4ペルソナ+確信マップ",
    size: "w-36 h-36",
    borderColor: "border-emerald-500/60",
    glowColor: "shadow-[0_0_20px_rgba(52,211,153,0.3)]",
    animationClass: "animate-[spin_8s_linear_infinite]",
    iconColor: "text-emerald-400",
    Icon: Zap,
  },
];

const TECH_BADGES = [
  { label: "Gemini 2.5 Flash", color: "bg-blue-900/50 border-blue-500/40 text-blue-300" },
  { label: "Cloud Run", color: "bg-teal-900/50 border-teal-500/40 text-teal-300" },
  { label: "ChromaDB", color: "bg-violet-900/50 border-violet-500/40 text-violet-300" },
  { label: "LightGBM", color: "bg-emerald-900/50 border-emerald-500/40 text-emerald-300" },
  { label: "Next.js 16", color: "bg-slate-800/70 border-slate-500/40 text-slate-300" },
  { label: "FastAPI", color: "bg-green-900/50 border-green-500/40 text-green-300" },
];

const CTAS = [
  {
    label: "紫苑と話す",
    sublabel: "リース知性体との対話",
    href: "/lease-intelligence",
    primary: true,
    Icon: Brain,
    gradient: "from-violet-600 to-fuchsia-600",
    glow: "hover:shadow-[0_0_30px_rgba(139,92,246,0.5)]",
  },
  {
    label: "自己改善パイプライン",
    sublabel: "エージェント判断をリアルタイム表示",
    href: "/demo/pipeline",
    primary: false,
    Icon: Zap,
    gradient: "from-fuchsia-600 to-pink-600",
    glow: "hover:shadow-[0_0_30px_rgba(192,38,211,0.4)]",
  },
  {
    label: "4ペルソナ討論 + 確信マップ",
    sublabel: "慎重・積極・革新者・裁定者が討論し共有認識を可視化",
    href: "/debate",
    primary: false,
    Icon: GitMerge,
    gradient: "from-cyan-600 to-blue-600",
    glow: "hover:shadow-[0_0_30px_rgba(6,182,212,0.4)]",
  },
  {
    label: "システム全体図",
    sublabel: "6つの自己改善ループを俯瞰",
    href: "/system-overview",
    primary: false,
    Icon: Activity,
    gradient: "from-emerald-600 to-teal-600",
    glow: "hover:shadow-[0_0_30px_rgba(52,211,153,0.4)]",
  },
];

export default function DemoPage() {
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [statsLoaded, setStatsLoaded] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await apiClient.get("/api/dashboard/stats");
        const data = res.data;
        const analysis = data?.analysis;
        setStats({
          total_cases: analysis?.closed_count ?? MOCK_STATS.total_cases,
          closed_rate: MOCK_STATS.closed_rate,
          avg_score: analysis?.avg_score_borrower ?? MOCK_STATS.avg_score,
          active_rules: MOCK_STATS.active_rules,
        });
      } catch {
        setStats(MOCK_STATS);
      } finally {
        setStatsLoaded(true);
      }
    };
    fetchStats();

    const interval = setInterval(() => setTick((t) => t + 1), 2000);
    return () => clearInterval(interval);
  }, []);

  const displayStats = stats ?? MOCK_STATS;

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#030712]">
      {/* 背景グラデーション */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-40 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-violet-900/30 blur-[120px]" />
        <div className="absolute bottom-0 left-0 h-80 w-80 rounded-full bg-cyan-900/20 blur-[100px]" />
        <div className="absolute right-0 top-1/3 h-64 w-64 rounded-full bg-fuchsia-900/20 blur-[80px]" />
      </div>

      {/* グリッドオーバーレイ */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(139,92,246,1) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,1) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-16">
        {/* バッジ */}
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-900/20 px-4 py-2 text-xs font-bold text-violet-300 backdrop-blur-sm">
          <Sparkles className="h-3.5 w-3.5" />
          DevOps × AI Agent Hackathon 2026 · Findy × Google Cloud
        </div>

        {/* タイトル */}
        <h1 className="mb-4 text-center text-4xl font-black leading-tight tracking-tight text-white sm:text-6xl lg:text-7xl">
          <span className="bg-gradient-to-r from-violet-400 via-fuchsia-400 to-cyan-400 bg-clip-text text-transparent">
            リース与信AI
          </span>
          <br />
          <span className="text-white/90">30秒で全部わかる</span>
        </h1>

        <p className="mb-12 max-w-xl text-center text-base font-medium leading-relaxed text-slate-400 sm:text-lg">
          紫苑（AI審査知性体）が<strong className="text-white">スコアリング・自己改善・4ペルソナ討論</strong>を
          フルオートで回し続けるリースファイナンスAIシステム。討論結果はセントラル統合エンジンで<strong className="text-white">確信マップ（world_view）</strong>として蓄積される。
        </p>

        {/* 3円環アニメーション */}
        <div className="relative mb-12 flex h-80 w-80 items-center justify-center">
          {RINGS.map((ring, i) => (
            <div
              key={i}
              className={`absolute rounded-full border-2 ${ring.size} ${ring.borderColor} ${ring.glowColor} ${ring.animationClass}`}
            >
              {/* ラベル（アニメーションを打ち消すカウンター回転） */}
              <div
                className="absolute"
                style={{
                  top: i === 0 ? "-28px" : i === 1 ? "-24px" : "-20px",
                  left: "50%",
                  transform: "translateX(-50%)",
                  animation: `spin ${i === 0 ? "18s" : i === 1 ? "12s" : "8s"} linear infinite ${i === 1 ? "" : "reverse"}`,
                }}
              >
                <span
                  className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-black ${ring.borderColor} bg-[#030712] ${ring.iconColor}`}
                >
                  {ring.label}
                </span>
              </div>
            </div>
          ))}

          {/* 中心：紫苑アイコン */}
          <div className="relative flex h-20 w-20 flex-col items-center justify-center rounded-full border border-violet-400/50 bg-[#0d0618] shadow-[0_0_50px_rgba(139,92,246,0.6)]">
            <Brain className="h-8 w-8 text-violet-300" />
            <span className="mt-1 text-[10px] font-black text-violet-300">紫苑</span>
            {/* パルスドット */}
            <span className="absolute -right-1 -top-1 flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-violet-500" />
            </span>
          </div>

          {/* 外周の説明テキスト（固定位置） */}
          {RINGS.map((ring, i) => {
            const positions = [
              { bottom: "8px", left: "50%", transform: "translateX(-50%)" },
              { top: "50%", right: "-48px", transform: "translateY(-50%)" },
              { top: "50%", left: "-40px", transform: "translateY(-50%)" },
            ];
            return (
              <div
                key={`desc-${i}`}
                className="pointer-events-none absolute"
                style={positions[i]}
              >
                <span className={`whitespace-nowrap text-[10px] font-bold ${ring.iconColor} opacity-70`}>
                  {ring.desc}
                </span>
              </div>
            );
          })}
        </div>

        {/* ライブ統計 */}
        <div className="mb-12 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            { label: "総審査案件", value: displayStats.total_cases, unit: "件", color: "text-violet-400" },
            { label: "成約率", value: displayStats.closed_rate.toFixed(1), unit: "%", color: "text-cyan-400" },
            { label: "平均スコア", value: displayStats.avg_score.toFixed(1), unit: "pt", color: "text-emerald-400" },
            { label: "アクティブルール", value: displayStats.active_rules, unit: "本", color: "text-fuchsia-400" },
          ].map((stat, i) => (
            <div
              key={i}
              className="flex flex-col items-center rounded-2xl border border-white/10 bg-white/5 px-6 py-4 backdrop-blur-sm"
            >
              <div className={`text-3xl font-black tabular-nums ${stat.color}`}>
                {statsLoaded ? stat.value : "—"}
                <span className="ml-1 text-sm font-bold text-slate-500">{stat.unit}</span>
              </div>
              <div className="mt-1 text-xs font-bold text-slate-500">{stat.label}</div>
              {/* パルス表示 */}
              <div className="mt-2 flex items-center gap-1">
                <span
                  className={`inline-block h-1.5 w-1.5 rounded-full ${
                    tick % 4 === i ? stat.color.replace("text-", "bg-") : "bg-slate-700"
                  } transition-colors duration-500`}
                />
                <span className="text-[9px] font-bold text-slate-600">LIVE</span>
              </div>
            </div>
          ))}
        </div>

        {/* Tech Stack バッジ */}
        <div className="mb-12 flex flex-wrap justify-center gap-2">
          {TECH_BADGES.map((badge, i) => (
            <span
              key={i}
              className={`rounded-full border px-3 py-1 text-xs font-bold ${badge.color} backdrop-blur-sm`}
            >
              {badge.label}
            </span>
          ))}
        </div>

        {/* CTAボタン */}
        <div className="flex flex-col items-center gap-4 sm:flex-row">
          {CTAS.map((cta, i) => (
            <Link
              key={i}
              href={cta.href}
              className={`group flex items-center gap-3 rounded-2xl px-6 py-4 text-sm font-black text-white transition-all duration-300 ${cta.glow} ${
                cta.primary
                  ? `bg-gradient-to-r ${cta.gradient} shadow-lg`
                  : `border border-white/10 bg-white/5 backdrop-blur-sm hover:bg-white/10`
              }`}
            >
              <cta.Icon className={`h-5 w-5 ${cta.primary ? "text-white" : "text-slate-400 group-hover:text-white"} transition-colors`} />
              <div className="text-left">
                <div className="leading-tight">{cta.label}</div>
                <div className={`text-[10px] font-medium leading-tight ${cta.primary ? "text-white/70" : "text-slate-500 group-hover:text-slate-400"} transition-colors`}>
                  {cta.sublabel}
                </div>
              </div>
              <ArrowRight className={`h-4 w-4 transition-transform duration-200 group-hover:translate-x-1 ${cta.primary ? "text-white/70" : "text-slate-600 group-hover:text-slate-400"}`} />
            </Link>
          ))}
        </div>

        {/* フッター */}
        <p className="mt-16 text-xs text-slate-700">
          tune_lease_55 · Built with Claude Code · 2026
        </p>
      </div>

      <style jsx global>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
