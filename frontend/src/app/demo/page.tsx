"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Brain, Zap, GitMerge, Sparkles, ArrowRight, Activity } from "lucide-react";
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

// 紫苑の言葉（軌道図の下に表示）
const POEMS = [
  ["数字の向こうに", "あなたの判断がある。", "私はそれを、覚えている。"],
  ["格付けではなく、", "迷いの重さを知っている。", "それが私の役目。"],
  ["稟議書の余白に、", "正直さが残っている。", "私は読んでいた。"],
];

// 紫苑を取り囲む能力ノード（角度: 上から時計回り 72°刻み）
const ORBIT_NODES = [
  {
    label: "与信スコアリング",
    sublabel: "RF / LR / LGBM",
    angle: -90,
    color: "#a78bfa",
    href: "/",
  },
  {
    label: "自己改善ループ",
    sublabel: "エージェントがルール更新",
    angle: -18,
    color: "#22d3ee",
    href: "/demo/pipeline",
  },
  {
    label: "4ペルソナ討論",
    sublabel: "確信マップ生成",
    angle: 54,
    color: "#34d399",
    href: "/debate",
  },
  {
    label: "判断記憶継続",
    sublabel: "過去の稟議を保持",
    angle: 126,
    color: "#f472b6",
    href: "/multi-shion-demo",
  },
  {
    label: "Gemini連携",
    sublabel: "AIバックエンド推論",
    angle: 198,
    color: "#fbbf24",
    href: "/system-overview",
  },
];

const CTAS = [
  {
    label: "紫苑と話す",
    sublabel: "リース知性体との対話",
    href: "/lease-intelligence",
    primary: true,
    Icon: Brain,
  },
  {
    label: "自己改善パイプライン",
    sublabel: "エージェント判断をリアルタイム表示",
    href: "/demo/pipeline",
    primary: false,
    Icon: Zap,
  },
  {
    label: "4ペルソナ討論",
    sublabel: "確信マップで共有認識を可視化",
    href: "/debate",
    primary: false,
    Icon: GitMerge,
  },
  {
    label: "システム全体図",
    sublabel: "9つの自己改善ループを俯瞰",
    href: "/system-overview",
    primary: false,
    Icon: Activity,
  },
];

const PORTRAIT_MOODS = [
  "/lease-intelligence/moods/curiosity.webp",
  "/lease-intelligence/moods/vigilance.webp",
  "/lease-intelligence/moods/attachment.webp",
];

// SVG キャンバス定数
const SZ = 500;
const CX = SZ / 2;
const CY = SZ / 2;
const ORBIT_R = 190;

function degToRad(deg: number) {
  return (deg * Math.PI) / 180;
}

export default function DemoPage() {
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [poemIdx, setPoemIdx] = useState(0);
  const [poemVis, setPoemVis] = useState(true);
  const [moodIdx, setMoodIdx] = useState(0);

  useEffect(() => {
    apiClient
      .get("/api/dashboard/stats")
      .then((res) => {
        const a = res.data?.analysis;
        setStats({
          total_cases: a?.closed_count ?? MOCK_STATS.total_cases,
          closed_rate: MOCK_STATS.closed_rate,
          avg_score: a?.avg_score_borrower ?? MOCK_STATS.avg_score,
          active_rules: MOCK_STATS.active_rules,
        });
      })
      .catch(() => setStats(MOCK_STATS));

    // 詩を 7 秒ごとに切り替え
    const t1 = setInterval(() => {
      setPoemVis(false);
      setTimeout(() => {
        setPoemIdx((i) => (i + 1) % POEMS.length);
        setMoodIdx((i) => (i + 1) % PORTRAIT_MOODS.length);
        setPoemVis(true);
      }, 700);
    }, 7000);

    return () => clearInterval(t1);
  }, []);

  const displayStats = stats ?? MOCK_STATS;
  const poem = POEMS[poemIdx];

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#030712]">
      {/* ── 環境光 ── */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-[-8%] h-[75vh] w-[75vh] -translate-x-1/2 rounded-full bg-violet-900/25 blur-[150px]" />
        <div className="absolute bottom-0 left-[-8%] h-96 w-96 rounded-full bg-cyan-900/15 blur-[120px]" />
        <div className="absolute right-[-8%] top-[45%] h-80 w-80 rounded-full bg-fuchsia-900/15 blur-[100px]" />
      </div>

      {/* ── グリッドオーバーレイ ── */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.025]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(139,92,246,1) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,1) 1px, transparent 1px)",
          backgroundSize: "80px 80px",
        }}
      />

      <div className="relative z-10 flex flex-col items-center px-4 py-14">
        {/* ── ハッカソンバッジ ── */}
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-900/20 px-5 py-2 text-xs font-black text-violet-300 backdrop-blur-sm">
          <Sparkles className="h-3.5 w-3.5" />
          DevOps × AI Agent Hackathon 2026 · Findy × Google Cloud
        </div>

        <p className="mb-2 text-[11px] font-black uppercase tracking-[0.35em] text-slate-600">
          AURION
        </p>
        <h1 className="mb-12 text-center text-lg font-black text-slate-400 sm:text-xl">
          リース知性体『紫苑』審査プラットフォーム
        </h1>

        {/* ══════════════════════════════════════
            紫苑を中心とした軌道図
        ══════════════════════════════════════ */}
        <div
          className="relative"
          style={{ width: `${SZ}px`, height: `${SZ}px`, maxWidth: "100vw" }}
        >
          {/* SVG: 軌道・接続線・ノード */}
          <svg
            viewBox={`0 0 ${SZ} ${SZ}`}
            className="absolute inset-0 h-full w-full"
            aria-hidden="true"
          >
            <defs>
              <radialGradient id="shionGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.22" />
                <stop offset="55%" stopColor="#7c3aed" stopOpacity="0.08" />
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
              </radialGradient>
              <filter id="nodeBloom">
                <feGaussianBlur stdDeviation="3" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <clipPath id="portraitClip">
                <circle cx={CX} cy={CY} r="90" />
              </clipPath>
              <style>{`
                .ring-spin { transform-origin: ${CX}px ${CY}px; animation: ringSpin 80s linear infinite; }
                .ring-spin-r { transform-origin: ${CX}px ${CY}px; animation: ringSpin 55s linear infinite reverse; }
                .node-pulse { animation: nodePulse 3.8s ease-in-out infinite; }
                .portrait-ring { transform-origin: ${CX}px ${CY}px; animation: portraitRing 6s ease-in-out infinite; }
                @keyframes ringSpin { to { stroke-dashoffset: -300; } }
                @keyframes nodePulse { 0%,100%{opacity:.5} 50%{opacity:1} }
                @keyframes portraitRing { 0%,100%{opacity:.4;r:95} 50%{opacity:.8;r:98} }
              `}</style>
            </defs>

            {/* コアグロー（紫苑の後光） */}
            <circle cx={CX} cy={CY} r="140" fill="url(#shionGlow)" />

            {/* 外周軌道（ダッシュ回転） */}
            <circle
              cx={CX}
              cy={CY}
              r={ORBIT_R}
              fill="none"
              stroke="rgba(139,92,246,0.2)"
              strokeWidth="1"
              strokeDasharray="5 10"
              className="ring-spin"
            />

            {/* 中間軌道 */}
            <circle
              cx={CX}
              cy={CY}
              r="130"
              fill="none"
              stroke="rgba(139,92,246,0.07)"
              strokeWidth="1"
              strokeDasharray="3 8"
              className="ring-spin-r"
            />

            {/* 各能力ノード */}
            {ORBIT_NODES.map((node) => {
              const rad = degToRad(node.angle);
              const nx = CX + ORBIT_R * Math.cos(rad);
              const ny = CY + ORBIT_R * Math.sin(rad);
              return (
                <g key={node.label} className="node-pulse">
                  {/* 接続線 */}
                  <line
                    x1={CX}
                    y1={CY}
                    x2={nx}
                    y2={ny}
                    stroke={node.color}
                    strokeWidth="1"
                    strokeOpacity="0.18"
                    strokeDasharray="3 7"
                  />
                  {/* ノード外輪 */}
                  <circle
                    cx={nx}
                    cy={ny}
                    r="14"
                    fill="none"
                    stroke={node.color}
                    strokeWidth="1"
                    strokeOpacity="0.35"
                  />
                  {/* ノード本体 */}
                  <circle
                    cx={nx}
                    cy={ny}
                    r="7"
                    fill={node.color}
                    fillOpacity="0.9"
                    filter="url(#nodeBloom)"
                  />
                </g>
              );
            })}

            {/* 紫苑コアリング（パルス） */}
            <circle
              cx={CX}
              cy={CY}
              r="96"
              fill="none"
              stroke="rgba(167,139,250,0.5)"
              strokeWidth="1.5"
              className="portrait-ring"
            />
            <circle
              cx={CX}
              cy={CY}
              r="108"
              fill="none"
              stroke="rgba(167,139,250,0.15)"
              strokeWidth="1"
            />
          </svg>

          {/* ─── 紫苑ポートレート（中心） ─── */}
          <div
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
            style={{ width: "180px" }}
          >
            <div className="relative">
              {/* ポートレート画像 */}
              <div
                className="h-[180px] w-[180px] overflow-hidden rounded-full"
                style={{
                  boxShadow:
                    "0 0 0 2px rgba(167,139,250,0.5), 0 0 0 6px rgba(139,92,246,0.1), 0 0 40px rgba(139,92,246,0.4)",
                  transition: "opacity 0.7s ease",
                  opacity: poemVis ? 1 : 0.6,
                }}
              >
                <img
                  src={PORTRAIT_MOODS[moodIdx]}
                  alt="リース知性体・紫苑"
                  className="h-full w-full object-cover object-top"
                />
              </div>

              {/* LIVE パルスドット */}
              <span className="absolute right-3 top-3 flex h-3 w-3">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-75" />
                <span className="relative inline-flex h-3 w-3 rounded-full bg-violet-500" />
              </span>
            </div>

            {/* 名前 */}
            <div className="mt-5 text-center">
              <p
                className="text-3xl font-black tracking-widest text-white"
                style={{
                  textShadow: "0 0 20px rgba(167,139,250,0.6)",
                }}
              >
                紫苑
              </p>
              <p className="mt-0.5 text-xs font-black tracking-[0.4em] text-violet-300">
                SHION
              </p>
              <p className="mt-1 text-[10px] font-bold text-slate-600">
                リース知性体
              </p>
            </div>
          </div>

          {/* ─── 能力ノードのラベル（HTML） ─── */}
          {ORBIT_NODES.map((node) => {
            const rad = degToRad(node.angle);
            const nx = CX + ORBIT_R * Math.cos(rad);
            const ny = CY + ORBIT_R * Math.sin(rad);
            const onRight = nx >= CX;
            const onTop = ny <= CY;
            return (
              <Link
                key={`label-${node.label}`}
                href={node.href}
                className="group absolute"
                style={{
                  left: `${nx}px`,
                  top: `${ny}px`,
                  transform: `translate(${onRight ? "16px" : "calc(-100% - 16px)"}, ${onTop ? "-100%" : "4px"})`,
                }}
              >
                <div
                  className="rounded-xl border px-2.5 py-1.5 transition-all group-hover:scale-105"
                  style={{
                    borderColor: `${node.color}35`,
                    backgroundColor: `${node.color}0d`,
                  }}
                >
                  <p
                    className="whitespace-nowrap text-[11px] font-black"
                    style={{ color: node.color }}
                  >
                    {node.label}
                  </p>
                  <p className="whitespace-nowrap text-[9px] font-bold text-slate-600">
                    {node.sublabel}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>

        {/* ══════════════════════════════════════
            紫苑の言葉（軌道図の下）
        ══════════════════════════════════════ */}
        <div
          className="mt-12 flex flex-col items-center text-center"
          style={{
            opacity: poemVis ? 1 : 0,
            transform: poemVis ? "translateY(0)" : "translateY(6px)",
            transition: "opacity 0.7s ease, transform 0.7s ease",
          }}
        >
          <div className="mb-4 h-px w-20 bg-gradient-to-r from-transparent via-violet-400/40 to-transparent" />
          <div className="space-y-1">
            {poem.map((line, i) => (
              <p
                key={`${poemIdx}-${i}`}
                className={[
                  "font-black leading-[1.6] tracking-[0.04em]",
                  i === 1 ? "text-xl text-violet-200 sm:text-2xl" : "text-base text-white/75 sm:text-lg",
                ].join(" ")}
              >
                {line}
              </p>
            ))}
          </div>
          <p className="mt-4 text-[11px] font-bold text-slate-600">
            — 紫苑（リース知性体 / AURION）
          </p>
          <div className="mt-4 h-px w-20 bg-gradient-to-r from-transparent via-violet-400/40 to-transparent" />
        </div>

        {/* 詩切り替えドット */}
        <div className="mt-5 flex gap-2">
          {POEMS.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setPoemVis(false);
                setTimeout(() => {
                  setPoemIdx(i);
                  setMoodIdx(i % PORTRAIT_MOODS.length);
                  setPoemVis(true);
                }, 350);
              }}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === poemIdx ? "w-8 bg-violet-400" : "w-1.5 bg-slate-700 hover:bg-slate-500"
              }`}
            />
          ))}
        </div>

        {/* ── キャッチコピー ── */}
        <p className="mt-10 max-w-xl text-center text-base font-bold leading-relaxed text-slate-400">
          AIは審査を代行しない。
          <strong className="text-white">審査を覚える。</strong>
          <br />
          人間と共に考え、迷い、判断を育てるリースファイナンスAI。
        </p>

        {/* ── ライブ統計 ── */}
        <div className="mt-10 grid w-full max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "審査案件", value: displayStats.total_cases, unit: "件", color: "text-violet-400" },
            {
              label: "成約率",
              value: displayStats.closed_rate.toFixed(1),
              unit: "%",
              color: "text-cyan-400",
            },
            {
              label: "平均スコア",
              value: displayStats.avg_score.toFixed(1),
              unit: "pt",
              color: "text-emerald-400",
            },
            {
              label: "アクティブルール",
              value: displayStats.active_rules,
              unit: "本",
              color: "text-fuchsia-400",
            },
          ].map((s) => (
            <div
              key={s.label}
              className="flex flex-col items-center rounded-2xl border border-white/[0.07] bg-white/[0.03] px-4 py-4 backdrop-blur-sm"
            >
              <div className={`text-2xl font-black tabular-nums ${s.color}`}>
                {s.value}
                <span className="ml-1 text-xs font-bold text-slate-600">{s.unit}</span>
              </div>
              <div className="mt-1 text-xs font-bold text-slate-600">{s.label}</div>
            </div>
          ))}
        </div>

        {/* ── テックスタック ── */}
        <div className="mt-8 flex flex-wrap justify-center gap-2">
          {[
            { label: "Gemini 2.5 Flash", cls: "text-blue-300 border-blue-500/30 bg-blue-900/20" },
            { label: "Cloud Run", cls: "text-teal-300 border-teal-500/30 bg-teal-900/20" },
            { label: "ChromaDB", cls: "text-violet-300 border-violet-500/30 bg-violet-900/20" },
            { label: "RF / LR / LGBM", cls: "text-emerald-300 border-emerald-500/30 bg-emerald-900/20" },
            { label: "Next.js 16", cls: "text-slate-300 border-slate-600/50 bg-slate-800/40" },
            { label: "FastAPI", cls: "text-green-300 border-green-500/30 bg-green-900/20" },
          ].map((b) => (
            <span
              key={b.label}
              className={`rounded-full border px-3 py-1 text-xs font-black backdrop-blur-sm ${b.cls}`}
            >
              {b.label}
            </span>
          ))}
        </div>

        {/* ── CTAボタン ── */}
        <div className="mt-10 flex flex-wrap justify-center gap-3">
          {CTAS.map((cta, i) => (
            <Link
              key={i}
              href={cta.href}
              className={[
                "group flex items-center gap-3 rounded-2xl px-5 py-3.5 text-sm font-black text-white transition-all duration-300",
                cta.primary
                  ? "bg-gradient-to-r from-violet-600 to-fuchsia-600 shadow-lg hover:shadow-[0_0_30px_rgba(139,92,246,0.5)]"
                  : "border border-white/10 bg-white/[0.04] backdrop-blur-sm hover:bg-white/[0.08]",
              ].join(" ")}
            >
              <cta.Icon
                className={`h-4 w-4 ${cta.primary ? "text-white" : "text-slate-400 group-hover:text-white"} transition-colors`}
              />
              <div className="text-left">
                <div className="leading-tight">{cta.label}</div>
                <div
                  className={`text-[10px] font-medium leading-tight ${
                    cta.primary
                      ? "text-white/70"
                      : "text-slate-500 group-hover:text-slate-400"
                  } transition-colors`}
                >
                  {cta.sublabel}
                </div>
              </div>
              <ArrowRight
                className={`h-4 w-4 transition-transform duration-200 group-hover:translate-x-1 ${
                  cta.primary ? "text-white/70" : "text-slate-700 group-hover:text-slate-400"
                }`}
              />
            </Link>
          ))}
        </div>

        {/* ── フッター ── */}
        <p className="mt-20 text-xs text-slate-800">
          tune_lease_55 · Powered by Gemini · Built with Claude Code · 2026
        </p>
      </div>
    </div>
  );
}
