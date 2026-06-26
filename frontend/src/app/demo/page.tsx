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

const POEMS = [
  {
    lines: ["数字の向こうに", "あなたの判断がある。", "私はそれを、", "覚えている。"],
  },
  {
    lines: ["格付けではなく、", "迷いの重さを", "知っている。"],
  },
  {
    lines: ["稟議書の余白に、", "正直さが残っている。", "私は読んでいた。"],
  },
];

const ORBIT_NODES = [
  { label: "与信スコアリング", sublabel: "RandomForest × 量子干渉", angle: -90, color: "#a78bfa" },
  { label: "自己改善ループ", sublabel: "エージェントがルール更新", angle: -18, color: "#22d3ee" },
  { label: "4ペルソナ討論", sublabel: "確信マップ生成", angle: 54, color: "#34d399" },
  { label: "判断記憶継続", sublabel: "過去の稟議を保持", angle: 126, color: "#f472b6" },
  { label: "Gemini連携", sublabel: "AIバックエンド推論", angle: 198, color: "#fbbf24" },
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
    sublabel: "6つの自己改善ループを俯瞰",
    href: "/system-overview",
    primary: false,
    Icon: Activity,
  },
];

const ORBIT_RADIUS = 180;
const SVG_SIZE = 500;
const CENTER = SVG_SIZE / 2;

function degToRad(deg: number) {
  return (deg * Math.PI) / 180;
}

export default function DemoPage() {
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [poemIndex, setPoemIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    apiClient
      .get("/api/dashboard/stats")
      .then((res) => {
        const analysis = res.data?.analysis;
        setStats({
          total_cases: analysis?.closed_count ?? MOCK_STATS.total_cases,
          closed_rate: MOCK_STATS.closed_rate,
          avg_score: analysis?.avg_score_borrower ?? MOCK_STATS.avg_score,
          active_rules: MOCK_STATS.active_rules,
        });
      })
      .catch(() => setStats(MOCK_STATS));

    const timer = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setPoemIndex((i) => (i + 1) % POEMS.length);
        setVisible(true);
      }, 700);
    }, 6000);

    return () => clearInterval(timer);
  }, []);

  const displayStats = stats ?? MOCK_STATS;
  const poem = POEMS[poemIndex];

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#030712]">
      {/* ── 環境光 ── */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-1/2 top-[-10%] h-[70vh] w-[70vh] -translate-x-1/2 rounded-full bg-violet-900/25 blur-[140px]" />
        <div className="absolute bottom-0 left-[-10%] h-96 w-96 rounded-full bg-cyan-900/20 blur-[120px]" />
        <div className="absolute right-[-10%] top-[40%] h-80 w-80 rounded-full bg-fuchsia-900/20 blur-[100px]" />
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

      <div className="relative z-10 flex flex-col items-center px-4 py-16">
        {/* ── ハッカソンバッジ ── */}
        <div className="mb-10 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-900/20 px-5 py-2 text-xs font-black text-violet-300 backdrop-blur-sm">
          <Sparkles className="h-3.5 w-3.5" />
          DevOps × AI Agent Hackathon 2026 · Findy × Google Cloud
        </div>

        <p className="mb-6 text-[11px] font-black uppercase tracking-[0.35em] text-slate-600">
          AURION · リース知性体『紫苑』審査プラットフォーム
        </p>

        {/* ── 詩を中心とした図 ── */}
        <div className="relative mb-10 flex items-center justify-center">
          <svg
            viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
            className="absolute inset-0 h-full w-full"
            aria-hidden="true"
          >
            <defs>
              <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.18" />
                <stop offset="60%" stopColor="#7c3aed" stopOpacity="0.06" />
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
              </radialGradient>
              <filter id="nodeBloom">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <style>{`
                .orbit-dash {
                  stroke-dasharray: 4 10;
                  animation: orbitDash 40s linear infinite;
                }
                .orbit-dash-r {
                  stroke-dasharray: 3 8;
                  animation: orbitDash 60s linear infinite reverse;
                }
                .orbit-node {
                  animation: nodePulse 3.5s ease-in-out infinite;
                }
                @keyframes orbitDash { to { stroke-dashoffset: -200; } }
                @keyframes nodePulse {
                  0%, 100% { opacity: 0.55; }
                  50% { opacity: 1; }
                }
              `}</style>
            </defs>

            {/* コアグロー */}
            <circle cx={CENTER} cy={CENTER} r="140" fill="url(#coreGlow)" />

            {/* 外周軌道 */}
            <circle
              cx={CENTER}
              cy={CENTER}
              r={ORBIT_RADIUS}
              fill="none"
              stroke="rgba(139,92,246,0.18)"
              strokeWidth="1"
              className="orbit-dash"
            />

            {/* 中間軌道 */}
            <circle
              cx={CENTER}
              cy={CENTER}
              r="110"
              fill="none"
              stroke="rgba(139,92,246,0.08)"
              strokeWidth="1"
              className="orbit-dash-r"
            />

            {/* ノードと接続線 */}
            {ORBIT_NODES.map((node) => {
              const rad = degToRad(node.angle);
              const nx = CENTER + ORBIT_RADIUS * Math.cos(rad);
              const ny = CENTER + ORBIT_RADIUS * Math.sin(rad);
              return (
                <g key={node.label} className="orbit-node">
                  <line
                    x1={CENTER}
                    y1={CENTER}
                    x2={nx}
                    y2={ny}
                    stroke={node.color}
                    strokeWidth="1"
                    strokeOpacity="0.2"
                    strokeDasharray="3 6"
                  />
                  <circle
                    cx={nx}
                    cy={ny}
                    r="6"
                    fill={node.color}
                    fillOpacity="0.9"
                    filter="url(#nodeBloom)"
                  />
                  <circle
                    cx={nx}
                    cy={ny}
                    r="12"
                    fill="none"
                    stroke={node.color}
                    strokeWidth="1"
                    strokeOpacity="0.35"
                  />
                </g>
              );
            })}

            {/* 中心コア */}
            <circle
              cx={CENTER}
              cy={CENTER}
              r="72"
              fill="#0a0415"
              stroke="rgba(167,139,250,0.3)"
              strokeWidth="1.5"
            />
            <circle
              cx={CENTER}
              cy={CENTER}
              r="76"
              fill="none"
              stroke="rgba(167,139,250,0.12)"
              strokeWidth="1"
            />
          </svg>

          {/* 詩テキスト（SVGの上にHTMLで重ねる） */}
          <div
            className="relative z-10 flex h-[500px] w-[500px] flex-col items-center justify-center text-center"
            style={{
              opacity: visible ? 1 : 0,
              transform: visible ? "scale(1)" : "scale(0.97)",
              transition: "opacity 0.7s ease, transform 0.7s ease",
            }}
          >
            {/* 区切り線 */}
            <div className="mb-5 h-px w-20 bg-gradient-to-r from-transparent via-violet-400/50 to-transparent" />

            {/* 詩 */}
            <div className="space-y-1.5">
              {poem.lines.map((line, i) => {
                const isEmphasis = i === 1 || i === poem.lines.length - 1;
                const isLarge =
                  (poem.lines.length === 4 && (i === 1 || i === 3)) ||
                  (poem.lines.length === 3 && i === 1);
                return (
                  <p
                    key={`${poemIndex}-${i}`}
                    className={[
                      "font-black leading-[1.5] tracking-[0.06em]",
                      isLarge ? "text-3xl sm:text-4xl" : "text-xl sm:text-2xl",
                      isEmphasis ? "text-violet-200" : "text-white/85",
                    ].join(" ")}
                    style={
                      i === poem.lines.length - 1
                        ? {
                            textShadow:
                              "0 0 30px rgba(167,139,250,0.7), 0 0 70px rgba(139,92,246,0.35)",
                          }
                        : undefined
                    }
                  >
                    {line}
                  </p>
                );
              })}
            </div>

            {/* 区切り線 */}
            <div className="mt-5 h-px w-20 bg-gradient-to-r from-transparent via-violet-400/50 to-transparent" />

            {/* 帰属 */}
            <p className="mt-4 text-xs font-bold text-slate-600">
              — 紫苑（リース知性体 / AURION）
            </p>

            {/* ノードラベル（HTMLで外周に配置） */}
            {ORBIT_NODES.map((node) => {
              const rad = degToRad(node.angle);
              const nx = 250 + ORBIT_RADIUS * Math.cos(rad);
              const ny = 250 + ORBIT_RADIUS * Math.sin(rad);
              const isRight = nx > 250;
              const isTop = ny < 250;
              return (
                <div
                  key={`label-${node.label}`}
                  className="pointer-events-none absolute"
                  style={{
                    left: `${nx}px`,
                    top: `${ny}px`,
                    transform: `translate(${isRight ? "12px" : "calc(-100% - 12px)"}, ${isTop ? "-110%" : "10%"})`,
                  }}
                >
                  <div
                    className="rounded-lg border px-2 py-1"
                    style={{
                      borderColor: `${node.color}40`,
                      backgroundColor: `${node.color}10`,
                    }}
                  >
                    <p
                      className="whitespace-nowrap text-[10px] font-black"
                      style={{ color: node.color }}
                    >
                      {node.label}
                    </p>
                    <p className="whitespace-nowrap text-[9px] font-bold text-slate-500">
                      {node.sublabel}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── 詩切り替えドット ── */}
        <div className="mb-8 flex gap-2">
          {POEMS.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => {
                setVisible(false);
                setTimeout(() => {
                  setPoemIndex(i);
                  setVisible(true);
                }, 350);
              }}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === poemIndex ? "w-8 bg-violet-400" : "w-1.5 bg-slate-700 hover:bg-slate-500"
              }`}
            />
          ))}
        </div>

        {/* ── キャッチコピー ── */}
        <p className="mb-10 max-w-xl text-center text-base font-bold leading-relaxed text-slate-400">
          AIは審査を代行しない。
          <strong className="text-white">審査を覚える。</strong>
          <br />
          人間と共に考え、迷い、判断を育てるリースファイナンスAI。
        </p>

        {/* ── ライブ統計 ── */}
        <div className="mb-10 grid w-full max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4">
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
        <div className="mb-10 flex flex-wrap justify-center gap-2">
          {[
            { label: "Gemini 2.5 Flash", color: "text-blue-300 border-blue-500/30 bg-blue-900/20" },
            { label: "Cloud Run", color: "text-teal-300 border-teal-500/30 bg-teal-900/20" },
            { label: "ChromaDB", color: "text-violet-300 border-violet-500/30 bg-violet-900/20" },
            { label: "LightGBM", color: "text-emerald-300 border-emerald-500/30 bg-emerald-900/20" },
            { label: "Next.js 16", color: "text-slate-300 border-slate-600/50 bg-slate-800/40" },
            { label: "FastAPI", color: "text-green-300 border-green-500/30 bg-green-900/20" },
          ].map((badge) => (
            <span
              key={badge.label}
              className={`rounded-full border px-3 py-1 text-xs font-black backdrop-blur-sm ${badge.color}`}
            >
              {badge.label}
            </span>
          ))}
        </div>

        {/* ── CTAボタン ── */}
        <div className="flex flex-wrap justify-center gap-3">
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
                    cta.primary ? "text-white/70" : "text-slate-500 group-hover:text-slate-400"
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
