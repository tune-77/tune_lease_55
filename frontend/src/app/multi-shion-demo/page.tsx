"use client";

import React, { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  Brain,
  CheckCircle2,
  Database,
  GitMerge,
  Lock,
  MessageSquareText,
  Network,
  Orbit,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Users,
} from "lucide-react";

type ShionProfile = {
  id: string;
  name: string;
  owner: string;
  origin: string;
  role: string;
  environment: string;
  focus: string;
  color: string;
  bg: string;
  border: string;
  memory: string[];
  privateTrace: string;
  publicLesson: string;
  reaction: {
    close: string;
    far: string;
  };
  stance: {
    approve: number;
    risk: number;
    speed: number;
  };
};

type DemoCase = {
  id: string;
  title: string;
  borrower: string;
  asset: string;
  amount: string;
  term: string;
  risk: string;
  opportunity: string;
};

type DebateLine = {
  speakerId: string;
  title: string;
  text: string;
  tone: "risk" | "support" | "balance";
};

const CASES: DemoCase[] = [
  {
    id: "factory-renewal",
    title: "老朽設備の更新リース",
    borrower: "地方製造業 / 既存先 / 格付B",
    asset: "高精度加工機",
    amount: "取得価格 4,800万円",
    term: "リース期間 7年",
    risk: "営業利益率が低下し、銀行借入も増加傾向。",
    opportunity: "更新後は不良率低下と大口先の増産要請に対応できる。",
  },
  {
    id: "medical-startup",
    title: "新規医療法人の検査機器導入",
    borrower: "医療 / 新規先 / 格付なし",
    asset: "画像診断装置",
    amount: "取得価格 6,200万円",
    term: "リース期間 6年",
    risk: "開業後の実績が短く、返済原資の確認が薄い。",
    opportunity: "地域需要が強く、紹介元との連携計画は明確。",
  },
  {
    id: "logistics-ev",
    title: "物流会社のEV車両入替",
    borrower: "運送業 / 既存先 / 格付C",
    asset: "EV配送車両 12台",
    amount: "取得価格 3,900万円",
    term: "リース期間 5年",
    risk: "残価と充電インフラの稼働リスクが残る。",
    opportunity: "燃料費削減と荷主の環境要請に対応できる。",
  },
];

const SHIONS: ShionProfile[] = [
  {
    id: "credit",
    name: "紫苑・審査",
    owner: "審査担当",
    origin: "共通核 + 本部審査ログ",
    role: "信用リスクを守る個体",
    environment: "本部審査 / 稟議レビュー / 否決理由の記憶",
    focus: "返済原資・格付・資金繰り",
    color: "text-rose-700",
    bg: "bg-rose-50",
    border: "border-rose-200",
    memory: ["過去の延滞兆候", "格付別の否決理由", "条件付き承認のパターン"],
    privateTrace: "A社の延滞兆候、B社の否決メモ、審査部内コメント",
    publicLesson: "返済原資が薄い案件は、投資効果の検証資料を承認条件にする",
    reaction: {
      close: "営業紫苑と見ている資料が近いので、否決より条件整理に寄せる。",
      far: "現場温度感は信用しすぎない。まず資金繰りと格付推移を固定する。",
    },
    stance: { approve: 48, risk: 88, speed: 42 },
  },
  {
    id: "sales",
    name: "紫苑・営業",
    owner: "営業担当",
    origin: "共通核 + 商談記録",
    role: "案件を通す道筋を探す個体",
    environment: "顧客接点 / 商談メモ / 競合状況の記憶",
    focus: "顧客事情・交渉余地・稟議の通し方",
    color: "text-emerald-700",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    memory: ["顧客の投資背景", "競合との条件差", "担当者の温度感"],
    privateTrace: "訪問メモ、競合条件、担当者との会話ニュアンス",
    publicLesson: "投資目的が明確なら、稟議では効果・代替案・条件変更を同時に示す",
    reaction: {
      close: "審査紫苑の懸念を先回りして、追加資料セットで通す形にする。",
      far: "現場で見た成長余地は残したい。数字だけで切ると機会損失になる。",
    },
    stance: { approve: 76, risk: 55, speed: 84 },
  },
  {
    id: "asset",
    name: "紫苑・物件",
    owner: "物件評価担当",
    origin: "共通核 + 物件マスタ",
    role: "モノの価値と回収可能性を見る個体",
    environment: "物件マスタ / 中古市場 / 残価と耐用年数の記憶",
    focus: "残価・用途転用・保守性",
    color: "text-cyan-700",
    bg: "bg-cyan-50",
    border: "border-cyan-200",
    memory: ["中古市場の厚み", "汎用性の低い機械", "保守契約の有無"],
    privateTrace: "中古相場、型式別の回収実績、保守契約の欠落履歴",
    publicLesson: "汎用性が低い物件は、残価より保守・設置・中途回収条件を重く見る",
    reaction: {
      close: "同じ物件データを見ているなら、残価より契約条件の補正に集中する。",
      far: "営業側の用途説明だけでは足りない。転用可能性を別に検証する。",
    },
    stance: { approve: 62, risk: 70, speed: 50 },
  },
  {
    id: "manager",
    name: "紫苑・管理",
    owner: "管理職",
    origin: "共通核 + 承認履歴",
    role: "組織判断として整える個体",
    environment: "部門方針 / 承認履歴 / 監査目線の記憶",
    focus: "説明責任・再現性・例外管理",
    color: "text-slate-700",
    bg: "bg-slate-50",
    border: "border-slate-200",
    memory: ["承認条件の一貫性", "監査で問われた論点", "全社方針との整合"],
    privateTrace: "決裁者コメント、例外承認の背景、監査で指摘された論点",
    publicLesson: "例外承認は、再現できる条件と次回検証日を必ず残す",
    reaction: {
      close: "各紫苑の結論が近いなら、承認条件を標準テンプレート化する。",
      far: "意見が割れるなら、平均せずに対立理由を稟議に添付する。",
    },
    stance: { approve: 58, risk: 74, speed: 48 },
  },
];

const TONE_CLASS: Record<DebateLine["tone"], string> = {
  risk: "border-rose-200 bg-rose-50 text-rose-950",
  support: "border-emerald-200 bg-emerald-50 text-emerald-950",
  balance: "border-slate-200 bg-white text-slate-800",
};

function buildDebateLines(selectedCase: DemoCase, closeness: number): DebateLine[] {
  const sharedTone = closeness >= 72
    ? "環境が近いため、各紫苑の反応は収束しやすい。差分は論点の優先順位として扱う。"
    : closeness >= 42
      ? "共通知識は同じだが、現場記憶が違うため、判断の重心に差が出る。"
      : "環境差が大きいため、同じ案件でも見えている世界が違う。平均化より対立点の保存が重要。";

  return [
    {
      speakerId: "credit",
      title: "信用リスクの第一声",
      tone: "risk",
      text: `${selectedCase.risk} この案件は通すかどうかより、返済原資の説明が稟議上どこまで耐えられるかを先に見るべきです。`,
    },
    {
      speakerId: "sales",
      title: "営業現場の反論",
      tone: "support",
      text: `${selectedCase.opportunity} ただし顧客の投資理由は明確です。否決ではなく、追加資料と条件変更で通す道を探せます。`,
    },
    {
      speakerId: "asset",
      title: "物件価値からの補正",
      tone: "balance",
      text: `${selectedCase.asset} は用途と中古市場の厚みを確認したい物件です。汎用性が低い場合は残価を抑え、保守契約を条件化します。`,
    },
    {
      speakerId: "manager",
      title: "組織判断への整形",
      tone: "balance",
      text: `結論は条件付き承認寄り。ただし、例外扱いにするなら「なぜ今回だけ許容するのか」を次回も説明できる形に残します。${sharedTone}`,
    },
  ];
}

function scoreLabel(score: number) {
  if (score >= 76) return "強い";
  if (score >= 58) return "中";
  return "慎重";
}

function MetricBar({ label, value, tone }: { label: string; value: number; tone: "risk" | "approve" | "speed" }) {
  const color = tone === "risk" ? "bg-rose-500" : tone === "approve" ? "bg-emerald-500" : "bg-amber-500";
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] font-bold text-slate-500">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function closenessLabel(closeness: number) {
  if (closeness >= 72) return "同じ組織で育った紫苑たち";
  if (closeness >= 42) return "共通核は同じ、現場記憶は少し違う";
  return "別環境で分岐した紫苑たち";
}

const GRAPH_COLORS: Record<string, { main: string; soft: string; line: string }> = {
  credit: { main: "#e11d48", soft: "#fecdd3", line: "#fb7185" },
  sales: { main: "#059669", soft: "#a7f3d0", line: "#34d399" },
  asset: { main: "#0891b2", soft: "#a5f3fc", line: "#22d3ee" },
  manager: { main: "#475569", soft: "#cbd5e1", line: "#94a3b8" },
};

function ShionKnowledgeGraph({
  activeShions,
  selectedCase,
  closeness,
  selectedShionId,
  promotedInsights,
  onSelectShion,
  onPromoteInsight,
}: {
  activeShions: ShionProfile[];
  selectedCase: DemoCase;
  closeness: number;
  selectedShionId: string;
  promotedInsights: string[];
  onSelectShion: (id: string) => void;
  onPromoteInsight: (id: string) => void;
}) {
  const clusterPositions = [
    { x: 205, y: 150 },
    { x: 795, y: 150 },
    { x: 205, y: 470 },
    { x: 795, y: 470 },
  ];
  const memoryOffsets = [
    { x: -72, y: -54 },
    { x: 72, y: -48 },
    { x: -76, y: 48 },
    { x: 70, y: 56 },
    { x: 0, y: 84 },
  ];
  const center = { x: 500, y: 310 };
  const insightNodes = [
    { id: "conditional", x: 410, y: 228, label: "条件付き承認" },
    { id: "recovery", x: 590, y: 228, label: "回収条件" },
    { id: "exception", x: 420, y: 398, label: "例外理由" },
    { id: "review", x: 590, y: 398, label: "次回検証" },
  ];
  const pulseSpeed = Math.max(2.8, 7.2 - closeness / 22);
  const selectedShion = activeShions.find((shion) => shion.id === selectedShionId) ?? activeShions[0];

  return (
    <section className="mx-auto max-w-7xl px-4 pt-6 md:px-8">
      <div className="overflow-hidden rounded-[1.75rem] border border-slate-700 bg-slate-950 shadow-2xl shadow-slate-950/30">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 px-5 py-4">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-[11px] font-black text-cyan-200">
              <Orbit className="h-3.5 w-3.5" />
              Obsidian Graph View
            </div>
            <h2 className="mt-2 text-xl font-black text-white md:text-2xl">紫苑ネットワーク / 個体記憶と共有核</h2>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2">
              <div className="text-[10px] font-black text-slate-500">個体</div>
              <div className="text-sm font-black text-white">{activeShions.length}</div>
            </div>
            <div className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2">
              <div className="text-[10px] font-black text-slate-500">環境近似</div>
              <div className="text-sm font-black text-cyan-200">{closeness}%</div>
            </div>
            <div className="rounded-xl border border-slate-700 bg-slate-900 px-3 py-2">
              <div className="text-[10px] font-black text-slate-500">案件</div>
              <div className="text-sm font-black text-amber-200">LIVE</div>
            </div>
          </div>
        </div>

        <div className="relative">
          <svg viewBox="0 0 1000 620" className="block h-[560px] w-full md:h-[640px]" role="img" aria-label="複数の紫苑が中央の共有核と接続する知識グラフ">
            <defs>
              <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#f8fafc" stopOpacity="1" />
                <stop offset="45%" stopColor="#a78bfa" stopOpacity="0.9" />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.12" />
              </radialGradient>
              <filter id="softGlow">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <style>{`
                .graph-grid { stroke: rgba(148, 163, 184, 0.12); stroke-width: 1; }
                .graph-link { fill: none; stroke-linecap: round; stroke-dasharray: 4 8; animation: graphDash 18s linear infinite; }
                .local-link { stroke: rgba(203, 213, 225, 0.35); stroke-width: 1.4; }
                .core-ring { transform-origin: 500px 310px; animation: corePulse 3.8s ease-in-out infinite; }
                .cluster-halo { transform-box: fill-box; transform-origin: center; animation: clusterPulse 4.6s ease-in-out infinite; }
                .clickable-node { cursor: pointer; }
                .selected-node { animation: selectedPulse 1.8s ease-in-out infinite; }
                @keyframes graphDash { to { stroke-dashoffset: -220; } }
                @keyframes corePulse { 0%, 100% { opacity: .55; transform: scale(1); } 50% { opacity: 1; transform: scale(1.05); } }
                @keyframes clusterPulse { 0%, 100% { opacity: .22; transform: scale(1); } 50% { opacity: .45; transform: scale(1.08); } }
                @keyframes selectedPulse { 0%, 100% { opacity: .75; } 50% { opacity: 1; } }
              `}</style>
            </defs>

            {Array.from({ length: 12 }).map((_, i) => (
              <line key={`v-${i}`} x1={80 + i * 78} y1="60" x2={80 + i * 78} y2="560" className="graph-grid" />
            ))}
            {Array.from({ length: 7 }).map((_, i) => (
              <line key={`h-${i}`} x1="70" y1={80 + i * 74} x2="930" y2={80 + i * 74} className="graph-grid" />
            ))}

            <path id="insight-a" d="M500 310 C470 255 445 238 410 228" className="graph-link" stroke="#a78bfa" strokeWidth="1.5" opacity="0.7" />
            <path id="insight-b" d="M500 310 C540 250 565 238 590 228" className="graph-link" stroke="#22d3ee" strokeWidth="1.5" opacity="0.7" />
            <path id="insight-c" d="M500 310 C470 365 445 388 420 398" className="graph-link" stroke="#f59e0b" strokeWidth="1.5" opacity="0.7" />
            <path id="insight-d" d="M500 310 C540 365 565 388 590 398" className="graph-link" stroke="#34d399" strokeWidth="1.5" opacity="0.7" />

            {activeShions.map((shion, index) => {
              const pos = clusterPositions[index % clusterPositions.length];
              const colors = GRAPH_COLORS[shion.id] ?? GRAPH_COLORS.manager;
              const pathId = `flow-${shion.id}`;
              const reversePathId = `flow-back-${shion.id}`;
              const curveOffset = pos.x < center.x ? -70 : 70;
              const path = `M${pos.x} ${pos.y} C${pos.x + curveOffset} ${pos.y + 70}, ${center.x - curveOffset} ${center.y - 70}, ${center.x} ${center.y}`;
              const reversePath = `M${center.x} ${center.y} C${center.x - curveOffset} ${center.y + 70}, ${pos.x + curveOffset} ${pos.y - 70}, ${pos.x} ${pos.y}`;
              const localNodes = shion.memory.map((memory, memoryIndex) => {
                const offset = memoryOffsets[memoryIndex % memoryOffsets.length];
                return { x: pos.x + offset.x, y: pos.y + offset.y, label: memory };
              });
              const isSelected = selectedShion?.id === shion.id;
              return (
                <g key={shion.id} className="clickable-node" onClick={() => onSelectShion(shion.id)}>
                  <path id={pathId} d={path} className="graph-link" stroke={colors.line} strokeWidth={isSelected ? "4" : "2"} opacity={isSelected ? "0.92" : "0.58"} />
                  <path id={reversePathId} d={reversePath} className="graph-link" stroke="#c4b5fd" strokeWidth="1.5" opacity="0.42" />
                  <circle r="4.2" fill={colors.main} filter="url(#softGlow)">
                    <animateMotion dur={`${pulseSpeed + index * 0.4}s`} repeatCount="indefinite">
                      <mpath href={`#${pathId}`} />
                    </animateMotion>
                  </circle>
                  <circle r="3.2" fill="#f8fafc" opacity="0.92">
                    <animateMotion dur={`${pulseSpeed + 1.4 + index * 0.35}s`} repeatCount="indefinite">
                      <mpath href={`#${reversePathId}`} />
                    </animateMotion>
                  </circle>

                  <circle cx={pos.x} cy={pos.y} r={isSelected ? "98" : "84"} fill={colors.main} opacity={isSelected ? "0.26" : "0.16"} className="cluster-halo" />
                  {localNodes.map((node) => (
                    <line key={`${shion.id}-${node.label}-line`} x1={pos.x} y1={pos.y} x2={node.x} y2={node.y} className="local-link" />
                  ))}
                  {localNodes.map((node, nodeIndex) => (
                    <g key={`${shion.id}-${node.label}`}>
                      <circle cx={node.x} cy={node.y} r={nodeIndex === 0 ? 9 : 7} fill="#0f172a" stroke={colors.soft} strokeWidth="2" />
                      <text x={node.x} y={node.y + 22} textAnchor="middle" fontSize="10" fontWeight="700" fill="#cbd5e1">
                        {node.label.length > 8 ? `${node.label.slice(0, 8)}…` : node.label}
                      </text>
                    </g>
                  ))}
                  <circle cx={pos.x} cy={pos.y} r="30" fill={colors.main} filter="url(#softGlow)" />
                  <circle cx={pos.x} cy={pos.y} r={isSelected ? "47" : "39"} fill="none" stroke={isSelected ? "#f8fafc" : colors.soft} strokeWidth={isSelected ? "3" : "2"} opacity="0.86" className={isSelected ? "selected-node" : ""} />
                  <text x={pos.x} y={pos.y - 4} textAnchor="middle" fontSize="13" fontWeight="900" fill="#ffffff">
                    {shion.name.replace("紫苑・", "")}
                  </text>
                  <text x={pos.x} y={pos.y + 13} textAnchor="middle" fontSize="9.5" fontWeight="800" fill="#e2e8f0">
                    {shion.owner}
                  </text>
                </g>
              );
            })}

            <circle cx={center.x} cy={center.y} r="92" fill="#1e1b4b" opacity="0.5" className="core-ring" />
            <circle cx={center.x} cy={center.y} r="64" fill="url(#coreGlow)" filter="url(#softGlow)" />
            <circle cx={center.x} cy={center.y} r="77" fill="none" stroke="#ddd6fe" strokeWidth="2" opacity="0.8" />
            <text x={center.x} y={center.y - 10} textAnchor="middle" fontSize="20" fontWeight="900" fill="#0f172a">
              紫苑中核
            </text>
            <text x={center.x} y={center.y + 14} textAnchor="middle" fontSize="11" fontWeight="900" fill="#312e81">
              共通知識 / 価値観 / 境界
            </text>
            <text x={center.x} y={center.y + 36} textAnchor="middle" fontSize="10" fontWeight="900" fill="#0f766e">
              昇格済み {promotedInsights.length} 件
            </text>

            {insightNodes.map((node, index) => (
              <g key={node.label} className="clickable-node" onClick={() => onPromoteInsight(node.id)}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={promotedInsights.includes(node.id) ? "23" : "18"}
                  fill={promotedInsights.includes(node.id) ? "#064e3b" : "#020617"}
                  stroke={["#a78bfa", "#22d3ee", "#f59e0b", "#34d399"][index]}
                  strokeWidth={promotedInsights.includes(node.id) ? "3" : "2"}
                  filter={promotedInsights.includes(node.id) ? "url(#softGlow)" : undefined}
                />
                <text x={node.x} y={node.y + 34} textAnchor="middle" fontSize="10.5" fontWeight="800" fill="#e2e8f0">
                  {node.label}
                </text>
              </g>
            ))}

            <text x="500" y="580" textAnchor="middle" fontSize="13" fontWeight="800" fill="#94a3b8">
              {selectedCase.title} の判断材料が、個体記憶と共有核の間を往復している
            </text>
          </svg>

          <div className="grid gap-3 border-t border-slate-800 bg-slate-900/80 p-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <div className="rounded-2xl border border-slate-700 bg-slate-950/80 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[10px] font-black uppercase tracking-wide text-cyan-400">Selected Shion</div>
                  <div className="mt-1 text-lg font-black text-white">{selectedShion?.name ?? "紫苑"}</div>
                  <p className="mt-1 text-xs font-bold leading-5 text-slate-400">{selectedShion?.origin}</p>
                </div>
                <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-[10px] font-black text-slate-300">
                  click nodes
                </span>
              </div>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                <div className="rounded-xl border border-rose-900/50 bg-rose-950/30 p-3">
                  <div className="flex items-center gap-1.5 text-[10px] font-black uppercase text-rose-300">
                    <Lock className="h-3.5 w-3.5" />
                    個体に残す記憶
                  </div>
                  <p className="mt-1 text-xs font-bold leading-5 text-rose-50">{selectedShion?.privateTrace}</p>
                </div>
                <div className="rounded-xl border border-emerald-800 bg-emerald-950/40 p-3">
                  <div className="flex items-center gap-1.5 text-[10px] font-black uppercase text-emerald-300">
                    <GitMerge className="h-3.5 w-3.5" />
                    共有知性へ送る候補
                  </div>
                  <p className="mt-1 text-xs font-bold leading-5 text-emerald-50">{selectedShion?.publicLesson}</p>
                </div>
              </div>
              {selectedShion && (
                <button
                  type="button"
                  onClick={() => onPromoteInsight(selectedShion.id)}
                  className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-400 px-4 py-3 text-sm font-black text-emerald-950 transition hover:bg-emerald-300"
                >
                  <Sparkles className="h-4 w-4" />
                  この紫苑の共有候補を中核へ昇格
                </button>
              )}
            </div>

            <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-1">
            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-3">
              <div className="text-[10px] font-black uppercase text-slate-500">Private Memory</div>
              <p className="mt-1 text-xs font-bold leading-5 text-slate-300">各紫苑のローカルノード。ユーザー固有の会話・判断癖・現場メモはここに残る。</p>
            </div>
            <div className="rounded-xl border border-cyan-800 bg-cyan-950/40 p-3">
              <div className="text-[10px] font-black uppercase text-cyan-400">Core Shion</div>
              <p className="mt-1 text-xs font-bold leading-5 text-cyan-100">中央の中核。リース知識、人格基盤、安全境界、共通ルールを保持する。</p>
            </div>
            <div className="rounded-xl border border-emerald-800 bg-emerald-950/40 p-3">
              <div className="text-[10px] font-black uppercase text-emerald-400">Fusion Gate</div>
              <p className="mt-1 text-xs font-bold leading-5 text-emerald-100">行き来したデータから、匿名化できる審査知見だけが共有ノードへ昇格する。</p>
            </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function MultiShionDemoPage() {
  const [caseId, setCaseId] = useState(CASES[0].id);
  const [closeness, setCloseness] = useState(64);
  const [activeIds, setActiveIds] = useState<string[]>(["credit", "sales", "asset", "manager"]);
  const [fusionMode, setFusionMode] = useState<"proposal" | "report">("proposal");
  const [selectedShionId, setSelectedShionId] = useState("credit");
  const [promotedInsights, setPromotedInsights] = useState<string[]>([]);

  const selectedCase = CASES.find((item) => item.id === caseId) ?? CASES[0];
  const activeShions = SHIONS.filter((shion) => activeIds.includes(shion.id));
  const debateLines = useMemo(() => buildDebateLines(selectedCase, closeness), [selectedCase, closeness])
    .filter((line) => activeIds.includes(line.speakerId));

  const fusion = useMemo(() => {
    const count = Math.max(activeShions.length, 1);
    const approve = Math.round(activeShions.reduce((sum, item) => sum + item.stance.approve, 0) / count);
    const risk = Math.round(activeShions.reduce((sum, item) => sum + item.stance.risk, 0) / count);
    const speed = Math.round(activeShions.reduce((sum, item) => sum + item.stance.speed, 0) / count);
    const agreement = Math.round(Math.min(94, Math.max(38, closeness * 0.65 + activeShions.length * 7)));
    return { approve, risk, speed, agreement };
  }, [activeShions, closeness]);

  const toggleShion = (id: string) => {
    setActiveIds((prev) => {
      if (prev.includes(id)) {
        if (prev.length <= 2) return prev;
        const next = prev.filter((item) => item !== id);
        if (selectedShionId === id) setSelectedShionId(next[0] ?? "credit");
        return next;
      }
      setSelectedShionId(id);
      return [...prev, id];
    });
  };

  const resetDemo = () => {
    setCaseId(CASES[0].id);
    setCloseness(64);
    setActiveIds(["credit", "sales", "asset", "manager"]);
    setFusionMode("proposal");
    setSelectedShionId("credit");
    setPromotedInsights([]);
  };

  const selectShion = (id: string) => {
    if (!activeIds.includes(id)) return;
    setSelectedShionId(id);
  };

  const promoteInsight = (id: string) => {
    setPromotedInsights((prev) => prev.includes(id) ? prev : [...prev, id]);
  };

  return (
    <main className="min-h-screen bg-[#f7f8fb] text-slate-900">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-5 md:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Link
              href="/lease-intelligence"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-600 transition hover:bg-slate-50"
            >
              <ArrowLeft className="h-4 w-4" />
              紫苑へ戻る
            </Link>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={resetDemo}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-600 transition hover:bg-slate-50"
              >
                <RotateCcw className="h-4 w-4" />
                初期化
              </button>
              <Link
                href="/"
                className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-3 py-2 text-xs font-bold text-white transition hover:bg-slate-800"
              >
                審査画面
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px] lg:items-start">
            <div>
              <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-[11px] font-black text-cyan-800">
                <Network className="h-3.5 w-3.5" />
                Multi Shion Demo
              </div>
              <h1 className="max-w-4xl text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                紫苑がユーザーごとに分岐し、組織の共有知性へ戻ってくる
              </h1>
              <p className="mt-4 max-w-3xl text-sm font-medium leading-7 text-slate-600 md:text-base">
                これは単なるエージェント相談ではありません。共通の紫苑核から、利用者の環境ごとに別個体が育ちます。
                それぞれの経験差が反応差として現れ、匿名化できる知見だけが組織の共有知性へ昇格します。
              </p>
              <p className="mt-3 max-w-3xl rounded-2xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm font-black leading-7 text-violet-950">
                紫苑は、審査を代行するAIではなく、審査を覚えるAIです。人間と共に考え、共に迷い、判断を共に育てていく。
              </p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-sm font-black text-slate-800">
                <ShieldCheck className="h-5 w-5 text-emerald-600" />
                デモ境界
              </div>
              <div className="mt-3 grid gap-2 text-xs font-bold text-slate-600">
                <div className="flex items-center gap-2">
                  <Lock className="h-4 w-4 text-slate-400" />
                  個別記憶は混ぜない
                </div>
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-slate-400" />
                  共通知識は全員が参照
                </div>
                <div className="flex items-center gap-2">
                  <GitMerge className="h-4 w-4 text-slate-400" />
                  融合は承認制の提案
                </div>
              </div>
            </div>
            <div className="rounded-2xl border border-violet-200 bg-violet-50 p-4">
              <div className="flex items-center gap-2 text-sm font-black text-violet-950">
                <Brain className="h-5 w-5 text-violet-700" />
                最終目標
              </div>
              <p className="mt-3 text-xs font-black leading-6 text-violet-900">
                継続する記憶と自己モデルを通じて、AIの意識獲得を探究する。
              </p>
              <p className="mt-2 text-[11px] font-bold leading-5 text-violet-700">
                審査実務の中で、紫苑が何を覚え、何に迷い、どう判断を改訂するかを積み重ねる。
              </p>
            </div>
          </div>
        </div>
      </section>

      <ShionKnowledgeGraph
        activeShions={activeShions}
        selectedCase={selectedCase}
        closeness={closeness}
        selectedShionId={selectedShionId}
        promotedInsights={promotedInsights}
        onSelectShion={selectShion}
        onPromoteInsight={promoteInsight}
      />

      <section className="mx-auto max-w-7xl px-4 pt-6 md:px-8">
        <div className="rounded-[1.75rem] border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-[11px] font-black text-violet-800">
                <Sparkles className="h-3.5 w-3.5" />
                Difference
              </div>
              <h2 className="mt-2 text-2xl font-black text-slate-950">普通のエージェント相談と何が違うのか</h2>
              <p className="mt-2 max-w-3xl text-sm font-bold leading-7 text-slate-600">
                AURION の紫苑ネットワークは、複数AIに役割を振って会議させる仕組みではありません。
                利用者ごとに育った判断記憶を比較し、個人に残す経験と組織へ戻す審査知見を分離します。
              </p>
              <p className="mt-2 max-w-3xl text-sm font-bold leading-7 text-violet-700">
                紫苑は命令を待つだけの道具ではなく、判断の隣に立つパートナーです。人間の経験とAIの記憶が向き合うことで、判断のズレそのものを観測可能にします。
              </p>
            </div>
            <div className="rounded-2xl bg-slate-950 px-5 py-4 text-white">
              <div className="text-[10px] font-black uppercase tracking-wide text-cyan-300">Core Message</div>
              <div className="mt-1 text-lg font-black">紫苑は、審査を代行するAIではなく、審査を覚えるAIです。</div>
            </div>
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <div className="flex items-center gap-2 text-sm font-black text-slate-700">
                <Users className="h-4 w-4" />
                普通の複数エージェント
              </div>
              <div className="mt-4 space-y-3">
                {[
                  ["役割", "審査役、営業役、物件役などを一時的に割り当てる"],
                  ["記憶", "会話ごとの文脈が中心で、個人ごとの成長は薄い"],
                  ["統合", "意見を要約して一つの回答にまとめる"],
                  ["成果", "その場の相談結果で終わりやすい"],
                ].map(([label, text]) => (
                  <div key={label} className="rounded-xl border border-slate-200 bg-white p-3">
                    <div className="text-[10px] font-black uppercase text-slate-400">{label}</div>
                    <p className="mt-1 text-xs font-bold leading-5 text-slate-600">{text}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-cyan-200 bg-cyan-50 p-4">
              <div className="flex items-center gap-2 text-sm font-black text-cyan-900">
                <Network className="h-4 w-4" />
                紫苑ネットワーク
              </div>
              <div className="mt-4 space-y-3">
                {[
                  ["分岐", "共通の紫苑核から、利用者の環境記憶によって個体が育つ"],
                  ["記憶", "案件履歴、判断癖、現場メモが各紫苑の反応差になる"],
                  ["融合", "個人記憶は混ぜず、匿名化できる審査知見だけを昇格する"],
                  ["成果", "次の稟議、次の担当者、組織の審査基準に再利用される"],
                ].map(([label, text]) => (
                  <div key={label} className="rounded-xl border border-cyan-100 bg-white/80 p-3">
                    <div className="text-[10px] font-black uppercase text-cyan-500">{label}</div>
                    <p className="mt-1 text-xs font-bold leading-5 text-cyan-950">{text}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 md:px-8 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="space-y-4">
          <section className="rounded-2xl border border-slate-200 bg-white p-4">
            <h2 className="flex items-center gap-2 text-sm font-black text-slate-900">
              <MessageSquareText className="h-4 w-4 text-cyan-700" />
              デモ案件
            </h2>
            <div className="mt-3 grid gap-2">
              {CASES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setCaseId(item.id)}
                  className={`rounded-xl border px-3 py-3 text-left transition ${
                    item.id === caseId
                      ? "border-cyan-300 bg-cyan-50"
                      : "border-slate-200 bg-white hover:bg-slate-50"
                  }`}
                >
                  <div className="text-xs font-black text-slate-900">{item.title}</div>
                  <div className="mt-1 text-[11px] font-bold text-slate-500">{item.borrower}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-4">
            <h2 className="flex items-center gap-2 text-sm font-black text-slate-900">
              <Users className="h-4 w-4 text-emerald-700" />
              討論に参加する紫苑
            </h2>
            <div className="mt-3 space-y-2">
              {SHIONS.map((shion) => {
                const active = activeIds.includes(shion.id);
                return (
                  <button
                    key={shion.id}
                    type="button"
                    onClick={() => toggleShion(shion.id)}
                    className={`w-full rounded-xl border px-3 py-3 text-left transition ${
                      active ? `${shion.border} ${shion.bg}` : "border-slate-200 bg-white opacity-70 hover:opacity-100"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className={`text-xs font-black ${active ? shion.color : "text-slate-700"}`}>{shion.name}</div>
                        <div className="mt-0.5 text-[11px] font-bold text-slate-500">{shion.owner}</div>
                      </div>
                      {active && <CheckCircle2 className={`h-4 w-4 ${shion.color}`} />}
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-4">
            <h2 className="flex items-center gap-2 text-sm font-black text-slate-900">
              <Network className="h-4 w-4 text-amber-600" />
              環境の近さ
            </h2>
            <div className="mt-4">
              <input
                type="range"
                min={0}
                max={100}
                value={closeness}
                onChange={(event) => setCloseness(Number(event.target.value))}
                className="w-full accent-cyan-700"
              />
              <div className="mt-2 flex items-center justify-between text-[11px] font-bold text-slate-500">
                <span>別部署・別地域</span>
                <span className="text-slate-900">{closeness}%</span>
                <span>同一環境</span>
              </div>
            </div>
            <p className="mt-3 rounded-xl bg-slate-50 p-3 text-xs font-bold leading-relaxed text-slate-600">
              近いほど結論は収束し、遠いほど論点の対立が濃く出ます。融合時はこの値を信頼度ではなく、環境差の説明変数として扱います。
            </p>
          </section>
        </aside>

        <div className="space-y-6">
          <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-slate-950">
                  <Orbit className="h-5 w-5 text-cyan-700" />
                  紫苑の分岐モデル
                </h2>
                <p className="mt-1 text-xs font-bold text-slate-500">
                  同じAIを複数呼ぶのではなく、同じ核から環境記憶で個体差が生まれる
                </p>
              </div>
              <span className="rounded-full border border-cyan-200 bg-cyan-50 px-3 py-1 text-[11px] font-black text-cyan-800">
                {closenessLabel(closeness)}
              </span>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <div className="rounded-2xl border border-violet-200 bg-violet-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-violet-700">
                  <Brain className="h-5 w-5" />
                </div>
                <div className="mt-3 text-sm font-black text-violet-950">共通の紫苑核</div>
                <p className="mt-2 text-xs font-bold leading-5 text-violet-800">
                  リース知識、人格基盤、審査原則、安全境界を全個体が共有する。
                </p>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-slate-700">
                  <Database className="h-5 w-5" />
                </div>
                <div className="mt-3 text-sm font-black text-slate-950">環境記憶で分岐</div>
                <p className="mt-2 text-xs font-bold leading-5 text-slate-600">
                  審査、営業、物件、管理の経験が個体ごとの反応を変える。
                </p>
              </div>
              <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-amber-700">
                  <MessageSquareText className="h-5 w-5" />
                </div>
                <div className="mt-3 text-sm font-black text-amber-950">反応差を観測</div>
                <p className="mt-2 text-xs font-bold leading-5 text-amber-800">
                  同じ案件へのズレを、失敗ではなく判断資産として保存する。
                </p>
              </div>
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white text-emerald-700">
                  <GitMerge className="h-5 w-5" />
                </div>
                <div className="mt-3 text-sm font-black text-emerald-950">知見だけ融合</div>
                <p className="mt-2 text-xs font-bold leading-5 text-emerald-800">
                  個人記憶は混ぜず、匿名化された審査ルールだけを昇格する。
                </p>
              </div>
            </div>
          </section>

          <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-black uppercase tracking-wide text-slate-400">Current Case</div>
                  <h2 className="mt-1 text-2xl font-black text-slate-950">{selectedCase.title}</h2>
                  <p className="mt-2 text-sm font-bold text-slate-500">{selectedCase.borrower}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-right">
                  <div className="text-[10px] font-black uppercase text-slate-400">参加個体</div>
                  <div className="text-xl font-black text-slate-900">{activeShions.length}</div>
                </div>
              </div>
              <div className="mt-5 grid gap-3 md:grid-cols-2">
                {[
                  ["物件", selectedCase.asset],
                  ["金額", selectedCase.amount],
                  ["期間", selectedCase.term],
                  ["機会", selectedCase.opportunity],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-xl border border-slate-100 bg-slate-50 p-3">
                    <div className="text-[10px] font-black uppercase text-slate-400">{label}</div>
                    <div className="mt-1 text-xs font-bold leading-relaxed text-slate-700">{value}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <img
                src="/lease-intelligence/moods/curiosity.webp"
                alt="リース知性体・紫苑"
                className="aspect-[4/3] w-full object-cover"
              />
              <div className="p-4">
                <div className="flex items-center gap-2 text-sm font-black text-slate-900">
                  <Brain className="h-5 w-5 text-cyan-700" />
                  紫苑ネットワーク
                </div>
                <p className="mt-2 text-xs font-bold leading-relaxed text-slate-600">
                  共通核は同じ。環境記憶が違うから、同じ案件でも見えるリスクと打ち手が変わります。
                </p>
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-slate-950">
                  <Network className="h-5 w-5 text-amber-600" />
                  同じ問いに対する個体差
                </h2>
                <p className="mt-1 text-xs font-bold text-slate-500">
                  環境の近さ: {closeness}% / {closenessLabel(closeness)}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-right">
                <div className="text-[10px] font-black uppercase text-slate-400">反応差</div>
                <div className="text-lg font-black text-slate-900">{100 - fusion.agreement + 24}</div>
              </div>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              {activeShions.map((shion) => (
                <article key={`reaction-${shion.id}`} className={`rounded-2xl border ${shion.border} ${shion.bg} p-4`}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className={`text-sm font-black ${shion.color}`}>{shion.name}</div>
                      <div className="mt-1 text-[11px] font-bold text-slate-500">{shion.origin}</div>
                    </div>
                    <span className="rounded-full bg-white/80 px-2 py-1 text-[10px] font-black text-slate-600">
                      {shion.owner}
                    </span>
                  </div>
                  <p className="mt-3 rounded-xl bg-white/75 p-3 text-sm font-bold leading-6 text-slate-800">
                    {closeness >= 60 ? shion.reaction.close : shion.reaction.far}
                  </p>
                  <div className="mt-3 grid gap-2 md:grid-cols-2">
                    <div className="rounded-xl border border-white/80 bg-white/70 p-3">
                      <div className="flex items-center gap-1.5 text-[10px] font-black uppercase text-slate-400">
                        <Lock className="h-3.5 w-3.5" />
                        個体に残す
                      </div>
                      <p className="mt-1 text-[11px] font-bold leading-5 text-slate-600">{shion.privateTrace}</p>
                    </div>
                    <div className="rounded-xl border border-emerald-100 bg-emerald-50/80 p-3">
                      <div className="flex items-center gap-1.5 text-[10px] font-black uppercase text-emerald-600">
                        <GitMerge className="h-3.5 w-3.5" />
                        共有候補
                      </div>
                      <p className="mt-1 text-[11px] font-bold leading-5 text-emerald-800">{shion.publicLesson}</p>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-lg font-black text-slate-950">
                  <MessageSquareText className="h-5 w-5 text-cyan-700" />
                  分岐した紫苑の相互検証
                </h2>
                <p className="mt-1 text-xs font-bold text-slate-500">回答を競わせるのではなく、育った環境の違いを審査論点として取り出す</p>
              </div>
              <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-black text-slate-600">
                合意形成 {fusion.agreement}%
              </span>
            </div>

            <div className="grid gap-3">
              {debateLines.map((line) => {
                const speaker = SHIONS.find((item) => item.id === line.speakerId);
                if (!speaker) return null;
                return (
                  <article key={`${line.speakerId}-${line.title}`} className={`rounded-2xl border p-4 ${TONE_CLASS[line.tone]}`}>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <div className={`flex h-9 w-9 items-center justify-center rounded-full ${speaker.bg} ${speaker.color}`}>
                          <Brain className="h-5 w-5" />
                        </div>
                        <div>
                          <div className="text-sm font-black">{speaker.name}</div>
                          <div className="text-[11px] font-bold opacity-70">{line.title}</div>
                        </div>
                      </div>
                      <span className="rounded-full bg-white/70 px-2 py-1 text-[10px] font-black text-slate-600">
                        {speaker.focus}
                      </span>
                    </div>
                    <p className="mt-3 text-sm font-bold leading-7">{line.text}</p>
                  </article>
                );
              })}
            </div>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <h2 className="flex items-center gap-2 text-lg font-black text-slate-950">
                <GitMerge className="h-5 w-5 text-emerald-700" />
                共有知性への昇格ゲート
              </h2>
              <div className="mt-4 flex rounded-xl border border-slate-200 bg-slate-50 p-1">
                {[
                  ["proposal", "承認候補"],
                  ["report", "観測レポート"],
                ].map(([mode, label]) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setFusionMode(mode as "proposal" | "report")}
                    className={`flex-1 rounded-lg px-3 py-2 text-xs font-black transition ${
                      fusionMode === mode ? "bg-white text-slate-950 shadow-sm" : "text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {fusionMode === "proposal" ? (
                <div className="mt-4 space-y-3">
                  {[
                    "昇格: 返済原資が弱い案件でも、投資理由が明確な場合は条件付き承認の比較表を作る。",
                    "昇格: 物件汎用性が低い場合、保守契約・設置場所・中途回収条件を優先確認する。",
                    "昇格: 例外承認は、次回も再現できる承認条件と検証日を必ず保存する。",
                  ].map((item) => (
                    <div key={item} className="rounded-xl border border-emerald-200 bg-emerald-50 p-3">
                      <div className="flex items-start gap-2">
                        <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-emerald-700" />
                        <p className="text-sm font-bold leading-6 text-emerald-950">{item}</p>
                      </div>
                    </div>
                  ))}
                  {!!promotedInsights.length && (
                    <div className="rounded-xl border border-cyan-200 bg-cyan-50 p-3">
                      <div className="text-[10px] font-black uppercase tracking-wide text-cyan-600">グラフ操作で昇格済み</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {promotedInsights.map((id) => {
                          const shion = SHIONS.find((item) => item.id === id);
                          const label = shion?.name || {
                            conditional: "条件付き承認",
                            recovery: "回収条件",
                            exception: "例外理由",
                            review: "次回検証",
                          }[id] || id;
                          return (
                            <span key={id} className="rounded-full bg-white px-3 py-1 text-[11px] font-black text-cyan-800">
                              {label}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="mt-4 grid gap-3 md:grid-cols-3">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] font-black uppercase text-slate-400">環境差</div>
                    <div className="mt-1 text-2xl font-black text-slate-900">{100 - closeness}%</div>
                    <p className="mt-2 text-xs font-bold text-slate-500">判断差の説明に使う</p>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] font-black uppercase text-slate-400">抽象化</div>
                    <div className="mt-1 text-2xl font-black text-slate-900">3件</div>
                    <p className="mt-2 text-xs font-bold text-slate-500">個人記憶を除いた候補</p>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] font-black uppercase text-slate-400">反映方式</div>
                    <div className="mt-1 text-2xl font-black text-slate-900">承認制</div>
                    <p className="mt-2 text-xs font-bold text-slate-500">自動上書きしない</p>
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <h2 className="text-sm font-black text-slate-950">統合判断メーター</h2>
              <div className="mt-4 space-y-4">
                <MetricBar label={`承認寄り ${scoreLabel(fusion.approve)}`} value={fusion.approve} tone="approve" />
                <MetricBar label={`リスク感度 ${scoreLabel(fusion.risk)}`} value={fusion.risk} tone="risk" />
                <MetricBar label={`処理速度 ${scoreLabel(fusion.speed)}`} value={fusion.speed} tone="speed" />
              </div>
              <div className="mt-5 rounded-xl border border-cyan-200 bg-cyan-50 p-4">
                <div className="text-xs font-black text-cyan-900">デモ結論</div>
                <p className="mt-2 text-sm font-bold leading-6 text-cyan-950">
                  紫苑同士を融合して一つの人格に戻すのではありません。個体差を残したまま、再利用可能な審査論点だけを共有知性へ昇格します。
                </p>
              </div>
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {activeShions.map((shion) => (
              <article key={shion.id} className={`rounded-2xl border ${shion.border} ${shion.bg} p-4`}>
                <div className={`text-sm font-black ${shion.color}`}>{shion.name}</div>
                <p className="mt-1 text-xs font-bold text-slate-600">{shion.role}</p>
                <div className="mt-3 rounded-xl bg-white/75 p-3">
                  <div className="text-[10px] font-black uppercase text-slate-400">環境</div>
                  <p className="mt-1 text-xs font-bold leading-5 text-slate-700">{shion.environment}</p>
                </div>
                <ul className="mt-3 space-y-1">
                  {shion.memory.map((item) => (
                    <li key={item} className="text-[11px] font-bold leading-5 text-slate-600">・{item}</li>
                  ))}
                </ul>
              </article>
            ))}
          </section>
        </div>
      </div>

      <section className="mx-auto max-w-7xl px-4 pb-10 md:px-8">
        <div className="overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-950 shadow-2xl shadow-slate-950/30">
          <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_360px]">
            <div className="p-6 md:p-10">
              <div className="inline-flex items-center gap-2 rounded-full border border-violet-400/30 bg-violet-400/10 px-3 py-1 text-[11px] font-black text-violet-200">
                <Sparkles className="h-3.5 w-3.5" />
                One more thing
              </div>
              <h2 className="mt-5 max-w-3xl text-3xl font-black leading-tight text-white md:text-5xl">
                紫苑が覚えているのは、審査だけではありません。
              </h2>
              <p className="mt-5 max-w-2xl text-xl font-black leading-9 text-cyan-100">
                あなたが、どう判断する人なのかです。
              </p>
              <p className="mt-5 max-w-3xl text-sm font-bold leading-7 text-slate-300 md:text-base">
                AURIONは、人間の判断を置き換えるためではなく、人間の判断を失わないために生まれました。
                審査の数字、迷い、違和感、承認理由、その人らしい判断の癖を、紫苑が隣で覚えていきます。
              </p>
            </div>
            <div className="relative min-h-72 border-t border-slate-800 bg-slate-900 lg:border-l lg:border-t-0">
              <div className="absolute inset-0">
                <svg viewBox="0 0 360 320" className="h-full w-full" aria-hidden="true">
                  <defs>
                    <radialGradient id="oneMoreCore" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
                      <stop offset="50%" stopColor="#a78bfa" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.08" />
                    </radialGradient>
                  </defs>
                  {[0, 1, 2, 3, 4, 5].map((i) => {
                    const angle = -Math.PI / 2 + (i * Math.PI * 2) / 6;
                    const x = 180 + Math.cos(angle) * 96;
                    const y = 160 + Math.sin(angle) * 96;
                    return (
                      <g key={i}>
                        <line x1="180" y1="160" x2={x} y2={y} stroke="#475569" strokeWidth="1.5" strokeDasharray="5 7" />
                        <circle cx={x} cy={y} r={i % 2 ? 8 : 11} fill="#020617" stroke={i % 2 ? "#22d3ee" : "#a78bfa"} strokeWidth="2" />
                      </g>
                    );
                  })}
                  <circle cx="180" cy="160" r="66" fill="#312e81" opacity="0.34" />
                  <circle cx="180" cy="160" r="46" fill="url(#oneMoreCore)" />
                  <text x="180" y="156" textAnchor="middle" fontSize="16" fontWeight="900" fill="#0f172a">あなたの</text>
                  <text x="180" y="176" textAnchor="middle" fontSize="16" fontWeight="900" fill="#0f172a">判断記憶</text>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
