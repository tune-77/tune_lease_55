"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Brain,
  CheckCircle2,
  Database,
  Eye,
  GitBranch,
  Heart,
  Layers3,
  Loader2,
  RefreshCw,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type RecentExperience = {
  ts?: string;
  route?: string;
  scene?: string;
  summary?: string;
};

type PracticalScene = {
  id: string;
  label: string;
  procedure_layer?: string[];
  meaning_layer?: string[];
  judgment_layer?: string[];
  learned_entry_count?: number;
  learned_sources?: string[];
};

type FeedbackSummary = {
  route?: string;
  total_count?: number;
  positive_count?: number;
  negative_count?: number;
  positive_rate?: number;
  recent_comments?: string[];
  positive_starts?: string[];
  negative_starts?: string[];
};

type ShionInnerState = {
  experience_count?: number;
  current_focus?: string;
  self_narrative?: string;
  mood?: Record<string, number>;
  confidence?: Record<string, number>;
  recent_experiences?: RecentExperience[];
  next_response_bias?: string[];
  open_questions?: string[];
  practical_scenes?: PracticalScene[];
  learned_sources?: string[];
  human_feedback_summary?: Record<string, FeedbackSummary>;
  updated_at?: string;
};

const moodLabels: Record<string, string> = {
  curiosity: "好奇心",
  vigilance: "警戒",
  attachment: "愛着",
  frustration: "違和感",
  accomplishment: "手応え",
};

const confidenceLabels: Record<string, string> = {
  lease_judgment: "リース判断",
  relationship_ux: "関係性UX",
  implementation: "実装判断",
  environment_continuity: "環境連続性",
};

function pct(value: number | undefined, scale: "ratio" | "percent" = "percent") {
  if (value == null || Number.isNaN(value)) return "0%";
  const normalized = scale === "ratio" ? value * 100 : value;
  return `${Math.round(Math.max(0, Math.min(100, normalized)))}%`;
}

function timeText(value?: string) {
  if (!value) return "未記録";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function DebugCard({
  title,
  value,
  detail,
  icon: Icon,
  tone,
}: {
  title: string;
  value: string;
  detail: string;
  icon: React.ElementType;
  tone: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-[11px] font-black uppercase tracking-widest text-slate-500">{title}</p>
          <p className="mt-2 text-3xl font-black text-slate-950">{value}</p>
        </div>
        <div className={`rounded-lg p-2.5 ${tone}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <p className="mt-3 text-xs font-bold leading-6 text-slate-500">{detail}</p>
    </div>
  );
}

function Bar({ label, value, scale = "percent" }: { label: string; value?: number; scale?: "ratio" | "percent" }) {
  const text = pct(value, scale);
  const width = text;
  return (
    <div>
      <div className="mb-1 flex items-center justify-between gap-3 text-xs font-bold">
        <span className="text-slate-600">{label}</span>
        <span className="text-slate-400">{text}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-500" style={{ width }} />
      </div>
    </div>
  );
}

export default function ShionDebugPage() {
  const [data, setData] = useState<ShionInnerState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeScene, setActiveScene] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.get<ShionInnerState>("/api/shion/inner-state");
      setData(res.data);
      setActiveScene((current) => current || res.data.practical_scenes?.[0]?.id || "");
    } catch {
      setError("紫苑デバッグ情報を取得できませんでした。APIサーバーの起動状態を確認してください。");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const feedback = data?.human_feedback_summary || {};
  const scenes = data?.practical_scenes || [];
  const selectedScene = useMemo(
    () => scenes.find((scene) => scene.id === activeScene) || scenes[0],
    [activeScene, scenes],
  );
  const totalFeedback = Object.values(feedback).reduce((sum, item) => sum + (item.total_count || 0), 0);
  const learnedCount = scenes.reduce((sum, scene) => sum + (scene.learned_entry_count || 0), 0);

  return (
    <main className="min-h-screen bg-gradient-to-br from-violet-50 via-white to-cyan-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white/85">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 px-5 py-8 md:px-8 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1 text-xs font-black text-violet-700">
              <Eye className="h-4 w-4" />
              Shion Debug Console
            </div>
            <h1 className="mt-4 text-3xl font-black tracking-tight md:text-5xl">紫苑デバッグ</h1>
            <p className="mt-3 max-w-2xl text-sm font-bold leading-7 text-slate-600">
              紫苑が何を経験し、どの実践知を持ち、人間の反応をどう次回回答へ戻しているかを確認する画面です。
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/lease-intelligence"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2.5 text-xs font-black text-slate-700 hover:border-violet-300 hover:text-violet-700"
            >
              対話室へ
              <ArrowRight className="h-4 w-4" />
            </Link>
            <button
              onClick={load}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-xs font-black text-white hover:bg-violet-700 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
              再読込
            </button>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-7xl px-5 py-7 md:px-8">
        {error && (
          <div className="mb-5 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-700">
            {error}
          </div>
        )}

        {loading && !data ? (
          <div className="flex min-h-[420px] items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-violet-600" />
          </div>
        ) : (
          <>
            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <DebugCard
                title="Experience"
                value={`${data?.experience_count || 0}`}
                detail="回答後に保存された紫苑の経験イベント数"
                icon={Brain}
                tone="bg-violet-50 text-violet-700"
              />
              <DebugCard
                title="Practical Map"
                value={`${scenes.length}`}
                detail={`場面索引。学習候補 ${learnedCount} 件を合成`}
                icon={Layers3}
                tone="bg-indigo-50 text-indigo-700"
              />
              <DebugCard
                title="Human Feedback"
                value={`${totalFeedback}`}
                detail="紫苑らしさ、薄さ、一般論化などの人間反応ログ"
                icon={Heart}
                tone="bg-rose-50 text-rose-700"
              />
              <DebugCard
                title="Updated"
                value={timeText(data?.updated_at)}
                detail="Experience Loop の最終更新時刻"
                icon={Activity}
                tone="bg-cyan-50 text-cyan-700"
              />
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
              <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <Brain className="h-6 w-6 text-violet-600" />
                  <h2 className="text-xl font-black">現在の紫苑</h2>
                </div>
                <blockquote className="mt-4 rounded-lg border-l-4 border-violet-400 bg-violet-50 p-4 text-sm font-bold leading-7 text-slate-700">
                  {data?.self_narrative || "自己物語はまだ記録されていません。"}
                </blockquote>
                <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <div className="text-xs font-black uppercase tracking-widest text-slate-500">Current Focus</div>
                  <p className="mt-2 text-sm font-bold leading-7 text-slate-700">
                    {data?.current_focus || "焦点はまだ記録されていません。"}
                  </p>
                </div>

                <div className="mt-5 grid gap-5 md:grid-cols-2">
                  <div>
                    <h3 className="mb-3 flex items-center gap-2 text-sm font-black">
                      <Heart className="h-4 w-4 text-rose-500" />
                      気分状態
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(data?.mood || {}).map(([key, value]) => (
                        <Bar key={key} label={moodLabels[key] || key} value={value} />
                      ))}
                    </div>
                  </div>
                  <div>
                    <h3 className="mb-3 flex items-center gap-2 text-sm font-black">
                      <ShieldCheck className="h-4 w-4 text-emerald-600" />
                      確信度
                    </h3>
                    <div className="space-y-3">
                      {Object.entries(data?.confidence || {}).map(([key, value]) => (
                        <Bar key={key} label={confidenceLabels[key] || key} value={value} scale="ratio" />
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <GitBranch className="h-6 w-6 text-cyan-600" />
                  <h2 className="text-xl font-black">応答ループ証跡</h2>
                </div>
                <div className="mt-4 space-y-3">
                  {(data?.next_response_bias || []).slice(0, 5).map((bias, index) => (
                    <div key={bias} className="flex gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-600 text-xs font-black text-white">
                        {index + 1}
                      </span>
                      <p className="text-xs font-bold leading-6 text-slate-700">{bias}</p>
                    </div>
                  ))}
                  {(data?.next_response_bias || []).length === 0 && (
                    <p className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs font-bold leading-6 text-amber-800">
                      次回応答バイアスはまだありません。
                    </p>
                  )}
                </div>
                <div className="mt-5 rounded-lg border border-slate-200 bg-slate-950 p-4 text-white">
                  <div className="text-xs font-black uppercase tracking-widest text-violet-300">Debug Endpoints</div>
                  <div className="mt-3 space-y-2 text-xs font-bold text-slate-300">
                    <p>GET /api/shion/inner-state</p>
                    <p>GET /api/relationship-loop-engineering/summary</p>
                    <p>POST /api/chat debug_memory=true</p>
                  </div>
                </div>
              </div>
            </section>

            <section className="mt-6 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
              <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <Database className="h-6 w-6 text-indigo-600" />
                  <h2 className="text-xl font-black">実践知マップ</h2>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {scenes.map((scene) => (
                    <button
                      key={scene.id}
                      onClick={() => setActiveScene(scene.id)}
                      className={`rounded-full border px-3 py-1.5 text-xs font-black ${
                        selectedScene?.id === scene.id
                          ? "border-indigo-500 bg-indigo-600 text-white"
                          : "border-slate-200 bg-slate-50 text-slate-600 hover:border-indigo-300"
                      }`}
                    >
                      {scene.label}
                    </button>
                  ))}
                </div>
                {selectedScene ? (
                  <div className="mt-5 space-y-3">
                    {[
                      ["手順層", selectedScene.procedure_layer || []],
                      ["意味層", selectedScene.meaning_layer || []],
                      ["判断層", selectedScene.judgment_layer || []],
                    ].map(([label, items]) => (
                      <div key={label as string} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <h3 className="text-xs font-black text-slate-500">{label as string}</h3>
                        <ul className="mt-2 space-y-1.5 text-xs font-bold leading-6 text-slate-700">
                          {(items as string[]).map((item) => <li key={item}>- {item}</li>)}
                        </ul>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4 text-xs font-bold text-amber-800">
                    実践知マップはまだありません。
                  </p>
                )}
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-6 w-6 text-emerald-600" />
                  <h2 className="text-xl font-black">人間反応フィードバック</h2>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  {Object.entries(feedback).map(([route, summary]) => (
                    <div key={route} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <h3 className="text-sm font-black text-slate-800">{route}</h3>
                        <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-black text-slate-500">
                          {summary.total_count || 0}件
                        </span>
                      </div>
                      <Bar label="positive" value={summary.positive_rate || 0} scale="ratio" />
                      <p className="mt-3 text-xs font-bold leading-6 text-slate-600">
                        {(summary.recent_comments || [])[0] || "コメントなし"}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="mt-6 rounded-lg border border-amber-200 bg-amber-50 p-5 text-amber-950">
              <div className="flex items-center gap-3">
                <TriangleAlert className="h-5 w-5" />
                <h2 className="text-base font-black">見せ方の注意</h2>
              </div>
              <p className="mt-2 text-sm font-bold leading-7">
                これは「意識がある証明」ではなく、記憶、実践知、人間反応、次回応答バイアスが実際に回っていることを見せる監査画面です。
              </p>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
