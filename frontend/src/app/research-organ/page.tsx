"use client";

import React, { useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Database,
  ExternalLink,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

type ResearchTopic = {
  key: string;
  title: string;
  query: string;
  validity_days: number;
  tags: string[];
};

type ResearchTopicsResponse = {
  adapter: string;
  label: string;
  default_output_dir: string;
  topics: ResearchTopic[];
};

type ResearchNote = {
  path: string;
  title: string;
  modified: number;
  size: number;
};

type ResearchRunResult = {
  ok: boolean;
  adapter: string;
  label: string;
  dry_run: boolean;
  topic: string;
  title: string;
  query: string;
  target_dir: string;
  path?: string;
  source_count?: number;
  model?: string;
  summary?: string[];
  use_cases?: string[];
  review_questions?: string[];
  summary_warning?: string;
};

const EXAMPLES = [
  "工作機械リースの市況と審査論点",
  "物流業界の2026年問題とリース審査で見るべき指標",
  "補助金制度変更が設備投資と資金繰りに与える影響",
];

function formatModified(value: number) {
  if (!value) return "";
  return new Date(value * 1000).toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ResearchOrganPage() {
  const [topics, setTopics] = useState<ResearchTopic[]>([]);
  const [adapterLabel, setAdapterLabel] = useState("Google AI Studio Researcher");
  const [outputDir, setOutputDir] = useState("Projects/tune_lease_55/Research/Auto Research");
  const [selectedTopic, setSelectedTopic] = useState("");
  const [customTopic, setCustomTopic] = useState("");
  const [notes, setNotes] = useState<ResearchNote[]>([]);
  const [researchRoot, setResearchRoot] = useState("");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<"dry" | "save" | null>(null);
  const [result, setResult] = useState<ResearchRunResult | null>(null);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");

  const selected = useMemo(
    () => topics.find((topic) => topic.key === selectedTopic) || null,
    [selectedTopic, topics],
  );
  const effectiveTopic = customTopic.trim() || selectedTopic;

  const loadInitial = async () => {
    setLoading(true);
    setError("");
    try {
      const [topicRes, notesRes] = await Promise.all([
        apiClient.get<ResearchTopicsResponse>("/api/research-organ/topics"),
        apiClient.get<{ notes: ResearchNote[]; research_root: string }>("/api/research-organ/notes?limit=5"),
      ]);
      setTopics(topicRes.data.topics || []);
      setAdapterLabel(topicRes.data.label || "Google AI Studio Researcher");
      setOutputDir(topicRes.data.default_output_dir || outputDir);
      setNotes(notesRes.data.notes || []);
      setResearchRoot(notesRes.data.research_root || "");
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "外部調査器官の初期化に失敗しました。");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInitial();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runResearch = async (dryRun: boolean) => {
    setRunning(dryRun ? "dry" : "save");
    setError("");
    setWarning("");
    setResult(null);
    try {
      const res = await apiClient.post<ResearchRunResult>("/api/research-organ/run", {
        topic: effectiveTopic,
        dry_run: dryRun,
      }, { timeout: 180000 });
      setResult(res.data);
      if (!dryRun) {
        try {
          const notesRes = await apiClient.get<{ notes: ResearchNote[]; research_root: string }>("/api/research-organ/notes?limit=5");
          setNotes(notesRes.data.notes || []);
          setResearchRoot(notesRes.data.research_root || "");
        } catch (notesErr: any) {
          setWarning(notesErr?.response?.data?.detail || notesErr?.message || "保存は完了しましたが、Researchノート一覧の再取得に失敗しました。更新ボタンで再読み込みできます。");
        }
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || "調査実行に失敗しました。Gemini APIキー、利用枠、safety filterを確認してください。");
    } finally {
      setRunning(null);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 p-4 text-slate-100 sm:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <section className="overflow-hidden rounded-3xl border border-violet-500/30 bg-[linear-gradient(135deg,rgba(76,29,149,0.75),rgba(14,116,144,0.35),rgba(15,23,42,0.96))] p-6 shadow-2xl shadow-violet-950/30 sm:p-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="text-[11px] font-black uppercase tracking-[0.28em] text-violet-200/80">SHION EXTERNAL RESEARCH ORGAN</p>
              <h1 className="mt-3 flex items-center gap-3 text-3xl font-black tracking-tight text-white sm:text-5xl">
                <span className="rounded-2xl bg-white/15 p-3 backdrop-blur">
                  <Search className="h-8 w-8 text-white" />
                </span>
                外部調査器官
              </h1>
              <p className="mt-4 text-sm font-bold leading-relaxed text-slate-200 sm:text-base">
                Google AI Studioで作った調査器官を、紫苑の外界入力として扱います。調査結果はその場で回答に混ぜず、要約・根拠確認・安全整理を通してObsidian Researchへ保存します。
              </p>
            </div>
            <div className="rounded-2xl border border-white/15 bg-white/10 px-4 py-3 backdrop-blur">
              <div className="text-[10px] font-black uppercase tracking-widest text-violet-100/70">Adapter</div>
              <div className="mt-1 text-sm font-black text-white">{adapterLabel}</div>
              <div className="mt-1 text-[11px] font-bold text-slate-300">Gemini Google Search → Research note → 紫苑RAG</div>
            </div>
          </div>
        </section>

        <section className="grid gap-5 lg:grid-cols-[1.25fr_0.75fr]">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl">
            <div className="mb-5 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-black text-white">調査テーマ</h2>
                <p className="mt-1 text-xs font-bold text-slate-400">定型テーマか任意テーマを選び、保存なし確認後にResearchへ保存します。</p>
              </div>
              <button
                type="button"
                onClick={loadInitial}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 px-3 py-2 text-xs font-black text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              >
                <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                更新
              </button>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              {topics.slice(0, 5).map((topic) => (
                <button
                  key={topic.key}
                  type="button"
                  onClick={() => {
                    setSelectedTopic(topic.key);
                    setCustomTopic("");
                    setResult(null);
                  }}
                  className={`rounded-2xl border p-4 text-left transition ${
                    selectedTopic === topic.key && !customTopic.trim()
                      ? "border-violet-400 bg-violet-500/15 shadow-lg shadow-violet-950/30"
                      : "border-slate-800 bg-slate-950/60 hover:border-slate-600"
                  }`}
                >
                  <div className="text-sm font-black text-white">{topic.title}</div>
                  <div className="mt-2 line-clamp-2 text-[11px] font-bold leading-relaxed text-slate-400">{topic.query}</div>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {topic.tags.slice(0, 3).map((tag) => (
                      <span key={tag} className="rounded-full bg-slate-800 px-2 py-1 text-[10px] font-bold text-slate-300">
                        {tag}
                      </span>
                    ))}
                  </div>
                </button>
              ))}
            </div>

            <div className="mt-5 rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
              <label className="text-xs font-black uppercase tracking-widest text-slate-500">任意テーマ</label>
              <textarea
                value={customTopic}
                onChange={(e) => {
                  setCustomTopic(e.target.value);
                  setResult(null);
                }}
                placeholder="例: 工作機械リースの市況と審査論点"
                className="mt-2 min-h-24 w-full resize-y rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-bold leading-relaxed text-white outline-none placeholder:text-slate-600 focus:border-violet-400"
                maxLength={160}
              />
              <div className="mt-3 flex flex-wrap gap-2">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => {
                      setCustomTopic(example);
                      setSelectedTopic("");
                    }}
                    className="rounded-full border border-slate-700 px-3 py-1.5 text-[11px] font-bold text-slate-300 hover:border-violet-400 hover:text-violet-200"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {selected && !customTopic.trim() && (
              <div className="mt-4 rounded-2xl border border-sky-500/20 bg-sky-500/10 p-4">
                <div className="text-xs font-black text-sky-200">選択中: {selected.title}</div>
                <p className="mt-1 text-xs leading-relaxed text-slate-300">{selected.query}</p>
              </div>
            )}

            <div className="mt-5 flex flex-col gap-3 sm:flex-row">
              <button
                type="button"
                onClick={() => runResearch(true)}
                disabled={Boolean(running)}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-800 px-4 py-3 text-sm font-black text-slate-100 hover:bg-slate-700 disabled:opacity-50"
              >
                {running === "dry" ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
                保存なしで接続確認
              </button>
              <button
                type="button"
                onClick={() => runResearch(false)}
                disabled={Boolean(running)}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-violet-500 px-4 py-3 text-sm font-black text-white shadow-lg shadow-violet-950/40 hover:bg-violet-400 disabled:opacity-50"
              >
                {running === "save" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                調査してResearch保存
              </button>
            </div>
          </div>

          <aside className="space-y-5">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="flex items-center gap-2 text-sm font-black text-white">
                <Database className="h-4 w-4 text-emerald-300" />
                保存先
              </h2>
              <p className="mt-2 break-all text-xs font-bold leading-relaxed text-slate-400">{researchRoot || outputDir}</p>
              <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3 text-xs font-bold leading-relaxed text-emerald-100">
                通常のiCloud Obsidian Vaultへ保存します。lease-wiki-vaultには保存しません。
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
              <h2 className="flex items-center gap-2 text-sm font-black text-white">
                <ShieldCheck className="h-4 w-4 text-violet-300" />
                安全ゲート
              </h2>
              <ul className="mt-3 space-y-2 text-xs font-bold leading-relaxed text-slate-400">
                <li>・参照URLがない調査結果は保存しない</li>
                <li>・一次情報、専門機関、補助情報を分ける</li>
                <li>・記事全文ではなく判断ノートへ圧縮する</li>
                <li>・保存後もneeds_human_reviewとして扱う</li>
              </ul>
            </div>
          </aside>
        </section>

        {error && (
          <section className="rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-rose-300" />
              <p className="text-sm font-bold leading-relaxed text-rose-100">{error}</p>
            </div>
          </section>
        )}

        {warning && (
          <section className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-300" />
              <p className="text-sm font-bold leading-relaxed text-amber-100">{warning}</p>
            </div>
          </section>
        )}

        {result && (
          <section className="rounded-2xl border border-emerald-500/30 bg-emerald-500/10 p-5">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-emerald-300" />
              <div>
                <h2 className="text-sm font-black text-emerald-100">{result.dry_run ? "接続確認OK" : "保存完了しました"}</h2>
                <p className="mt-1 text-sm font-bold text-white">{result.title}</p>
                <p className="mt-2 break-all text-xs font-bold text-emerald-100/80">{result.path || result.target_dir}</p>
                {!result.dry_run && (
                  <p className="mt-2 text-xs font-bold text-emerald-100/80">
                    source_count: {result.source_count ?? 0} / model: {result.model || "-"}
                  </p>
                )}
                {result.summary_warning && (
                  <p className="mt-3 text-xs font-bold text-amber-100">{result.summary_warning}</p>
                )}
                {!!result.summary?.length && (
                  <div className="mt-4 rounded-xl border border-emerald-300/20 bg-slate-950/35 p-3">
                    <div className="text-[10px] font-black uppercase tracking-widest text-emerald-100/60">調査要約</div>
                    <ul className="mt-2 space-y-1.5 text-xs font-bold leading-relaxed text-emerald-50/90">
                      {result.summary.map((item) => (
                        <li key={item} className="flex gap-2">
                          <span className="text-emerald-300">・</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {!!result.use_cases?.length && (
                  <div className="mt-3 rounded-xl border border-cyan-300/20 bg-cyan-500/10 p-3">
                    <div className="text-[10px] font-black uppercase tracking-widest text-cyan-100/60">今後どう役立つか</div>
                    <ul className="mt-2 space-y-1.5 text-xs font-bold leading-relaxed text-cyan-50/90">
                      {result.use_cases.map((item) => (
                        <li key={item} className="flex gap-2">
                          <span className="text-cyan-300">・</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {!!result.review_questions?.length && (
                  <div className="mt-3 rounded-xl border border-violet-300/20 bg-violet-500/10 p-3">
                    <div className="text-[10px] font-black uppercase tracking-widest text-violet-100/60">次に確認すること</div>
                    <ul className="mt-2 space-y-1.5 text-xs font-bold leading-relaxed text-violet-50/90">
                      {result.review_questions.map((item) => (
                        <li key={item} className="flex gap-2">
                          <span className="text-violet-300">・</span>
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <h2 className="flex items-center gap-2 text-lg font-black text-white">
                <BookOpen className="h-5 w-5 text-sky-300" />
                最近のResearchノート
              </h2>
              <p className="mt-1 text-xs font-bold text-slate-500">保存済みの外部調査を、紫苑RAGが次の判断で参照できます。</p>
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {notes.length === 0 && (
              <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm font-bold text-slate-500">
                まだResearchノートが見つかりません。
              </div>
            )}
            {notes.slice(0, 5).map((note) => (
              <div key={note.path} className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="line-clamp-2 text-sm font-black text-white">{note.title}</h3>
                    <p className="mt-2 truncate text-[11px] font-bold text-slate-500" title={note.path}>{note.path}</p>
                  </div>
                  <ExternalLink className="h-4 w-4 flex-shrink-0 text-slate-600" />
                </div>
                <div className="mt-3 text-[11px] font-bold text-slate-500">{formatModified(note.modified)}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
