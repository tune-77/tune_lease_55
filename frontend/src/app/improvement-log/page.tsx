"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  AlertCircle,
  CheckCircle2,
  ClipboardList,
  RefreshCw,
  Search,
  ShieldCheck,
  Wrench,
  XCircle,
  Clock,
  GitCommit,
  Sparkles,
} from "lucide-react";

type PendingRecipe = {
  id: string;
  rev: string;
  title: string;
  files: { path: string; changes: { find: string; replace: string }[] }[];
  safety?: string;
  risk_level?: string;
  intelligence_comment?: string;
  shion_recommendation?: "auto" | "discuss" | "review";
  shion_reason?: string;
  generated_at?: string;
};

type RecipeStatus = {
  pending_count: number;
  approved_count: number;
  applied_count: number;
  rejected_count: number;
  codex_auto_queue?: {
    status?: string;
    queued_count?: number;
    safe_count?: number;
    maybe_count?: number;
    manual_or_blocked_count?: number;
  };
  note?: string;
};

type ImprovementItem = {
  id: string;
  title: string;
  status: string;
  priority?: string;
  category?: string;
  recommended_order?: number;
  canonical_key?: string;
  group_id?: string;
  duplicate_count?: number;
  reason?: string;
  auto_fix_policy?: { reason?: string; risk?: string };
};

type ImprovementLog = {
  date: string;
  generated_at: string;
  status: string;
  approved: number;
  auto_fix_candidates: number;
  needs_review: number;
  parked?: number;
  rejected: number;
  applied: number;
  items: ImprovementItem[];
  obsidian_compliance?: {
    status?: string;
    violations?: unknown[];
    route_sensitive_ids?: string[];
  };
  recursive_self_improvement?: {
    source?: string;
    generated_at?: string;
    canonical_candidate_count?: number;
    ranked_queue_count?: number;
    suppressed_count?: number;
    measurement_summary?: {
      pdca_rate?: number;
      response_changed_rate?: number;
      repeat_issue_rate?: number;
      reuse_rate?: number;
      noise_rate?: number;
      prompt_total?: number;
      prompt_previous_diff_count?: number;
    };
  };
  source?: string;
};

type PipelineSummary = {
  run_date: string | null;
  applied_count: number;
  needs_review_count: number;
  failed_count: number;
  commit_result: { success: boolean; message?: string; pr_url?: string | null } | null;
};

type GapItem = {
  id: string;
  title: string;
  priority: "critical" | "high" | "medium" | "low" | string;
  category: string;
  evidence?: string[];
  impact?: string;
  recommended_action?: string;
  suggested_program?: string;
  guardrail?: string;
  source_refs?: string[];
};

type LedgerRule = {
  rev_id: string;
  type: string;
  pending_review: boolean;
  description: string;
  source?: string;
  target?: string;
  risk?: string;
  auto_fix_allowed?: boolean;
  affected_files?: string[];
  applied_at?: string;
  manual_reason?: string;
};

type GapAnalysis = {
  available: boolean;
  generated_at?: string;
  mode?: string;
  source?: string;
  counts?: Record<string, number>;
  items: GapItem[];
};

type PromptFeedbackSummary = {
  source?: string;
  summary?: {
    total: number;
    pdca_count: number;
    pdca_rate: number;
    previous_diff_count: number;
    previous_diff_rate: number;
    avg_response_len: number;
    avg_prompt_base_len: number;
    avg_prompt_final_len: number;
    avg_prompt_diff_added: number;
    avg_prompt_diff_removed: number;
    avg_prompt_diff_context: number;
    by_surface: Record<string, {
      count: number;
      pdca_rate: number;
      avg_response_len: number;
      avg_prompt_diff_added: number;
      avg_prompt_diff_removed: number;
      response_changed_rate: number;
    }>;
  };
};

type OperationalTrustSummary = {
  status: "ok" | "attention" | string;
  attention: string[];
  memory_usage: {
    source: string;
    total: number;
    recent_days: number;
    recent_total: number;
    pdca_applied_count: number;
    judgment_learning_count: number;
    latest_timestamp: string;
    by_surface: Record<string, number>;
    recent_items: {
      timestamp: string;
      surface: string;
      knowledge_ref_count: number;
      pdca_applied: boolean;
      judgment_learning_used: boolean;
      question_hash: string;
    }[];
  };
  pdca_rules: {
    source: string;
    active: number;
    expiring_soon: number;
    expired: number;
    inactive: number;
    manual_rule_count: number;
    rules: {
      rule: string;
      source: string;
      status: string;
      expires_at: string;
      days_left: number | null;
    }[];
  };
  knowledge_corrections: {
    available: boolean;
    source?: string;
    total: number;
    needs_review: number;
    items: {
      path: string;
      name: string;
      status: string;
      updated_at: string;
    }[];
  };
};

const STATUS_LABELS: Record<string, { label: string; className: string }> = {
  APPROVED: { label: "承認", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  AUTO_FIX_CANDIDATE: { label: "自動修正候補", className: "bg-blue-50 text-blue-700 border-blue-200" },
  NEEDS_REVIEW: { label: "要確認", className: "bg-amber-50 text-amber-700 border-amber-200" },
  needs_review: { label: "要確認", className: "bg-amber-50 text-amber-700 border-amber-200" },
  PARKED: { label: "保留", className: "bg-slate-50 text-slate-500 border-slate-200" },
  REJECTED: { label: "拒否", className: "bg-rose-50 text-rose-700 border-rose-200" },
  APPLIED: { label: "適用済", className: "bg-slate-100 text-slate-700 border-slate-300" },
  RULE_REGISTERED: { label: "今後ルール化済", className: "bg-indigo-50 text-indigo-700 border-indigo-200" },
  RULE_REVIEW: { label: "今後ルール要確認", className: "bg-violet-50 text-violet-700 border-violet-200" },
  SKIPPED: { label: "スキップ", className: "bg-slate-50 text-slate-500 border-slate-200" },
};

const CATEGORY_LABELS: Record<string, string> = {
  quick_ui: "UI",
  obsidian_chat: "Obsidian/Chat",
  logic_light: "軽量ロジック",
  data_quality: "運用品質",
  db_api: "DB/API",
  external: "外部連携",
  infra: "インフラ",
  planning: "仕様整理",
};

export default function ImprovementLogPage() {
  const [activeTab, setActiveTab] = useState<"improvements" | "recipes" | "ledger">("improvements");
  const [data, setData] = useState<ImprovementLog | null>(null);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [gapAnalysis, setGapAnalysis] = useState<GapAnalysis | null>(null);
  const [promptSummary, setPromptSummary] = useState<PromptFeedbackSummary | null>(null);
  const [trustSummary, setTrustSummary] = useState<OperationalTrustSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("NEEDS_REVIEW");
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});
  const [pendingRecipes, setPendingRecipes] = useState<PendingRecipe[]>([]);
  const [recipesLoading, setRecipesLoading] = useState(false);
  const [dismissedRecipes, setDismissedRecipes] = useState<Set<string>>(new Set());
  const [recipeStatus, setRecipeStatus] = useState<RecipeStatus | null>(null);
  const [recipeError, setRecipeError] = useState("");
  const [ledgerRules, setLedgerRules] = useState<LedgerRule[]>([]);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [ledgerError, setLedgerError] = useState("");
  const [approvingRuleIds, setApprovingRuleIds] = useState<Set<string>>(new Set());

  const fetchLedgerRules = useCallback(async () => {
    setLedgerLoading(true);
    setLedgerError("");
    try {
      const res = await apiClient.get<{ rules: LedgerRule[] }>("/api/rule-engine/rules");
      setLedgerRules(res.data.rules ?? []);
    } catch {
      setLedgerError("今後の自動修正ルールの取得に失敗しました");
    } finally {
      setLedgerLoading(false);
    }
  }, []);

  const handleApproveRule = useCallback(async (revId: string) => {
    setApprovingRuleIds((prev) => new Set(prev).add(revId));
    try {
      await apiClient.patch(`/api/rule-engine/rules/${revId}/approve`);
      setLedgerRules((prev) =>
        prev.map((r) => (r.rev_id === revId ? { ...r, pending_review: false } : r))
      );
    } catch {
      setLedgerError(`${revId} の承認に失敗しました`);
    } finally {
      setApprovingRuleIds((prev) => {
        const next = new Set(prev);
        next.delete(revId);
        return next;
      });
    }
  }, []);

  const fetchRecipes = useCallback(async () => {
    setRecipesLoading(true);
    setRecipeError("");
    try {
      const [res, statusRes] = await Promise.all([
        apiClient.get<{ recipes: PendingRecipe[] }>("/api/recipes/pending"),
        apiClient.get<RecipeStatus>("/api/recipes/status"),
      ]);
      setPendingRecipes(res.data.recipes ?? []);
      setRecipeStatus(statusRes.data ?? null);
    } catch (error) {
      setPendingRecipes([]);
      setRecipeStatus(null);
      setRecipeError("今回の修正案の状態を取得できませんでした");
    } finally {
      setRecipesLoading(false);
    }
  }, []);

  const handleRecipeAction = useCallback(
    async (recipe: PendingRecipe, action: "approve" | "reject") => {
      setRecipeError("");
      try {
        await apiClient.post(`/api/recipes/${recipe.id}/${action}`);
        setDismissedRecipes((prev) => new Set(prev).add(recipe.id));
        await fetchRecipes();
      } catch (error) {
        setRecipeError(action === "approve" ? "今回の修正案を適用待ちへ送れませんでした" : "今回の修正案の破棄に失敗しました");
      }
    },
    [fetchRecipes]
  );

  const fetchLog = useCallback(async () => {
    setLoading(true);
    try {
      const [logRes, summaryRes, gapsRes, promptRes, trustRes] = await Promise.all([
        apiClient.get<ImprovementLog>("/api/improvement-log"),
        apiClient.get<PipelineSummary>("/api/improvement-pipeline/summary"),
        apiClient.get<GapAnalysis>("/api/lease-system-gaps"),
        apiClient.get<PromptFeedbackSummary>("/api/prompt-feedback/summary"),
        apiClient.get<OperationalTrustSummary>("/api/operational-trust/summary"),
      ]);
      setData(logRes.data);
      setSummary(summaryRes.data);
      setGapAnalysis(gapsRes.data);
      setPromptSummary(promptRes.data || null);
      setTrustSummary(trustRes.data || null);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const activityDate = new Date().toLocaleDateString("sv-SE");
    const activityKey = `lease-intelligence-activity:improvement_log:${activityDate}`;
    if (!window.sessionStorage.getItem(activityKey)) {
      apiClient.post("/api/lease-intelligence/activity", {
        surface: "improvement_log",
        action: "page_view",
        event_id: activityKey,
      }).then(() => window.sessionStorage.setItem(activityKey, "1")).catch(() => {});
    }
  }, []);

  useEffect(() => {
    fetchLog();
  }, [fetchLog]);

  useEffect(() => {
    fetchRecipes();
  }, [fetchRecipes]);

  useEffect(() => {
    fetchLedgerRules();
  }, [fetchLedgerRules]);

  const handleReview = useCallback(
    async (item: ImprovementItem, action: "approved" | "rejected" | "deferred") => {
      const itemKey = item.canonical_key || item.id || item.title;
      setActionLoading((prev) => ({ ...prev, [itemKey]: true }));
      try {
        await apiClient.post("/api/improvement-log/review", {
          key: item.canonical_key || item.id || "",
          title: item.title,
          action,
        });
        await fetchLog();
      } catch {
        // 失敗時は何もしない（再fetchで状態は保持される）
      } finally {
        setActionLoading((prev) => ({ ...prev, [itemKey]: false }));
      }
    },
    [fetchLog]
  );

  const handleRegisterPromptRule = useCallback(
    async (item: ImprovementItem) => {
      const itemKey = item.canonical_key || item.id || item.title;
      setActionLoading((prev) => ({ ...prev, [itemKey]: true }));
      const reason = item.auto_fix_policy?.reason || item.reason || item.title || "";
      const rule = `${item.title || item.id || "改善項目"}: ${reason}`.trim();
      try {
        await apiClient.post("/api/prompt-feedback/rules/register", {
          title: item.title || item.id || "改善項目",
          rule,
          key: item.canonical_key || item.id || item.title || "",
          canonical_key: item.canonical_key || item.id || item.title || "",
          source: "improvement-log",
          surface: item.category || "",
          reason,
        });
        await fetchLog();
      } catch {
        // 失敗時は何もしない（再fetchで状態は保持される）
      } finally {
        setActionLoading((prev) => ({ ...prev, [itemKey]: false }));
      }
    },
    [fetchLog]
  );

  const filteredItems = useMemo(() => {
    const items = data?.items ?? [];
    return items.filter((item) => {
      const matchesStatus = status === "ALL" || item.status === status || (status === "NEEDS_REVIEW" && item.status === "needs_review");
      const needle = query.trim().toLowerCase();
      const matchesQuery =
        !needle ||
        item.id.toLowerCase().includes(needle) ||
        (item.title || "").toLowerCase().includes(needle) ||
        (item.canonical_key || "").toLowerCase().includes(needle);
      return matchesStatus && matchesQuery;
    });
  }, [data?.items, query, status]);

  const obsidianStatus = data?.obsidian_compliance?.status || "unknown";
  const obsidianViolations = data?.obsidian_compliance?.violations?.length || 0;

  const visibleRecipes = pendingRecipes.filter((r) => !dismissedRecipes.has(r.id));

  return (
    <main className="min-h-screen bg-slate-50 p-4 md:p-6">
      <div className="mx-auto max-w-6xl space-y-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900 text-white">
              <ClipboardList className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">改善パイプライン ログ</h1>
              <p className="text-sm text-slate-500">
                {data?.date ? `最終実行: ${data.date}` : "最新の改善レポートを読み込みます"}
              </p>
            </div>
          </div>
          <button
            onClick={activeTab === "improvements" ? fetchLog : fetchRecipes}
            className="ml-auto inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            <RefreshCw className="h-4 w-4" />
            更新
          </button>
        </div>

        {/* タブナビゲーション */}
        <div className="flex gap-1 rounded-lg border border-slate-200 bg-white p-1">
          <button
            onClick={() => setActiveTab("improvements")}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-semibold transition-colors ${
              activeTab === "improvements"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            改善候補リスト
          </button>
          <button
            onClick={() => setActiveTab("recipes")}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-semibold transition-colors ${
              activeTab === "recipes"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            今回の修正案
            {visibleRecipes.length > 0 && (
              <span className="ml-2 inline-flex items-center justify-center rounded-full bg-amber-500 px-1.5 text-xs font-bold text-white">
                {visibleRecipes.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("ledger")}
            className={`flex-1 rounded-md px-4 py-2 text-sm font-semibold transition-colors ${
              activeTab === "ledger"
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            今後の自動修正ルール
            {ledgerRules.filter((r) => r.pending_review).length > 0 && (
              <span className="ml-2 inline-flex items-center justify-center rounded-full bg-indigo-500 px-1.5 text-xs font-bold text-white">
                {ledgerRules.filter((r) => r.pending_review).length}
              </span>
            )}
          </button>
        </div>

        {/* 今回の修正案タブ */}
        {activeTab === "recipes" && (
          <section className="space-y-3">
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <span className="rounded-full bg-amber-50 px-2 py-1 font-semibold text-amber-700">
                  承認待ち {recipeStatus?.pending_count ?? visibleRecipes.length}
                </span>
                <span className="rounded-full bg-blue-50 px-2 py-1 font-semibold text-blue-700">
                  適用待ち {recipeStatus?.approved_count ?? 0}
                </span>
                <span className="rounded-full bg-emerald-50 px-2 py-1 font-semibold text-emerald-700">
                  適用済 {recipeStatus?.applied_count ?? 0}
                </span>
                <span className="rounded-full bg-rose-50 px-2 py-1 font-semibold text-rose-700">
                  却下 {recipeStatus?.rejected_count ?? 0}
                </span>
              </div>
              {recipeStatus?.codex_auto_queue && (
                <p className="mt-2 text-xs text-slate-500">
                  Codex自動キュー: {recipeStatus.codex_auto_queue.status || "-"} / safe {recipeStatus.codex_auto_queue.safe_count ?? 0} / maybe {recipeStatus.codex_auto_queue.maybe_count ?? 0} / manual {recipeStatus.codex_auto_queue.manual_or_blocked_count ?? 0}
                </p>
              )}
              <p className="mt-2 text-xs text-slate-500">
                今回の修正案は、この実行で作られた1回限りの修正パッチです。「適用待ちへ送る」と承認済みフォルダへ移り、実適用は別処理で実行します。
              </p>
              {recipeError && (
                <p className="mt-2 text-xs font-semibold text-rose-600">{recipeError}</p>
              )}
            </div>
            {recipesLoading ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                読み込み中...
              </div>
            ) : visibleRecipes.length === 0 ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                承認待ちの今回の修正案はありません。安全な自動修正候補が生成された時だけここに表示されます。
              </div>
            ) : (
              visibleRecipes.map((recipe) => (
                <RecipeCard
                  key={recipe.id}
                  recipe={recipe}
                  onApprove={() => handleRecipeAction(recipe, "approve")}
                  onReject={() => handleRecipeAction(recipe, "reject")}
                />
              ))
            )}
          </section>
        )}

        {/* 今後の自動修正ルールタブ */}
        {activeTab === "ledger" && (
          <section className="space-y-3">
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                <span className="rounded-full bg-indigo-50 px-2 py-1 font-semibold text-indigo-700">
                  承認待ち {ledgerRules.filter((r) => r.pending_review).length}
                </span>
                <span className="rounded-full bg-emerald-50 px-2 py-1 font-semibold text-emerald-700">
                  承認済み {ledgerRules.filter((r) => !r.pending_review).length}
                </span>
                <span className="rounded-full bg-slate-100 px-2 py-1 font-semibold text-slate-600">
                  合計 {ledgerRules.length}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                今後の自動修正ルールは、次回以降も同じ種類の修正に使う継続ルールです。「自動適用を許可」すると batch_apply の対象になります。
              </p>
              {ledgerError && (
                <p className="mt-2 text-xs font-semibold text-rose-600">{ledgerError}</p>
              )}
            </div>
            {ledgerLoading ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                読み込み中...
              </div>
            ) : ledgerRules.length === 0 ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                今後の自動修正ルールがありません
              </div>
            ) : (
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[900px] text-sm">
                    <thead className="bg-slate-100 text-left text-xs text-slate-500">
                      <tr>
                        <th className="px-4 py-3">REV-ID</th>
                        <th className="px-4 py-3">種別</th>
                        <th className="px-4 py-3">説明</th>
                        <th className="px-4 py-3">リスク</th>
                        <th className="px-4 py-3">状態</th>
                        <th className="px-4 py-3">操作</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {ledgerRules.map((rule) => {
                        const isApproving = approvingRuleIds.has(rule.rev_id);
                        const riskClass =
                          rule.risk === "high"
                            ? "bg-rose-100 text-rose-700"
                            : rule.risk === "medium"
                            ? "bg-amber-100 text-amber-700"
                            : "bg-emerald-100 text-emerald-700";
                        return (
                          <tr key={rule.rev_id} className="align-top hover:bg-slate-50">
                            <td className="px-4 py-3 font-mono text-xs font-bold text-slate-600">
                              {rule.rev_id}
                            </td>
                            <td className="px-4 py-3 text-xs text-slate-500">{rule.type}</td>
                            <td className="px-4 py-3">
                              <div className="text-sm text-slate-800">{rule.description}</div>
                              {rule.applied_at && (
                                <div className="mt-0.5 text-[11px] text-slate-400">
                                  適用済: {rule.applied_at}
                                </div>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {rule.risk && (
                                <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${riskClass}`}>
                                  {rule.risk}
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {rule.pending_review ? (
                                <span className="inline-flex rounded-full border border-indigo-200 bg-indigo-50 px-2 py-1 text-xs font-semibold text-indigo-700">
                                  承認待ち
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700">
                                  ✅ 承認済み
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {rule.pending_review ? (
                                <button
                                  onClick={() => handleApproveRule(rule.rev_id)}
                                  disabled={isApproving}
                                  className="rounded border border-indigo-300 bg-indigo-50 px-3 py-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-40"
                                >
                                  {isApproving ? "処理中..." : "自動適用を許可"}
                                </button>
                              ) : (
                                <span className="text-xs text-slate-300">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}

        {/* 改善案タブ */}
        {activeTab === "improvements" && (
        <>

        {/* 朝報告サマリーカード */}
        {summary && (
          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-700">
              <GitCommit className="h-4 w-4" />
              パイプライン実行サマリー
              {summary.run_date && (
                <span className="ml-1 text-xs font-normal text-slate-400">{summary.run_date}</span>
              )}
            </div>
            <div className="flex flex-wrap gap-3">
              <SummaryChip
                label="自動適用"
                value={summary.applied_count}
                color="emerald"
                icon={<CheckCircle2 className="h-3.5 w-3.5" />}
              />
              <SummaryChip
                label="要確認"
                value={summary.needs_review_count}
                color="amber"
                icon={<AlertCircle className="h-3.5 w-3.5" />}
              />
              <SummaryChip
                label="失敗"
                value={summary.failed_count}
                color="rose"
                icon={<XCircle className="h-3.5 w-3.5" />}
              />
              <div className="flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium border-slate-200 bg-slate-50 text-slate-600">
                <GitCommit className="h-3.5 w-3.5" />
                コミット:{" "}
                {summary.commit_result?.success
                  ? <span className="text-emerald-600">成功</span>
                  : <span className="text-slate-400">{summary.commit_result?.message || "なし"}</span>}
              </div>
            </div>
          </section>
        )}

        {promptSummary?.summary && (
          <section className="rounded-lg border border-cyan-200 bg-cyan-50 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-cyan-900">
              <ShieldCheck className="h-4 w-4" />
              プロンプト改善ループ
              {promptSummary.source && (
                <span className="ml-1 text-xs font-normal text-cyan-700">{promptSummary.source}</span>
              )}
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <MiniMetric label="総件数" value={promptSummary.summary.total} />
              <MiniMetric label="PDCA反映率" value={`${promptSummary.summary.pdca_rate}%`} />
              <MiniMetric label="前回差分率" value={`${promptSummary.summary.previous_diff_rate}%`} />
              <MiniMetric label="平均応答長" value={promptSummary.summary.avg_response_len} />
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              {Object.entries(promptSummary.summary.by_surface || {}).slice(0, 4).map(([surface, stats]) => (
                <div key={surface} className="rounded-lg border border-cyan-100 bg-white p-3">
                  <div className="text-sm font-semibold text-slate-900">{surface}</div>
                  <div className="mt-1 text-xs text-slate-600">
                    {stats.count}件 / PDCA {stats.pdca_rate}% / 変化率 {stats.response_changed_rate}%
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    平均長 {stats.avg_response_len} / diff +{stats.avg_prompt_diff_added} -{stats.avg_prompt_diff_removed}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {trustSummary && (
          <section className="rounded-lg border border-emerald-200 bg-white p-4">
            <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <ShieldCheck className="h-4 w-4 text-emerald-600" />
                  実務安心運用
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                    trustSummary.status === "ok"
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-amber-100 text-amber-800"
                  }`}>
                    {trustSummary.status === "ok" ? "OK" : "要確認"}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  記憶使用・PDCA期限・Knowledge訂正候補を読み取り専用で監査します。
                </p>
              </div>
              {trustSummary.attention.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {trustSummary.attention.map((item) => (
                    <span key={item} className="rounded-full bg-amber-50 px-2 py-1 text-[11px] font-semibold text-amber-800">
                      {trustAttentionLabel(item)}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <TrustMetric label="監査ログ" value={`${trustSummary.memory_usage.recent_total}件`} detail={`直近${trustSummary.memory_usage.recent_days}日`} />
              <TrustMetric label="PDCA適用" value={`${trustSummary.memory_usage.pdca_applied_count}件`} detail="応答ログ内" />
              <TrustMetric label="有効PDCA" value={`${trustSummary.pdca_rules.active}件`} detail={`期限近 ${trustSummary.pdca_rules.expiring_soon} / 期限切れ ${trustSummary.pdca_rules.expired}`} />
              <TrustMetric label="訂正候補" value={`${trustSummary.knowledge_corrections.needs_review}件`} detail={`全${trustSummary.knowledge_corrections.total}件`} />
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-bold text-slate-700">直近の記憶使用ログ</div>
                <div className="mt-2 space-y-1.5">
                  {trustSummary.memory_usage.recent_items.length === 0 ? (
                    <div className="text-xs text-slate-500">直近ログはありません</div>
                  ) : trustSummary.memory_usage.recent_items.slice(-4).map((item, index) => (
                    <div key={`${item.timestamp}-${index}`} className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                      <span className="font-mono text-slate-400">{item.timestamp || "-"}</span>
                      <span className="rounded-full bg-white px-2 py-0.5 font-semibold text-slate-700">{item.surface}</span>
                      <span>refs {item.knowledge_ref_count}</span>
                      {item.pdca_applied && <span className="text-emerald-700">PDCA</span>}
                      {item.judgment_learning_used && <span className="text-indigo-700">判断学習</span>}
                      {item.question_hash && <span className="font-mono text-slate-400">#{item.question_hash}</span>}
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-bold text-slate-700">Knowledge訂正レビュー</div>
                <div className="mt-2 space-y-1.5">
                  {!trustSummary.knowledge_corrections.available ? (
                    <div className="text-xs text-slate-500">Vaultを確認できません</div>
                  ) : trustSummary.knowledge_corrections.items.length === 0 ? (
                    <div className="text-xs text-slate-500">訂正候補はありません</div>
                  ) : trustSummary.knowledge_corrections.items.slice(0, 4).map((item) => (
                    <div key={item.path} className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                      <span className={`rounded-full px-2 py-0.5 font-semibold ${
                        item.status === "needs_review"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-white text-slate-600"
                      }`}>
                        {item.status}
                      </span>
                      <span className="max-w-[22rem] truncate">{item.name}</span>
                      <span className="font-mono text-slate-400">{item.updated_at}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}

        {data?.recursive_self_improvement?.measurement_summary && (
          <section className="rounded-lg border border-indigo-200 bg-indigo-50 p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-indigo-900">
              <Sparkles className="h-4 w-4" />
              再帰的自己改善
              {data.recursive_self_improvement.generated_at && (
                <span className="ml-1 text-xs font-normal text-indigo-700">
                  {data.recursive_self_improvement.generated_at}
                </span>
              )}
            </div>
            <div className="grid gap-3 md:grid-cols-5">
              <MiniMetric label="PDCA反映率" value={`${data.recursive_self_improvement.measurement_summary.pdca_rate ?? 0}%`} />
              <MiniMetric label="応答変化率" value={`${data.recursive_self_improvement.measurement_summary.response_changed_rate ?? 0}%`} />
              <MiniMetric label="再発率" value={`${data.recursive_self_improvement.measurement_summary.repeat_issue_rate ?? 0}%`} />
              <MiniMetric label="再利用率" value={`${data.recursive_self_improvement.measurement_summary.reuse_rate ?? 0}%`} />
              <MiniMetric label="ノイズ率" value={`${data.recursive_self_improvement.measurement_summary.noise_rate ?? 0}%`} />
            </div>
            <div className="mt-3 flex flex-wrap gap-2 text-xs text-indigo-800">
              <span className="rounded-full bg-white px-2 py-1">
                候補 {data.recursive_self_improvement.canonical_candidate_count ?? 0}
              </span>
              <span className="rounded-full bg-white px-2 py-1">
                キュー {data.recursive_self_improvement.ranked_queue_count ?? 0}
              </span>
              <span className="rounded-full bg-white px-2 py-1">
                抑制 {data.recursive_self_improvement.suppressed_count ?? 0}
              </span>
            </div>
            {data.recursive_self_improvement.source && (
              <p className="mt-3 break-all text-[11px] text-indigo-500">
                {data.recursive_self_improvement.source}
              </p>
            )}
          </section>
        )}

        {gapAnalysis?.available && (
          <section className="rounded-lg border border-slate-200 bg-white p-4">
            <div className="mb-3 flex flex-col gap-1 md:flex-row md:items-center md:justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
                  <AlertCircle className="h-4 w-4 text-rose-500" />
                  不足項目・改善診断
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  本体非連動の読み取り専用診断。スコア・DB・モデルは変更しません。
                </p>
              </div>
              <div className="flex flex-wrap gap-2 text-xs">
                <span className="rounded-full bg-rose-50 px-2.5 py-1 font-bold text-rose-700">
                  Critical {gapAnalysis.counts?.critical ?? 0}
                </span>
                <span className="rounded-full bg-amber-50 px-2.5 py-1 font-bold text-amber-700">
                  High {gapAnalysis.counts?.high ?? 0}
                </span>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 font-bold text-slate-600">
                  Total {gapAnalysis.items?.length ?? 0}
                </span>
              </div>
            </div>
            <div className="space-y-3">
              {(gapAnalysis.items || []).map((item) => (
                <div key={item.id} className="rounded-md border border-slate-200 bg-slate-50 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs font-bold text-slate-500">{item.id}</span>
                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${gapPriorityClass(item.priority)}`}>
                      {item.priority}
                    </span>
                    <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-bold text-slate-500">
                      {item.category}
                    </span>
                  </div>
                  <h2 className="mt-2 text-sm font-bold text-slate-900">{item.title}</h2>
                  {item.impact && <p className="mt-1 text-xs leading-relaxed text-slate-600">{item.impact}</p>}
                  {item.recommended_action && (
                    <p className="mt-2 text-xs leading-relaxed text-slate-700">
                      <span className="font-bold">次の対応:</span> {item.recommended_action}
                    </p>
                  )}
                  {item.evidence?.length ? (
                    <div className="mt-2 text-[11px] leading-relaxed text-slate-500">
                      {item.evidence.slice(0, 2).map((line, index) => (
                        <div key={index}>・{line}</div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
            {gapAnalysis.source && (
              <p className="mt-3 break-all text-[11px] text-slate-400">{gapAnalysis.source}</p>
            )}
          </section>
        )}

        <div className="grid gap-3 md:grid-cols-6">
          <Stat label="適用済" value={data?.applied ?? 0} icon={<CheckCircle2 className="h-4 w-4" />} />
          <Stat label="承認" value={data?.approved ?? 0} icon={<CheckCircle2 className="h-4 w-4" />} />
          <Stat label="自動修正候補" value={data?.auto_fix_candidates ?? 0} icon={<Wrench className="h-4 w-4" />} />
          <Stat label="要確認" value={data?.needs_review ?? 0} icon={<AlertCircle className="h-4 w-4" />} />
          <Stat label="保留" value={data?.parked ?? 0} icon={<Clock className="h-4 w-4" />} />
          <Stat label="拒否" value={data?.rejected ?? 0} icon={<XCircle className="h-4 w-4" />} />
        </div>

        <section className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="ID・タイトル・canonical_keyで検索"
                className="w-full rounded-md border border-slate-300 py-2 pl-9 pr-3 text-sm outline-none focus:border-slate-500"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {["ALL", "AUTO_FIX_CANDIDATE", "NEEDS_REVIEW", "PARKED", "REJECTED", "APPLIED"].map((key) => (
                <button
                  key={key}
                  onClick={() => setStatus(key)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    status === key ? "bg-slate-900 text-white" : "border border-slate-300 bg-white text-slate-600"
                  }`}
                >
                  {key === "ALL" ? "すべて" : STATUS_LABELS[key]?.label || key}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1">
              <ShieldCheck className="h-3.5 w-3.5" />
              Obsidian: {obsidianStatus} / violations {obsidianViolations}
            </span>
            {data?.source && <span className="rounded-full bg-slate-100 px-2 py-1">{data.source}</span>}
            <span className="rounded-full bg-slate-100 px-2 py-1">{filteredItems.length}件表示</span>
          </div>
        </section>

        <section className="overflow-hidden rounded-lg border border-slate-200 bg-white">
          {loading ? (
            <div className="p-10 text-center text-sm text-slate-500">読み込み中...</div>
          ) : filteredItems.length === 0 ? (
            <div className="p-10 text-center text-sm text-slate-500">該当する改善案がありません</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1000px] text-sm">
                <thead className="bg-slate-100 text-left text-xs text-slate-500">
                  <tr>
                    <th className="px-4 py-3">順</th>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">タイトル</th>
                    <th className="px-4 py-3">分類</th>
                    <th className="px-4 py-3">状態</th>
                    <th className="px-4 py-3">理由</th>
                    <th className="px-4 py-3">操作</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredItems.map((item) => {
                    const statusStyle = STATUS_LABELS[item.status] || {
                      label: item.status || "-",
                      className: "bg-slate-50 text-slate-600 border-slate-200",
                    };
                    const itemKey = item.canonical_key || item.id || item.title;
                    const isNeedsReview = item.status === "NEEDS_REVIEW" || item.status === "needs_review";
                    const isActing = !!actionLoading[itemKey];
                    return (
                      <tr key={`${item.id}-${item.status}`} className="align-top hover:bg-slate-50">
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">{item.recommended_order ?? "-"}</td>
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">{item.id}</td>
                        <td className="px-4 py-3">
                          <div className="font-medium text-slate-800">{item.title || "-"}</div>
                          <div className="mt-1 text-xs text-slate-400">
                            {item.canonical_key || "-"}
                            {item.duplicate_count ? ` / duplicates ${item.duplicate_count}` : ""}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-xs text-slate-600">
                          {CATEGORY_LABELS[item.category || ""] || item.category || "-"}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`inline-flex rounded-full border px-2 py-1 text-xs font-semibold ${statusStyle.className}`}>
                            {statusStyle.label}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs leading-relaxed text-slate-600">
                          {item.auto_fix_policy?.reason || item.reason || "-"}
                        </td>
                        <td className="px-4 py-3">
                          {isNeedsReview ? (
                            <div className="flex gap-1.5">
                              <ActionButton
                                label="レビュー承認"
                                onClick={() => handleReview(item, "approved")}
                                disabled={isActing}
                                variant="approve"
                              />
                              <ActionButton
                                label="今回は却下"
                                onClick={() => handleReview(item, "rejected")}
                                disabled={isActing}
                                variant="reject"
                              />
                              <ActionButton
                                label="後で見る"
                                onClick={() => handleReview(item, "deferred")}
                                disabled={isActing}
                                variant="defer"
                              />
                              <ActionButton
                                label="今後ルール化"
                                onClick={() => handleRegisterPromptRule(item)}
                                disabled={isActing}
                                variant="learn"
                                icon={<Sparkles className="h-3.5 w-3.5" />}
                              />
                            </div>
                          ) : (
                            <span className="text-xs text-slate-300">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
        </>
        )}
      </div>
    </main>
  );
}

function RecipeCard({
  recipe,
  onApprove,
  onReject,
}: {
  recipe: PendingRecipe;
  onApprove: () => void;
  onReject: () => void;
}) {
  const [acting, setActing] = useState(false);
  const totalChanges = recipe.files.reduce((sum, f) => sum + f.changes.length, 0);
  const riskLevel = recipe.risk_level ?? "low";
  const riskBadge =
    riskLevel === "high"
      ? "bg-rose-100 text-rose-700 border-rose-200"
      : riskLevel === "medium"
      ? "bg-amber-100 text-amber-700 border-amber-200"
      : "bg-emerald-100 text-emerald-700 border-emerald-200";

  const handle = async (action: () => Promise<void> | void) => {
    setActing(true);
    try {
      await action();
    } finally {
      setActing(false);
    }
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs font-bold text-slate-500">{recipe.rev ?? recipe.id}</span>
        <span className={`rounded-full border px-2 py-0.5 text-[10px] font-bold ${riskBadge}`}>
          {riskLevel}
        </span>
      </div>
      <h3 className="mt-2 text-sm font-bold text-slate-900">{recipe.title}</h3>
      <p className="mt-1 text-xs text-slate-500">
        変更箇所: {totalChanges}件 /{" "}
        {recipe.files.map((f) => f.path.split("/").pop()).join(", ")}
      </p>
      {recipe.shion_recommendation && (
        <div className="mt-2 flex items-center gap-2">
          <ShionBadge recommendation={recipe.shion_recommendation} />
          {recipe.shion_reason && (
            <span className="text-[11px] text-slate-500">{recipe.shion_reason}</span>
          )}
        </div>
      )}
      {recipe.intelligence_comment && (
        <div className="mt-2 rounded border border-purple-200 bg-purple-50 px-2.5 py-1.5 text-[11px] text-purple-700">
          {recipe.intelligence_comment}
        </div>
      )}
      <div className="mt-3 flex gap-2">
        <button
          onClick={() => handle(onApprove)}
          disabled={acting}
          className="rounded border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-40"
        >
          適用待ちへ送る
        </button>
        <button
          onClick={() => handle(onReject)}
          disabled={acting}
          className="rounded border border-rose-300 bg-rose-50 px-3 py-1.5 text-xs font-semibold text-rose-700 hover:bg-rose-100 disabled:opacity-40"
        >
          ❌ 却下
        </button>
      </div>
    </div>
  );
}

const SHION_BADGE_STYLES: Record<string, { label: string; className: string }> = {
  auto:    { label: "自動修正可", className: "bg-blue-100 text-blue-700 border-blue-200" },
  discuss: { label: "要相談",     className: "bg-orange-100 text-orange-700 border-orange-200" },
  review:  { label: "要確認",     className: "bg-slate-100 text-slate-500 border-slate-200" },
};

function ShionBadge({ recommendation }: { recommendation: "auto" | "discuss" | "review" }) {
  const style = SHION_BADGE_STYLES[recommendation] ?? SHION_BADGE_STYLES.review;
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold ${style.className}`}>
      紫苑: {style.label}
    </span>
  );
}

function Stat({ label, value, icon }: { label: string; value: number; icon: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-slate-900">{value}</div>
    </div>
  );
}

function SummaryChip({
  label,
  value,
  color,
  icon,
}: {
  label: string;
  value: number;
  color: "emerald" | "amber" | "rose";
  icon: React.ReactNode;
}) {
  const colorMap = {
    emerald: "border-emerald-200 bg-emerald-50 text-emerald-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
    rose: "border-rose-200 bg-rose-50 text-rose-700",
  };
  return (
    <div className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${colorMap[color]}`}>
      {icon}
      {label}: <span className="font-bold">{value}</span>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-cyan-100 bg-white p-3">
      <div className="text-xs font-medium text-slate-500">{label}</div>
      <div className="mt-1 text-lg font-bold text-slate-900">{value}</div>
    </div>
  );
}

function TrustMetric({ label, value, detail }: { label: string; value: React.ReactNode; detail: string }) {
  return (
    <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-3">
      <div className="text-xs font-medium text-emerald-800">{label}</div>
      <div className="mt-1 text-lg font-bold text-slate-900">{value}</div>
      <div className="mt-1 text-[11px] text-slate-500">{detail}</div>
    </div>
  );
}

function trustAttentionLabel(item: string) {
  const labels: Record<string, string> = {
    knowledge_corrections_need_review: "Knowledge訂正レビュー",
    pdca_rules_expired: "PDCA期限切れ",
    pdca_rules_expiring_soon: "PDCA期限近い",
    memory_usage_log_not_recent: "監査ログ未更新",
  };
  return labels[item] || item;
}

const ACTION_STYLES = {
  approve: "border-emerald-300 bg-emerald-50 text-emerald-700 hover:bg-emerald-100",
  reject: "border-rose-300 bg-rose-50 text-rose-700 hover:bg-rose-100",
  defer: "border-slate-300 bg-slate-50 text-slate-600 hover:bg-slate-100",
  learn: "border-cyan-300 bg-cyan-50 text-cyan-700 hover:bg-cyan-100",
};

function gapPriorityClass(priority: string) {
  const key = String(priority || "").toLowerCase();
  if (key === "critical") return "bg-rose-100 text-rose-800";
  if (key === "high") return "bg-amber-100 text-amber-800";
  if (key === "medium") return "bg-sky-100 text-sky-800";
  return "bg-slate-200 text-slate-700";
}

function ActionButton({
  label,
  onClick,
  disabled,
  variant,
  icon,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  variant: "approve" | "reject" | "defer" | "learn";
  icon?: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded border px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${ACTION_STYLES[variant]}`}
    >
      {icon ? <span className="mr-1 inline-flex align-middle">{icon}</span> : null}
      {label}
    </button>
  );
}
