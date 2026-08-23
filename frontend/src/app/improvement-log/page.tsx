"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import {
  AlertCircle,
  AlertTriangle,
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
  Eye,
  Scale,
  MessageCircleHeart,
  TrendingDown,
  BookOpenCheck,
  Trash2,
  PenLine,
} from "lucide-react";
import LoopEngineeringCard from "@/components/analysis/LoopEngineeringCard";

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
  codex_auto_queue_detail?: QueueStatus;
  shion_error_repair_queue?: QueueStatus;
  shion_error_repair_result?: QueueResultStatus;
  surfaces?: Record<string, string>;
  note?: string;
};

type QueueStatus = {
  available?: boolean;
  path?: string;
  generated_at?: string;
  status?: string;
  queued_count?: number;
  safe_count?: number;
  maybe_count?: number;
  manual_or_blocked_count?: number;
  items?: {
    id?: string;
    title?: string;
    target_module?: string;
    execution_status?: string;
  }[];
  manual_or_blocked?: {
    id?: string;
    title?: string;
    reason?: string;
  }[];
};

type QueueResultStatus = {
  available?: boolean;
  path?: string;
  generated_at?: string;
  total?: number;
  succeeded?: number;
  failed?: number;
  results?: {
    id?: string;
    title?: string;
    exit_code?: number;
    backend?: string;
    stderr?: string;
  }[];
};

type ShionActionLedgerEntry = {
  timestamp?: string;
  action?: string;
  summary?: string;
  risk_level?: string;
  requires_user_approval?: boolean;
  user_approved?: boolean | null;
  target?: string | null;
  result?: string;
};

type ShionActionLedgerSummary = {
  generated_at?: string;
  days?: number;
  total?: number;
  by_action?: Record<string, number>;
  pending_approval_count?: number;
  pending_approval?: ShionActionLedgerEntry[];
  recent?: ShionActionLedgerEntry[];
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
  detail?: string;
  raw_preview?: string;
  source?: string;
  source_event_id?: string;
  source_ts?: string;
  source_surface?: string;
  event_id?: string;
  recorded_at?: string;
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
    shion_review_loop?: {
      status?: string;
      label?: string;
      steps?: string[];
    };
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

type ImprovementTriageRecord = {
  canonical_key: string;
  item_id?: string;
  source_event_id?: string;
  title?: string;
  decision?: string;
  reason?: string;
  rule_decision?: string;
  classified_by?: string;
  decided_at?: string;
  approved_at?: string;
  codex_request_draft?: string;
};

type ImprovementTriageResponse = {
  records?: ImprovementTriageRecord[];
  counts?: Record<string, number>;
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
  pending_llm?: boolean;
  category?: string;
  description: string;
  source?: string;
  target?: string;
  risk?: string;
  auto_fix_allowed?: boolean;
  affected_files?: string[];
  applied_at?: string;
  manual_reason?: string;
};

type ImprovementLogTab = "improvements" | "recipes" | "ledger";

type RelatedFeatureAction = {
  label: string;
  hint: string;
  href?: string;
  tab?: ImprovementLogTab;
  keywords: string[];
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

type ScreeningInputAssistSummary = {
  source?: string;
  summary?: {
    event_count: number;
    session_count: number;
    by_action: Record<string, number>;
    search_count: number;
    copy_count: number;
    copy_rate: number | null;
    score_submit_count: number;
    submitted_after_copy_count: number;
    submitted_after_copy_rate: number | null;
    avg_copied_fields: number | null;
    avg_confirm_fields: number | null;
    avg_changed_after_copy: number | null;
    avg_elapsed_after_copy_ms: number | null;
  };
  recent_events?: {
    id?: string;
    ts?: string;
    action?: string;
    industry_sub?: string;
    asset_name?: string;
    source_company_name?: string;
    copied_field_count?: number;
    changed_after_copy_count?: number;
  }[];
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

type JudgmentAssetPromotionCandidate = {
  id: string;
  candidate_type: string;
  research_topic: string;
  claim: string;
  effective_claim: string;
  edited_claim?: string;
  evidence_path?: string;
  promotion_status: string;
  verified_status: string;
  use_count: number;
  useful_count: number;
  neutral_count: number;
  rejected_count: number;
  edit_count: number;
  last_feedback_at?: string;
  verification_note?: string;
  score: number;
};

type JudgmentAssetPromotionSummary = {
  count: number;
  active_count: number;
  promotion_policy: string;
  candidates: JudgmentAssetPromotionCandidate[];
};

type AgenticSkillInboxItem = {
  id: string;
  source_event_id?: string;
  tool_name: string;
  candidate_type: string;
  claim: string;
  status: string;
  review_decision?: string;
  review_note?: string;
  created_at?: string;
  promotion_policy?: string;
  case_context?: {
    company_name?: string;
    industry_cat?: string;
    asset_name?: string;
    score?: number;
  };
};

type AgenticSkillInboxSummary = {
  count: number;
  promotion_policy: string;
  items: AgenticSkillInboxItem[];
};

type AgenticSkillFlowCheck = {
  status: "ok" | "warn" | "empty" | string;
  checks: { name: string; status: string; message: string }[];
  summary: {
    usage_events: number;
    result_events: number;
    reviewable_results: number;
    linked_reviewable_results: number;
    inbox_items: number;
    open_inbox_items: number;
    review_decisions: number;
  };
  guardrail: string;
};

type AgenticSkillNextActions = {
  mode: string;
  status: string;
  proposals: {
    priority: string;
    type: string;
    title: string;
    reason: string;
    human_action: string;
    score?: {
      impact: string;
      risk: string;
      effort: string;
      evidence: string;
      recommendation: string;
    };
    target?: unknown;
  }[];
  guardrail: string;
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
  expired: { label: "期限切れ", className: "bg-slate-100 text-slate-400 border-slate-200" },
  EXPIRED: { label: "期限切れ", className: "bg-slate-100 text-slate-400 border-slate-200" },
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

const LEDGER_TYPE_LABELS: Record<string, string> = {
  patch_json: "JSONパッチ",
  llm_diff: "LLM差分",
  manual: "手動対応",
  rag_boost_adjust: "RAG調整",
};

const RELATED_FEATURE_ACTIONS: RelatedFeatureAction[] = [
  {
    label: "今回の修正案",
    tab: "recipes",
    hint: "この改善から作られた1回限りの修正パッチを確認します",
    keywords: ["改善ログからの機能連携強化", "機能連携", "直接遷移", "修正案", "自動修正", "パッチ", "適用待ち"],
  },
  {
    label: "自動修正ルール",
    tab: "ledger",
    hint: "今後も使う継続ルールや承認待ちルールを確認します",
    keywords: ["改善ログからの機能連携強化", "機能連携", "直接遷移", "ルール", "PDCA", "今後ルール", "再発防止"],
  },
  {
    label: "紫苑対話",
    href: "/lease-intelligence",
    hint: "紫苑の対話・記憶・判断資産連携を確認します",
    keywords: ["紫苑", "対話", "記憶", "回答", "自己言及", "具体性", "深掘り", "判断資産"],
  },
  {
    label: "帰還データ検疫",
    href: "/cloudrun-return-review",
    hint: "Cloud Runから戻った改善入力の検疫画面を開きます",
    keywords: ["Cloud Run", "cloudrun", "GCS", "帰還", "検疫", "cloudrun_input", "cloudrun_gcs_input"],
  },
  {
    label: "審査・分析",
    href: "/screening",
    hint: "審査案件の入力・分析画面を開きます",
    keywords: ["審査", "スコア", "案件", "判断", "物件", "稟議"],
  },
  {
    label: "結果登録",
    href: "/register",
    hint: "成約・失注など審査結果の登録画面を開きます",
    keywords: ["結果登録", "成約", "失注", "実績", "支払い"],
  },
  {
    label: "ループ証跡",
    href: "/loop-proof",
    hint: "改善ループや判断資産化の証跡を確認します",
    keywords: ["ループ", "証跡", "再利用", "改善履歴", "判断資産"],
  },
  {
    label: "運用情報",
    href: "/operations",
    hint: "システム概要・DevOps・記憶運用を統合画面で確認します",
    keywords: ["システム", "概要", "監視", "導線", "機能連携", "デプロイ", "Cloudflare", "環境", "運用", "DevOps"],
  },
  {
    label: "DevOps詳細",
    href: "/devops",
    hint: "Cloud RunやCloudflareなど運用状態を詳細確認します",
    keywords: ["デプロイ", "Cloudflare", "環境", "運用", "DevOps"],
  },
];

function relatedFeatureActionsFor(item: ImprovementItem): RelatedFeatureAction[] {
  const haystack = [
    item.title,
    item.reason,
    item.detail,
    item.raw_preview,
    item.category,
    item.source,
    item.source_surface,
    item.source_event_id,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return RELATED_FEATURE_ACTIONS.filter((action) =>
    action.keywords.some((keyword) => haystack.includes(keyword.toLowerCase()))
  ).slice(0, 3);
}

function formatRate(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatDurationMs(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value < 1000) return `${Math.round(value)}ms`;
  const seconds = Math.round(value / 1000);
  if (seconds < 60) return `${seconds}秒`;
  return `${Math.floor(seconds / 60)}分${String(seconds % 60).padStart(2, "0")}秒`;
}

function isTodayLocalDate(value?: string) {
  if (!value) return false;
  const today = new Date().toLocaleDateString("sv-SE");
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value.slice(0, 10) === today;
  return parsed.toLocaleDateString("sv-SE") === today;
}

function screeningInputAssistVerdict(summary?: ScreeningInputAssistSummary["summary"]) {
  const sessionCount = summary?.session_count ?? 0;
  const searchCount = summary?.search_count ?? 0;
  const copyRate = summary?.copy_rate ?? null;
  const submittedAfterCopyRate = summary?.submitted_after_copy_rate ?? null;
  const avgChangedAfterCopy = summary?.avg_changed_after_copy ?? null;
  if (!summary || sessionCount < 5 || searchCount < 3) {
    return {
      label: "保留",
      className: "border-amber-200 bg-amber-50 text-amber-800",
      icon: <Clock className="h-4 w-4" />,
      reason: "まだ判断に足る利用データが少ないため、採否は保留です。",
      nextAction: "まずは審査画面で数件使い、検索・コピー・提出までのログを増やします。",
    };
  }
  if ((copyRate ?? 0) >= 0.25 && (submittedAfterCopyRate ?? 0) >= 0.6 && (avgChangedAfterCopy ?? 0) <= 3) {
    return {
      label: "採用",
      className: "border-emerald-200 bg-emerald-50 text-emerald-800",
      icon: <CheckCircle2 className="h-4 w-4" />,
      reason: "検索後にコピーされ、コピー後も大きな手戻りなく審査提出まで進んでいます。",
      nextAction: "次はコピー候補の精度を上げ、平均提出時間の短縮を継続して見ます。",
    };
  }
  if (searchCount >= 5 && (copyRate ?? 0) < 0.1) {
    return {
      label: "却下候補",
      className: "border-rose-200 bg-rose-50 text-rose-800",
      icon: <XCircle className="h-4 w-4" />,
      reason: "検索されてもコピーにつながっておらず、候補品質か表示位置を見直す必要があります。",
      nextAction: "類似案件のランキング理由と差分プレビューを改善してから再計測します。",
    };
  }
  return {
    label: "保留",
    className: "border-amber-200 bg-amber-50 text-amber-800",
    icon: <AlertTriangle className="h-4 w-4" />,
    reason: "利用は始まっていますが、採用判断にはコピー率・提出率・手戻りのいずれかがまだ弱い状態です。",
    nextAction: "コピー後に変更された項目を見て、要確認フィールドの設計を調整します。",
  };
}

// batch_apply.py のスキップ条件と同じ判定。
// 「承認済み」でも manual / pending_llm は自動適用されないことを画面上で区別する
type LedgerEffectiveStatus = "applied" | "pending_review" | "manual_only" | "awaiting_apply";

function isAutoApplyExempt(rule: LedgerRule): boolean {
  return rule.type === "manual" || (rule.type === "llm_diff" && !!rule.pending_llm);
}

function ledgerEffectiveStatus(rule: LedgerRule): LedgerEffectiveStatus {
  if (rule.applied_at) return "applied";
  if (rule.pending_review) return "pending_review";
  if (isAutoApplyExempt(rule)) return "manual_only";
  return "awaiting_apply";
}

const AGENTIC_SKILL_LABELS: Record<string, string> = {
  structure_judgment_asset_candidate: "判断資産候補化",
  validate_lease_source_summary: "情報源検証",
  convert_research_to_screening_insights: "審査確認点化",
  build_screening_decision_flow: "判断分岐整理",
  write_scqa_report: "SCQA整理",
};

export default function ImprovementLogPage() {
  const [activeTab, setActiveTab] = useState<ImprovementLogTab>("improvements");
  const [data, setData] = useState<ImprovementLog | null>(null);
  const [summary, setSummary] = useState<PipelineSummary | null>(null);
  const [gapAnalysis, setGapAnalysis] = useState<GapAnalysis | null>(null);
  const [promptSummary, setPromptSummary] = useState<PromptFeedbackSummary | null>(null);
  const [screeningInputAssistSummary, setScreeningInputAssistSummary] = useState<ScreeningInputAssistSummary | null>(null);
  const [trustSummary, setTrustSummary] = useState<OperationalTrustSummary | null>(null);
  const [triageRecords, setTriageRecords] = useState<ImprovementTriageRecord[]>([]);
  const [copiedFixKey, setCopiedFixKey] = useState("");
  const [expandedFixKey, setExpandedFixKey] = useState("");
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("NEEDS_REVIEW");
  const [actionLoading, setActionLoading] = useState<Record<string, boolean>>({});
  const [pendingRecipes, setPendingRecipes] = useState<PendingRecipe[]>([]);
  const [recipesLoading, setRecipesLoading] = useState(false);
  const [dismissedRecipes, setDismissedRecipes] = useState<Set<string>>(new Set());
  const [recipeStatus, setRecipeStatus] = useState<RecipeStatus | null>(null);
  const [actionLedgerSummary, setActionLedgerSummary] = useState<ShionActionLedgerSummary | null>(null);
  const [showCodexRequestDetails, setShowCodexRequestDetails] = useState(true);
  const [recipeError, setRecipeError] = useState("");
  const [ledgerRules, setLedgerRules] = useState<LedgerRule[]>([]);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [ledgerError, setLedgerError] = useState("");
  const [ledgerTypeFilter, setLedgerTypeFilter] = useState("ALL");
  const [approvingRuleIds, setApprovingRuleIds] = useState<Set<string>>(new Set());
  const [hiddenImprovementKeys, setHiddenImprovementKeys] = useState<Set<string>>(new Set());
  const [isCloudRunHost, setIsCloudRunHost] = useState(false);
  const [judgmentAssetPromotion, setJudgmentAssetPromotion] = useState<JudgmentAssetPromotionSummary | null>(null);
  const [judgmentAssetPromotionLoading, setJudgmentAssetPromotionLoading] = useState(false);
  const [judgmentAssetPromotionError, setJudgmentAssetPromotionError] = useState("");
  const [judgmentAssetPromotionMessage, setJudgmentAssetPromotionMessage] = useState("");
  const [judgmentAssetActionLoading, setJudgmentAssetActionLoading] = useState<Record<string, boolean>>({});
  const [agenticSkillInbox, setAgenticSkillInbox] = useState<AgenticSkillInboxSummary | null>(null);
  const [agenticSkillInboxLoading, setAgenticSkillInboxLoading] = useState(false);
  const [agenticSkillInboxError, setAgenticSkillInboxError] = useState("");
  const [agenticSkillInboxMessage, setAgenticSkillInboxMessage] = useState("");
  const [agenticSkillReviewLoading, setAgenticSkillReviewLoading] = useState<Record<string, boolean>>({});
  const [agenticSkillDrafts, setAgenticSkillDrafts] = useState<Record<string, string>>({});
  const [agenticSkillFlowCheck, setAgenticSkillFlowCheck] = useState<AgenticSkillFlowCheck | null>(null);
  const [agenticSkillNextActions, setAgenticSkillNextActions] = useState<AgenticSkillNextActions | null>(null);

  const fetchJudgmentAssetPromotion = useCallback(async () => {
    setJudgmentAssetPromotionLoading(true);
    setJudgmentAssetPromotionError("");
    try {
      const res = await apiClient.get<JudgmentAssetPromotionSummary>("/api/judgment-assets/promotion-candidates", {
        params: { limit: 6 },
      });
      setJudgmentAssetPromotion(res.data);
    } catch {
      setJudgmentAssetPromotion(null);
      setJudgmentAssetPromotionError("判断資産の昇格候補を取得できませんでした");
    } finally {
      setJudgmentAssetPromotionLoading(false);
    }
  }, []);

  const handleJudgmentAssetAction = useCallback(async (
    candidate: JudgmentAssetPromotionCandidate,
    action: "promote" | "hold" | "reject",
  ) => {
    setJudgmentAssetActionLoading((prev) => ({ ...prev, [candidate.id]: true }));
    setJudgmentAssetPromotionError("");
    setJudgmentAssetPromotionMessage("");
    try {
      if (action === "promote") {
        const res = await apiClient.post(`/api/judgment-assets/promotion-candidates/${candidate.id}/promote`);
        const status = res.data?.promotion?.status || "";
        setJudgmentAssetPromotionMessage(
          status === "queued_for_local_promotion"
            ? "Cloud Run上では昇格申請として記録しました。正規判断資産への反映はローカル側で行います。"
            : "正規判断資産へ昇格しました。次回の紫苑レビューで使われます。"
        );
      } else {
        await apiClient.post(`/api/judgment-assets/promotion-candidates/${candidate.id}/review`, {
          action,
          comment: `improvement-log UI ${action}`,
        });
        setJudgmentAssetPromotionMessage(action === "hold" ? "判断資産候補を保留しました。" : "判断資産候補を捨てました。");
      }
      await fetchJudgmentAssetPromotion();
    } catch {
      setJudgmentAssetPromotionError(action === "promote" ? "判断資産への昇格に失敗しました" : "判断資産候補のレビュー更新に失敗しました");
    } finally {
      setJudgmentAssetActionLoading((prev) => ({ ...prev, [candidate.id]: false }));
    }
  }, [fetchJudgmentAssetPromotion]);

  const fetchAgenticSkillInbox = useCallback(async () => {
    setAgenticSkillInboxLoading(true);
    setAgenticSkillInboxError("");
    try {
      const [res, flowRes, nextActionsRes] = await Promise.all([
        apiClient.get<AgenticSkillInboxSummary>("/api/judgment-assets/agentic-skill-inbox", {
          params: { limit: 10, status: "candidate" },
        }),
        apiClient.get<AgenticSkillFlowCheck>("/api/judgment-assets/agentic-skill-flow-check"),
        apiClient.get<AgenticSkillNextActions>("/api/judgment-assets/agentic-skill-next-actions", {
          params: { limit: 3 },
        }),
      ]);
      setAgenticSkillInbox(res.data);
      setAgenticSkillFlowCheck(flowRes.data);
      setAgenticSkillNextActions(nextActionsRes.data);
      const drafts: Record<string, string> = {};
      (res.data.items || []).forEach((item) => {
        drafts[item.id] = item.claim || "";
      });
      setAgenticSkillDrafts(drafts);
    } catch {
      setAgenticSkillInbox(null);
      setAgenticSkillFlowCheck(null);
      setAgenticSkillNextActions(null);
      setAgenticSkillInboxError("紫苑ADKレビュー箱を取得できませんでした");
    } finally {
      setAgenticSkillInboxLoading(false);
    }
  }, []);

  const handleAgenticSkillReview = useCallback(async (
    item: AgenticSkillInboxItem,
    decision: "adopted" | "revised" | "held" | "rejected",
  ) => {
    setAgenticSkillReviewLoading((prev) => ({ ...prev, [item.id]: true }));
    setAgenticSkillInboxError("");
    setAgenticSkillInboxMessage("");
    try {
      const draft = (agenticSkillDrafts[item.id] || item.claim || "").trim();
      await apiClient.post(`/api/judgment-assets/agentic-skill-inbox/${item.id}/review`, {
        decision,
        note: `improvement-log UI ${decision}`,
        edited_claim: decision === "revised" ? draft : "",
      });
      setAgenticSkillInboxMessage(
        decision === "adopted"
          ? "ADK候補を採用として記録しました。自動昇格はしていません。"
          : decision === "revised"
            ? "ADK候補を修正済みとして記録しました。"
            : decision === "held"
              ? "ADK候補を保留しました。"
              : "ADK候補を却下しました。"
      );
      await fetchAgenticSkillInbox();
    } catch {
      setAgenticSkillInboxError("紫苑ADKレビュー箱の更新に失敗しました");
    } finally {
      setAgenticSkillReviewLoading((prev) => ({ ...prev, [item.id]: false }));
    }
  }, [agenticSkillDrafts, fetchAgenticSkillInbox]);

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
      const [res, statusRes, actionLedgerRes] = await Promise.all([
        apiClient.get<{ recipes: PendingRecipe[] }>("/api/recipes/pending"),
        apiClient.get<RecipeStatus>("/api/recipes/status"),
        apiClient.get<ShionActionLedgerSummary>("/api/shion/action-ledger/summary", {
          params: { days: 7 },
        }).catch(() => null),
      ]);
      setPendingRecipes(res.data.recipes ?? []);
      setRecipeStatus(statusRes.data ?? null);
      setActionLedgerSummary(actionLedgerRes?.data ?? null);
    } catch (error) {
      setPendingRecipes([]);
      setRecipeStatus(null);
      setActionLedgerSummary(null);
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

  const handleRecipeApproveAndApply = useCallback(
    async (recipe: PendingRecipe) => {
      setRecipeError("");
      try {
        const res = await apiClient.post<{ status: string; message?: string }>(
          `/api/recipes/${recipe.id}/approve-and-apply`,
        );
        setDismissedRecipes((prev) => new Set(prev).add(recipe.id));
        const status = res.data?.status || "";
        const message = res.data?.message || "";
        if (status !== "applied") {
          setRecipeError(`自動適用は完了しませんでした: ${status}${message ? ` / ${message}` : ""}`);
        }
        await fetchRecipes();
      } catch (err: any) {
        const detail = err?.response?.data?.detail || "承認後の自動適用に失敗しました。";
        setRecipeError(String(detail));
      }
    },
    [fetchRecipes]
  );

  const fetchLog = useCallback(async () => {
    setLoading(true);
    try {
      const [logRes, summaryRes, gapsRes, promptRes, inputAssistRes, trustRes, triageRes] = await Promise.all([
        apiClient.get<ImprovementLog>("/api/improvement-log"),
        apiClient.get<PipelineSummary>("/api/improvement-pipeline/summary"),
        apiClient.get<GapAnalysis>("/api/lease-system-gaps"),
        apiClient.get<PromptFeedbackSummary>("/api/prompt-feedback/summary"),
        apiClient.get<ScreeningInputAssistSummary>("/api/screening-input-assist-events/summary", {
          params: { limit: 500 },
        }).catch(() => null),
        apiClient.get<OperationalTrustSummary>("/api/operational-trust/summary"),
        apiClient.get<ImprovementTriageResponse>("/api/improvement/triage"),
      ]);
      setData(logRes.data);
      setSummary(summaryRes.data);
      setGapAnalysis(gapsRes.data);
      setPromptSummary(promptRes.data || null);
      setScreeningInputAssistSummary(inputAssistRes?.data || null);
      setTrustSummary(trustRes.data || null);
      setTriageRecords(triageRes.data?.records || []);
    } catch {
      setData(null);
      setScreeningInputAssistSummary(null);
      setTriageRecords([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setIsCloudRunHost(window.location.hostname.endsWith(".run.app"));
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
    fetchJudgmentAssetPromotion();
  }, [fetchJudgmentAssetPromotion]);

  useEffect(() => {
    fetchAgenticSkillInbox();
  }, [fetchAgenticSkillInbox]);

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
        const rawContext = item.raw_preview || item.detail || "";
        await apiClient.post("/api/improvement-log/review", {
          key: item.canonical_key || item.id || "",
          title: item.title,
          action,
          reason: [
            item.reason || item.auto_fix_policy?.reason || `UI経由で${action}`,
            item.source_event_id ? `source_event_id: ${item.source_event_id}` : "",
            rawContext ? `原文: ${rawContext}` : "",
          ].filter(Boolean).join("\n"),
        });
        setHiddenImprovementKeys((prev) => new Set(prev).add(itemKey));
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
        setHiddenImprovementKeys((prev) => new Set(prev).add(itemKey));
        await fetchLog();
      } catch {
        // 失敗時は何もしない（再fetchで状態は保持される）
      } finally {
        setActionLoading((prev) => ({ ...prev, [itemKey]: false }));
      }
    },
    [fetchLog]
  );

  const handleDeleteImprovement = useCallback(
    async (item: ImprovementItem) => {
      const itemKey = item.canonical_key || item.id || item.title;
      setActionLoading((prev) => ({ ...prev, [itemKey]: true }));
      try {
        await apiClient.post("/api/improvement-log/delete", {
          key: item.canonical_key || item.id || item.title || "",
          title: item.title || item.id || "改善項目",
        });
        setHiddenImprovementKeys((prev) => new Set(prev).add(itemKey));
        await fetchLog();
      } catch {
        // 失敗時は一覧に残す
      } finally {
        setActionLoading((prev) => ({ ...prev, [itemKey]: false }));
      }
    },
    [fetchLog]
  );

  const filteredItems = useMemo(() => {
    const items = data?.items ?? [];
    return items.filter((item) => {
      const itemKey = item.canonical_key || item.id || item.title;
      if (hiddenImprovementKeys.has(itemKey)) return false;
      const matchesStatus = status === "ALL" || item.status === status || (status === "NEEDS_REVIEW" && item.status === "needs_review");
      const needle = query.trim().toLowerCase();
      const matchesQuery =
        !needle ||
        item.id.toLowerCase().includes(needle) ||
        (item.title || "").toLowerCase().includes(needle) ||
        (item.canonical_key || "").toLowerCase().includes(needle);
      return matchesStatus && matchesQuery;
    });
  }, [data?.items, hiddenImprovementKeys, query, status]);

  const obsidianStatus = data?.obsidian_compliance?.status || "unknown";
  const obsidianViolations = data?.obsidian_compliance?.violations?.length || 0;

  const visibleRecipes = pendingRecipes.filter((r) => !dismissedRecipes.has(r.id));
  const codexRequestDrafts = useMemo(
    () => (actionLedgerSummary?.recent || [])
      .filter((entry) => entry.action === "codex_request_drafted")
      .sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || ""))),
    [actionLedgerSummary?.recent]
  );
  const actionLedgerCheckItems = useMemo(
    () => (actionLedgerSummary?.pending_approval || []).filter(
      (entry) => !["codex_request_drafted", "implementation_observed"].includes(entry.action || "")
    ),
    [actionLedgerSummary?.pending_approval]
  );

  const todayFixQueue = useMemo(() => {
    const items = data?.items || [];
    return triageRecords
      .filter((record) => record.decision === "today" && isTodayLocalDate(record.approved_at))
      .map((record) => {
        const matchedItem = items.find((item) =>
          item.canonical_key === record.canonical_key ||
          item.id === record.item_id ||
          item.source_event_id === record.source_event_id
        );
        return { record, item: matchedItem };
      })
      .sort((a, b) => String(b.record.approved_at || b.record.decided_at || "").localeCompare(String(a.record.approved_at || a.record.decided_at || "")));
  }, [data?.items, triageRecords]);

  const ledgerTypes = useMemo(() => {
    const types = new Set(ledgerRules.map((r) => r.type || "unknown"));
    return ["ALL", ...Array.from(types).sort()];
  }, [ledgerRules]);

  const filteredLedgerRules = useMemo(
    () => ledgerRules.filter((r) => ledgerTypeFilter === "ALL" || (r.type || "unknown") === ledgerTypeFilter),
    [ledgerRules, ledgerTypeFilter]
  );

  // パイプライン自体の障害検出（batch_apply・朝報告が止まっている可能性）
  const pipelineAlerts = useMemo(
    () => ledgerRules.filter((r) => r.category === "pipeline_fix" && r.pending_review),
    [ledgerRules]
  );

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
            onClick={() => {
              if (activeTab === "improvements") {
                fetchLog();
                fetchJudgmentAssetPromotion();
                fetchRecipes();
              } else if (activeTab === "recipes") {
                fetchRecipes();
              } else {
                fetchLedgerRules();
                fetchRecipes();
              }
            }}
            className="ml-auto inline-flex items-center gap-2 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
          >
            <RefreshCw className="h-4 w-4" />
            更新
          </button>
        </div>

        {pipelineAlerts.length > 0 && (
          <div className="rounded-lg border border-rose-300 bg-rose-50 p-4">
            <div className="flex items-center gap-2 text-sm font-bold text-rose-800">
              <AlertTriangle className="h-4 w-4" />
              改善パイプライン自体に障害が検出されています
            </div>
            <ul className="mt-2 space-y-1 text-xs text-rose-700">
              {pipelineAlerts.map((r) => (
                <li key={r.rev_id}>
                  <span className="font-mono font-bold">{r.rev_id}</span>: {r.description}
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-rose-600">
              パイプラインが失敗している間は、承認済みルールの自動適用や朝の改善レポート生成が止まっている可能性があります。先にこちらの復旧を確認してください。
            </p>
          </div>
        )}

        <section className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
                <ShieldCheck className="h-4 w-4" />
                自動修復・確認ログの所在
              </div>
              <p className="mt-1 text-xs font-semibold leading-5 text-slate-500">
                件数の出どころをここに集約しています。自動修正候補、紫苑の軽微エラー修復、人間確認が必要な箱を分けて表示します。
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                fetchRecipes();
                fetchLedgerRules();
              }}
              className="inline-flex shrink-0 items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              状態更新
            </button>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-lg border border-blue-100 bg-blue-50 p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-black text-blue-900">Codex自動改善キュー</div>
                <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-black text-blue-700">
                  {recipeStatus?.codex_auto_queue_detail?.status || recipeStatus?.codex_auto_queue?.status || "未生成"}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs font-bold">
                <div className="rounded bg-white p-2 text-blue-800">
                  <div className="text-lg font-black">{recipeStatus?.codex_auto_queue_detail?.queued_count ?? recipeStatus?.codex_auto_queue?.queued_count ?? 0}</div>
                  実行候補
                </div>
                <div className="rounded bg-white p-2 text-emerald-700">
                  <div className="text-lg font-black">{recipeStatus?.codex_auto_queue_detail?.safe_count ?? recipeStatus?.codex_auto_queue?.safe_count ?? 0}</div>
                  safe
                </div>
                <div className="rounded bg-white p-2 text-amber-700">
                  <div className="text-lg font-black">{recipeStatus?.codex_auto_queue_detail?.manual_or_blocked_count ?? recipeStatus?.codex_auto_queue?.manual_or_blocked_count ?? 0}</div>
                  保留
                </div>
              </div>
              <p className="mt-2 break-all text-[11px] font-semibold text-blue-700">
                {recipeStatus?.codex_auto_queue_detail?.path || recipeStatus?.surfaces?.codex_queue || "reports/codex_auto_queue_*.json"}
              </p>
            </div>

            <div className="rounded-lg border border-emerald-100 bg-emerald-50 p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-black text-emerald-900">紫苑の軽微エラー修復</div>
                <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-black text-emerald-700">
                  {recipeStatus?.shion_error_repair_queue?.status || "未生成"}
                </span>
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs font-bold">
                <div className="rounded bg-white p-2 text-emerald-800">
                  <div className="text-lg font-black">{recipeStatus?.shion_error_repair_queue?.queued_count ?? 0}</div>
                  実行候補
                </div>
                <div className="rounded bg-white p-2 text-emerald-700">
                  <div className="text-lg font-black">{recipeStatus?.shion_error_repair_queue?.safe_count ?? 0}</div>
                  safe
                </div>
                <div className="rounded bg-white p-2 text-rose-700">
                  <div className="text-lg font-black">{recipeStatus?.shion_error_repair_result?.failed ?? 0}</div>
                  失敗
                </div>
              </div>
              <p className="mt-2 break-all text-[11px] font-semibold text-emerald-700">
                {recipeStatus?.shion_error_repair_queue?.path || "reports/shion_error_repair_queue_*.json"}
              </p>
              {recipeStatus?.shion_error_repair_result?.available && (
                <p className="mt-1 break-all text-[11px] font-semibold text-emerald-700">
                  結果: {recipeStatus.shion_error_repair_result.path}
                </p>
              )}
            </div>

            <div className="rounded-lg border border-amber-100 bg-amber-50 p-3">
              <div className="text-xs font-black text-amber-900">人間確認が必要な箱</div>
              <div className="mt-3 space-y-2 text-xs font-bold text-amber-800">
                <button
                  type="button"
                  onClick={() => setActiveTab("recipes")}
                  className="flex w-full items-center justify-between rounded bg-white px-3 py-2 text-left hover:bg-amber-100"
                >
                  <span>今回の修正案</span>
                  <span>{recipeStatus?.pending_count ?? visibleRecipes.length}件</span>
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("ledger")}
                  className="flex w-full items-center justify-between rounded bg-white px-3 py-2 text-left hover:bg-amber-100"
                >
                  <span>今後の自動修正ルール</span>
                  <span>{ledgerRules.filter((r) => ledgerEffectiveStatus(r) === "pending_review").length}件</span>
                </button>
                <div className="flex items-center justify-between rounded bg-white px-3 py-2">
                  <span>紫苑行動ログ（要確認）</span>
                  <span>{actionLedgerCheckItems.length}件</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-black text-slate-800">自動修正候補の探索履歴</div>
              <div className="mt-3 flex items-end gap-2">
                <span className="text-3xl font-black text-slate-900">
                  {actionLedgerSummary?.by_action?.codex_request_drafted ?? 0}
                </span>
                <span className="pb-1 text-xs font-bold text-slate-500">件 / 直近{actionLedgerSummary?.days ?? 7}日</span>
              </div>
              <p className="mt-2 text-[11px] font-semibold leading-5 text-slate-500">
                これは「自動修正できる候補を探した履歴」です。0件の探索も記録されますが、承認待ちの依頼ではありません。
              </p>
              <button
                type="button"
                onClick={() => setShowCodexRequestDetails((value) => !value)}
                className="mt-3 inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-1.5 text-[11px] font-black text-slate-700 hover:bg-slate-100"
              >
                <Eye className="h-3.5 w-3.5" />
                  {showCodexRequestDetails ? "探索履歴を閉じる" : `探索履歴を見る (${codexRequestDrafts.length}件)`}
              </button>
              {showCodexRequestDetails && (
                <div className="mt-3 max-h-72 space-y-2 overflow-y-auto pr-1">
                  {codexRequestDrafts.length === 0 ? (
                    <div className="rounded bg-white p-2 text-[11px] font-semibold text-slate-500">
                      直近期間の探索履歴はありません。
                    </div>
                  ) : (
                    codexRequestDrafts.map((entry, index) => (
                      <div key={`${entry.timestamp}-${index}`} className="rounded bg-white p-2 text-[11px] leading-5 text-slate-600">
                        <div className="flex flex-wrap items-center gap-2 font-black text-slate-800">
                          <span>{entry.timestamp || "-"}</span>
                          {entry.result && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">{entry.result}</span>}
                        </div>
                        <div className="mt-1 font-semibold">{entry.summary || entry.action}</div>
                        {entry.target && (
                          <div className="mt-1 break-all font-mono text-[10px] text-slate-400">
                            {entry.target}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {todayFixQueue.length > 0 && (
          <section className="overflow-hidden rounded-2xl border border-emerald-200 bg-white shadow-sm">
            <div className="flex flex-col gap-3 border-b border-emerald-100 bg-emerald-50 p-4 md:flex-row md:items-center md:justify-between">
              <div className="flex items-start gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-600 text-white">
                  <Wrench className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-base font-black text-emerald-950">今日直すリスト</h2>
                  <p className="mt-1 text-sm font-bold leading-6 text-emerald-800">
                    `/lease-intelligence` で `修正` を押して、修正キューに入った自己提案だけを表示しています。
                  </p>
                </div>
              </div>
              <span className="rounded-full bg-white px-3 py-1 text-xs font-black text-emerald-800">
                修正キュー {todayFixQueue.length}件
              </span>
            </div>
            <div className="grid gap-3 p-4">
              {todayFixQueue.map(({ record, item }) => {
                const key = record.canonical_key || record.item_id || record.title || "";
                const title = record.title || item?.title || record.item_id || "修正キュー項目";
                const reason = record.reason || item?.auto_fix_policy?.reason || item?.reason || "";
                const copied = copiedFixKey === key;
                const expanded = expandedFixKey === key;
                return (
                  <article key={`${key}-${record.approved_at || record.decided_at}`} className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 text-[10px] font-black">
                          {record.item_id && <span className="rounded-full bg-slate-900 px-2 py-1 text-white">{record.item_id}</span>}
                          <span className="rounded-full bg-emerald-100 px-2 py-1 text-emerald-800">修正キュー</span>
                          {record.approved_at && <span className="rounded-full bg-white px-2 py-1 text-emerald-700">{record.approved_at}</span>}
                        </div>
                        <h3 className="mt-3 text-sm font-black leading-6 text-slate-900">{title}</h3>
                        {reason && <p className="mt-1 text-xs font-bold leading-5 text-slate-600">{reason}</p>}
                        <p className="mt-2 break-all text-[11px] font-semibold text-slate-400">{record.canonical_key}</p>
                      </div>
                      <div className="flex shrink-0 flex-wrap gap-2">
                        {record.codex_request_draft && (
                          <>
                            <button
                              type="button"
                              onClick={() => setExpandedFixKey(expanded ? "" : key)}
                              className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-white px-3 py-2 text-xs font-black text-emerald-700 transition hover:bg-emerald-50"
                            >
                              <Eye className="h-4 w-4" />
                              {expanded ? "内容を閉じる" : "内容を見る"}
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                navigator.clipboard?.writeText(record.codex_request_draft || "").then(() => {
                                  setCopiedFixKey(key);
                                  window.setTimeout(() => setCopiedFixKey(""), 1600);
                                }).catch(() => {});
                              }}
                              className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-black text-white transition hover:bg-emerald-700"
                            >
                              <ClipboardList className="h-4 w-4" />
                              {copied ? "コピー済み" : "依頼文コピー"}
                            </button>
                          </>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            setActiveTab("improvements");
                            setStatus("ALL");
                            setQuery(record.canonical_key || record.item_id || title);
                          }}
                          className="inline-flex items-center gap-1 rounded-lg border border-emerald-200 bg-white px-3 py-2 text-xs font-black text-emerald-700 transition hover:bg-emerald-50"
                        >
                          <Search className="h-4 w-4" />
                          一覧で確認
                        </button>
                      </div>
                    </div>
                    {expanded && record.codex_request_draft && (
                      <pre className="mt-3 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-emerald-200 bg-white p-3 text-[11px] leading-5 text-slate-700">
                        {record.codex_request_draft}
                      </pre>
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        )}

        <section className="overflow-hidden rounded-2xl border border-amber-200 bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-amber-100 bg-amber-50 p-4 md:flex-row md:items-center md:justify-between">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500 text-white">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-black text-amber-950">判断資産レビュー・昇格</h2>
                <p className="mt-1 text-sm font-bold leading-6 text-amber-800">
                  紫苑レビューで使われ、人間が評価した候補だけを、正規判断資産へ昇格します。Cloud Run検疫とは別の人間承認ゲートです。
                </p>
                <p className="mt-1 text-xs font-bold leading-5 text-amber-700">
                  {isCloudRunHost
                    ? "Cloud Run版では昇格申請を記録します。正本への反映はローカル版で行います。"
                    : "ローカル/Cloudflareローカル版では、この画面から正本の判断資産へ昇格できます。"}
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs font-black">
              <span className="rounded-full bg-white px-3 py-1 text-amber-800">
                正規判断資産 {judgmentAssetPromotion?.active_count ?? 0}件
              </span>
              <span className="rounded-full bg-white px-3 py-1 text-amber-800">
                昇格候補 {judgmentAssetPromotion?.count ?? 0}件
              </span>
            </div>
          </div>
          <div className="p-4">
            {judgmentAssetPromotionError && (
              <div className="mb-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm font-bold text-rose-700">
                {judgmentAssetPromotionError}
              </div>
            )}
            {judgmentAssetPromotionMessage && (
              <div className="mb-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm font-bold text-emerald-700">
                {judgmentAssetPromotionMessage}
              </div>
            )}
            {judgmentAssetPromotionLoading ? (
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-8 text-center text-sm font-bold text-slate-500">
                判断資産の昇格候補を読み込み中...
              </div>
            ) : !judgmentAssetPromotion?.candidates?.length ? (
              <div className="rounded-xl border border-dashed border-amber-200 bg-amber-50/40 p-6 text-center">
                <BookOpenCheck className="mx-auto h-7 w-7 text-amber-500" />
                <p className="mt-2 text-sm font-black text-amber-950">今すぐ昇格する候補はありません</p>
                <p className="mt-1 text-xs font-bold text-amber-700">
                  審査分析画面で判断資産候補に「効いた」または「修正」を返すと、ここに出ます。
                </p>
              </div>
            ) : (
              <div className="grid gap-3">
                {judgmentAssetPromotion.candidates.map((candidate) => {
                  const isBusy = !!judgmentAssetActionLoading[candidate.id];
                  const displayClaim = candidate.edited_claim || candidate.effective_claim || candidate.claim;
                  return (
                    <article key={candidate.id} className="rounded-xl border border-amber-100 bg-amber-50/50 p-4">
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2 text-[10px] font-black">
                            <span className="rounded-full bg-slate-900 px-2 py-1 text-white">JA-{candidate.id.slice(0, 8)}</span>
                            <span className="rounded-full bg-amber-100 px-2 py-1 text-amber-800">{candidate.candidate_type}</span>
                            <span className="rounded-full bg-white px-2 py-1 text-slate-600">{candidate.research_topic || "manual"}</span>
                            <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700">効いた {candidate.useful_count}</span>
                            <span className="rounded-full bg-blue-50 px-2 py-1 text-blue-700">修正 {candidate.edit_count}</span>
                            <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">使用 {candidate.use_count}</span>
                            {candidate.rejected_count > 0 && (
                              <span className="rounded-full bg-rose-50 px-2 py-1 text-rose-700">違う {candidate.rejected_count}</span>
                            )}
                          </div>
                          <p className="mt-3 text-sm font-bold leading-7 text-slate-900">{displayClaim}</p>
                          <p className="mt-2 break-all text-[11px] font-semibold text-slate-500">
                            出典: {candidate.evidence_path || "manual"} / 状態: {candidate.verified_status}
                          </p>
                        </div>
                        <div className="flex shrink-0 flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => handleJudgmentAssetAction(candidate, "promote")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-black text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <CheckCircle2 className="h-4 w-4" />
                            {isCloudRunHost ? "昇格申請" : "昇格する"}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleJudgmentAssetAction(candidate, "hold")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-black text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <Clock className="h-4 w-4" />
                            保留
                          </button>
                          <button
                            type="button"
                            onClick={() => handleJudgmentAssetAction(candidate, "reject")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-black text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <Trash2 className="h-4 w-4" />
                            捨てる
                          </button>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        <section className="overflow-hidden rounded-2xl border border-cyan-200 bg-white shadow-sm">
          <div className="flex flex-col gap-3 border-b border-cyan-100 bg-cyan-50 p-4 md:flex-row md:items-center md:justify-between">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-cyan-600 text-white">
                <PenLine className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-black text-cyan-950">紫苑ADKレビュー箱</h2>
                <p className="mt-1 text-sm font-bold leading-6 text-cyan-800">
                  紫苑が内側で作った判断資産候補・審査確認点・判断フローを、人間レビュー用に隔離しています。
                </p>
                <p className="mt-1 text-xs font-bold leading-5 text-cyan-700">
                  採用しても自動昇格・スコア変更・RAG反映はしません。使えそうな候補だけ、後続の判断資産レビューへ回します。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs font-black">
              <span className="rounded-full bg-white px-3 py-1 text-cyan-800">
                未レビュー {agenticSkillInbox?.count ?? 0}件
              </span>
              <button
                type="button"
                onClick={fetchAgenticSkillInbox}
                disabled={agenticSkillInboxLoading}
                className="inline-flex items-center gap-1 rounded-full bg-cyan-700 px-3 py-1 text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${agenticSkillInboxLoading ? "animate-spin" : ""}`} />
                更新
              </button>
            </div>
          </div>
          <div className="p-4">
            {agenticSkillInboxError && (
              <div className="mb-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm font-bold text-rose-700">
                {agenticSkillInboxError}
              </div>
            )}
            {agenticSkillInboxMessage && (
              <div className="mb-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm font-bold text-emerald-700">
                {agenticSkillInboxMessage}
              </div>
            )}
            {agenticSkillFlowCheck && (
              <div className={`mb-3 rounded-xl border p-3 ${
                agenticSkillFlowCheck.status === "ok"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : agenticSkillFlowCheck.status === "empty"
                    ? "border-slate-200 bg-slate-50 text-slate-600"
                    : "border-amber-200 bg-amber-50 text-amber-800"
              }`}>
                <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                  <div className="flex items-center gap-2 text-sm font-black">
                    {agenticSkillFlowCheck.status === "ok" ? (
                      <CheckCircle2 className="h-4 w-4" />
                    ) : agenticSkillFlowCheck.status === "empty" ? (
                      <Clock className="h-4 w-4" />
                    ) : (
                      <AlertTriangle className="h-4 w-4" />
                    )}
                    一連の流れ: {agenticSkillFlowCheck.status === "ok" ? "正常" : agenticSkillFlowCheck.status === "empty" ? "未使用" : "要確認"}
                  </div>
                  <div className="flex flex-wrap gap-1.5 text-[11px] font-black">
                    <span className="rounded-full bg-white/80 px-2 py-1">使用 {agenticSkillFlowCheck.summary.usage_events}</span>
                    <span className="rounded-full bg-white/80 px-2 py-1">候補 {agenticSkillFlowCheck.summary.inbox_items}</span>
                    <span className="rounded-full bg-white/80 px-2 py-1">未レビュー {agenticSkillFlowCheck.summary.open_inbox_items}</span>
                    <span className="rounded-full bg-white/80 px-2 py-1">レビュー {agenticSkillFlowCheck.summary.review_decisions}</span>
                  </div>
                </div>
                {agenticSkillFlowCheck.status === "warn" && (
                  <div className="mt-2 grid gap-1 text-[11px] font-bold">
                    {agenticSkillFlowCheck.checks.filter((check) => check.status === "warn").slice(0, 3).map((check) => (
                      <div key={check.name}>{check.name}: {check.message}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {agenticSkillNextActions?.proposals?.length ? (
              <div className="mb-3 rounded-xl border border-violet-200 bg-violet-50 p-3 text-violet-900">
                <div className="mb-2 flex items-center gap-2 text-sm font-black">
                  <Sparkles className="h-4 w-4" />
                  紫苑からの提案
                </div>
                <div className="grid gap-2">
                  {agenticSkillNextActions.proposals.slice(0, 3).map((proposal, index) => (
                    <div key={`${proposal.type}-${index}`} className="rounded-lg bg-white/80 p-3">
                      <div className="flex flex-wrap items-center gap-2 text-[11px] font-black">
                        <span className="rounded-full bg-violet-100 px-2 py-1 text-violet-800">{proposal.priority}</span>
                        {proposal.score?.recommendation && (
                          <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700">{proposal.score.recommendation}</span>
                        )}
                        <span className="text-slate-900">{proposal.title}</span>
                      </div>
                      {proposal.score && (
                        <div className="mt-2 flex flex-wrap gap-1.5 text-[10px] font-black text-slate-600">
                          <span className="rounded-full bg-slate-100 px-2 py-1">impact {proposal.score.impact}</span>
                          <span className="rounded-full bg-slate-100 px-2 py-1">risk {proposal.score.risk}</span>
                          <span className="rounded-full bg-slate-100 px-2 py-1">effort {proposal.score.effort}</span>
                          <span className="rounded-full bg-slate-100 px-2 py-1">evidence {proposal.score.evidence}</span>
                        </div>
                      )}
                      <p className="mt-1 text-xs font-bold leading-5 text-violet-800">{proposal.reason}</p>
                      <p className="mt-1 text-[11px] font-black text-slate-600">次: {proposal.human_action}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            {agenticSkillInboxLoading ? (
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-8 text-center text-sm font-bold text-slate-500">
                紫苑ADKレビュー箱を読み込み中...
              </div>
            ) : !agenticSkillInbox?.items?.length ? (
              <div className="rounded-xl border border-dashed border-cyan-200 bg-cyan-50/40 p-6 text-center">
                <PenLine className="mx-auto h-7 w-7 text-cyan-600" />
                <p className="mt-2 text-sm font-black text-cyan-950">未レビューのADK候補はありません</p>
                <p className="mt-1 text-xs font-bold text-cyan-700">
                  紫苑が審査中にagentic skillを使い、候補化したものだけがここに出ます。
                </p>
              </div>
            ) : (
              <div className="grid gap-3">
                {agenticSkillInbox.items.map((item) => {
                  const isBusy = !!agenticSkillReviewLoading[item.id];
                  const draft = agenticSkillDrafts[item.id] ?? item.claim ?? "";
                  const caseContext = item.case_context || {};
                  return (
                    <article key={item.id} className="rounded-xl border border-cyan-100 bg-cyan-50/50 p-4">
                      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2 text-[10px] font-black">
                            <span className="rounded-full bg-slate-900 px-2 py-1 text-white">ADK-{item.id.slice(0, 8)}</span>
                            <span className="rounded-full bg-cyan-100 px-2 py-1 text-cyan-800">
                              {AGENTIC_SKILL_LABELS[item.tool_name] || item.tool_name}
                            </span>
                            <span className="rounded-full bg-white px-2 py-1 text-slate-600">{item.candidate_type}</span>
                            {caseContext.company_name && (
                              <span className="rounded-full bg-white px-2 py-1 text-slate-600">{caseContext.company_name}</span>
                            )}
                            {caseContext.score !== undefined && (
                              <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-600">score {caseContext.score}</span>
                            )}
                          </div>
                          <textarea
                            value={draft}
                            onChange={(event) => setAgenticSkillDrafts((prev) => ({ ...prev, [item.id]: event.target.value }))}
                            rows={3}
                            className="mt-3 w-full resize-y rounded-lg border border-cyan-100 bg-white px-3 py-2 text-sm font-bold leading-7 text-slate-900 outline-none transition focus:border-cyan-400 focus:ring-2 focus:ring-cyan-100"
                          />
                          <p className="mt-2 break-all text-[11px] font-semibold text-slate-500">
                            出典: {item.source_event_id || "agentic_skill"} / 作成: {item.created_at || "-"} / 状態: {item.status}
                          </p>
                        </div>
                        <div className="flex shrink-0 flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => handleAgenticSkillReview(item, "adopted")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-black text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <CheckCircle2 className="h-4 w-4" />
                            採用
                          </button>
                          <button
                            type="button"
                            onClick={() => handleAgenticSkillReview(item, "revised")}
                            disabled={isBusy || !draft.trim()}
                            className="inline-flex items-center gap-1 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs font-black text-blue-700 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <PenLine className="h-4 w-4" />
                            修正
                          </button>
                          <button
                            type="button"
                            onClick={() => handleAgenticSkillReview(item, "held")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-black text-slate-600 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <Clock className="h-4 w-4" />
                            保留
                          </button>
                          <button
                            type="button"
                            onClick={() => handleAgenticSkillReview(item, "rejected")}
                            disabled={isBusy}
                            className="inline-flex items-center gap-1 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-black text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <Trash2 className="h-4 w-4" />
                            却下
                          </button>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>
        </section>
        <div className="space-y-3">
          <LoopEngineeringCard
            icon={Eye}
            title="紫苑の自己提案: 画面利用"
            description="通常の保留ではなく、紫苑が画面利用状況から出す仮説です"
            analyzeEndpoint="/api/usage-loop/propose"
            proposalsEndpoint="/api/usage-loop/proposals"
            buttonLabel="紫苑に自己提案させる"
            proposalKindLabel="紫苑の自己提案"
            fields={[
              { key: "target_page", label: "対象" },
              { key: "reason", label: "理由" },
            ]}
          />
          <LoopEngineeringCard
            icon={Scale}
            title="紫苑の自己提案: 審査判断乖離"
            description="通常の保留ではなく、紫苑が審査フィードバックから出す確認仮説です"
            analyzeEndpoint="/api/judgment-divergence/analyze"
            proposalsEndpoint="/api/judgment-divergence/proposals"
            buttonLabel="紫苑に自己提案させる"
            proposalKindLabel="紫苑の自己提案"
            fields={[
              { key: "observation", label: "観察" },
              { key: "review_point", label: "確認観点" },
            ]}
          />
          <LoopEngineeringCard
            icon={MessageCircleHeart}
            title="紫苑の自己提案: 人間反応"
            description="通常の保留ではなく、紫苑が回答評価の傾向から出す応答改善仮説です"
            analyzeEndpoint="/api/feedback-pattern/analyze"
            proposalsEndpoint="/api/feedback-pattern/proposals"
            buttonLabel="紫苑に自己提案させる"
            proposalKindLabel="紫苑の自己提案"
            fields={[
              { key: "pattern", label: "傾向" },
              { key: "suggestion", label: "提案" },
            ]}
          />
          <LoopEngineeringCard
            icon={TrendingDown}
            title="紫苑の自己提案: 実績ドリフト"
            description="通常の保留ではなく、紫苑が支払い実績とスコア帯の乖離から出す再校正仮説です"
            analyzeEndpoint="/api/outcome-drift/analyze"
            proposalsEndpoint="/api/outcome-drift/proposals"
            buttonLabel="紫苑に自己提案させる"
            proposalKindLabel="紫苑の自己提案"
            fields={[
              { key: "observation", label: "観察" },
              { key: "review_point", label: "確認観点" },
            ]}
          />
          <LoopEngineeringCard
            icon={BookOpenCheck}
            title="紫苑の自己提案: ナレッジ穴探し"
            description="通常の保留ではなく、紫苑が知識参照0件の質問から出す調査仮説です"
            analyzeEndpoint="/api/knowledge-gap/analyze"
            proposalsEndpoint="/api/knowledge-gap/proposals"
            buttonLabel="紫苑に自己提案させる"
            proposalKindLabel="紫苑の自己提案"
            fields={[
              { key: "reason", label: "理由" },
              { key: "search_hint", label: "検索キーワード案" },
            ]}
          />
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
                  自動改善キュー（claude実行・gemini予備）: {recipeStatus.codex_auto_queue.status || "-"} / safe {recipeStatus.codex_auto_queue.safe_count ?? 0} / maybe {recipeStatus.codex_auto_queue.maybe_count ?? 0} / manual {recipeStatus.codex_auto_queue.manual_or_blocked_count ?? 0}
                </p>
              )}
              <p className="mt-2 text-xs text-slate-500">
                今回の修正案は、この実行で作られた1回限りの修正パッチです。「承認して自動適用」はローカル作業ツリーへ即時適用し、「適用待ちへ送る」は承認済みフォルダへ移して後で処理します。
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
                  isCloudRunHost={isCloudRunHost}
                  onApprove={() => handleRecipeAction(recipe, "approve")}
                  onApproveAndApply={() => handleRecipeApproveAndApply(recipe)}
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
                  承認待ち {ledgerRules.filter((r) => ledgerEffectiveStatus(r) === "pending_review").length}
                </span>
                <span className="rounded-full bg-blue-50 px-2 py-1 font-semibold text-blue-700">
                  適用待ち {ledgerRules.filter((r) => ledgerEffectiveStatus(r) === "awaiting_apply").length}
                </span>
                <span className="rounded-full bg-emerald-50 px-2 py-1 font-semibold text-emerald-700">
                  適用済み {ledgerRules.filter((r) => ledgerEffectiveStatus(r) === "applied").length}
                </span>
                <span className="rounded-full bg-amber-50 px-2 py-1 font-semibold text-amber-700">
                  自動適用対象外 {ledgerRules.filter((r) => ledgerEffectiveStatus(r) === "manual_only").length}
                </span>
                <span className="rounded-full bg-slate-100 px-2 py-1 font-semibold text-slate-600">
                  合計 {ledgerRules.length}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">
                今後の自動修正ルールは、次回以降も同じ種類の修正に使う継続ルールです。「自動適用を許可」すると次回の朝パイプライン（batch_apply）で1回だけ自動適用され、「適用済み」になります。種別が「手動対応」のルールは承認しても自動適用されず、人の実装が必要です。
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {ledgerTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => setLedgerTypeFilter(type)}
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      ledgerTypeFilter === type
                        ? "bg-slate-900 text-white"
                        : "border border-slate-300 bg-white text-slate-600"
                    }`}
                  >
                    {type === "ALL"
                      ? `すべて ${ledgerRules.length}`
                      : `${LEDGER_TYPE_LABELS[type] || type} ${ledgerRules.filter((r) => (r.type || "unknown") === type).length}`}
                  </button>
                ))}
              </div>
              {ledgerError && (
                <p className="mt-2 text-xs font-semibold text-rose-600">{ledgerError}</p>
              )}
            </div>
            {ledgerLoading ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                読み込み中...
              </div>
            ) : filteredLedgerRules.length === 0 ? (
              <div className="rounded-lg border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">
                {ledgerTypeFilter === "ALL" ? "今後の自動修正ルールがありません" : "この種別のルールはありません"}
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
                      {filteredLedgerRules.map((rule) => {
                        const isApproving = approvingRuleIds.has(rule.rev_id);
                        const effectiveStatus = ledgerEffectiveStatus(rule);
                        const exempt = isAutoApplyExempt(rule);
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
                            <td className="px-4 py-3 text-xs text-slate-500">
                              {LEDGER_TYPE_LABELS[rule.type] || rule.type}
                            </td>
                            <td className="px-4 py-3">
                              <div className="text-sm text-slate-800">{rule.description}</div>
                              {exempt && rule.manual_reason && (
                                <div className="mt-0.5 text-[11px] text-slate-400">
                                  手動対応の理由: {rule.manual_reason}
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
                              {effectiveStatus === "applied" ? (
                                <div>
                                  <span className="inline-flex items-center gap-1 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700">
                                    ✅ 適用済み
                                  </span>
                                  <div className="mt-0.5 text-[11px] text-slate-400">{rule.applied_at}</div>
                                </div>
                              ) : effectiveStatus === "pending_review" ? (
                                <div className="flex flex-col items-start gap-1">
                                  <span className="inline-flex rounded-full border border-indigo-200 bg-indigo-50 px-2 py-1 text-xs font-semibold text-indigo-700">
                                    承認待ち
                                  </span>
                                  {exempt && (
                                    <span className="inline-flex rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700">
                                      自動適用対象外
                                    </span>
                                  )}
                                </div>
                              ) : effectiveStatus === "manual_only" ? (
                                <span
                                  className="inline-flex rounded-full border border-amber-200 bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700"
                                  title="承認済みですが、このルールは自動適用の対象外です。人が実装するまで改善は反映されません"
                                >
                                  自動適用対象外
                                </span>
                              ) : (
                                <span
                                  className="inline-flex rounded-full border border-blue-200 bg-blue-50 px-2 py-1 text-xs font-semibold text-blue-700"
                                  title="承認済みで、次回の朝パイプライン（batch_apply）で自動適用される予定です"
                                >
                                  適用待ち（次回自動適用）
                                </span>
                              )}
                            </td>
                            <td className="px-4 py-3">
                              {rule.pending_review ? (
                                <button
                                  onClick={() => handleApproveRule(rule.rev_id)}
                                  disabled={isApproving}
                                  title={
                                    exempt
                                      ? "承認しても自動適用はされません（手動対応が必要な項目）。確認済みとして承認待ちから外します"
                                      : "次回の朝パイプライン（batch_apply）で1回だけ自動適用されます"
                                  }
                                  className="rounded border border-indigo-300 bg-indigo-50 px-3 py-1.5 text-xs font-semibold text-indigo-700 hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-40"
                                >
                                  {isApproving ? "処理中..." : exempt ? "確認済みにする" : "自動適用を許可"}
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

        {screeningInputAssistSummary?.summary && (
          <section className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            {(() => {
              const verdict = screeningInputAssistVerdict(screeningInputAssistSummary.summary);
              return (
                <div className={`mb-3 rounded-lg border p-3 ${verdict.className}`}>
                  <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                    <div>
                      <div className="flex items-center gap-2 text-sm font-black">
                        {verdict.icon}
                        追跡判定: {verdict.label}
                      </div>
                      <p className="mt-1 text-xs font-bold leading-5">{verdict.reason}</p>
                    </div>
                    <div className="rounded-lg bg-white/70 px-3 py-2 text-[11px] font-bold leading-5">
                      success_metric: 平均提出時間 / コピー後提出率 / コピー後変更数
                    </div>
                  </div>
                  <p className="mt-2 text-xs font-bold leading-5">{verdict.nextAction}</p>
                </div>
              );
            })()}
            <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-semibold text-emerald-900">
                  <ClipboardList className="h-4 w-4" />
                  審査入力補助の効果測定
                  {screeningInputAssistSummary.source && (
                    <span className="ml-1 text-xs font-normal text-emerald-700">
                      {screeningInputAssistSummary.source}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-xs text-emerald-800">
                  過去案件コピーと入力中の確認観点が、審査実行までの行動に効いているかを確認します。
                </p>
              </div>
              <a
                href="/screening"
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-emerald-200 bg-white px-3 py-2 text-xs font-black text-emerald-700 transition hover:bg-emerald-100"
              >
                <Search className="h-3.5 w-3.5" />
                審査画面へ
              </a>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <MiniMetric label="セッション" value={screeningInputAssistSummary.summary.session_count} />
              <MiniMetric label="検索→コピー率" value={formatRate(screeningInputAssistSummary.summary.copy_rate)} />
              <MiniMetric label="コピー後提出率" value={formatRate(screeningInputAssistSummary.summary.submitted_after_copy_rate)} />
              <MiniMetric label="平均提出時間" value={formatDurationMs(screeningInputAssistSummary.summary.avg_elapsed_after_copy_ms)} />
            </div>
            <div className="mt-3 grid gap-3 md:grid-cols-4">
              <MiniMetric label="検索回数" value={screeningInputAssistSummary.summary.search_count} />
              <MiniMetric label="コピー回数" value={screeningInputAssistSummary.summary.copy_count} />
              <MiniMetric label="平均コピー項目" value={screeningInputAssistSummary.summary.avg_copied_fields ?? "-"} />
              <MiniMetric label="コピー後変更" value={screeningInputAssistSummary.summary.avg_changed_after_copy ?? "-"} />
            </div>
            <div className="mt-3 rounded-lg border border-emerald-100 bg-white p-3">
              <div className="text-xs font-bold text-slate-700">直近イベント</div>
              <div className="mt-2 space-y-1.5">
                {(screeningInputAssistSummary.recent_events || []).length === 0 ? (
                  <div className="text-xs text-slate-500">まだイベントはありません</div>
                ) : (screeningInputAssistSummary.recent_events || []).slice(-6).reverse().map((event, index) => (
                  <div key={event.id || `${event.ts}-${index}`} className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                    <span className="font-mono text-slate-400">{event.ts || "-"}</span>
                    <span className="rounded-full bg-emerald-50 px-2 py-0.5 font-semibold text-emerald-800">{event.action || "-"}</span>
                    {(event.industry_sub || event.asset_name) && (
                      <span>{[event.industry_sub, event.asset_name].filter(Boolean).join(" / ")}</span>
                    )}
                    {event.source_company_name && <span className="text-slate-500">copy元 {event.source_company_name}</span>}
                    {event.copied_field_count != null && <span>copy {event.copied_field_count}</span>}
                    {event.changed_after_copy_count != null && <span>変更 {event.changed_after_copy_count}</span>}
                  </div>
                ))}
              </div>
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
            <div className="mt-3 rounded-lg border border-indigo-200 bg-white p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-black text-emerald-700">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {data.recursive_self_improvement.shion_review_loop?.label || "紫苑チェックで閉ループ化済み"}
                </span>
                <span className="text-xs font-semibold text-slate-600">
                  改善候補を紫苑で見て、人間の評価を戻し、次の改善PMレポートへ返せます。
                </span>
              </div>
              <div className="mt-2 grid gap-2 md:grid-cols-4">
                {(data.recursive_self_improvement.shion_review_loop?.steps || [
                  "審査分析画面で紫苑レビュー",
                  "役に立った/要修正/違うを記録",
                  "判断資産候補と改善ログへ戻す",
                  "次回の改善PMレポートで再確認",
                ]).map((step, index) => (
                  <div key={`${index}-${step}`} className="rounded-md bg-indigo-50 px-2.5 py-2 text-[11px] font-semibold text-indigo-900">
                    <span className="mr-1 text-indigo-500">{index + 1}.</span>
                    {step}
                  </div>
                ))}
              </div>
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

          <p className="mt-3 text-xs text-slate-500">
            承認・却下・保留・ルール化は「要確認」の項目に表示されます。削除は表示中の項目を一覧から隠します。
          </p>
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
                    const relatedActions = relatedFeatureActionsFor(item);
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
                          {item.source === "cloudrun_gcs_input" && (
                            <div className="mt-1 text-xs text-sky-700">
                              Cloud Run入力
                              {item.source_event_id ? ` / event ${item.source_event_id.slice(0, 8)}` : ""}
                              {item.source_ts || item.recorded_at ? ` / ${item.source_ts || item.recorded_at}` : ""}
                            </div>
                          )}
                          {item.source === "shion_promise" && (
                            <div className="mt-1 text-xs text-fuchsia-700">
                              紫苑の約束（自動下調べ対象）
                            </div>
                          )}
                          {relatedActions.length > 0 && (
                            <div className="mt-2 flex flex-wrap items-center gap-1.5">
                              <span className="text-[11px] font-bold text-slate-400">関連導線</span>
                              {relatedActions.map((action) =>
                                action.tab ? (
                                  <button
                                    key={`${itemKey}-${action.label}`}
                                    type="button"
                                    onClick={() => setActiveTab(action.tab!)}
                                    title={action.hint}
                                    className="rounded-full border border-sky-200 bg-sky-50 px-2 py-0.5 text-[11px] font-bold text-sky-700 transition hover:bg-sky-100"
                                  >
                                    {action.label}
                                  </button>
                                ) : (
                                  <a
                                    key={`${itemKey}-${action.label}`}
                                    href={action.href}
                                    title={action.hint}
                                    className="rounded-full border border-sky-200 bg-white px-2 py-0.5 text-[11px] font-bold text-sky-700 transition hover:bg-sky-50"
                                  >
                                    {action.label}
                                  </a>
                                )
                              )}
                            </div>
                          )}
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
                          <div>{item.auto_fix_policy?.reason || item.reason || "-"}</div>
                          {(item.raw_preview || item.detail) && (
                            <details className="mt-2 rounded-lg border border-slate-200 bg-white p-2">
                              <summary className="cursor-pointer font-bold text-sky-700">原文を見る</summary>
                              <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-relaxed text-slate-700">
                                {item.raw_preview || item.detail}
                              </pre>
                            </details>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {isNeedsReview ? (
                            <div className="flex flex-wrap gap-1.5">
                              <ActionButton
                                label="レビュー承認"
                                onClick={() => handleReview(item, "approved")}
                                disabled={isActing}
                                variant="approve"
                                title="この改善案を実装対象として承認します（ledger と Obsidian に記録されます）"
                              />
                              <ActionButton
                                label="今回は却下"
                                onClick={() => handleReview(item, "rejected")}
                                disabled={isActing}
                                variant="reject"
                                title="今回の提案としては不採用にします。永久拒否ではなく、却下の記録が残ります"
                              />
                              <ActionButton
                                label="後で見る"
                                onClick={() => handleReview(item, "deferred")}
                                disabled={isActing}
                                variant="defer"
                                title="判断を保留します。項目は要確認のままリストに残ります"
                              />
                              <ActionButton
                                label="今後ルール化"
                                onClick={() => handleRegisterPromptRule(item)}
                                disabled={isActing}
                                variant="learn"
                                icon={<Sparkles className="h-3.5 w-3.5" />}
                                title="同種の問題を防ぐPDCAルールとして登録し、AIのプロンプトに注入されます（有効期限つき）"
                              />
                              <ActionButton
                                label="削除"
                                onClick={() => handleDeleteImprovement(item)}
                                disabled={isActing}
                                variant="delete"
                                icon={<Trash2 className="h-3.5 w-3.5" />}
                                title="この改善候補を一覧から削除します（監査ログには deleted として残ります）"
                              />
                            </div>
                          ) : (
                            <ActionButton
                              label="削除"
                              onClick={() => handleDeleteImprovement(item)}
                              disabled={isActing}
                              variant="delete"
                              icon={<Trash2 className="h-3.5 w-3.5" />}
                              title="この改善候補を一覧から削除します（監査ログには deleted として残ります）"
                            />
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
  isCloudRunHost,
  onApprove,
  onApproveAndApply,
  onReject,
}: {
  recipe: PendingRecipe;
  isCloudRunHost: boolean;
  onApprove: () => void;
  onApproveAndApply: () => void;
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
          onClick={() => handle(onApproveAndApply)}
          disabled={acting || isCloudRunHost}
          title="この修正パッチを承認し、ローカル作業ツリーへ即時適用します。gitがcleanでない場合や安全チェック失敗時は止まります"
          className="rounded border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700 hover:bg-blue-100 disabled:opacity-40"
        >
          {isCloudRunHost ? "自動適用はローカルのみ" : "承認して自動適用"}
        </button>
        <button
          onClick={() => handle(onApprove)}
          disabled={acting}
          title="この修正パッチを承認済みフォルダへ移動します。この時点ではコードは変わらず、実適用は別処理が実行します"
          className="rounded border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 hover:bg-emerald-100 disabled:opacity-40"
        >
          適用待ちへ送る
        </button>
        <button
          onClick={() => handle(onReject)}
          disabled={acting}
          title="この修正パッチを破棄します（却下フォルダへ移動し、適用されません）"
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
  delete: "border-slate-300 bg-white text-slate-500 hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700",
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
  title,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  variant: "approve" | "reject" | "defer" | "learn" | "delete";
  icon?: React.ReactNode;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded border px-2 py-1 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${ACTION_STYLES[variant]}`}
    >
      {icon ? <span className="mr-1 inline-flex align-middle">{icon}</span> : null}
      {label}
    </button>
  );
}
