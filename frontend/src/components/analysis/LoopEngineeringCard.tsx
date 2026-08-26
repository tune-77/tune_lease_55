"use client";

import React, { useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import { Loader2, Sparkles, FileText, Copy, Check, type LucideIcon } from "lucide-react";

type GenericProposal = {
  title: string;
  priority?: "high" | "medium" | "low" | string;
  generated_at?: string;
  status?: string;
  evidence_layer?: string;
  evidence_layer_label?: string;
  quality_score?: number;
  proposal_maturity?: string;
  review_status_label?: string;
  effect_tracking?: string;
  genetic_profile?: {
    fitness_score?: number;
    mutation_rate?: number;
    selection_status?: string;
    inheritance_policy?: string;
  };
  [key: string]: unknown;
};

type FieldSpec = { key: string; label: string };

const SHION_SELF_HYPOTHESIS_FIELDS: FieldSpec[] = [
  { key: "hypothesis", label: "仮説" },
  { key: "evidence", label: "根拠" },
  { key: "proposed_change", label: "変更案" },
  { key: "success_metric", label: "成功指標" },
  { key: "verification_plan", label: "検証方法" },
  { key: "risk", label: "リスク" },
];

type LoopEngineeringCardProps = {
  icon: LucideIcon;
  title: string;
  description: string;
  analyzeEndpoint: string;
  proposalsEndpoint: string;
  buttonLabel: string;
  fields: FieldSpec[];
  proposalKindLabel?: string;
  enableRequestDraft?: boolean;
};

function buildShionRequestDraft(proposal: GenericProposal): string {
  const title = String(proposal.title || "").trim();
  const targetPage = String(proposal.target_page || "").trim();
  const hypothesis = String(proposal.hypothesis || "").trim();
  const evidence = String(proposal.evidence || "").trim();
  const proposedChange = String(proposal.proposed_change || "").trim();
  const verificationPlan = String(proposal.verification_plan || "").trim();
  const risk = String(proposal.risk || "").trim();

  const lines = [`紫苑依頼文: ${title} を小さく実装してください。`];
  if (targetPage) lines.push(`- 対象: ${targetPage}`);
  if (hypothesis) lines.push(`- 目的: ${hypothesis}${evidence ? `（${evidence}）` : ""}`);
  if (proposedChange) lines.push(`- 変更案: ${proposedChange}`);
  if (verificationPlan) lines.push(`- 検証方法: ${verificationPlan}`);
  if (risk) lines.push(`- リスク: ${risk}`);
  lines.push("- 変更範囲: 対象ファイルを最小限に。DB/API分岐/スコアリング/認証/デプロイ設定には触らない");
  lines.push("- data/・models/・.claude/state・.streamlit/secrets.toml は変更禁止");
  lines.push("- gitship/deploy の要否は User が判断する（自動実行しない）");
  return lines.join("\n");
}

const PRIORITY_STYLE: Record<string, string> = {
  high: "border-rose-200 bg-rose-50 text-rose-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-slate-200 bg-slate-50 text-slate-600",
};

const MATURITY_STYLE: Record<string, string> = {
  ready_for_review: "border-emerald-200 bg-emerald-50 text-emerald-700",
  watch: "border-sky-200 bg-sky-50 text-sky-700",
  thin: "border-slate-200 bg-white text-slate-500",
};

export default function LoopEngineeringCard({
  icon: Icon,
  title,
  description,
  analyzeEndpoint,
  proposalsEndpoint,
  buttonLabel,
  fields,
  proposalKindLabel = "改善案",
  enableRequestDraft = false,
}: LoopEngineeringCardProps) {
  const [proposals, setProposals] = useState<GenericProposal[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [openDraftIndex, setOpenDraftIndex] = useState<number | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const copyDraft = async (index: number, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex((current) => (current === index ? null : current)), 1500);
    } catch {
      // クリップボードが使えない環境では無視（テキストエリアから手動コピー可能）
    }
  };

  const fetchProposals = async () => {
    try {
      const res = await apiClient.get<{ proposals: GenericProposal[] }>(proposalsEndpoint);
      setProposals(res.data?.proposals || []);
    } catch {
      setProposals([]);
    }
  };

  useEffect(() => {
    fetchProposals();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [proposalsEndpoint]);

  const runAnalyze = async () => {
    setLoading(true);
    setMessage(null);
    try {
      const res = await apiClient.post<{ generated: boolean; reason?: string }>(analyzeEndpoint);
      if (res.data?.generated) {
        setMessage(res.data?.reason || "新しい改善案を考えました。");
        await fetchProposals();
      } else {
        setMessage(res.data?.reason || "改善案を生成できませんでした。");
      }
    } catch {
      setMessage("改善案の生成に失敗しました。");
    } finally {
      setLoading(false);
    }
  };

  const fieldsForProposal = (proposal: GenericProposal) => {
    const hasHypothesisSchema = SHION_SELF_HYPOTHESIS_FIELDS.some((field) => proposal[field.key]);
    return hasHypothesisSchema ? SHION_SELF_HYPOTHESIS_FIELDS : fields;
  };

  return (
    <div className="rounded-lg border border-indigo-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5 text-indigo-600" />
          <div>
            <h2 className="text-sm font-bold text-slate-900">{title}</h2>
            <p className="text-xs text-slate-500">{description}</p>
          </div>
        </div>
        <button
          onClick={runAnalyze}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-xs font-bold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          {buttonLabel}
        </button>
      </div>

      {message && <p className="mt-3 text-xs font-bold text-indigo-700">{message}</p>}

      <div className="mt-3 space-y-2">
        {proposals.length === 0 ? (
          <p className="text-xs text-slate-400">まだ{proposalKindLabel}はありません。ボタンを押すと生成されます。</p>
        ) : (
          proposals.map((proposal, index) => (
            <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="mb-1 flex flex-wrap items-center gap-1.5">
                    <span className="rounded-full border border-indigo-200 bg-indigo-50 px-2 py-0.5 text-[10px] font-black text-indigo-700">
                      {proposalKindLabel}
                    </span>
                    {proposal.priority && (
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${
                          PRIORITY_STYLE[proposal.priority] || PRIORITY_STYLE.medium
                        }`}
                      >
                        {proposal.priority}
                      </span>
                    )}
                    {(proposal.evidence_layer_label || proposal.evidence_layer) && (
                      <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 text-[10px] font-black text-violet-700">
                        {String(proposal.evidence_layer_label || proposal.evidence_layer)}
                      </span>
                    )}
                    {(proposal.review_status_label || proposal.proposal_maturity) && (
                      <span
                        className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${
                          MATURITY_STYLE[String(proposal.proposal_maturity || "")] || MATURITY_STYLE.watch
                        }`}
                      >
                        {String(proposal.review_status_label || proposal.proposal_maturity)}
                      </span>
                    )}
                    {typeof proposal.quality_score === "number" && (
                      <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-black text-slate-600">
                        quality {proposal.quality_score}
                      </span>
                    )}
                    {typeof proposal.genetic_profile?.fitness_score === "number" && (
                      <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-black text-emerald-700">
                        fitness {proposal.genetic_profile.fitness_score}
                      </span>
                    )}
                    {typeof proposal.genetic_profile?.mutation_rate === "number" && (
                      <span className="rounded-full border border-fuchsia-200 bg-fuchsia-50 px-2 py-0.5 text-[10px] font-black text-fuchsia-700">
                        mutation {Math.round(proposal.genetic_profile.mutation_rate * 100)}%
                      </span>
                    )}
                    {proposal.genetic_profile?.inheritance_policy && (
                      <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-black text-slate-600">
                        {proposal.genetic_profile.inheritance_policy}
                      </span>
                    )}
                  </div>
                  <p className="text-sm font-bold text-slate-900">{proposal.title}</p>
                </div>
              </div>
              {fieldsForProposal(proposal).map((field) => {
                const value = proposal[field.key];
                if (!value) return null;
                return (
                  <p key={field.key} className="mt-1 text-xs leading-relaxed text-slate-600">
                    <span className="font-bold text-indigo-600">{field.label}: </span>
                    {String(value)}
                  </p>
                );
              })}
              {proposal.effect_tracking && (
                <p className="mt-2 border-t border-slate-200 pt-2 text-xs leading-relaxed text-slate-500">
                  <span className="font-bold text-slate-700">効き方の追跡: </span>
                  {String(proposal.effect_tracking)}
                </p>
              )}
              {enableRequestDraft && (
                <div className="mt-2 border-t border-slate-200 pt-2">
                  <button
                    type="button"
                    onClick={() => setOpenDraftIndex((current) => (current === index ? null : index))}
                    className="inline-flex items-center gap-1 rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-[11px] font-bold text-emerald-700 hover:bg-emerald-100"
                  >
                    <FileText className="h-3 w-3" />
                    採用したい: 紫苑依頼文を作る
                  </button>
                  {openDraftIndex === index && (
                    <div className="mt-2">
                      <textarea
                        readOnly
                        value={buildShionRequestDraft(proposal)}
                        rows={7}
                        className="w-full resize-y rounded-md border border-emerald-200 bg-emerald-50/50 px-2 py-1.5 text-[11px] leading-relaxed text-slate-800 outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => copyDraft(index, buildShionRequestDraft(proposal))}
                        className="mt-1 inline-flex items-center gap-1 rounded-md bg-emerald-600 px-2 py-1 text-[11px] font-bold text-white hover:bg-emerald-700"
                      >
                        {copiedIndex === index ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                        {copiedIndex === index ? "コピーしました" : "コピー"}
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
