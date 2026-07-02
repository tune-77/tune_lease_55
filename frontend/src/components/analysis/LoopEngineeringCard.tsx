"use client";

import React, { useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import { Loader2, Sparkles, type LucideIcon } from "lucide-react";

type GenericProposal = {
  title: string;
  priority?: "high" | "medium" | "low" | string;
  generated_at?: string;
  status?: string;
  [key: string]: unknown;
};

type FieldSpec = { key: string; label: string };

type LoopEngineeringCardProps = {
  icon: LucideIcon;
  title: string;
  description: string;
  analyzeEndpoint: string;
  proposalsEndpoint: string;
  buttonLabel: string;
  fields: FieldSpec[];
};

const PRIORITY_STYLE: Record<string, string> = {
  high: "border-rose-200 bg-rose-50 text-rose-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-slate-200 bg-slate-50 text-slate-600",
};

export default function LoopEngineeringCard({
  icon: Icon,
  title,
  description,
  analyzeEndpoint,
  proposalsEndpoint,
  buttonLabel,
  fields,
}: LoopEngineeringCardProps) {
  const [proposals, setProposals] = useState<GenericProposal[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

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
          <p className="text-xs text-slate-400">まだ改善案がありません。ボタンを押すと生成されます。</p>
        ) : (
          proposals.map((proposal, index) => (
            <div key={index} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <p className="text-sm font-bold text-slate-900">{proposal.title}</p>
                {proposal.priority && (
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${
                      PRIORITY_STYLE[proposal.priority] || PRIORITY_STYLE.medium
                    }`}
                  >
                    {proposal.priority}
                  </span>
                )}
              </div>
              {fields.map((field) => {
                const value = proposal[field.key];
                if (!value) return null;
                return (
                  <p key={field.key} className="mt-1 text-xs leading-relaxed text-slate-600">
                    <span className="font-bold text-indigo-600">{field.label}: </span>
                    {String(value)}
                  </p>
                );
              })}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
