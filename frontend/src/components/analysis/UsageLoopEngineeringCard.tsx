"use client";

import React, { useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import { Eye, Loader2, Sparkles } from "lucide-react";

type UsageProposal = {
  title: string;
  target_page?: string;
  reason?: string;
  priority?: "high" | "medium" | "low" | string;
  generated_at?: string;
};

const PRIORITY_STYLE: Record<string, string> = {
  high: "border-rose-200 bg-rose-50 text-rose-700",
  medium: "border-amber-200 bg-amber-50 text-amber-700",
  low: "border-slate-200 bg-slate-50 text-slate-600",
};

export default function UsageLoopEngineeringCard() {
  const [proposals, setProposals] = useState<UsageProposal[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const fetchProposals = async () => {
    try {
      const res = await apiClient.get<{ proposals: UsageProposal[] }>("/api/usage-loop/proposals");
      setProposals(res.data?.proposals || []);
    } catch {
      setProposals([]);
    }
  };

  useEffect(() => {
    fetchProposals();
  }, []);

  const runPropose = async () => {
    setLoading(true);
    setMessage(null);
    try {
      const res = await apiClient.post<{ generated: boolean; reason?: string }>("/api/usage-loop/propose");
      if (res.data?.generated) {
        setMessage("画面利用状況から新しい改善案を考えました。");
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
          <Eye className="h-5 w-5 text-indigo-600" />
          <div>
            <h2 className="text-sm font-bold text-slate-900">画面利用ループエンジニアリング</h2>
            <p className="text-xs text-slate-500">
              紫苑がUserの画面利用状況を観察し、Geminiで改善案を考える機能
            </p>
          </div>
        </div>
        <button
          onClick={runPropose}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-xs font-bold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          利用状況から改善案を考える
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
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-black ${
                    PRIORITY_STYLE[proposal.priority || "medium"] || PRIORITY_STYLE.medium
                  }`}
                >
                  {proposal.priority || "medium"}
                </span>
              </div>
              {proposal.target_page && (
                <p className="mt-1 text-[11px] font-bold text-indigo-600">対象: {proposal.target_page}</p>
              )}
              {proposal.reason && <p className="mt-1 text-xs leading-relaxed text-slate-600">{proposal.reason}</p>}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
