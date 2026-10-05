"use client";

// REV-470: 現場メモから推定した経営者の定性シグナル（不安・自信・切迫感・説明の曖昧さ/食い違い/一貫性）。
// 参考表示専用。スコア・判定には反映しない。根拠としてメモの該当文を引用する。
import React, { useEffect, useState } from "react";
import { MessageSquareQuote } from "lucide-react";
import { fetchInterviewSignals, type InterviewSignalsResult } from "@/lib/interviewSignals";

type Props = {
  memo: string;
};

const LEVEL_STYLE: Record<string, string> = {
  強: "bg-violet-100 text-violet-800",
  中: "bg-violet-50 text-violet-700",
  弱: "bg-slate-100 text-slate-600",
};

export default function InterviewSignalsCard({ memo }: Props) {
  // 取得済みの結果をどのメモに対するものかと一緒に持ち、メモが変わったら読み込み中として扱う
  const [fetched, setFetched] = useState<{ memo: string; data: InterviewSignalsResult | null } | null>(null);

  useEffect(() => {
    if (!memo.trim()) return;
    let cancelled = false;
    fetchInterviewSignals(memo).then((data) => {
      if (!cancelled) setFetched({ memo, data });
    });
    return () => {
      cancelled = true;
    };
  }, [memo]);

  const loading = Boolean(memo.trim()) && fetched?.memo !== memo;
  const result = loading ? null : fetched?.data ?? null;

  if (result && !result.enabled) return null;

  return (
    <div className="rounded-2xl border border-violet-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-black text-violet-900">
          <MessageSquareQuote className="h-4 w-4" aria-hidden="true" />
          現場メモの定性シグナル
        </h3>
        <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-[11px] font-bold text-amber-800">
          推測・参考表示（スコア・判定には未反映）
        </span>
      </div>

      {!memo.trim() ? (
        <p className="mt-3 text-sm text-slate-500">
          「審査入力」の現場メモが空のため、読み取るものがありません。面談の様子を書くと、ここに参考として表示されます。
        </p>
      ) : loading ? (
        <p className="mt-3 text-sm text-slate-500">メモを読み取っています...</p>
      ) : !result ? (
        <p className="mt-3 text-sm text-slate-500">読み取りに失敗しました（審査結果には影響ありません）。</p>
      ) : result.signals.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">メモから目立ったシグナルは読み取れませんでした。</p>
      ) : (
        <ul className="mt-3 space-y-3">
          {result.signals.map((signal) => (
            <li key={signal.key} className="rounded-xl border border-slate-100 bg-slate-50 p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-slate-800">{signal.label}</span>
                <span className={`rounded-full px-2 py-0.5 text-[11px] font-black ${LEVEL_STYLE[signal.level] ?? LEVEL_STYLE["弱"]}`}>
                  {signal.level}
                </span>
              </div>
              <ul className="mt-2 space-y-1">
                {signal.evidence.map((item, index) => (
                  <li key={index} className="text-xs leading-5 text-slate-600">
                    <span className="text-slate-400">メモ: </span>「{item.quote}」
                    <span className="ml-1 text-slate-400">（手がかり: {item.cue}）</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}

      {result && (
        <p className="mt-3 text-[11px] leading-5 text-slate-500">
          {result.disclaimer}
          {result.excluded_sentence_count
            ? ` 年齢・性別・国籍など属性に触れる${result.excluded_sentence_count}文は読み取りに使っていません。`
            : ""}
        </p>
      )}
    </div>
  );
}
