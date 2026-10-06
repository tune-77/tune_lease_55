"use client";

// REV-488: 紫苑の過去イラストをランダムに1枚出す（Gemini 呼び出しなし・毎朝の生成は停止中）。
// 画像は public ではなく API（/api/shion/illustrations/file/...）から配信する。
// next start はビルド後に public へ足したファイルを 404 にするため。
import { useEffect, useState } from "react";
import Image from "next/image";
import { X } from "lucide-react";
import { apiClient } from "@/lib/api";

type Illustration = { available: boolean; date?: string; url?: string };

function useShionIllustration(mode: "daily" | "random", enabled = true): Illustration | null {
  const [illustration, setIllustration] = useState<Illustration | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    apiClient
      .get<Illustration>("/api/shion/illustrations/random", { params: { mode } })
      .then(({ data }) => {
        if (!cancelled) setIllustration(data);
      })
      .catch(() => {
        if (!cancelled) setIllustration({ available: false });
      });
    return () => {
      cancelled = true;
    };
  }, [mode, enabled]);
  return illustration;
}

type Props = { mode: "daily" | "random"; className?: string };

/** 日付つきのイラスト1枚。取れなければ何も出さない。 */
export default function ShionIllustration({ mode, className = "" }: Props) {
  const illustration = useShionIllustration(mode);
  if (!illustration?.available || !illustration.url) return null;
  return (
    <figure className={className}>
      <Image
        src={illustration.url}
        alt={`紫苑のイラスト ${illustration.date ?? ""}`}
        width={640}
        height={360}
        unoptimized
        className="aspect-[16/9] w-full rounded-lg border border-violet-100 object-cover"
      />
      <figcaption className="mt-1 text-right text-[10px] text-slate-400">{illustration.date}</figcaption>
    </figure>
  );
}

const TODAY_KEY = "shion-today-illustration-date";

function localDateKey(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

/**
 * 「今日の紫苑」（/chat 上部と紫苑対話室）。その日最初に開いた時だけ1回出す。
 * 表示済みは共通キーで持つので、どちらかで見たらその日は両方とも出ない。
 */
export function TodayShionCard({ className = "" }: { className?: string }) {
  // 初回描画は画像取得前なので何も出さず、サーバー描画との食い違いは起きない
  const [open, setOpen] = useState(() => {
    try {
      return typeof window !== "undefined" && window.localStorage.getItem(TODAY_KEY) !== localDateKey();
    } catch {
      return false; // localStorage が使えない環境では出さない
    }
  });
  useEffect(() => {
    if (!open) return;
    try {
      window.localStorage.setItem(TODAY_KEY, localDateKey()); // 今日はもう出した
    } catch {
      // 記録できなくても表示は続ける
    }
  }, [open]);
  const illustration = useShionIllustration("daily", open);
  if (!open || !illustration?.available || !illustration.url) return null;
  return (
    <div className={`flex-shrink-0 rounded-xl border border-violet-100 bg-violet-50/60 p-3 ${className}`}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <p className="text-sm font-bold text-violet-800">
          今日の紫苑
          <span className="ml-2 text-[11px] font-normal text-slate-400">{illustration.date} のイラスト</span>
        </p>
        <button
          type="button"
          onClick={() => setOpen(false)}
          aria-label="今日の紫苑を閉じる"
          className="rounded-md p-1 text-slate-400 hover:bg-white hover:text-slate-600"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <Image
        src={illustration.url}
        alt={`今日の紫苑 ${illustration.date ?? ""}`}
        width={640}
        height={360}
        unoptimized
        className="mx-auto aspect-[16/9] w-full max-w-md rounded-lg object-cover"
      />
    </div>
  );
}
