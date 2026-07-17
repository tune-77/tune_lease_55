"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useSidebar } from "@/context/SidebarContext";
import { apiClient } from "@/lib/api";
import { ArrowLeft, Home, Sparkles } from "lucide-react";

const ACTIVITY_KEY = "shion-concierge-activity-v1";
const MAX_ACTIVITY_ITEMS = 30;

type ActivityItem = {
  path: string;
  title: string;
  ts: number;
  count?: number;
};

const PAGE_TITLES: Record<string, string> = {
  "/": "リース知性体 紫苑システム",
  "/screening": "審査・分析",
  "/home": "ダッシュボード",
  "/chat": "紫苑チャット",
  "/demo": "ハッカソンデモ",
  "/demo-home": "デモホーム",
  "/demo/judgment-evolution": "1000件早送りデモ",
  "/demo/knowledge-loop": "知識ループ確認",
  "/devops": "DevOpsサイクル",
  "/improvement-log": "改善PMレポート",
  "/research-organ": "外部調査器官",
  "/voice-chat": "リアルタイム会話",
  "/cases": "過去案件",
  "/shion-memory-system": "紫苑の記憶システム",
  "/shion-core": "知性体コア",
  "/shion-debug": "紫苑デバッグ",
  "/system-overview": "System Overview",
  "/report": "審査レポート",
  "/batch": "バッチ審査",
  "/lease-intelligence": "リース知性体との対話",
  "/multi-shion-demo": "マルチ紫苑デモ",
  "/chat-compare": "紫苑/一般 比較",
  "/shion-identity-check": "自己同一性検査",
  "/cloudrun-return-review": "帰還データ検疫",
  "/register": "結果登録",
};

const HACKATHON_RETURN_ROUTES = new Set([
  "/screening",
  "/lease-intelligence",
  "/register",
  "/demo",
  "/demo/judgment-evolution",
  "/demo/knowledge-loop",
  "/system-overview",
  "/devops",
  "/improvement-log",
  "/shion-memory-system",
  "/cloudrun-return-review",
  "/chat-compare",
  "/shion-identity-check",
  "/multi-shion-demo",
]);

function pageTitle(path: string) {
  return PAGE_TITLES[path] || path.replace(/^\//, "") || "リース知性体 紫苑システム";
}

export default function ContentWrapper({ children }: { children: React.ReactNode }) {
  const { isCollapsed } = useSidebar();
  const pathname = usePathname();
  const router = useRouter();

  const marginClass = isCollapsed ? 'lg:ml-20' : 'lg:ml-64';
  const showHackathonReturn = HACKATHON_RETURN_ROUTES.has(pathname || "");

  useEffect(() => {
    if (typeof window === "undefined" || !pathname) return;
    try {
      const raw = window.localStorage.getItem(ACTIVITY_KEY);
      const existing = raw ? (JSON.parse(raw) as ActivityItem[]) : [];
      const previous = existing.find((item) => item.path === pathname);
      const nextItem = {
        path: pathname,
        title: pageTitle(pathname),
        ts: Date.now(),
        count: Math.max(1, Number(previous?.count || 0) + 1),
      };
      const deduped = existing.filter((item) => item.path !== pathname);
      window.localStorage.setItem(
        ACTIVITY_KEY,
        JSON.stringify([nextItem, ...deduped].slice(0, MAX_ACTIVITY_ITEMS)),
      );
    } catch {
      // localStorage is advisory only.
    }
    // 画面利用ループエンジニアリング(Observe): サーバー側にも訪問イベントを送る。
    // 失敗しても画面表示には影響させない。
    apiClient.post("/api/usage-loop/visit", { path: pathname }).catch(() => {});
  }, [pathname]);

  return (
    <main className={`flex-1 min-h-screen relative transition-all duration-300 overflow-x-hidden pb-[env(safe-area-inset-bottom)] ${marginClass}`}>
      {children}
      {showHackathonReturn && (
        <nav
          aria-label="ハッカソン戻り導線"
          className="fixed bottom-4 right-4 z-50 flex max-w-[calc(100vw-2rem)] flex-wrap items-center justify-end gap-2 rounded-2xl border border-slate-200 bg-white/92 p-2 shadow-2xl shadow-slate-950/15 backdrop-blur-md"
        >
          <button
            type="button"
            onClick={() => router.back()}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-black text-slate-700 transition hover:bg-slate-50"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            前画面へ戻る
          </button>
          <Link
            href="/demo-home"
            className="inline-flex items-center gap-1.5 rounded-xl bg-slate-950 px-3 py-2 text-xs font-black text-white transition hover:bg-slate-800"
          >
            <Sparkles className="h-3.5 w-3.5 text-yellow-300" />
            デモホーム
          </Link>
          <Link
            href="/screening"
            className="hidden items-center gap-1.5 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-black text-emerald-900 transition hover:bg-emerald-100 sm:inline-flex"
          >
            <Home className="h-3.5 w-3.5" />
            審査へ戻る
          </Link>
        </nav>
      )}
    </main>
  );
}
