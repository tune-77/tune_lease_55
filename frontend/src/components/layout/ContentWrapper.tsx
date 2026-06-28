"use client";

import React, { useEffect } from "react";
import { usePathname } from "next/navigation";
import { useSidebar } from "@/context/SidebarContext";

const ACTIVITY_KEY = "shion-concierge-activity-v1";
const MAX_ACTIVITY_ITEMS = 12;

type ActivityItem = {
  path: string;
  title: string;
  ts: number;
};

const PAGE_TITLES: Record<string, string> = {
  "/": "リース知性体 紫苑システム",
  "/screening": "審査・分析",
  "/home": "ダッシュボード",
  "/chat": "紫苑チャット",
  "/research-organ": "外部調査器官",
  "/voice-chat": "リアルタイム会話",
  "/cases": "過去案件",
  "/shion-memory-system": "紫苑の記憶システム",
  "/shion-debug": "紫苑デバッグ",
  "/system-overview": "System Overview",
  "/report": "審査レポート",
  "/batch": "バッチ審査",
  "/lease-intelligence": "リース知性体との対話",
};

function pageTitle(path: string) {
  return PAGE_TITLES[path] || path.replace(/^\//, "") || "リース知性体 紫苑システム";
}

export default function ContentWrapper({ children }: { children: React.ReactNode }) {
  const { isCollapsed } = useSidebar();
  const pathname = usePathname();

  const marginClass = isCollapsed ? 'lg:ml-20' : 'lg:ml-64';

  useEffect(() => {
    if (typeof window === "undefined" || !pathname) return;
    try {
      const raw = window.localStorage.getItem(ACTIVITY_KEY);
      const existing = raw ? (JSON.parse(raw) as ActivityItem[]) : [];
      const nextItem = { path: pathname, title: pageTitle(pathname), ts: Date.now() };
      const deduped = existing.filter((item) => item.path !== pathname);
      window.localStorage.setItem(
        ACTIVITY_KEY,
        JSON.stringify([nextItem, ...deduped].slice(0, MAX_ACTIVITY_ITEMS)),
      );
    } catch {
      // localStorage is advisory only.
    }
  }, [pathname]);

  return (
    <main className={`flex-1 min-h-screen relative transition-all duration-300 overflow-x-hidden pb-[env(safe-area-inset-bottom)] ${marginClass}`}>
      {children}
    </main>
  );
}
