"use client";

import React from "react";
import { LifeBuoy, ExternalLink } from "lucide-react";
import Link from "next/link";

type FeatureItem = {
  icon: string;
  name: string;
  href: string;
  description: string;
  tags?: string[];
};

type FeatureGroup = {
  title: string;
  items: FeatureItem[];
};

const FEATURES: FeatureGroup[] = [
  {
    title: "🔍 審査・スコアリング",
    items: [
      {
        icon: "📋",
        name: "リースくんウィザード",
        href: "/lease-kun",
        description: "質問形式で審査に必要な情報を収集。初回入力・簡易審査に最適。",
        tags: ["入力", "AI"],
      },
      {
        icon: "📊",
        name: "定量審査",
        href: "/quantitative",
        description: "売上高・自己資本比率等の財務指標を入力して自動スコアリング。",
        tags: ["財務", "スコア"],
      },
      {
        icon: "💬",
        name: "定性審査",
        href: "/qualitative",
        description: "事業内容・経営者評価等の定性情報を記録・評価。",
        tags: ["定性", "スコア"],
      },
      {
        icon: "📄",
        name: "審査レポート",
        href: "/report",
        description: "スコアリング結果・業界比較・Q_risk分析をまとめたレポートを生成。",
        tags: ["レポート"],
      },
    ],
  },
  {
    title: "📈 分析・ダッシュボード",
    items: [
      {
        icon: "📊",
        name: "業種別成約率",
        href: "/industry-winrate",
        description: "past_cases DBから業種別の成約率を集計してグラフ表示。",
        tags: ["分析"],
      },
      {
        icon: "🏢",
        name: "営業部別成約率",
        href: "/sales-dept-winrate",
        description: "営業部ごとの成約率・平均スコア・特性メモを表示。",
        tags: ["分析", "営業"],
      },
      {
        icon: "📉",
        name: "履歴分析・ダッシュボード",
        href: "/history-dash",
        description: "時系列での審査件数・スコア推移・成約率トレンドを可視化。",
        tags: ["分析", "時系列"],
      },
      {
        icon: "🫧",
        name: "ビジュアルインサイト",
        href: "/visual",
        description: "バブルチャート・ヒートマップ・サンキー図で案件を多角分析。",
        tags: ["可視化"],
      },
      {
        icon: "🌌",
        name: "知識宇宙マップ",
        href: "/knowledge-space",
        description: "Obsidianナレッジベースを3D空間でインタラクティブに探索。",
        tags: ["ナレッジ", "3D"],
      },
    ],
  },
  {
    title: "📚 参照・ナレッジ",
    items: [
      {
        icon: "📖",
        name: "法定耐用年数一覧",
        href: "/useful-life",
        description: "国税庁の耐用年数表をカテゴリ別・検索可能な形式で提供。",
        tags: ["参照"],
      },
      {
        icon: "🏭",
        name: "業種別リース物件例",
        href: "/industry-assets",
        description: "業種ごとの代表的なリース設備・取得価格目安・耐用年数一覧。",
        tags: ["参照", "物件"],
      },
      {
        icon: "📐",
        name: "残価設定ガイドライン",
        href: "/residual-guide",
        description: "物件カテゴリ別の推奨残価率と月額計算シミュレーション。",
        tags: ["計算", "残価"],
      },
      {
        icon: "🧾",
        name: "固定資産税シミュレーター",
        href: "/tax-calc",
        description: "取得価格・耐用年数から固定資産税の年次推移を試算。",
        tags: ["計算", "税"],
      },
      {
        icon: "💰",
        name: "補助金情報",
        href: "/subsidies",
        description: "リース設備に活用できる主要補助金の一覧と検索。",
        tags: ["補助金"],
      },
      {
        icon: "❓",
        name: "リース知識 FAQ",
        href: "/faq",
        description: "リースとレンタルの違い・審査ポイント・補助金活用等のQ&A。",
        tags: ["FAQ"],
      },
      {
        icon: "📣",
        name: "営業向け説明ガイド",
        href: "/sales-guide",
        description: "承認・条件付き承認・否決別の顧客説明トーキングポイント。",
        tags: ["営業", "ガイド"],
      },
    ],
  },
  {
    title: "🤖 AI・エージェント",
    items: [
      {
        icon: "💬",
        name: "AIチャット",
        href: "/chat",
        description: "リース審査に関する質問をAIに相談。Obsidianナレッジと連携。",
        tags: ["AI", "チャット"],
      },
      {
        icon: "🤝",
        name: "エージェントハブ",
        href: "/agent-hub",
        description: "軍師AI・審査エージェント等の多機能AIを統合管理。",
        tags: ["AI", "エージェント"],
      },
      {
        icon: "🗣️",
        name: "エージェント議論",
        href: "/debate",
        description: "複数AIが審査案件について多角的に議論して合意形成。",
        tags: ["AI", "議論"],
      },
    ],
  },
];

export default function HelpPage() {
  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <LifeBuoy className="text-blue-500 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">システム機能一覧</h1>
          <p className="text-sm text-slate-500">リース審査AIシステムで利用できる機能の全体像。各ページへのリンク付き。</p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">🚀 はじめての方へ</p>
        <p className="text-xs text-blue-700">
          まず <Link href="/lease-kun" className="underline font-medium">リースくんウィザード</Link> で審査情報を入力し、
          <Link href="/report" className="underline font-medium ml-1">審査レポート</Link> でスコアを確認してください。
          詳細な財務分析は <Link href="/quantitative" className="underline font-medium ml-1">定量審査</Link> で行えます。
        </p>
      </div>

      {FEATURES.map((group) => (
        <div key={group.title} className="space-y-3">
          <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">{group.title}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {group.items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-start gap-3 bg-white border border-slate-200 rounded-xl p-3 shadow-sm hover:border-blue-300 hover:shadow-md transition-all group"
              >
                <span className="text-2xl mt-0.5">{item.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="font-semibold text-slate-800 text-sm group-hover:text-blue-600 transition-colors">{item.name}</p>
                    <ExternalLink size={11} className="text-slate-300 group-hover:text-blue-400" />
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{item.description}</p>
                  {item.tags && (
                    <div className="flex gap-1 mt-1.5 flex-wrap">
                      {item.tags.map((t) => (
                        <span key={t} className="text-xs px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded">{t}</span>
                      ))}
                    </div>
                  )}
                </div>
              </Link>
            ))}
          </div>
        </div>
      ))}

      <div className="text-center pt-2">
        <Link href="/improvement-log" className="text-xs text-slate-400 hover:text-slate-600 underline">
          改善パイプラインログを見る →
        </Link>
      </div>
    </div>
  );
}
