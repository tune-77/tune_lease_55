"use client";

import React from "react";
import { LifeBuoy, ExternalLink, Info, AlertTriangle, CheckCircle2, TrendingDown } from "lucide-react";
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
        name: "営業部別分析",
        href: "/department",
        description: "営業部ごとの成約率・平均スコア・平均金利・業種構成を表示。",
        tags: ["分析", "営業"],
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
        href: "/subsidy",
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

      {/* スコア判定クイックリファレンス */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">📊 スコア判定クイックリファレンス</h2>
        <div className="grid grid-cols-3 gap-3">
          {[
            { range: '70点以上', label: '承認推奨', color: 'bg-emerald-50 border-emerald-200 text-emerald-800', icon: <CheckCircle2 className="w-4 h-4 text-emerald-500" /> },
            { range: '60〜69点', label: '条件付き承認', color: 'bg-amber-50 border-amber-200 text-amber-800', icon: <AlertTriangle className="w-4 h-4 text-amber-500" /> },
            { range: '60点未満', label: '否決', color: 'bg-rose-50 border-rose-200 text-rose-800', icon: <TrendingDown className="w-4 h-4 text-rose-500" /> },
          ].map(s => (
            <div key={s.range} className={`rounded-xl border p-3 text-center ${s.color}`}>
              <div className="flex justify-center mb-1">{s.icon}</div>
              <p className="text-lg font-black">{s.range}</p>
              <p className="text-xs font-bold mt-0.5">{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 主要財務指標 閾値一覧 */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">📐 主要指標の審査基準値</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50">
                <th className="border border-slate-200 px-3 py-2 text-left font-black text-slate-600">指標</th>
                <th className="border border-slate-200 px-3 py-2 text-center font-black text-emerald-700">良好</th>
                <th className="border border-slate-200 px-3 py-2 text-center font-black text-amber-700">要注意</th>
                <th className="border border-slate-200 px-3 py-2 text-center font-black text-rose-700">高リスク</th>
              </tr>
            </thead>
            <tbody>
              {[
                ['営業利益率', '5%以上', '2〜5%', '2%未満'],
                ['自己資本比率', '20%以上', '10〜20%', '10%未満'],
                ['算出済みPD', '3%以下', '3〜8%', '8%超'],
                ['流動比率', '100%以上', '80〜100%', '80%未満'],
                ['負債比率', '60%以下', '60〜80%', '80%超'],
                ['Q_risk（量子干渉リスク）', '35未満', '35〜59', '60以上'],
              ].map(([label, good, warn, risk]) => (
                <tr key={label} className="even:bg-white odd:bg-slate-50/40">
                  <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{label}</td>
                  <td className="border border-slate-200 px-3 py-2 text-center text-emerald-700 font-bold">{good}</td>
                  <td className="border border-slate-200 px-3 py-2 text-center text-amber-700 font-bold">{warn}</td>
                  <td className="border border-slate-200 px-3 py-2 text-center text-rose-700 font-bold">{risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 標準ワークフロー */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">🔄 標準審査フロー</h2>
        <div className="flex flex-col sm:flex-row gap-2">
          {[
            { step: '1', label: '情報入力', desc: 'リースくんウィザードまたは定量審査で基本情報を入力', href: '/lease-kun' },
            { step: '2', label: 'スコアリング', desc: '既存先RF・新規先ロジスティック回帰を軸にスコア算出。PDは算出済みの場合のみ参照', href: '/' },
            { step: '3', label: 'レポート確認', desc: '審査レポート・業種比較・営業ガイドを確認', href: '/report' },
            { step: '4', label: 'AI相談', desc: '疑問点をAIチャットまたはエージェント議論で深掘り', href: '/chat' },
          ].map(s => (
            <Link key={s.step} href={s.href} className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-3 hover:border-blue-300 hover:bg-blue-50 transition-all group">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-5 h-5 rounded-full bg-indigo-600 text-white text-[10px] font-black flex items-center justify-center flex-shrink-0">{s.step}</span>
                <p className="font-black text-sm text-slate-700 group-hover:text-indigo-700">{s.label}</p>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">{s.desc}</p>
            </Link>
          ))}
        </div>
      </div>

      {/* モデル説明 */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">🧠 AIモデル構成</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          {[
            { name: 'スコアリングモデル', model: 'RandomForest / LogisticRegression', role: '既存先はRandomForest、新規先はロジスティック回帰を軸に審査スコア（0〜100pt）を算出。', color: 'border-indigo-200 bg-indigo-50' },
            { name: '比較分析モデル', model: 'LGBM / RF / LR', role: '定量分析では3モデルを比較し、特徴量重要度と係数から判断材料を補強。', color: 'border-rose-200 bg-rose-50' },
            { name: '量子干渉リスク（Q_risk）', model: 'quantum_analysis_module', role: '複数リスク要因の非線形相互作用を検出。35以上で要注意、60以上で強警戒。', color: 'border-violet-200 bg-violet-50' },
          ].map(m => (
            <div key={m.name} className={`rounded-xl border p-3 ${m.color}`}>
              <p className="font-black text-slate-800 mb-1">{m.name}</p>
              <p className="text-[10px] font-mono text-slate-500 mb-1.5">{m.model}</p>
              <p className="text-slate-600 leading-relaxed">{m.role}</p>
            </div>
          ))}
        </div>
        <div className="flex items-start gap-2 bg-blue-50 border border-blue-200 rounded-xl p-3 text-xs text-blue-700">
          <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <p>スコアリングモデルと高リスク財務パターン警告は<strong>独立して動作</strong>します。後者は高リスク格付先との財務類似度を示す補助指標で、実際のデフォルト確率（PD）ではありません。PDがある場合だけPDとして参照してください。</p>
        </div>
      </div>

      {/* よくある質問 */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h2 className="text-sm font-bold text-slate-700 border-b border-slate-200 pb-2">❓ よくある質問</h2>
        <div className="space-y-2">
          {[
            { q: 'スコアが70点を超えているのに条件付き承認になるのはなぜ？', a: 'AIスコアは参考値です。Q_risk（量子干渉リスク）が高い・算出済みPDや高リスク財務パターン警告が強い・担当者による定性評価で懸念がある場合は最終判断が変わります。' },
            { q: 'AIチャットとエージェント議論の違いは？', a: 'AIチャット（めぶきちゃん）は1対1の対話型相談。エージェント議論は複数AIが異なる立場から審査案件を多角的に評価し合意形成を行います。' },
            { q: '業種別成約率のデータはいつ更新されますか？', a: '審査案件のステータス（成約/失注）が登録されるたびにリアルタイムで更新されます。' },
            { q: 'レポートを印刷したい', a: 'レポートページの「コピー」ボタンでMarkdownテキストをクリップボードにコピーできます。その後、任意のエディタ・ドキュメントに貼り付けて印刷してください。' },
            { q: 'スマートフォンから使えますか？', a: '専用スマホUI（モバイルアプリ版）を提供しています。ヤナミ/雪風ペルソナとの審査相談が可能です。' },
          ].map((item, i) => (
            <details key={i} className="group border border-slate-200 rounded-xl overflow-hidden">
              <summary className="flex items-start gap-3 px-4 py-3 cursor-pointer bg-white hover:bg-slate-50 transition-colors">
                <span className="text-indigo-500 font-black text-xs mt-0.5 shrink-0">Q.</span>
                <span className="text-sm font-bold text-slate-700 flex-1">{item.q}</span>
              </summary>
              <div className="px-4 py-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-600 leading-relaxed">
                <span className="font-black text-slate-500 mr-1">A.</span>{item.a}
              </div>
            </details>
          ))}
        </div>
      </div>

      <div className="text-center pt-2">
        <Link href="/improvement-log" className="text-xs text-slate-400 hover:text-slate-600 underline">
          改善パイプラインログを見る →
        </Link>
      </div>
    </div>
  );
}
