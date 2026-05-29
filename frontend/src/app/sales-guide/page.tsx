"use client";

import React, { useState } from "react";
import { Megaphone, ChevronDown, ChevronRight, CheckCircle2, AlertTriangle, XCircle } from "lucide-react";

type GuideItem = {
  verdict: "承認" | "条件付き承認" | "否決";
  scoreRange: string;
  label: string;
  color: string;
  bgColor: string;
  icon: React.ReactNode;
  summary: string;
  talkingPoints: string[];
  risksToAcknowledge: string[];
  nextActions: string[];
};

const GUIDE: GuideItem[] = [
  {
    verdict: "承認",
    scoreRange: "70pt以上",
    label: "承認",
    color: "#16a34a",
    bgColor: "#f0fdf4",
    icon: <CheckCircle2 size={20} className="text-green-600" />,
    summary: "財務指標・物件条件ともに問題なし。スムーズに契約提案へ移行できます。",
    talkingPoints: [
      "「財務状況・設備条件ともに審査基準を満たしています」",
      "「リース期間・金額は○○円/月のご提案が可能です」",
      "「ご契約後すぐにご利用開始いただけます」",
    ],
    risksToAcknowledge: [],
    nextActions: [
      "見積書・リース契約書の発行",
      "物件発注手続きの開始",
      "契約締結・検収スケジュールの確認",
    ],
  },
  {
    verdict: "条件付き承認",
    scoreRange: "55〜69pt",
    label: "条件付き承認（要審議）",
    color: "#d97706",
    bgColor: "#fffbeb",
    icon: <AlertTriangle size={20} className="text-amber-500" />,
    summary: "一定の懸念はあるが、条件を調整することで成約の可能性があります。",
    talkingPoints: [
      "「ご状況を確認させていただいた上で、最適なプランをご提案できます」",
      "「リース期間の調整や保証条件のご相談が可能です」",
      "「追加資料をご用意いただくことでよりスムーズに進められます」",
    ],
    risksToAcknowledge: [
      "財務指標（自己資本比率・流動比率等）が業種平均を下回っている",
      "リース期間が法定耐用年数に対して長めに設定されている",
      "Q_riskが高く財務データの精査が必要",
    ],
    nextActions: [
      "リース期間の短縮を検討（例: 5年→3年）",
      "前受金・頭金の設定で残額リスクを低減",
      "代表者保証の追加検討",
      "最新決算書・試算表・事業計画書の追加収集",
      "二次審査（現地確認）の実施検討",
    ],
  },
  {
    verdict: "否決",
    scoreRange: "54pt以下",
    label: "否決",
    color: "#dc2626",
    bgColor: "#fef2f2",
    icon: <XCircle size={20} className="text-red-600" />,
    summary: "現時点では審査基準を満たせませんでした。代替手段や将来的な再申請を検討します。",
    talkingPoints: [
      "「現時点では条件が整っていないため、慎重に対応させていただきます」",
      "「将来的な財務改善後に再度ご検討いただける場合がございます」",
      "「他の資金調達手段についてもご案内できます」",
    ],
    risksToAcknowledge: [
      "複数の財務指標が審査基準を大きく下回っている",
      "業種リスクが高く回収困難なケースに該当",
      "物件の残価が極めて低くリスクが大きい",
    ],
    nextActions: [
      "割賦販売・分割払い等の代替提案",
      "補助金・助成金活用の案内",
      "半年〜1年後の財務改善後の再申請案内",
      "物件・リース期間・金額の大幅見直し",
    ],
  },
];

function GuideCard({ item }: { item: GuideItem }) {
  const [openSection, setOpenSection] = useState<string | null>("talkingPoints");

  const sections = [
    { key: "talkingPoints", label: "💬 顧客への説明文例", items: item.talkingPoints },
    ...(item.risksToAcknowledge.length > 0
      ? [{ key: "risks", label: "⚠️ 確認すべきリスク", items: item.risksToAcknowledge }]
      : []),
    { key: "nextActions", label: "📋 次のアクション", items: item.nextActions },
  ];

  return (
    <div
      className="rounded-xl border shadow-sm overflow-hidden"
      style={{ borderColor: item.color, borderLeftWidth: 4 }}
    >
      <div className="flex items-center gap-3 px-4 py-3" style={{ background: item.bgColor }}>
        {item.icon}
        <div>
          <span className="font-bold text-slate-800">{item.label}</span>
          <span className="ml-2 text-xs text-slate-500">スコア {item.scoreRange}</span>
        </div>
      </div>
      <div className="bg-white px-4 py-3">
        <p className="text-sm text-slate-600 mb-3">{item.summary}</p>
        <div className="space-y-1">
          {sections.map((sec) => (
            <div key={sec.key} className="border border-slate-100 rounded-lg overflow-hidden">
              <button
                className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-slate-50 text-sm font-medium text-slate-700"
                onClick={() => setOpenSection(openSection === sec.key ? null : sec.key)}
              >
                {openSection === sec.key ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {sec.label}
              </button>
              {openSection === sec.key && (
                <ul className="px-4 pb-3 pt-1 space-y-1.5 bg-slate-50">
                  {sec.items.map((it, i) => (
                    <li key={i} className="text-xs text-slate-600 flex items-start gap-2">
                      <span className="mt-0.5 text-slate-400">•</span>
                      <span>{it}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function SalesGuidePage() {
  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div className="flex items-start gap-3">
        <Megaphone className="text-blue-600 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">審査結果 営業向け説明ガイド</h1>
          <p className="text-sm text-slate-500">審査結果を顧客・社内に説明する際のトーキングポイントと次のアクションをまとめています。</p>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-800">
        <p className="font-semibold mb-1">📌 使い方</p>
        <p className="text-xs text-blue-700">審査結果画面のスコアを確認し、該当する判定区分のガイドを参照してください。顧客との商談前に「説明文例」を確認しておくと円滑です。</p>
      </div>

      <div className="space-y-4">
        {GUIDE.map((item) => <GuideCard key={item.verdict} item={item} />)}
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500">
        <p className="font-semibold text-slate-600 mb-1">⚠️ 注意事項</p>
        <p>このガイドは社内参考用です。審査結果の最終判断は審査部門の承認が必要です。顧客への回答前に上長確認を行ってください。</p>
      </div>
    </div>
  );
}
