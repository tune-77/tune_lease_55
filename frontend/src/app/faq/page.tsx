"use client";

import React, { useState } from "react";
import { HelpCircle, ChevronDown, ChevronRight } from "lucide-react";

type FaqItem = {
  q: string;
  a: string;
  tags?: string[];
};

type FaqSection = {
  title: string;
  icon: string;
  items: FaqItem[];
};

const FAQ_DATA: FaqSection[] = [
  {
    title: "リース基礎知識",
    icon: "📖",
    items: [
      {
        q: "リースとレンタルの違いは何ですか？",
        a: "リースは特定の物件を特定の期間使用する契約で、通常2〜10年程度の中長期契約です。物件はリース会社が購入し、ユーザーは月額リース料を支払います。レンタルは短期（日〜月単位）で既製品を貸し出す形態です。リースは中途解約が原則不可で残価リスクはリース会社が負担しますが、レンタルは柔軟に返却できます。",
        tags: ["基礎", "比較"],
      },
      {
        q: "リース期間はどのくらいが適切ですか？",
        a: "法定耐用年数の60〜80%が標準的な目安です。例えば法定耐用年数4年のPCなら2〜4年、5年のトラックなら3〜5年が一般的です。期間が耐用年数の70%を超えると満了時の残余価値が低下し、審査スコアに影響する場合があります。",
        tags: ["期間", "耐用年数"],
      },
      {
        q: "ファイナンスリースとオペレーティングリースの違いは？",
        a: "ファイナンスリースはリース料総額が物件価格のほぼ全額をカバーし、中途解約不可です。会計上はリース資産として計上されます。オペレーティングリースは物件の一部期間のみを使用し、残価をリース会社が負担するため月額が低くなりますが、リース会社の残価リスク管理が重要です。",
        tags: ["基礎", "会計"],
      },
    ],
  },
  {
    title: "リース対象物件",
    icon: "📦",
    items: [
      {
        q: "リースできない物件はありますか？",
        a: "以下はリース対象外となる場合があります：①土地・建物等の不動産、②消耗品・在庫商品、③著作権・ソフトウェアライセンスのみ（ハード一体型は可）、④廃棄・処分が困難な物件、⑤中古品（取り扱いのある会社もあり）。不動産は割賦購入やローンが適します。",
        tags: ["対象外", "物件"],
      },
      {
        q: "ソフトウェアはリースできますか？",
        a: "ハードウェアとセット導入する場合は対象になるケースがあります。クラウドサービス・SaaSは原則リース対象外です。パッケージソフトのライセンスのみは対象外ですが、導入コンサル・カスタマイズ費用を含む一体型提案であれば審査可能なケースがあります。",
        tags: ["IT", "ソフトウェア"],
      },
      {
        q: "中古設備のリースは可能ですか？",
        a: "取扱会社によって異なりますが、査定が難しいため慎重な審査が必要です。残存価値が明確な場合（メーカー再生品・認定中古品など）は対象になることがあります。取得価格の証明書類が必要になります。",
        tags: ["中古", "設備"],
      },
    ],
  },
  {
    title: "審査・スコアリング",
    icon: "🔍",
    items: [
      {
        q: "審査スコアはどのように計算されますか？",
        a: "財務指標（自己資本比率・流動比率・売上高成長率等）、物件スコア（耐用年数・残価率・リース期間比率）、業種特性（業界平均との比較）、Q_risk（財務整合性チェック）などを総合的に評価します。スコアは0〜100ptで、70pt以上が承認の目安です。",
        tags: ["スコア", "審査"],
      },
      {
        q: "Q_risk（量子整合性リスク）とは何ですか？",
        a: "財務諸表上の複数指標間の整合性を検査するスコアです。例えば売上高に対する減価償却費・賃借料の比率が業種平均から大きく乖離している場合に高くなります。35以上が要注意ラインで、財務データの信頼性確認や追加書類取得を推奨します。",
        tags: ["Q_risk", "スコア"],
      },
      {
        q: "PD（推定倒産確率）の見方は？",
        a: "自己資本比率・流動比率・営業利益率から算出した簡易デフォルト確率です。目安：10%未満=低リスク、10〜30%=要注意、30%超=高リスク。あくまで財務指標ベースの参考値であり、実際の倒産確率とは異なります。",
        tags: ["PD", "スコア"],
      },
      {
        q: "条件付き承認（要審議）の場合、どのような対策がありますか？",
        a: "主な対策：①リース期間を短縮して月額負担を下げる、②前受金・頭金を入れることで残額リスクを低減、③代表者保証を追加、④最新決算書・事業計画書等の追加書類を取得、⑒物件を変更して残価を高める。スコアが65〜70pt付近の境界ケースでは二次審査での現地確認が有効です。",
        tags: ["条件付き", "対策"],
      },
    ],
  },
  {
    title: "法定耐用年数・金利",
    icon: "📅",
    items: [
      {
        q: "主要設備の法定耐用年数の目安は？",
        a: "【情報機器】サーバー4〜5年、PC4年、複合機5年。【車両】普通乗用車6年、トラック4〜5年。【機械設備】一般機械10年、建設機械6〜8年。【工具】一般工具4〜5年。詳細は「法定耐用年数一覧」ページを参照してください。",
        tags: ["耐用年数", "設備"],
      },
      {
        q: "リース金利（リース料率）はどのように決まりますか？",
        a: "基準金利（長期プライムレート等）に信用リスクスプレッドを加算して決まります。物件の残価率・リース期間・借手の信用力・業種リスクが主な要素です。当システムでは動的金利提案機能（開発中）で最適金利帯を提示します。",
        tags: ["金利", "料率"],
      },
    ],
  },
  {
    title: "補助金・助成金",
    icon: "💰",
    items: [
      {
        q: "補助金を活用したリースは可能ですか？",
        a: "可能なケースがあります。ものづくり補助金・IT導入補助金はリース併用実績があります。ただし補助金公募要領で「リース可」と明記されているか必ず事前確認が必要です。補助対象設備の所有権はリース会社にあるため、補助金事務局への事前相談を推奨します。",
        tags: ["補助金", "助成金"],
      },
      {
        q: "補助金採択案件は審査スコアに影響しますか？",
        a: "補助金採択は物件の実需確認・行政による事業計画検証済みとみなせるため、審査上プラス評価になります。目安として+2〜5pt程度の効果があります。補助金申請書類・採択通知書を追加書類として提出してください。",
        tags: ["補助金", "スコア"],
      },
    ],
  },
];

function FaqAccordion({ item }: { item: FaqItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        className="w-full flex items-start gap-3 px-4 py-3 text-left bg-white hover:bg-slate-50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        {open ? (
          <ChevronDown className="text-slate-400 mt-0.5 shrink-0" size={16} />
        ) : (
          <ChevronRight className="text-slate-400 mt-0.5 shrink-0" size={16} />
        )}
        <span className="text-sm font-medium text-slate-700">{item.q}</span>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-1 bg-slate-50 border-t border-slate-100">
          <p className="text-sm text-slate-600 leading-relaxed">{item.a}</p>
          {item.tags && (
            <div className="flex gap-1.5 mt-2 flex-wrap">
              {item.tags.map((t) => (
                <span key={t} className="text-xs px-2 py-0.5 bg-blue-50 text-blue-600 rounded">{t}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function FaqPage() {
  const [activeSection, setActiveSection] = useState<string | null>(null);

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      {/* ヘッダー */}
      <div className="flex items-start gap-3">
        <HelpCircle className="text-slate-500 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">リース知識 FAQ</h1>
          <p className="text-sm text-slate-500">営業・審査担当者向けのよくある質問と回答。</p>
        </div>
      </div>

      {/* セクションフィルタ */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setActiveSection(null)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            activeSection === null ? "bg-slate-700 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
          }`}
        >
          すべて
        </button>
        {FAQ_DATA.map((sec) => (
          <button
            key={sec.title}
            onClick={() => setActiveSection(activeSection === sec.title ? null : sec.title)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
              activeSection === sec.title ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-blue-50"
            }`}
          >
            {sec.icon} {sec.title}
          </button>
        ))}
      </div>

      {/* FAQ本体 */}
      {FAQ_DATA.filter((sec) => !activeSection || sec.title === activeSection).map((sec) => (
        <div key={sec.title} className="space-y-2">
          <h2 className="text-sm font-bold text-slate-700 flex items-center gap-2">
            <span>{sec.icon}</span> {sec.title}
          </h2>
          {sec.items.map((item) => (
            <FaqAccordion key={item.q} item={item} />
          ))}
        </div>
      ))}

      <p className="text-xs text-slate-400 text-center pt-2">
        内容は随時更新されます。最新情報は各担当部署または公式資料を確認してください。
      </p>
    </div>
  );
}
