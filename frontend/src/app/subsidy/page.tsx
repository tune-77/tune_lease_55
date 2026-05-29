"use client";

import React, { useState } from 'react';
import { Gift, ChevronDown, ChevronUp, ExternalLink, Zap, Cpu, Factory, Leaf, Building2, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';

type SubsidyItem = {
  name: string;
  shortName: string;
  icon: React.ReactNode;
  color: string;
  maxAmount: string;
  rate: string;
  target: string;
  deadline: string;
  leaseRelation: string;
  leaseAdvantage: string;
  points: string[];
  caution?: string;
};

const subsidies: SubsidyItem[] = [
  {
    name: 'ものづくり・商工業・サービス補助金（ものづくり補助金）',
    shortName: 'ものづくり補助金',
    icon: <Factory className="w-5 h-5" />,
    color: 'text-blue-600',
    maxAmount: '通常枠: 1,250万円 / グローバル展開型: 3,000万円',
    rate: '1/2〜2/3（中小企業の場合）',
    target: '革新的な製品・サービス開発や生産プロセス改善に取り組む中小企業・小規模事業者',
    deadline: '随時公募（年数回）',
    leaseRelation: 'リース取得の設備も補助対象。補助金受給後に設備をリース会社へ譲渡（セール＆リースバック）する形態も認められる場合あり。',
    leaseAdvantage: '補助金で設備取得コストを削減 → リース料率の計算ベースが下がるため実質リース料を圧縮できる',
    points: [
      '補助対象経費: 機械装置・システム構築費、技術導入費、専門家経費等',
      'リース取得の場合: リース会社が設備を取得し補助金申請は事業者が実施',
      '加点要件: DX・グリーン化に対応する設備は加点',
      '交付決定前の発注・契約は補助対象外',
    ],
    caution: 'リース取得の場合、リース会社との事前調整と申請機関への確認が必須',
  },
  {
    name: 'IT導入補助金',
    shortName: 'IT導入補助金',
    icon: <Cpu className="w-5 h-5" />,
    color: 'text-indigo-600',
    maxAmount: '通常枠A: 150万円未満 / B: 150〜450万円 / デジタル化基盤枠: 350万円',
    rate: '1/2〜3/4',
    target: '業務効率化・DX推進のためのITツール（ソフトウェア・クラウド・ハード）を導入する中小企業',
    deadline: '随時（年複数回）',
    leaseRelation: 'PCやサーバー等のハードウェアはデジタル化基盤導入枠（旧IT導入補助金2022）で補助対象。ソフトウェアリセール・SaaS契約は対象外の場合が多い。',
    leaseAdvantage: 'PCリース案件で補助金を活用することで実質負担額を削減。4年リース＋補助金の組み合わせが人気。',
    points: [
      'デジタル化基盤枠: PC・タブレット・プリンター・スキャナー等が対象',
      'IT導入支援事業者（認定ベンダー）経由での申請が必須',
      'ハードウェアのみの申請も可能（ソフトウェアと同時購入不要）',
      'リース取得でも補助対象になることが多い（要確認）',
    ],
  },
  {
    name: '事業再構築補助金',
    shortName: '事業再構築補助金',
    icon: <TrendingUp className="w-5 h-5" />,
    color: 'text-emerald-600',
    maxAmount: '最大1億円（成長枠・グリーン成長枠）',
    rate: '1/2〜2/3',
    target: 'ポストコロナ・事業転換・新分野展開・業態転換に取り組む中小企業',
    deadline: '随時（複数回公募）',
    leaseRelation: '新事業に必要な設備のリース取得が補助対象。既存事業廃止後の新設備には特に有効。',
    leaseAdvantage: '大型設備投資（製造ライン・飲食設備等）を補助金＋リースの組み合わせで初期投資を最小化',
    points: [
      '補助対象経費: 建物費、機械装置、システム構築費、外注費等',
      '事業計画書の作成・認定経営革新等支援機関の確認が必要',
      '付加価値額年率3%以上・給与支給総額年率2%以上の達成が条件',
      'リース取得設備は「リース料総額」ではなく「設備取得額」が補助対象',
    ],
    caution: '補助事業実施期間中は設備の転売・目的外使用が禁止',
  },
  {
    name: '省エネルギー投資促進支援事業費補助金（省エネ補助金）',
    shortName: '省エネ補助金',
    icon: <Leaf className="w-5 h-5" />,
    color: 'text-green-600',
    maxAmount: '最大15億円（大規模省エネ）/ 中小: 1億円程度',
    rate: '1/3〜1/2',
    target: '省エネ効果が見込まれる設備（高効率機器）に更新する事業者',
    deadline: '年1〜2回公募（METI）',
    leaseRelation: '省エネ設備（高効率空調・照明・ボイラー・コンプレッサー等）のリースが対象。リース取得でも申請可能。',
    leaseAdvantage: '省エネ設備リース＋補助金で初期コストを最小化。太陽光・蓄電池との組み合わせも有効。',
    points: [
      '省エネ効果: 原油換算で年1kl以上の削減見込みが目安',
      '対象設備: 産業用ヒートポンプ・高効率照明・変圧器・コンプレッサー等',
      'リース取得の場合はリース会社が申請者となるケースもある',
      'SII（省エネルギーセンター）への事前登録が必要',
    ],
  },
  {
    name: '小規模事業者持続化補助金',
    shortName: '持続化補助金',
    icon: <Building2 className="w-5 h-5" />,
    color: 'text-orange-600',
    maxAmount: '通常枠: 50万円 / 特別枠（創業・後継者等）: 200万円',
    rate: '2/3',
    target: '販路開拓・業務効率化に取り組む小規模事業者（従業員5人以下等）',
    deadline: '年複数回（商工会議所経由）',
    leaseRelation: '機械装置費・広告宣伝費等が対象。少額リースの場合は補助額が小さいため、単体設備購入の方が合理的なケースもある。',
    leaseAdvantage: '少額設備（PC・POS・調理機器等）の購入＋リース検討に有効',
    points: [
      '商工会議所・商工会の支援を受けて計画書を作成',
      '設備費・広告費・委託費・展示会出展費等が対象',
      '補助上限が低いためリースより購入との組み合わせが多い',
    ],
    caution: '補助金受給後5年以内の設備の目的外使用・転売は返還対象',
  },
  {
    name: '太陽光・再生可能エネルギー関連補助金（ZEB・ZEH等）',
    shortName: '再エネ補助金',
    icon: <Zap className="w-5 h-5" />,
    color: 'text-amber-600',
    maxAmount: '事業規模による（数百万〜数億円）',
    rate: '1/3〜1/2',
    target: '再生可能エネルギー設備（太陽光・蓄電池・燃料電池等）を導入する事業者',
    deadline: '年1〜2回（環境省・METI等）',
    leaseRelation: '太陽光発電設備リース（ソーラーリース）は補助対象になるケースあり。PPAとの違いに注意。',
    leaseAdvantage: '太陽光リースと補助金の組み合わせで初期投資ゼロ近くまで圧縮可能。17年の法定耐用年数を活かした長期リースが多い。',
    points: [
      'ZEB（Nearly ZEB等）認証を受けた建物の設備は補助率が高い',
      'PPAモデルは通常補助金対象外（事業者がシステムを所有しないため）',
      'リースで取得した場合は所有者（リース会社）側の申請が必要な場合あり',
      '売電収入がある場合は補助金の一部返還が求められることあり',
    ],
  },
];

function SubsidyCard({ sub }: { sub: SubsidyItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-slate-200 rounded-2xl overflow-hidden bg-white shadow-sm">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-start gap-3 px-5 py-4 text-left hover:bg-slate-50 transition-colors"
      >
        <span className={`mt-0.5 flex-shrink-0 ${sub.color}`}>{sub.icon}</span>
        <div className="flex-1 min-w-0">
          <div className="font-black text-slate-800 text-sm leading-snug">{sub.shortName}</div>
          <div className="text-xs text-slate-500 mt-0.5 truncate">{sub.name}</div>
          <div className="flex flex-wrap gap-2 mt-1.5">
            <span className="text-[10px] font-bold text-slate-500 bg-slate-100 rounded px-2 py-0.5">最大 {sub.maxAmount.split(' / ')[0]}</span>
            <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 rounded px-2 py-0.5">補助率 {sub.rate}</span>
          </div>
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-slate-400 flex-shrink-0 mt-1" /> : <ChevronDown className="w-4 h-4 text-slate-400 flex-shrink-0 mt-1" />}
      </button>

      {open && (
        <div className="px-5 pb-5 border-t border-slate-100 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
            {[
              { label: '補助上限額', value: sub.maxAmount },
              { label: '補助率', value: sub.rate },
              { label: '対象', value: sub.target },
              { label: '公募時期', value: sub.deadline },
            ].map(r => (
              <div key={r.label} className="bg-slate-50 rounded-xl p-3">
                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">{r.label}</p>
                <p className="text-xs font-bold text-slate-700 leading-snug">{r.value}</p>
              </div>
            ))}
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-xl p-3">
            <p className="text-[10px] font-black text-blue-600 uppercase tracking-widest mb-1.5">リースとの関係</p>
            <p className="text-xs text-slate-700 leading-relaxed">{sub.leaseRelation}</p>
            <div className="mt-2 flex items-start gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-blue-500 flex-shrink-0 mt-0.5" />
              <p className="text-xs font-bold text-blue-700">{sub.leaseAdvantage}</p>
            </div>
          </div>

          <div>
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">審査上の確認ポイント</p>
            <ul className="space-y-1">
              {sub.points.map((p, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-slate-600">
                  <span className="mt-1 w-1.5 h-1.5 rounded-full bg-slate-300 flex-shrink-0" />
                  {p}
                </li>
              ))}
            </ul>
          </div>

          {sub.caution && (
            <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-xl">
              <AlertCircle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-amber-700 font-bold">{sub.caution}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function SubsidyPage() {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <Gift className="text-emerald-500" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">補助金情報</h1>
          <p className="text-sm text-slate-500">リース審査に関連する主要補助金の概要・リースとの関係・審査上の注意点をまとめています。</p>
        </div>
      </div>

      <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-2xl text-sm text-emerald-800">
        <p className="font-black mb-1">補助金 × リースの基本的な考え方</p>
        <p className="text-xs leading-relaxed">補助金の活用により設備取得コストが削減されると、リースの計算ベース（取得価額）が下がり、月次リース料の圧縮につながります。また補助金受給後にリースバックを組み合わせることで資金繰り改善と設備近代化を同時に実現できるケースがあります。ただし補助金の種類によって「リース取得可否」「申請者」「転売制限」が異なるため、個別案件ごとに担当者・申請機関に確認してください。</p>
      </div>

      <div className="space-y-3">
        {subsidies.map(sub => (
          <SubsidyCard key={sub.shortName} sub={sub} />
        ))}
      </div>

      <div className="text-center text-xs text-slate-400 pt-4 border-t border-slate-100">
        ※ 補助金の内容・要件は公募ごとに変更される場合があります。最新情報は各省庁・申請機関の公式サイトをご確認ください。
        <br />
        ※ 本情報は一般的な審査実務への参考情報であり、個別申請の可否を保証するものではありません。
      </div>
    </div>
  );
}
