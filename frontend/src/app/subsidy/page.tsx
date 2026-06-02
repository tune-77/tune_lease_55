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
  officialUrl: string;
  points: string[];
  caution?: string;
};

const subsidies: SubsidyItem[] = [
  {
    name: 'ものづくり・商業・サービス生産性向上促進補助金',
    shortName: 'ものづくり補助金',
    icon: <Factory className="w-5 h-5" />,
    color: 'text-blue-600',
    maxAmount: '公募回・類型・従業員規模で変動',
    rate: '類型・事業者規模で変動',
    target: '革新的な新製品・新サービス開発、生産プロセス改善、海外需要開拓などに取り組む中小企業等',
    deadline: '公募回ごとに確認',
    leaseRelation: '機械装置・システム構築費が中心。リースで組む場合は、補助対象経費になる契約形態・所有者・支払方法を公募要領と事務局で確認する。',
    leaseAdvantage: '設備投資案件で採択可能性がある場合、自己資金負担・リース料負担の説明材料になる',
    officialUrl: 'https://portal.monodukuri-hojo.jp/',
    points: [
      '補助対象経費・補助率・上限額は公募回の公募要領で確認',
      '交付決定前の発注・契約・支払は原則として補助対象外になりやすい',
      '機械装置、専用システム、試作・生産性向上設備との相性が良い',
      '賃上げ・付加価値向上などの基本要件を確認',
    ],
    caution: 'リース可否は制度名だけで判断しない。必ず該当回の公募要領・事務局確認を前提にする。',
  },
  {
    name: 'デジタル化・AI導入補助金2026',
    shortName: 'デジタル化・AI導入補助金',
    icon: <Cpu className="w-5 h-5" />,
    color: 'text-indigo-600',
    maxAmount: '枠・ITツール・ハードウェア区分で変動',
    rate: '公募要領で確認',
    target: '業務効率化、デジタル化、AI導入に使う登録ITツール等を導入する中小企業等',
    deadline: '公募回ごとに確認',
    leaseRelation: '補助対象は登録ITツール・対象ハードウェア等の要件に従う。リース・サブスク・クラウド利用料の扱いは枠ごとに確認が必要。',
    leaseAdvantage: 'ソフト・POS・会計・受発注・AI活用など、設備投資よりIT投資寄りの案件で確認価値が高い',
    officialUrl: 'https://it-shien.smrj.go.jp/',
    points: [
      'IT導入支援事業者と登録ITツールの確認が先',
      'ハードウェアは対象区分・同時導入条件・上限額を確認',
      '汎用PC単体などは対象外または条件付きになりやすい',
      '見積前に公式サイトの公募要領・対象ツール検索を確認',
    ],
    caution: '旧「IT導入補助金」の感覚で説明しない。2026年の公募名称・枠・対象経費で確認する。',
  },
  {
    name: '中小企業省力化投資補助金',
    shortName: '省力化投資補助金',
    icon: <Zap className="w-5 h-5" />,
    color: 'text-cyan-600',
    maxAmount: '一般型・カタログ注文型で変動',
    rate: '一般型・カタログ注文型で変動',
    target: '人手不足解消に効果のあるロボット、IoT、設備・システム等を導入する中小企業等',
    deadline: '公募回ごとに確認',
    leaseRelation: '対象製品・設備、導入方法、支払方法は一般型/カタログ注文型で確認する。リース案件では所有者・補助対象経費・支払時期の確認が必要。',
    leaseAdvantage: '自動化・省人化設備のリース提案で、返済原資改善や投資回収説明に使いやすい',
    officialUrl: 'https://shoryokuka.smrj.go.jp/',
    points: [
      'ロボット、IoT、清掃・配膳・検査・搬送など省力化設備と相性が良い',
      'カタログ注文型は対象製品登録の有無を確認',
      '一般型は個別設備・システム導入の事業計画が重要',
      '賃上げ・生産性向上要件を確認',
    ],
    caution: '対象製品に見えても、登録状況・型番・導入形態で対象外になることがある。',
  },
  {
    name: '中小企業新事業進出補助金',
    shortName: '新事業進出補助金',
    icon: <TrendingUp className="w-5 h-5" />,
    color: 'text-emerald-600',
    maxAmount: '最大9,000万円（概要値。詳細は公募要領）',
    rate: '1/2〜2/3（概要値。詳細は公募要領）',
    target: '既存事業と異なる新市場・高付加価値事業への進出に取り組む中小企業等',
    deadline: '公募回ごとに確認',
    leaseRelation: '新事業に必要な機械装置・システム構築費等が論点。リースの場合は補助対象経費にできるか事前確認が必要。',
    leaseAdvantage: '新規ライン、新店舗、新サービス設備など、大型投資の妥当性を説明する材料になる',
    officialUrl: 'https://shinjigyou-shinshutsu.smrj.go.jp/',
    points: [
      '旧・事業再構築補助金の感覚でなく、新事業進出補助金として確認',
      '新市場・高付加価値事業への進出要件を確認',
      '建物費、機械装置・システム構築費など幅広いが、公募要領で対象経費を確認',
      '賃上げ・付加価値向上などの要件を確認',
    ],
    caution: '旧ページURLではなく、最新の公式サイト・公募要領を確認する。',
  },
  {
    name: '省エネルギー投資促進支援事業費補助金（省エネ補助金）',
    shortName: '省エネ補助金',
    icon: <Leaf className="w-5 h-5" />,
    color: 'text-green-600',
    maxAmount: '事業類型・設備区分で変動',
    rate: '事業類型・設備区分で変動',
    target: '工場・事業場等で省エネ効果が見込まれる高効率設備へ更新する事業者',
    deadline: '公募回ごとに確認',
    leaseRelation: '高効率空調、照明、ボイラー、コンプレッサー等が論点。リース導入可否・共同申請要否は公募要領で確認する。',
    leaseAdvantage: '省エネ設備リース＋補助金で初期コストを最小化。太陽光・蓄電池との組み合わせも有効。',
    officialUrl: 'https://www.enecho.meti.go.jp/category/saving_and_new/saving/enterprise/support/index.html',
    points: [
      '省エネ効果の算定、既存設備との比較、証憑が重要',
      '対象設備・指定設備の該当可否を確認',
      '交付決定前の契約・発注は避ける',
      'パートナー金融機関確認書など、回によって必要書類がある',
    ],
  },
  {
    name: '小規模事業者持続化補助金',
    shortName: '持続化補助金',
    icon: <Building2 className="w-5 h-5" />,
    color: 'text-orange-600',
    maxAmount: '通常枠・創業型等で変動',
    rate: '枠・事業者要件で変動',
    target: '販路開拓・業務効率化に取り組む小規模事業者（従業員5人以下等）',
    deadline: '公募回ごとに確認',
    leaseRelation: '販路開拓に必要な機械装置、広報、展示会、店舗改装等が中心。リース料そのものを補助対象にできるかは公募要領で確認。',
    leaseAdvantage: '少額設備・店舗改装・販路開拓に関連するリース/購入比較の相談材料になる',
    officialUrl: 'https://www.chusho.meti.go.jp/keiei/shokibo/jizoku/',
    points: [
      '商工会議所・商工会の支援を受けて計画書を作成',
      '販路開拓に資する取り組みかが重要',
      '汎用性の高い備品・PC等は対象外になりやすい',
      '補助上限が比較的小さいため、リースより購入・少額投資との比較が必要',
    ],
    caution: '設備導入だけでは弱い。販路開拓・売上拡大とのつながりを確認する。',
  },
  {
    name: '脱炭素社会の構築に向けたESGリース促進事業',
    shortName: 'ESGリース補助金',
    icon: <Leaf className="w-5 h-5" />,
    color: 'text-teal-600',
    maxAmount: '対象設備・リース契約で変動',
    rate: '補助率・補助額は年度要領で確認',
    target: '脱炭素機器をリースで導入する中小企業等',
    deadline: '年度・予算枠で確認',
    leaseRelation: 'リース前提の制度。対象機器、指定リース事業者、補助金のリース料低減反映方法を確認する。',
    leaseAdvantage: '脱炭素・省エネ設備をリースで提案する際に、最初に確認すべき制度',
    officialUrl: 'https://esg-lease.or.jp/',
    points: [
      '対象機器かどうかを公式HP・指定リース事業者で確認',
      '補助金がリース料低減に反映されるか確認',
      '省エネ効果・CO2削減効果の説明資料を準備',
      '他補助金との併用可否を確認',
    ],
    caution: '「ESG」一般論ではなく、この制度の対象機器・指定リース事業者に該当するかを見る。',
  },
  {
    name: '環境対応車・トラック関連補助金',
    shortName: '環境対応車補助金',
    icon: <Zap className="w-5 h-5" />,
    color: 'text-sky-600',
    maxAmount: '車種・年度・制度で変動',
    rate: '制度ごとに確認',
    target: '環境性能の高いトラック、バス、商用車等を導入する事業者',
    deadline: '年度・予算枠で確認',
    leaseRelation: '車両リースで使える制度もあるが、制度ごとに申請者・所有者・使用者・車両登録条件が異なる。',
    leaseAdvantage: '運送業・建設業などの車両更新案件で、導入コストと環境対応を同時に説明できる',
    officialUrl: 'https://www.levo.or.jp/subsidy/diesel/',
    points: [
      '車両区分、燃料種別、初度登録、使用の本拠地を確認',
      'リース会社が申請者になる制度か、使用者が申請者になる制度か確認',
      '国・自治体・業界団体の制度が分かれるため、案件所在地で再確認',
      '予算消化で早期終了する制度が多い',
    ],
    caution: '車両補助金は年度更新・予算終了が早い。見積時点で必ず公式HPを確認する。',
  },
  {
    name: 'ZEB・再生可能エネルギー関連補助金',
    shortName: 'ZEB・再エネ補助金',
    icon: <Zap className="w-5 h-5" />,
    color: 'text-amber-600',
    maxAmount: '事業・建物区分・設備区分で変動',
    rate: '制度ごとに確認',
    target: 'ZEB化、省CO2建築、再エネ・蓄電池等の導入に取り組む事業者',
    deadline: '公募回ごとに確認',
    leaseRelation: '太陽光・蓄電池・空調等をリースで導入する場合、所有者・PPA・リース・自己所有の違いが補助対象可否に直結する。',
    leaseAdvantage: '建物設備更新、太陽光・蓄電池、空調更新などの大型案件で、投資回収とCO2削減の説明材料になる',
    officialUrl: 'https://www.env.go.jp/earth/ondanka/zeb.html',
    points: [
      'ZEB、Nearly ZEB、ZEB Ready 等の区分を確認',
      'PPA、リース、自己所有で対象経費・申請者が変わる',
      '補助対象設備と補助対象外工事を切り分ける',
      '売電・自家消費・蓄電池併設の条件を確認',
    ],
    caution: '再エネ補助金は制度が細かく分かれるため、このカードは入口。案件ごとに環境省・SII・自治体制度を確認する。',
  },
];

function SubsidyCard({ sub }: { sub: SubsidyItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-slate-200 rounded-2xl overflow-hidden bg-white shadow-sm">
      <div className="flex items-start gap-2 px-5 py-4 hover:bg-slate-50 transition-colors">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="flex min-w-0 flex-1 items-start gap-3 text-left"
        >
          <span className={`mt-0.5 flex-shrink-0 ${sub.color}`}>{sub.icon}</span>
          <div className="min-w-0 flex-1">
            <div className="font-black text-slate-800 text-sm leading-snug">{sub.shortName}</div>
            <div className="text-xs text-slate-500 mt-0.5 truncate">{sub.name}</div>
            <div className="flex flex-wrap gap-2 mt-1.5">
              <span className="text-[10px] font-bold text-slate-500 bg-slate-100 rounded px-2 py-0.5">上限 {sub.maxAmount.split(' / ')[0]}</span>
              <span className="text-[10px] font-bold text-emerald-700 bg-emerald-50 rounded px-2 py-0.5">補助率 {sub.rate}</span>
            </div>
          </div>
        </button>
        <a
          href={sub.officialUrl}
          target="_blank"
          rel="noreferrer"
          className="mt-0.5 inline-flex h-8 flex-shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 text-[11px] font-black text-slate-700 transition hover:border-emerald-300 hover:bg-emerald-50 hover:text-emerald-700"
        >
          公式HP
          <ExternalLink className="h-3 w-3" />
        </a>
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
          aria-label={open ? `${sub.shortName}を閉じる` : `${sub.shortName}を開く`}
        >
          {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

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
