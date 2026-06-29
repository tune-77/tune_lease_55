"use client";

import React, { useState } from 'react';
import {
  HelpCircle, ChevronDown, ChevronUp, BookOpen, Cpu, Zap,
  Building2, TrendingDown, Clock, RefreshCw, Wrench,
  Ban, RotateCcw, Factory, Percent, Landmark, Package, Trash2, HardHat
} from 'lucide-react';

type FaqItem = {
  q: string;
  a: React.ReactNode;
};

type FaqSection = {
  id: string;
  title: string;
  icon: React.ReactNode;
  color: string;
  items: FaqItem[];
};

function FaqAccordion({ items }: { items: FaqItem[] }) {
  const [open, setOpen] = useState<number | null>(null);
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="border border-slate-200 rounded-xl overflow-hidden">
          <button
            onClick={() => setOpen(open === i ? null : i)}
            className="w-full flex items-start justify-between gap-3 px-5 py-4 text-left bg-white hover:bg-slate-50 transition-colors"
          >
            <span className="text-sm font-black text-slate-700 leading-snug">{item.q}</span>
            {open === i
              ? <ChevronUp className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
              : <ChevronDown className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />}
          </button>
          {open === i && (
            <div className="px-5 pb-5 bg-slate-50 border-t border-slate-100 text-sm text-slate-600 leading-relaxed space-y-2">
              {item.a}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

const sections: FaqSection[] = [
  // REV-067: リースとレンタルの違い
  {
    id: 'lease-vs-rental',
    title: 'リースとレンタルの違い',
    icon: <RefreshCw className="w-5 h-5" />,
    color: 'text-blue-600',
    items: [
      {
        q: 'リースとレンタルは何が違いますか？',
        a: (
          <div className="space-y-3">
            <p>大きく分けて <strong>契約期間・物件の選択・費用の性質</strong> が異なります。</p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-100">
                    <th className="border border-slate-200 px-3 py-2 text-left font-black">項目</th>
                    <th className="border border-slate-200 px-3 py-2 text-left font-black text-blue-700">リース</th>
                    <th className="border border-slate-200 px-3 py-2 text-left font-black text-emerald-700">レンタル</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['契約期間', '中長期（2〜7年）固定', '短期・日単位〜月単位、更新可'],
                    ['物件の選択', 'ユーザーが選定・指定', 'レンタル会社の在庫から選択'],
                    ['所有権', 'リース会社が所有', 'レンタル会社が所有'],
                    ['修繕・保守', '原則ユーザー負担', 'レンタル会社が対応'],
                    ['中途解約', '原則不可（違約金あり）', '比較的容易'],
                    ['会計処理', 'ファイナンスリースは資産計上', 'オフバランス（費用処理）'],
                    ['用途', '設備・機器の長期利用', '建機・イベント等の短期利用'],
                  ].map(([item, lease, rental]) => (
                    <tr key={item} className="even:bg-white odd:bg-slate-50/50">
                      <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{item}</td>
                      <td className="border border-slate-200 px-3 py-2 text-blue-700">{lease}</td>
                      <td className="border border-slate-200 px-3 py-2 text-emerald-700">{rental}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ),
      },
      {
        q: 'ファイナンスリースとオペレーティングリースの違いは？',
        a: (
          <div className="space-y-2">
            <p><strong>ファイナンスリース</strong>：リース期間中に物件の経済的価値の大部分をユーザーが享受するもの。解約不能・フルペイアウトが要件。借手の貸借対照表に資産として計上（IFRS16号）。</p>
            <p><strong>オペレーティングリース</strong>：上記以外のリース。解約権・残価設定がある場合が多く、オフバランス処理が可能なケースあり。</p>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">※ 2019年以降IFRS16号の適用により、上場企業は原則すべてのリースをオンバランス化が必要。</p>
          </div>
        ),
      },
      {
        q: 'リース料はどのように決まりますか？',
        a: (
          <p>リース料 ＝ 物件価格 ÷ リース料率 × 期間。リース料率は<strong>調達コスト（基準金利）＋物件スプレッド＋借手リスクスプレッド</strong>で構成されます。詳しくは「金利決定ロジック」のFAQをご覧ください。</p>
        ),
      },
    ],
  },

  // REV-068: 金利決定ロジック
  {
    id: 'rate-logic',
    title: '金利・リース料率の決定ロジック',
    icon: <Zap className="w-5 h-5" />,
    color: 'text-amber-600',
    items: [
      {
        q: 'リース金利はどのように計算されますか？',
        a: (
          <div className="space-y-3">
            <p>当システムでは以下の4要素の合計で提案金利を算出します：</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                { label: '① 基準金利', desc: '日銀政策金利・TIBOR等を元にした市場調達コスト。期間が長いほど高くなる傾向。', color: 'bg-slate-100 text-slate-700' },
                { label: '② 物件スプレッド', desc: '物件の流動性・残存価値リスクに応じた上乗せ。医療機器・ITは高め、車両・建機は低め。', color: 'bg-blue-50 text-blue-700' },
                { label: '③ 格付スプレッド', desc: '借手企業の信用格付に応じた上乗せ。格付が低いほど（高スコアほど）スプレッドが大きい。', color: 'bg-indigo-50 text-indigo-700' },
                { label: '④ リスク補正', desc: 'AIスコアを元にした動的調整。スコアが高い場合はマイナス補正（金利低下）も発生。', color: 'bg-emerald-50 text-emerald-700' },
              ].map(i => (
                <div key={i.label} className={`p-3 rounded-lg ${i.color}`}>
                  <p className="font-black text-xs mb-1">{i.label}</p>
                  <p className="text-xs leading-relaxed">{i.desc}</p>
                </div>
              ))}
            </div>
            <p className="text-xs font-bold text-slate-600 bg-slate-100 px-3 py-2 rounded-lg">
              提案金利 ＝ 基準金利 ＋ 物件スプレッド ＋ 格付スプレッド ＋ リスク補正
            </p>
          </div>
        ),
      },
      {
        q: 'スコアが改善すると金利はどれくらい下がりますか？',
        a: (
          <div className="space-y-2">
            <p>概算では、<strong>スコア10pt改善ごとに金利が約0.15〜0.20%低下</strong>します（リスク補正の変化による）。</p>
            <p>動的金利提案エンジン（サイドバー → 動的金利提案エンジン）で実際のスコア感度グラフを確認できます。</p>
          </div>
        ),
      },
      {
        q: '基準金利はいつ更新されますか？',
        a: (
          <p>基準金利マスタ（設定 → 基準金利マスタ）で期間ごとに管理しています。市場金利変動時に管理者が手動で更新します。自動連動機能は現在未実装です。</p>
        ),
      },
    ],
  },

  {
    id: 'q-risk',
    title: 'Q_riskの考え方',
    icon: <Cpu className="w-5 h-5" />,
    color: 'text-violet-600',
    items: [
      {
        q: 'Q_riskとは何ですか？',
        a: (
          <div className="space-y-3">
            <p>
              <strong>既存スコアだけでは説明できない成約・失注の歪みを見つける探索シグナル</strong>です。
              旧来の財務矛盾スコアや減点係数として固定せず、スコアリング外で成約を動かす要因を探します。
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                { label: '高スコア失注', desc: 'スコアは強いのに、金利・競合・条件提示後離脱で取れない案件。' },
                { label: '低スコア成約', desc: 'スコアは弱いのに、銀行支援・前受金・補助金・物件換金性で取れる案件。' },
                { label: '同スコア帯の分岐', desc: '同じスコア帯でも、営業部・業種細分・物件・提案順序で結果が割れる案件。' },
                { label: '非スコア因子', desc: '価格、競合、銀行支援、補助金タイミング、営業導線、顧客の急ぎ度。' },
              ].map(i => (
                <div key={i.label} className="rounded-lg bg-white border border-violet-100 p-3">
                  <p className="text-xs font-black text-violet-700 mb-1">{i.label}</p>
                  <p className="text-xs text-slate-600 leading-relaxed">{i.desc}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 bg-white border border-slate-200 rounded-lg p-2">
              Q_riskはスコアを自動で下げるための指標ではありません。成約の正体を探すため、優先的に深掘りする対象を示します。
            </p>
          </div>
        ),
      },
    ],
  },

  // REV-069: 高額工事費リース審査基準
  {
    id: 'construction-faq',
    title: '高額工事費・建設機械リースの審査',
    icon: <Wrench className="w-5 h-5" />,
    color: 'text-stone-600',
    items: [
      {
        q: '工事費用をリースに含めることはできますか？',
        a: (
          <div className="space-y-2">
            <p>設備の<strong>据付工事費・搬入費</strong>は、リース物件の取得に不可欠なものであれば原則としてリース対象額に含めることができます。</p>
            <p>ただし、以下の工事費は対象外となる場合があります：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li>建物本体の改修・改造工事（不動産工事）</li>
              <li>電気・配管など既存設備への恒久的変更工事</li>
              <li>物件の撤去・廃棄費用（リース終了時費用）</li>
            </ul>
          </div>
        ),
      },
      {
        q: '工事費込みで1億円以上の大型案件の審査基準は？',
        a: (
          <div className="space-y-2">
            <p>大型案件（1億円超）では通常審査に加えて以下が追加要件となります：</p>
            <div className="grid grid-cols-1 gap-2">
              {[
                { label: 'スコア要件', desc: 'AIスコア 65pt以上（70pt未満は条件付き承認）' },
                { label: '自己資本比率', desc: '20%以上を目安（15%未満は担保要請）' },
                { label: '担保・保証', desc: '代表者連帯保証または物件担保が原則必須' },
                { label: '財務書類', desc: '3期分の決算書 + 直近の試算表' },
                { label: '物件価値確認', desc: '中古市場価格・残存価値の確認が必要' },
              ].map(r => (
                <div key={r.label} className="flex gap-3 text-xs">
                  <span className="font-black text-stone-700 w-28 flex-shrink-0">{r.label}</span>
                  <span className="text-slate-600">{r.desc}</span>
                </div>
              ))}
            </div>
          </div>
        ),
      },
      {
        q: '建設機械リース特有のリスクは何ですか？',
        a: (
          <div className="space-y-2">
            <p>建設機械リースでは以下のリスクに注意が必要です：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>稼働率依存リスク</strong>：工事受注減による機械の遊休化</li>
              <li><strong>損耗リスク</strong>：稼働環境による急激な価値低下</li>
              <li><strong>法規制リスク</strong>：排ガス規制強化による旧型機の価値下落</li>
              <li><strong>季節変動</strong>：冬季の工事停止による資金繰り悪化</li>
            </ul>
            <p className="text-[11px] text-slate-500">これらを踏まえ、建設機械の物件スプレッドは比較的低め（残存価値が高く流動性があるため）に設定されています。</p>
          </div>
        ),
      },
    ],
  },

  // REV-079: PD用語説明
  {
    id: 'pd-explanation',
    title: 'PD（デフォルト確率）の解説',
    icon: <TrendingDown className="w-5 h-5" />,
    color: 'text-rose-600',
    items: [
      {
        q: 'PD（デフォルト確率）とは何ですか？',
        a: (
          <div className="space-y-2">
            <p><strong>PD（Probability of Default）</strong>とは、借手企業が将来一定期間内に<strong>債務不履行（デフォルト）を起こす確率</strong>を統計モデルで推定した指標です。</p>
            <p>当システムでは過去の審査データを学習したMLモデル（既存先RandomForest・新規先ロジスティック回帰を軸に、分析ではLGBMも参照）が、財務指標・格付・業種等から各案件のPDを計算します。</p>
          </div>
        ),
      },
      {
        q: 'PDの数値をどう解釈すればよいですか？',
        a: (
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-xs">
              {[
                { range: '0〜1%', label: '非常に低リスク', color: 'bg-emerald-50 border-emerald-200 text-emerald-700' },
                { range: '1〜3%', label: '低〜標準リスク', color: 'bg-sky-50 border-sky-200 text-sky-700' },
                { range: '3〜6%', label: '注意が必要', color: 'bg-amber-50 border-amber-200 text-amber-700' },
                { range: '6〜10%', label: '高リスク', color: 'bg-orange-50 border-orange-200 text-orange-700' },
                { range: '10%超', label: '否決検討水準', color: 'bg-rose-50 border-rose-200 text-rose-700' },
              ].map(d => (
                <div key={d.range} className={`p-2 rounded-lg border ${d.color}`}>
                  <p className="font-black">{d.range}</p>
                  <p className="text-[10px] mt-0.5">{d.label}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500">※ 上記は目安であり、物件・期間・保証条件によって判断が変わります。PDのみで機械的に判断せず、定性情報と合わせて総合判断してください。</p>
          </div>
        ),
      },
      {
        q: 'PDとAIスコアの関係は？',
        a: (
          <div className="space-y-2">
            <p>AIスコア（100点満点）はPDを含む複数指標の総合評価です。概ね以下の関係があります：</p>
            <div className="text-xs space-y-1">
              {[
                ['70pt以上', '承認推奨', 'PD概ね3%以下'],
                ['60〜69pt', '条件付き承認', 'PD 3〜8%程度'],
                ['60pt未満', '否決推奨', 'PD 8%超の場合多い'],
              ].map(([score, status, pd]) => (
                <div key={score} className="flex gap-3">
                  <span className="font-black text-indigo-700 w-20 flex-shrink-0">{score}</span>
                  <span className="text-slate-700 w-24 flex-shrink-0">{status}</span>
                  <span className="text-slate-500">{pd}</span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-500">PDはスコアの重要構成要素ですが、営業利益率・自己資本比率・格付なども同様に加味されます。</p>
          </div>
        ),
      },
      {
        q: 'スプレッドとPDの関係は？',
        a: (
          <p>リース金利のリスクスプレッド部分はPDと正の相関があります。PD1%の増加で、概ね<strong>金利0.1〜0.3%程度の上昇</strong>が生じます（モデル係数による）。担保・保証でPDリスクをカバーすることで、金利低減交渉が可能な場合があります。</p>
        ),
      },
    ],
  },

  // REV-098: 一般的なリース知識FAQ
  {
    id: 'general-faq',
    title: 'リース基礎知識 FAQ',
    icon: <BookOpen className="w-5 h-5" />,
    color: 'text-indigo-600',
    items: [
      {
        q: 'リースのメリット・デメリットを教えてください',
        a: (
          <div className="space-y-2">
            <div>
              <p className="font-black text-emerald-700 text-xs mb-1">メリット</p>
              <ul className="list-disc ml-4 text-xs space-y-0.5">
                <li>初期投資を抑えて最新設備を導入できる</li>
                <li>月次の定額費用（リース料）として管理しやすい</li>
                <li>陳腐化リスクを軽減（期間終了後に新型へ乗り換え可）</li>
                <li>設備の評価・調達をリース会社に任せられる</li>
              </ul>
            </div>
            <div>
              <p className="font-black text-rose-700 text-xs mb-1">デメリット</p>
              <ul className="list-disc ml-4 text-xs space-y-0.5">
                <li>中途解約が原則できない（残リース料相当の違約金）</li>
                <li>総支払額は購入より高くなる場合がある</li>
                <li>物件を所有できない（所有権はリース会社）</li>
                <li>改造・カスタマイズに制限がある</li>
              </ul>
            </div>
          </div>
        ),
      },
      {
        q: 'リース期間はどのように決めればよいですか？',
        a: (
          <div className="space-y-2">
            <p>一般的には<strong>法定耐用年数の70〜120%</strong>の範囲でリース期間を設定します。</p>
            <p>実務では以下の観点で期間を選定します：</p>
            <ul className="list-disc ml-4 text-xs space-y-0.5">
              <li>月次リース料の負担が事業CFに見合うか</li>
              <li>技術革新サイクル（ITは短め、重機は長め）</li>
              <li>プロジェクト期間や融資返済スケジュールとの整合</li>
            </ul>
          </div>
        ),
      },
      {
        q: 'リース残価とは何ですか？',
        a: (
          <p><strong>残価（Residual Value）</strong>とはリース期間終了時点の物件の想定市場価値です。オペレーティングリースでは残価をリース会社が保証する代わりにリース料が低くなります。残価設定が高い案件では、期間中の物件管理状態が重要です。</p>
        ),
      },
      {
        q: '審査で最も重視される財務指標は何ですか？',
        a: (
          <div className="space-y-1">
            <p>当システムの重要度順：</p>
            <div className="grid grid-cols-1 gap-1 text-xs">
              {[
                { rank: '①', label: '営業利益率', desc: '事業の本質的な収益力を示す。目安5%以上' },
                { rank: '②', label: '自己資本比率', desc: '財務安定性。目安20%以上' },
                { rank: '③', label: '流動比率', desc: '短期支払能力。目安100%以上' },
                { rank: '④', label: 'インタレストカバレッジ', desc: '金利支払能力。目安2倍以上' },
                { rank: '⑤', label: '業歴・格付', desc: '定性評価として加点' },
              ].map(r => (
                <div key={r.rank} className="flex gap-2">
                  <span className="font-black text-indigo-600 w-4 flex-shrink-0">{r.rank}</span>
                  <span className="font-bold text-slate-700 w-32 flex-shrink-0">{r.label}</span>
                  <span className="text-slate-500">{r.desc}</span>
                </div>
              ))}
            </div>
          </div>
        ),
      },
    ],
  },

  // REV-121: 法定耐用年数一覧
  {
    id: 'useful-life',
    title: '主要設備の法定耐用年数一覧',
    icon: <Clock className="w-5 h-5" />,
    color: 'text-teal-600',
    items: [
      {
        q: '情報機器・IT設備の法定耐用年数を教えてください',
        a: (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-teal-50">
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-teal-800">設備種別</th>
                  <th className="border border-slate-200 px-3 py-2 text-center font-black text-teal-800">耐用年数</th>
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-teal-800">備考</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['サーバー・工業用コンピュータ', '5年', '減価償却資産の耐用年数省令 別表第一'],
                  ['パソコン（PC・タブレット）', '4年', '事務用機器として'],
                  ['コピー機・プリンター', '5年', ''],
                  ['デジタル複合機', '5年', ''],
                  ['電話・ビジネスフォン', '10年', '金属製'],
                  ['ルーター・ネットワーク機器', '5年', '電子計算機として'],
                  ['デジタルカメラ・映像機器', '5年', ''],
                  ['ソフトウェア（購入）', '5年（3年）', '複写販売用は3年'],
                ].map(([item, years, note]) => (
                  <tr key={item} className="even:bg-white odd:bg-slate-50/40">
                    <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{item}</td>
                    <td className="border border-slate-200 px-3 py-2 text-center font-black text-teal-700">{years}</td>
                    <td className="border border-slate-200 px-3 py-2 text-slate-500">{note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      },
      {
        q: '医療機器・工作機械の法定耐用年数は？',
        a: (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-rose-50">
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-rose-800">設備種別</th>
                  <th className="border border-slate-200 px-3 py-2 text-center font-black text-rose-800">耐用年数</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['CT・MRI等（大型医療機器）', '6〜7年'],
                  ['レントゲン・X線装置', '6年'],
                  ['内視鏡・手術機器', '5〜7年'],
                  ['工作機械（金属加工）', '10年'],
                  ['プレス機・鍛造機械', '10〜15年'],
                  ['建設用クレーン', '10〜15年'],
                  ['フォークリフト', '4年'],
                  ['大型トラック（10t以上）', '5年'],
                  ['乗用車', '6年'],
                  ['太陽光発電設備', '17年'],
                ].map(([item, years]) => (
                  <tr key={item} className="even:bg-white odd:bg-slate-50/40">
                    <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{item}</td>
                    <td className="border border-slate-200 px-3 py-2 text-center font-black text-rose-700">{years}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      },
    ],
  },

  // REV-122: リースバック
  {
    id: 'leaseback',
    title: 'リースバックとは',
    icon: <Building2 className="w-5 h-5" />,
    color: 'text-purple-600',
    items: [
      {
        q: 'リースバックの仕組みを教えてください',
        a: (
          <div className="space-y-2">
            <p><strong>セール＆リースバック（Sale and Leaseback）</strong>とは、企業が自社で所有する設備・不動産をリース会社に売却し、同時にその設備をリース会社からリースバックして使用し続ける取引です。</p>
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-xs">
              <p className="font-black text-purple-700 mb-2">資金調達の流れ</p>
              <p className="text-slate-600">企業 → （物件売却）→ リース会社 → （売却代金）→ 企業</p>
              <p className="text-slate-600 mt-1">企業 ← （物件を継続使用）← リース会社 ← （月次リース料）← 企業</p>
            </div>
          </div>
        ),
      },
      {
        q: 'リースバックのメリットとデメリットは？',
        a: (
          <div className="space-y-2">
            <div>
              <p className="font-black text-emerald-700 text-xs mb-1">メリット</p>
              <ul className="list-disc ml-4 text-xs space-y-0.5">
                <li>固定資産を現金化して運転資金・設備投資資金を確保できる</li>
                <li>貸借対照表上の固定資産が減少し、財務指標（ROA等）が改善</li>
                <li>銀行融資枠を使わずに資金調達できる</li>
                <li>引き続き設備を使用できるため事業継続に支障なし</li>
              </ul>
            </div>
            <div>
              <p className="font-black text-rose-700 text-xs mb-1">デメリット・注意点</p>
              <ul className="list-disc ml-4 text-xs space-y-0.5">
                <li>総コストは通常の融資より高くなることが多い</li>
                <li>IFRS16号適用企業は売却後もリース負債として計上が必要な場合あり</li>
                <li>物件の所有権がなくなるため、担保提供・処分ができなくなる</li>
                <li>資金難のサインとみなされ、取引先・金融機関の信頼に影響する可能性</li>
              </ul>
            </div>
          </div>
        ),
      },
      {
        q: 'リースバック案件の審査上の注意点は？',
        a: (
          <div className="space-y-2">
            <p>リースバック案件は通常の新規リース案件と異なる審査観点が必要です：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>資金使途の確認</strong>：売却資金が事業改善に使われるか（単なる延命でないか）</li>
              <li><strong>物件の適正価格評価</strong>：帳簿価格vs市場価格の乖離チェック</li>
              <li><strong>財務悪化の兆候確認</strong>：なぜリースバックが必要か（資金繰り悪化サインか）</li>
              <li><strong>リース終了後の対応</strong>：再リース・返却・買取のいずれかを事前確認</li>
            </ul>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">リースバックは財務改善の有効手段ですが、当システムのAIスコアに加えて担当者の定性判断を重視してください。</p>
          </div>
        ),
      },
    ],
  },

  // REV-058/059: リース対象外物件の明確化
  {
    id: 'non-leaseable',
    title: 'リース対象外となる物件・資産',
    icon: <Ban className="w-5 h-5" />,
    color: 'text-red-600',
    items: [
      {
        q: 'リースにできない物件・資産の種類は？',
        a: (
          <div className="space-y-3">
            <p>以下に該当する物件・資産は原則としてリースの対象外です。</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {[
                { cat: '不動産', items: ['土地（原則対象外）', '建物・構築物本体', '内装工事（固着度が高いもの）'] },
                { cat: '消耗品・短命資産', items: ['消耗品（文具・用紙等）', '耐用年数1年未満の資産', '一回使用で費消されるもの'] },
                { cat: '無形資産', items: ['特許権・商標権・のれん', 'ソフトウェアライセンス（SaaS型）', '営業権・顧客リスト'] },
                { cat: 'その他', items: ['生き物・農産物・水産物', '有価証券・金融商品', '既に所有権のない資産（他社リース中のもの）'] },
              ].map(g => (
                <div key={g.cat} className="bg-red-50 border border-red-100 rounded-lg p-3">
                  <p className="font-black text-red-700 text-xs mb-1.5">{g.cat}</p>
                  <ul className="space-y-0.5">
                    {g.items.map(i => (
                      <li key={i} className="text-xs text-slate-600 flex items-start gap-1.5">
                        <span className="mt-1.5 w-1 h-1 rounded-full bg-red-300 flex-shrink-0" />
                        {i}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">※ 建物付属設備（電気・空調・給排水設備等）は取り外し可能かつ独立して使用できるものはリース対象になる場合があります。</p>
          </div>
        ),
      },
      {
        q: '「固着した物件」はなぜリース対象外なのですか？',
        a: (
          <div className="space-y-2">
            <p>リースは<strong>所有権がリース会社にあり、使用権のみユーザーが持つ</strong>取引形態です。そのため、取り外しや移動ができない固着設備（建物と一体化した内装・基礎工事等）は、リース会社が担保として回収・処分できないためリース不可となります。</p>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs">
              <p className="font-black text-amber-700 mb-1">判断基準の目安</p>
              <ul className="list-disc ml-4 space-y-0.5 text-slate-600">
                <li>取り外しに建物解体が必要 → 対象外</li>
                <li>独立して機能・使用できる → 対象の可能性あり</li>
                <li>撤去・移設コストが物件価値の50%超 → 要協議</li>
              </ul>
            </div>
          </div>
        ),
      },
      {
        q: '中古品・既存設備はリース対象になりますか？',
        a: (
          <div className="space-y-2">
            <p>中古品はリース対象になる場合がありますが、以下の条件を確認する必要があります：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>残存耐用年数</strong>：リース期間より長い残存年数があること</li>
              <li><strong>市場価格の確認</strong>：適正な中古市場価格を査定できること</li>
              <li><strong>所有権の明確化</strong>：売主に確実な所有権があること（担保・抵当権等がないこと）</li>
              <li><strong>動作確認</strong>：設備が正常に稼働しており、保守可能であること</li>
            </ul>
          </div>
        ),
      },
    ],
  },

  // REV-063: リース物件の処分・返却・再リース
  {
    id: 'lease-end',
    title: 'リース期間満了・返却・再リース',
    icon: <RotateCcw className="w-5 h-5" />,
    color: 'text-teal-600',
    items: [
      {
        q: 'リース期間が終了したらどうなりますか？',
        a: (
          <div className="space-y-3">
            <p>リース期間満了時には主に以下の3つの選択肢があります：</p>
            <div className="grid grid-cols-1 gap-2">
              {[
                { icon: '①', label: '返却', desc: 'リース会社に物件を返却して契約終了。次の最新機器へのリースに乗り換え可能。', color: 'bg-sky-50 border-sky-200 text-sky-800' },
                { icon: '②', label: '再リース（継続使用）', desc: '同じ物件を通常より低いリース料で継続してリース。再リース料は通常、元のリース料の1/10〜1/5程度。', color: 'bg-emerald-50 border-emerald-200 text-emerald-800' },
                { icon: '③', label: '買取（購入）', desc: '残価（帳簿価額・市場価値等）で物件を購入。オペレーティングリースでは選択肢にない場合あり。', color: 'bg-amber-50 border-amber-200 text-amber-800' },
              ].map(opt => (
                <div key={opt.icon} className={`p-3 rounded-lg border ${opt.color}`}>
                  <p className="font-black text-sm mb-1">{opt.icon} {opt.label}</p>
                  <p className="text-xs leading-relaxed">{opt.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ),
      },
      {
        q: '中途解約はできますか？違約金はどのくらいですか？',
        a: (
          <div className="space-y-2">
            <p>ファイナンスリースは原則として<strong>中途解約不可</strong>です。やむを得ない事情での解約時は、残存リース料相当額（残存リース料総額 − 利息相当額）が違約金として請求されます。</p>
            <div className="bg-rose-50 border border-rose-200 rounded-lg p-3 text-xs">
              <p className="font-black text-rose-700 mb-1">違約金の目安</p>
              <p className="text-slate-600">残存リース料の80〜100%程度が一般的。残期間が長いほど負担が大きい。</p>
              <p className="text-slate-600 mt-1">例: 月額リース料10万円・残20ヶ月の場合 → 160〜200万円程度</p>
            </div>
            <p className="text-[11px] text-slate-500">オペレーティングリースは物件や契約条件により中途解約条項が設けられることがあります。契約書の確認が必要です。</p>
          </div>
        ),
      },
      {
        q: '返却時の物件の状態・原状回復義務はどうなりますか？',
        a: (
          <div className="space-y-2">
            <p>返却時は<strong>通常の使用による自然消耗を超えた損耗・破損</strong>がある場合、修繕費用をユーザーが負担します。</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li>通常の使用による摩耗・劣化：ユーザー負担なし</li>
              <li>改造・無断加工による損傷：ユーザー負担（原状回復義務）</li>
              <li>事故・水没等による損傷：ユーザー負担（保険でカバー可能なケースあり）</li>
              <li>データ消去：IT機器の場合、ユーザーが完全消去を証明する義務あり</li>
            </ul>
          </div>
        ),
      },
    ],
  },

  // REV-064: 業種別リース物件例
  {
    id: 'industry-items',
    title: '業種別リース物件の例',
    icon: <Factory className="w-5 h-5" />,
    color: 'text-orange-600',
    items: [
      {
        q: '製造業・建設業でよく使われるリース物件は？',
        a: (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-orange-50">
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-orange-800">業種</th>
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-orange-800">代表的なリース物件</th>
                  <th className="border border-slate-200 px-3 py-2 text-center font-black text-orange-800">耐用年数目安</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['金属加工・製造', '工作機械・プレス機・NC旋盤・溶接機', '10〜15年'],
                  ['食品製造', '冷凍冷蔵庫・包装機・充填機・殺菌装置', '8〜12年'],
                  ['建設・土木', '油圧ショベル・クレーン・フォークリフト・高所作業車', '6〜15年'],
                  ['印刷・出版', 'オフセット印刷機・デジタル印刷機・断裁機', '8〜10年'],
                  ['物流・倉庫', 'フォークリフト・自動倉庫システム・仕分け機', '4〜12年'],
                ].map(([ind, items, years]) => (
                  <tr key={ind} className="even:bg-white odd:bg-slate-50/40">
                    <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{ind}</td>
                    <td className="border border-slate-200 px-3 py-2 text-slate-600">{items}</td>
                    <td className="border border-slate-200 px-3 py-2 text-center font-bold text-orange-700">{years}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      },
      {
        q: 'サービス業・医療・飲食業でよく使われるリース物件は？',
        a: (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-indigo-50">
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-indigo-800">業種</th>
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-indigo-800">代表的なリース物件</th>
                  <th className="border border-slate-200 px-3 py-2 text-center font-black text-indigo-800">耐用年数目安</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['医療・クリニック', 'CT・MRI・レントゲン・内視鏡・手術ロボット', '5〜7年'],
                  ['飲食・ホテル', '業務用厨房機器・食洗機・製氷機・エスプレッソマシン', '6〜10年'],
                  ['美容・エステ', '美容医療機器・脱毛レーザー・エステ機器', '5〜8年'],
                  ['小売・店舗', 'POS端末・セルフレジ・冷蔵ショーケース・電子棚札', '5〜10年'],
                  ['オフィス・IT', 'サーバー・PC・複合機・ネットワーク機器', '4〜8年'],
                ].map(([ind, items, years]) => (
                  <tr key={ind} className="even:bg-white odd:bg-slate-50/40">
                    <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{ind}</td>
                    <td className="border border-slate-200 px-3 py-2 text-slate-600">{items}</td>
                    <td className="border border-slate-200 px-3 py-2 text-center font-bold text-indigo-700">{years}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      },
    ],
  },

  // REV-066: 残価設定ガイドライン
  {
    id: 'residual-value',
    title: '残価（リジデュアルバリュー）の設定ガイドライン',
    icon: <Percent className="w-5 h-5" />,
    color: 'text-violet-600',
    items: [
      {
        q: '残価とは何ですか？リースにどう影響しますか？',
        a: (
          <div className="space-y-2">
            <p><strong>残価（Residual Value / RV）</strong>とはリース期間終了時点での物件の推定市場価値です。オペレーティングリースでは残価をリース会社が保証するため、ユーザーの月次リース料を低く抑えられます。</p>
            <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 text-xs">
              <p className="font-black text-violet-700 mb-1">リース料計算への影響</p>
              <p className="text-slate-600">月次リース料 ＝ （物件価格 − 残価） ÷ リース期間 ＋ 金利コスト</p>
              <p className="text-slate-600 mt-1">残価が高いほど月次リース料が低くなる（ただし残価リスクはリース会社が負担）</p>
            </div>
          </div>
        ),
      },
      {
        q: '物件別の残価設定目安を教えてください',
        a: (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-violet-50">
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-violet-800">物件カテゴリ</th>
                  <th className="border border-slate-200 px-3 py-2 text-center font-black text-violet-800">残価率目安（取得価格比）</th>
                  <th className="border border-slate-200 px-3 py-2 text-left font-black text-violet-800">ポイント</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['車両（商用・乗用）', '20〜40%', '市場流動性が高く残価は安定'],
                  ['建設機械・フォークリフト', '25〜45%', 'ブランド・稼働時間が価値に影響'],
                  ['工作機械・製造設備', '15〜30%', '機種・精度・産業需要に依存'],
                  ['PC・IT機器', '5〜15%', '陳腐化が速く残価は低め'],
                  ['医療機器', '20〜35%', '認可維持・メーカーサポートが重要'],
                  ['太陽光発電設備', '30〜50%', '売電契約・FITが価値を左右'],
                ].map(([cat, rv, note]) => (
                  <tr key={cat} className="even:bg-white odd:bg-slate-50/40">
                    <td className="border border-slate-200 px-3 py-2 font-bold text-slate-700">{cat}</td>
                    <td className="border border-slate-200 px-3 py-2 text-center font-black text-violet-700">{rv}</td>
                    <td className="border border-slate-200 px-3 py-2 text-slate-600">{note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      },
      {
        q: '残価リスクとはどのようなリスクですか？',
        a: (
          <div className="space-y-2">
            <p>残価リスクとは、<strong>期間満了時の実際の市場価値が設定残価を下回るリスク</strong>です。オペレーティングリースではリース会社がこのリスクを負います。</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>技術革新リスク</strong>：IT機器・医療機器は新技術により旧型の価値が急落</li>
              <li><strong>市場需給リスク</strong>：景気後退で中古市場全体の価格が下落</li>
              <li><strong>法規制リスク</strong>：環境規制強化（排ガス・フロン規制等）で特定物件が流通不可に</li>
              <li><strong>損耗リスク</strong>：使用状況が悪く想定以上に価値が劣化</li>
            </ul>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">審査上は「残価設定が高いオペレーティングリース案件」では物件の市場流動性・将来需要を特に慎重に評価してください。</p>
          </div>
        ),
      },
    ],
  },

  // REV-077: 固定資産税とリースの関係
  {
    id: 'property-tax',
    title: '固定資産税とリースの関係',
    icon: <Landmark className="w-5 h-5" />,
    color: 'text-slate-600',
    items: [
      {
        q: 'リース物件の固定資産税は誰が払いますか？',
        a: (
          <div className="space-y-2">
            <p>リース物件の<strong>固定資産税はリース会社（所有者）が納税義務者</strong>となります。ただし、その固定資産税相当額はリース料に含まれてユーザーが実質的に負担します。</p>
            <div className="bg-slate-100 border border-slate-200 rounded-lg p-3 text-xs">
              <p className="font-black text-slate-700 mb-1">リース料の内訳（概念）</p>
              <div className="space-y-1 text-slate-600">
                <p>リース料 ＝ 物件取得原価の回収分</p>
                <p>　　　　　＋ 固定資産税相当額</p>
                <p>　　　　　＋ 金利（資金調達コスト）</p>
                <p>　　　　　＋ リース会社の利益・諸経費</p>
              </div>
            </div>
          </div>
        ),
      },
      {
        q: '固定資産税はどのくらいの金額になりますか？',
        a: (
          <div className="space-y-2">
            <p>固定資産税の税率は原則<strong>1.4%（評価額に対して）</strong>です。減価償却により評価額は毎年減少します。</p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs">
              <p className="font-black text-slate-700 mb-1">概算例（取得価格1,000万円の設備の場合）</p>
              <div className="space-y-0.5 text-slate-600">
                <p>初年度: 評価額≒700万円 × 1.4% ＝ 約98,000円/年</p>
                <p>3年目: 評価額≒400万円 × 1.4% ＝ 約56,000円/年</p>
                <p>5年目: 評価額≒200万円 × 1.4% ＝ 約28,000円/年</p>
              </div>
            </div>
          </div>
        ),
      },
      {
        q: 'ファイナンスリースとオペレーティングリースで固定資産税の扱いは違いますか？',
        a: (
          <div className="space-y-2">
            <p>所有権がリース会社にある場合、<strong>どちらのリース形態でも固定資産税の納税義務者はリース会社</strong>です。ただし会計処理上の扱いに注意が必要です。</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="font-black text-blue-700 mb-1">ファイナンスリース</p>
                <p className="text-slate-600">固定資産税相当額はリース料の一部として費用計上。リース資産・負債は借手が計上（IFRS/日本基準）。</p>
              </div>
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                <p className="font-black text-emerald-700 mb-1">オペレーティングリース</p>
                <p className="text-slate-600">固定資産税相当額を含むリース料全額を費用（支払リース料）として計上。資産・負債はオフバランス。</p>
              </div>
            </div>
            <p className="text-[11px] text-slate-500">※ 所有権移転ファイナンスリースでは最終的にユーザーが所有者となるため、以降の固定資産税はユーザーが直接納税します。</p>
          </div>
        ),
      },
    ],
  },

  // IT機器（REV-121補完）
  {
    id: 'it-lease',
    title: 'IT機器・PCリースの審査',
    icon: <Cpu className="w-5 h-5" />,
    color: 'text-sky-600',
    items: [
      {
        q: 'PCやサーバーのリース審査で特有のポイントは？',
        a: (
          <div className="space-y-2">
            <p>IT機器リースでは以下の点に注意が必要です：</p>
            <ul className="list-disc ml-4 text-xs space-y-0.5">
              <li><strong>陳腐化スピード</strong>：PCは3〜4年で市場価値がほぼゼロになるため残価リスクに注意</li>
              <li><strong>データ消去義務</strong>：リース終了時にデータ消去を確実に実施する必要がある</li>
              <li><strong>保守契約との連動</strong>：メーカーサポート期間内のリース期間設定が望ましい</li>
              <li><strong>一括vs分散</strong>：大量PCの一括リース（100台超）は個別審査より総額管理が重要</li>
            </ul>
          </div>
        ),
      },
      {
        q: 'クラウド移行後の機器はリース対象になりますか？',
        a: (
          <p>物理的な設備（サーバー・ストレージ・ネットワーク機器等）はリース対象です。ただし、クラウドサービス自体（SaaS・IaaS等のサブスクリプション費用）はリース対象外です。ハイブリッド環境では物理機器部分のみリース対象として切り出してご相談ください。</p>
        ),
      },
    ],
  },
  {
    id: 'new-business',
    title: '新規事業・創業融資',
    icon: '🚀',
    color: 'text-violet-500',
    items: [
      {
        q: '新規事業計画書はどのように審査されますか？（REV-040）',
        a: (
          <div className="space-y-2">
            <p>新規事業のリース審査では、既存事業の財務実績と新規事業計画の双方を評価します。主なチェックポイントは以下のとおりです：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>事業計画の合理性</strong>：売上・費用の根拠が明示されているか。「楽観シナリオのみ」は評価を下げます。</li>
              <li><strong>既存キャッシュフローとの切り離し</strong>：新規事業が赤字でも既存事業で返済できるか確認します。</li>
              <li><strong>担保・保証の有無</strong>：新規事業単体での信用が低い場合、保証人・担保の有無が大きく影響します。</li>
              <li><strong>市場の裏付け</strong>：業界統計・競合分析・LOI（意向書）など客観的な根拠があると評価が高まります。</li>
              <li><strong>経営者の経験・実績</strong>：関連業種での就業歴や過去の起業経験がプラスに評価されます。</li>
            </ul>
          </div>
        ),
      },
      {
        q: '創業間もない企業でもリースは利用できますか？',
        a: (
          <div className="space-y-2">
            <p>創業後2〜3期以内の企業は財務実績が限られるため、以下の条件が揃う場合に審査を進めます：</p>
            <ul className="list-disc ml-4 text-xs space-y-0.5">
              <li>代表者個人の信用情報が良好であること</li>
              <li>親会社・グループ会社の保証が取れること</li>
              <li>物件の汎用性が高く残価リスクが低いこと（IT機器・標準車両など）</li>
              <li>取得価額が小口（概ね500万円以下）であること</li>
            </ul>
            <p className="text-xs text-slate-500 mt-1">※ 大型設備（1,000万円超）は原則として3期以上の決算書が必要です。</p>
          </div>
        ),
      },
      {
        q: '事業計画書の数値が甘いと審査に影響しますか？',
        a: (
          <p>はい、影響します。根拠のない高成長率や費用の過小見積もりは審査担当者の信頼を損ないます。保守的なベースシナリオと楽観シナリオを両方提示し、最悪ケースでも返済可能であることを示す方が評価されます。定性評価スコア「経営者の信頼性」にも反映されます。</p>
        ),
      },
      {
        q: 'フランチャイズ加盟の場合、審査はどう変わりますか？',
        a: (
          <div className="space-y-2">
            <p>フランチャイズ（FC）は本部の実績・ブランド力が信用補完として機能します。評価上のポイント：</p>
            <ul className="list-disc ml-4 text-xs space-y-0.5">
              <li>本部のFC加盟契約書・開業サポート内容の確認</li>
              <li>同FC他加盟店の平均売上・損益データの参照</li>
              <li>本部保証やロイヤリティの資金計画への影響確認</li>
            </ul>
          </div>
        ),
      },
    ],
  },

  // リース対象外資産・判断基準
  {
    id: 'non-leaseable',
    title: 'リース対象外・対象判断の基準',
    icon: <Ban className="w-5 h-5" />,
    color: 'text-rose-600',
    items: [
      {
        q: 'リースできない資産（リース対象外）はどれですか？',
        a: (
          <div className="space-y-3">
            <p>以下の資産はリース取引の対象外です：</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {[
                ['土地・建物（不動産）', '動産リースの対象外。不動産リースは別途スキームが必要。'],
                ['消耗品・原材料', '使用により価値がゼロになる物品はリース不可。'],
                ['有価証券・商品在庫', '金融資産・流動資産はリース対象外。'],
                ['無形資産（ソフトウェア単体）', '単体のSaaSや保守サービスはリース不可（ハードとセットなら可）。'],
                ['廃棄物処理費・解体費用', 'サービス費用であり物件取得に該当しない。'],
                ['リース会社の業種規制品目', '武器・賭博機器等は法令・社内規程で禁止。'],
              ].map(([item, reason]) => (
                <div key={item} className="bg-rose-50 border border-rose-100 rounded-lg p-2.5">
                  <p className="font-black text-rose-700 mb-1">{item}</p>
                  <p className="text-slate-600">{reason}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">※ 工事費・設置費用は物件価格の一部として含められる場合があります（下記Q参照）。</p>
          </div>
        ),
      },
      {
        q: 'リース対象になるかどうかの判断基準を教えてください',
        a: (
          <div className="space-y-2">
            <p>リース会社が物件をリース対象と判断する際の主な基準は以下の4点です：</p>
            <div className="space-y-2 text-xs">
              {[
                ['① 特定性・独立性', '識別可能な固定資産として管理できること（シリアル番号・型番等）。'],
                ['② 換金性（流動性）', 'リース終了時または途中解約時に売却・処分が可能であること。汎用性が高いほど評価UP。'],
                ['③ 取得費用の明確性', '物件取得原価が明確に積算できること（工事費が主体の場合は難しい）。'],
                ['④ 経済的耐用年数', '物件の実用的な使用可能年数が確認できること。消耗品・短命品は対象外。'],
              ].map(([label, desc]) => (
                <div key={label} className="flex gap-2 p-2.5 bg-slate-50 border border-slate-200 rounded-lg">
                  <span className="font-black text-slate-700 shrink-0">{label}</span>
                  <span className="text-slate-600">{desc}</span>
                </div>
              ))}
            </div>
          </div>
        ),
      },
      {
        q: '工事費・設置費用もリースに含められますか？',
        a: (
          <div className="space-y-2">
            <p>一定の条件のもとで、工事費・設置費用を物件取得価額に含めてリースできます。</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>対象となる費用</strong>：物件の設置・据付工事費（配管・電気配線含む）、輸送費、試運転費用</li>
              <li><strong>目安</strong>：工事費が物件本体価格の<strong>50%以内</strong>であれば一体でリース可能なケースが多い</li>
              <li><strong>注意点</strong>：工事費が主体（物件価格より高い）の場合はリース組成が困難。建設工事そのものは対象外</li>
            </ul>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs">
              <p className="font-black text-amber-700 mb-1">高額工事費用を含む案件の審査上の確認ポイント</p>
              <ul className="list-disc ml-4 text-slate-600 space-y-0.5">
                <li>工事完了後の設備が独立した資産として認識できるか</li>
                <li>撤去・移設が現実的に可能か（換金性の確認）</li>
                <li>工事業者への直接支払いスキームとするか、ユーザー立替払後にリース会社が買取るかを整理</li>
              </ul>
            </div>
          </div>
        ),
      },
    ],
  },

  // リース終了時・処分FAQ
  {
    id: 'end-of-lease',
    title: 'リース終了・中途解約・物件処分',
    icon: <Trash2 className="w-5 h-5" />,
    color: 'text-slate-600',
    items: [
      {
        q: 'リース終了時の選択肢はどれくらいありますか？',
        a: (
          <div className="space-y-2">
            <p>リース期間満了時の主な選択肢は以下の3つです：</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
              {[
                { title: '① 返却', desc: '物件をリース会社に返却して契約終了。最もシンプル。', color: 'bg-blue-50 border-blue-200 text-blue-800' },
                { title: '② 再リース', desc: '同物件のリース期間を延長（通常1年単位）。月額は下がるケースが多い。', color: 'bg-emerald-50 border-emerald-200 text-emerald-800' },
                { title: '③ 買取り', desc: '時価または残価でユーザーが物件を購入。所有権が移転する。', color: 'bg-amber-50 border-amber-200 text-amber-800' },
              ].map(opt => (
                <div key={opt.title} className={`rounded-lg p-3 border ${opt.color}`}>
                  <p className="font-black mb-1">{opt.title}</p>
                  <p className="text-slate-600">{opt.desc}</p>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 bg-slate-100 rounded p-2">※ 選択可能な選択肢はリース種別・契約内容によって異なります。契約書の「期間終了後の処理」条項を事前に確認してください。</p>
          </div>
        ),
      },
      {
        q: 'リース物件の返却時に費用は発生しますか？',
        a: (
          <div className="space-y-2">
            <p>返却時の費用負担は契約内容によりますが、一般的に以下のケースで費用が発生します：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>原状回復費用</strong>：使用による通常摩耗を超える損傷がある場合。特にOA機器・フォークリフトで問題になりやすい</li>
              <li><strong>輸送・撤去費用</strong>：返却のための運搬・据付解除費用（契約に規定がある場合はユーザー負担）</li>
              <li><strong>データ消去費用</strong>：PC・複合機等では個人情報保護の観点からリース会社指定の消去が必要な場合がある</li>
            </ul>
          </div>
        ),
      },
      {
        q: '中途解約はできますか？解約金の計算方法は？',
        a: (
          <div className="space-y-2">
            <p>ファイナンスリースは原則として<strong>中途解約不可</strong>です。ただし、やむを得ない事情がある場合は解約損害金（違約金）の支払いにより解約が認められるケースがあります。</p>
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs">
              <p className="font-black text-slate-700 mb-1">解約損害金の一般的な計算方式</p>
              <p className="text-slate-600">残存リース料の現在価値合計 − リース物件の処分価額</p>
              <p className="text-slate-500 mt-1">（物件価値が高いほど損害金は少なくなる傾向）</p>
            </div>
            <p className="text-xs text-slate-600">オペレーティングリースの場合は解約条件が緩いケースが多いですが、残価リスクの補填を求められる場合があります。</p>
          </div>
        ),
      },
    ],
  },

  // 内装・建物付帯設備のリース
  {
    id: 'construction-lease',
    title: '内装・建物付帯設備・ソフトウェアのリース',
    icon: <HardHat className="w-5 h-5" />,
    color: 'text-orange-600',
    items: [
      {
        q: '店舗の内装工事費はリースできますか？',
        a: (
          <div className="space-y-2">
            <p>内装工事費は原則として<strong>リース対象外</strong>ですが、設備機器部分を切り出してリース組成することが可能です。</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                <p className="font-black text-emerald-700 mb-1">リース可能な部分</p>
                <ul className="list-disc ml-3 text-slate-600 space-y-0.5">
                  <li>厨房機器・調理設備</li>
                  <li>空調機器（個別設置型）</li>
                  <li>サイン・照明設備</li>
                  <li>POS・レジシステム</li>
                </ul>
              </div>
              <div className="bg-rose-50 border border-rose-200 rounded-lg p-3">
                <p className="font-black text-rose-700 mb-1">リース困難な部分</p>
                <ul className="list-disc ml-3 text-slate-600 space-y-0.5">
                  <li>壁・床・天井の仕上げ工事</li>
                  <li>建物躯体への埋込み設備</li>
                  <li>撤去不能な造作工事</li>
                  <li>開業準備費・デザイン費</li>
                </ul>
              </div>
            </div>
          </div>
        ),
      },
      {
        q: 'ソフトウェアのみのリースは可能ですか？',
        a: (
          <div className="space-y-2">
            <p><strong>単体のソフトウェア（ライセンス・SaaS）はリース対象外</strong>です。ただし以下のケースでは組み込みが可能です：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>ハードウェアと一体型</strong>：PC・サーバー等のハードにプリインストールされたソフトはセットでリース可能</li>
              <li><strong>組込みシステム</strong>：産業機器に組み込まれた制御ソフトは物件の一部として扱える場合がある</li>
              <li><strong>オンプレ型業務システム</strong>：サーバーと一体で構成するシステムはハード込みでリース可能（SaaSは不可）</li>
            </ul>
          </div>
        ),
      },
      {
        q: '太陽光発電設備・蓄電池はリース対象になりますか？',
        a: (
          <div className="space-y-2">
            <p>太陽光パネル・蓄電池は<strong>リース対象として認められます</strong>。ただし以下の点を確認してください：</p>
            <ul className="list-disc ml-4 text-xs space-y-1">
              <li><strong>FIT（固定価格買取制度）との整合</strong>：リース会社が設備所有者となるためFITの名義調整が必要</li>
              <li><strong>物件価値の評価</strong>：FIT終了後の残存価値・売電収入の見通しが審査に影響</li>
              <li><strong>設置場所の権利</strong>：屋根・土地の使用権（賃貸借契約等）が確保されていること</li>
              <li><strong>O&M（運営保守）コスト</strong>：発電量低下リスクや維持管理費を資金計画に組み込むこと</li>
            </ul>
          </div>
        ),
      },
    ],
  },
];

export default function FaqPage() {
  const [activeSection, setActiveSection] = useState<string | null>(null);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <HelpCircle className="text-indigo-500" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">ナレッジベース・FAQ</h1>
          <p className="text-sm text-slate-500">リース審査に関する基礎知識・用語解説・審査基準をまとめています。</p>
        </div>
      </div>

      {/* セクションナビ */}
      <div className="flex flex-wrap gap-2">
        {sections.map(s => (
          <button
            key={s.id}
            onClick={() => {
              setActiveSection(activeSection === s.id ? null : s.id);
              setTimeout(() => {
                document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }, 50);
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border transition-all
              ${activeSection === s.id
                ? 'bg-indigo-600 text-white border-indigo-600'
                : 'bg-white text-slate-600 border-slate-200 hover:border-indigo-300 hover:text-indigo-600'}`}
          >
            <span className={activeSection === s.id ? 'text-white' : s.color}>{s.icon}</span>
            {s.title}
          </button>
        ))}
      </div>

      {/* FAQセクション */}
      <div className="space-y-8">
        {sections.map(s => (
          <div key={s.id} id={s.id} className="scroll-mt-4">
            <div className="flex items-center gap-2 mb-4">
              <span className={s.color}>{s.icon}</span>
              <h2 className={`text-base font-black ${s.color}`}>{s.title}</h2>
            </div>
            <FaqAccordion items={s.items} />
          </div>
        ))}
      </div>

      <div className="text-center text-xs text-slate-400 pt-4 border-t border-slate-100">
        ※ 本FAQの内容は一般的なリース審査実務に基づいています。個別案件の判断は担当者の総合的な判断に委ねられます。
      </div>
    </div>
  );
}
