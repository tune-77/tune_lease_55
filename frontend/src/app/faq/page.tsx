"use client";

import React, { useState } from 'react';
import {
  HelpCircle, ChevronDown, ChevronUp, BookOpen, Cpu, Zap,
  Building2, TrendingDown, Clock, RefreshCw, Wrench
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
            <p>当システムでは過去の審査データを学習したMLモデル（LightGBM）が、財務指標・格付・業種等から各案件のPDを計算します。</p>
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
