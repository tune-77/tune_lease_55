"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Send, ArrowRight, ArrowLeft, Bot, Activity, CheckCircle, ChevronDown } from 'lucide-react';
import axios from 'axios';
import { API_BASE } from '../../lib/api';

// --- 型定義 ---
type Message = {
  role: 'bot' | 'user' | 'humor';
  text: React.ReactNode;
};

// --- 初期データ ---
const STEPS = [
  "企業・業種", "取引と競合", "リース物件", "損益計算", "資産情報",
  "経費・減価償却", "信用情報", "契約条件", "定性評価", "最終確認"
];

// --- メインコンポーネント ---
export default function LeaseKunWizard() {
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<Message[]>([
    { role: 'bot', text: 'はじめまして！リースくんです 🎩 まず企業名と業種から教えてね！' }
  ]);
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // --- フォームステート ---
  const [formData, setFormData] = useState({
    // Step 0
    company_no: '', company_name: '',
    industry_major: 'D 建設業', industry_sub: '06 総合工事業',
    // Step 1
    main_bank: 'メイン先', competitor: '競合なし',
    num_competitors: '未入力', deal_source: 'その他', deal_occurrence: '不明',
    // Step 2
    asset_name: 'IT・OA機器',
    // Step 3 (PL)
    nenshu: '', gross_profit: '', op_profit: '', ord_profit: '', net_income: '',
    // Step 4 (BS)
    total_assets: '', net_assets: '', machines: '', other_assets: '',
    // Step 5 (経費)
    depreciation: '', dep_expense: '', rent: '', rent_expense: '',
    // Step 6 (信用)
    grade: '②4-6 (標準)', contracts: '', bank_credit: '', lease_credit: '',
    // Step 7 (契約)
    customer_type: '新規先', contract_type: '一般', deal_source2: 'その他',
    lease_term: 60, acceptance_year: new Date().getFullYear(), acquisition_cost: '',
    // Step 8 (定性)
    qual_corr_company_history: '未選択',
    qual_corr_customer_stability: '未選択',
    qual_corr_repayment_history: '未選択',
    qual_corr_business_future: '未選択',
    qual_corr_equipment_purpose: '未選択',
    qual_corr_main_bank: '未選択',
    passion_text: '',
    // Step 9
    intuition: 3
  });

  // 自動スクロール
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history, step]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleNext = (e: React.FormEvent) => {
    e.preventDefault();
    if (step >= STEPS.length - 1) {
      submitScore();
      return;
    }

    let answerText = '';
    let nextBotText = '';

    switch(step) {
      case 0:
        answerText = `${formData.company_name || '（企業名未入力）'} / ${formData.industry_major} / ${formData.industry_sub}`;
        nextBotText = `次は取引状況について。当行メイン先？競合はいる？`;
        break;
      case 1:
        answerText = `${formData.main_bank} / ${formData.competitor} / 商談: ${formData.deal_source}`;
        nextBotText = `何をリースするのかな？（物件名）`;
        break;
      case 2:
        answerText = formData.asset_name || "その他";
        nextBotText = `損益計算書(P/L)の数値を入力してね！売上高は必須だよ。`;
        break;
      case 3:
        if (!formData.nenshu || Number(formData.nenshu) <= 0) return alert("売上高は必須です！");
        answerText = `売上: ${formData.nenshu}千円 / 営業利益: ${formData.op_profit || 0}千円`;
        nextBotText = `貸借対照表(B/S)！総資産は必須。機械やその他の内訳もあれば。`;
        break;
      case 4:
        if (!formData.total_assets || Number(formData.total_assets) <= 0) return alert("総資産は必須です！");
        answerText = `総資産: ${formData.total_assets}千円 / 純資産: ${formData.net_assets || 0}千円`;
        nextBotText = `減価償却や地代家賃などの経費項目はある？（なければ空欄かスキップでOK！）`;
        break;
      case 5:
        answerText = `償却: ${formData.depreciation || 0}千円 / 家賃: ${formData.rent || 0}千円`;
        nextBotText = `対象の格付や与信残高を教えてね。`;
        break;
      case 6:
        answerText = `格付: ${formData.grade} / 銀行与信: ${formData.bank_credit || 0}千円`;
        nextBotText = `今回の契約期間や取得価格はどうなってる？`;
        break;
      case 7:
        if (!formData.acquisition_cost || Number(formData.acquisition_cost) <= 0) return alert("取得価格(千円)は必須です！");
        answerText = `${formData.customer_type} / ${formData.lease_term}ヶ月 / ${formData.acquisition_cost}千円`;
        nextBotText = `定性的な評価項目（6点）を教えて。難しければ「未選択」でも審査はできるよ。`;
        break;
      case 8:
        answerText = `定性評価 入力完了`;
        nextBotText = `最後！！直感スコア（1〜5）を教えて。これで審査を実行するよ。`;
        break;
    }

    const addedMessages: Message[] = [{ role: 'user', text: answerText }];

    if (Math.random() < 0.3) {
      const humors = [
        "業種によって「良い数字」の基準は変わります。比較する相手を間違えないように。",
        "数字の向こう側にある現場を、想像しながら読んでいます。",
        "審査は減点ゲームではなく、可能性を見つける作業です。"
      ];
      addedMessages.push({ role: 'humor', text: humors[Math.floor(Math.random() * humors.length)] });
    }
    addedMessages.push({ role: 'bot', text: nextBotText });

    setHistory(prev => [...prev, ...addedMessages]);
    setStep(s => s + 1);
  };

  const submitScore = async () => {
    setLoading(true);
    setHistory(prev => [...prev,
      { role: 'user', text: `直感: ${formData.intuition}点。これで審査よろしく！` },
      { role: 'bot', text: '了解！FastAPIのフル審査エンジンにデータを送っています... 🚀' }
    ]);

    try {
      const payload = {
        company_no:                   formData.company_no,
        company_name:                 formData.company_name,
        industry_major:               formData.industry_major,
        industry_sub:                 formData.industry_sub,
        main_bank:                    formData.main_bank,
        competitor:                   formData.competitor,
        num_competitors:              formData.num_competitors,
        deal_source:                  formData.deal_source,
        deal_occurrence:              formData.deal_occurrence,
        customer_type:                formData.customer_type,
        contract_type:                formData.contract_type,
        grade:                        formData.grade,
        nenshu:                       Number(formData.nenshu || 0),
        gross_profit:                 Number(formData.gross_profit || 0),
        op_profit:                    Number(formData.op_profit || 0),
        ord_profit:                   Number(formData.ord_profit || 0),
        net_income:                   Number(formData.net_income || 0),
        total_assets:                 Number(formData.total_assets || 1),
        net_assets:                   Number(formData.net_assets || 0),
        machines:                     Number(formData.machines || 0),
        other_assets:                 Number(formData.other_assets || 0),
        depreciation:                 Number(formData.depreciation || 0),
        dep_expense:                  Number(formData.dep_expense || 0),
        rent:                         Number(formData.rent || 0),
        rent_expense:                 Number(formData.rent_expense || 0),
        bank_credit:                  Number(formData.bank_credit || 0),
        lease_credit:                 Number(formData.lease_credit || 0),
        contracts:                    Number(formData.contracts || 0),
        acquisition_cost:             Number(formData.acquisition_cost || 0),
        lease_term:                   Number(formData.lease_term || 60),
        acceptance_year:              Number(formData.acceptance_year || new Date().getFullYear()),
        qual_corr_company_history:    formData.qual_corr_company_history,
        qual_corr_customer_stability: formData.qual_corr_customer_stability,
        qual_corr_repayment_history:  formData.qual_corr_repayment_history,
        qual_corr_business_future:    formData.qual_corr_business_future,
        qual_corr_equipment_purpose:  formData.qual_corr_equipment_purpose,
        qual_corr_main_bank:          formData.qual_corr_main_bank,
        passion_text:                 formData.passion_text,
        intuition:                    Number(formData.intuition),
      };

      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/score/full`, payload);

      setHistory(prev => [...prev, {
        role: 'humor',
        text: (
          <span>
            <b>🎉 審査完了！</b><br/>
            総合スコア: {res.data.score?.toFixed(1)}点<br/>
            判定: {res.data.hantei}<br/>
            借手スコア: {res.data.score_borrower?.toFixed(1)}点<br/><br/>
            詳細は「📋 審査・分析」タブから確認してね！
          </span>
        )
      }]);
    } catch (e) {
      setHistory(prev => [...prev, { role: 'humor', text: 'エラー発生！APIサーバーが立ち上がっているか確認してね。' }]);
    } finally {
      setLoading(false);
    }
  };

  const goBack = () => {
    if (step === 0) return;
    setStep(s => s - 1);
    setHistory(prev => {
      const nw = [...prev];
      nw.pop(); nw.pop();
      return nw;
    });
  };

  // 定性評価のオプション群
  const qualOpts = {
    qual_corr_company_history:    { label: "設立・経営年数",     opts: ["未選択","20年以上","10年〜20年","5年〜10年","3年〜5年","3年未満"] },
    qual_corr_customer_stability: { label: "顧客安定性",         opts: ["未選択","非常に安定（大口・長期）","安定（分散良好）","普通","やや不安定（集中あり）","不安定・依存大"] },
    qual_corr_repayment_history:  { label: "返済履歴",           opts: ["未選択","5年以上問題なし","3年以上問題なし","遅延少ない","遅延・リスケあり","問題あり・要確認"] },
    qual_corr_business_future:    { label: "事業将来性",         opts: ["未選択","有望（成長・ニーズ確実）","やや有望","普通","やや懸念","懸念（縮小・競争激化）"] },
    qual_corr_equipment_purpose:  { label: "設備目的",           opts: ["未選択","収益直結・受注必須","生産性向上・省力化","更新・維持・法定対応","やや不明確","不明確・要説明"] },
    qual_corr_main_bank:          { label: "メイン銀行関係",     opts: ["未選択","メイン先で取引良好・支援表明","メイン先","サブ扱い・取引あり","取引浅い・他社メイン","取引なし・不安"] },
  };

  const sel = "w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-sm font-bold text-[#1A1A2E] appearance-none outline-none focus:border-[#E8A838]";
  const inp = "w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-sm outline-none focus:border-[#E8A838]";
  const inpReq = "w-full bg-amber-50 border-2 border-[#E8A838] rounded-xl p-3 text-sm outline-none font-bold";
  const lbl = "text-[11px] font-black text-slate-500 mb-1 block";

  return (
    <div className="md:min-h-[calc(100vh-2rem)] flex items-center justify-center bg-slate-900 md:py-8 w-full">
      <div className="w-full h-[100dvh] md:max-w-[400px] md:h-[800px] bg-[#f4f1ec] md:rounded-[3rem] md:shadow-2xl overflow-hidden md:border-[12px] border-slate-800 relative flex flex-col">
        {/* スマホのノッチ（PC視聴時のみ） */}
        <div className="hidden md:flex absolute top-0 inset-x-0 mx-auto w-32 h-6 bg-slate-800 rounded-b-2xl z-20 justify-center items-center">
          <div className="w-12 h-1 bg-slate-900 rounded-full mt-1"></div>
        </div>

        {/* ヘッダー */}
        <div className="bg-gradient-to-r from-[#1A1A2E] to-[#2d2d4e] w-full pt-12 md:pt-10 pb-4 px-4 shadow flex justify-between items-center shrink-0 z-10 text-[#E8A838]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-white rounded-full flex justify-center items-center shadow-inner overflow-hidden border-2 border-[#E8A838]">
              <img src="https://api.dicebear.com/7.x/bottts/svg?seed=LeaseApp&backgroundColor=E8A838" />
            </div>
            <div>
              <h3 className="font-black text-sm tracking-widest uppercase">Lease-Wizard</h3>
              <p className="text-[10px] opacity-80 mt-0.5">Step {step + 1} / {STEPS.length}</p>
            </div>
          </div>
          <div className="text-[10px] font-bold bg-[#E8A838] text-slate-900 px-2 py-1 rounded-sm">
            {STEPS[step]}
          </div>
        </div>

        {/* 進行バー */}
        <div className="h-1 bg-slate-200 shrink-0">
          <div
            className="h-full bg-gradient-to-r from-orange-400 to-[#E8A838] transition-all duration-300"
            style={{ width: `${((step) / STEPS.length) * 100}%` }}
          />
        </div>

        {/* チャット履歴エリア */}
        <div ref={scrollRef} className="flex-1 w-full p-4 overflow-y-auto space-y-4 scrollbar-hide">
          {history.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'bot' && (
                <div className="bg-white border-2 border-[#1A1A2E] text-[#1A1A2E] rounded-2xl rounded-tl-none p-3 shadow-sm text-sm font-medium max-w-[85%] leading-relaxed">
                  {msg.text}
                </div>
              )}
              {msg.role === 'user' && (
                <div className="bg-[#1A1A2E] text-white rounded-2xl rounded-tr-none py-2 px-4 shadow-sm text-sm max-w-[85%] text-right leading-relaxed">
                  {msg.text}
                </div>
              )}
              {msg.role === 'humor' && (
                <div className="bg-[#FFF8E8] border border-[#E8A838] text-amber-900 rounded-xl py-3 px-4 shadow-sm text-sm w-full mx-2 my-2">
                  <div className="text-[10px] font-bold text-[#E8A838] mb-1">💬 リースくんのつぶやき</div>
                  <div>{msg.text}</div>
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border-2 border-[#1A1A2E] text-[#1A1A2E] rounded-2xl p-3 flex items-center gap-2">
                <Activity className="animate-spin w-4 h-4 text-orange-500" />
                <span className="text-sm">エンジン実行中...</span>
              </div>
            </div>
          )}
          <div className="h-2"></div>
        </div>

        {/* 下部フォームエリア */}
        {!loading && (
        <form onSubmit={handleNext} className="w-full bg-white border-t-2 border-[#1A1A2E] p-4 shrink-0 shadow-[0_-4px_15px_rgba(0,0,0,0.05)] rounded-t-2xl z-20">
          <div className="mb-4 space-y-3 max-h-[40vh] overflow-y-auto scrollbar-hide pb-2 px-1">

            {/* Step 0: 企業・業種 */}
            {step === 0 && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>企業番号</label>
                    <input type="text" name="company_no" value={formData.company_no} onChange={handleChange} placeholder="例: 123456" className={inp} />
                  </div>
                  <div>
                    <label className={lbl}>企業名</label>
                    <input type="text" name="company_name" value={formData.company_name} onChange={handleChange} placeholder="例: 株式会社○○" className={inp} />
                  </div>
                </div>
                <div className="relative">
                  <label className={lbl}>業種（大分類）</label>
                  <select name="industry_major" value={formData.industry_major} onChange={handleChange} className={sel}>
                    <option>A 農業，林業</option>
                    <option>B 漁業</option>
                    <option>C 鉱業，採石業，砂利採取業</option>
                    <option>D 建設業</option>
                    <option>E 製造業</option>
                    <option>F 電気・ガス・熱供給・水道業</option>
                    <option>G 情報通信業</option>
                    <option>H 運輸業，郵便業</option>
                    <option>I 卸売業，小売業</option>
                    <option>J 金融業，保険業</option>
                    <option>K 不動産業，物品賃貸業</option>
                    <option>L 学術研究，専門・技術サービス業</option>
                    <option>M 宿泊業，飲食サービス業</option>
                    <option>N 生活関連サービス業，娯楽業</option>
                    <option>O 教育，学習支援業</option>
                    <option>P 医療，福祉</option>
                    <option>Q 複合サービス事業</option>
                    <option>R サービス業（他に分類されないもの）</option>
                  </select>
                  <ChevronDown className="absolute right-3 top-8 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>
                <div className="relative">
                  <label className={lbl}>業種（中分類）</label>
                  <input type="text" name="industry_sub" value={formData.industry_sub} onChange={handleChange} placeholder="例: 06 総合工事業" className={inp} />
                </div>
              </>
            )}

            {/* Step 1: 取引・競合 */}
            {step === 1 && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>取引区分</label>
                    <select name="main_bank" value={formData.main_bank} onChange={handleChange} className={sel}>
                      <option>メイン先</option><option>非メイン先</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>顧客区分</label>
                    <select name="customer_type" value={formData.customer_type} onChange={handleChange} className={sel}>
                      <option>既存先</option><option>新規先</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>競合状況</label>
                    <select name="competitor" value={formData.competitor} onChange={handleChange} className={sel}>
                      <option>競合なし</option><option>競合あり</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>競合社数</label>
                    <select name="num_competitors" value={formData.num_competitors} onChange={handleChange} className={sel}>
                      <option>未入力</option><option>0社</option><option>1社</option><option>2社</option><option>3社以上</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className={lbl}>商談ソース</label>
                    <select name="deal_source" value={formData.deal_source} onChange={handleChange} className={sel}>
                      <option>銀行紹介</option><option>その他</option>
                    </select>
                  </div>
                  <div>
                    <label className={lbl}>発生経緯</label>
                    <select name="deal_occurrence" value={formData.deal_occurrence} onChange={handleChange} className={sel}>
                      <option>不明</option><option>指名</option><option>相見積もり</option>
                    </select>
                  </div>
                </div>
              </>
            )}

            {/* Step 2: 物件 */}
            {step === 2 && (
              <div>
                <label className={lbl}>物件選択</label>
                <select name="asset_name" value={formData.asset_name} onChange={handleChange} className={sel}>
                  <option>建設機械</option>
                  <option>IT・OA機器</option>
                  <option>医療機器</option>
                  <option>車両・運搬車</option>
                  <option>製造設備・工作機械</option>
                  <option>オフィス家具・内装</option>
                  <option>飲食店設備</option>
                  <option>太陽光・省エネ設備</option>
                  <option>その他・未選択</option>
                </select>
              </div>
            )}

            {/* Step 3: P/L */}
            {step === 3 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="number" name="nenshu" value={formData.nenshu} onChange={handleChange} placeholder="売上高 (千円) ※必須" className={inpReq} required />
                </div>
                <input type="number" name="gross_profit" value={formData.gross_profit} onChange={handleChange} placeholder="売上総利益 (千)" className={inp} />
                <input type="number" name="op_profit" value={formData.op_profit} onChange={handleChange} placeholder="営業利益 (千)" className={inp} />
                <input type="number" name="ord_profit" value={formData.ord_profit} onChange={handleChange} placeholder="経常利益 (千)" className={inp} />
                <input type="number" name="net_income" value={formData.net_income} onChange={handleChange} placeholder="当期純利益 (千)" className={inp} />
              </div>
            )}

            {/* Step 4: B/S */}
            {step === 4 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="number" name="total_assets" value={formData.total_assets} onChange={handleChange} placeholder="総資産 (千円) ※必須" className={inpReq} required />
                </div>
                <div className="col-span-2">
                  <input type="number" name="net_assets" value={formData.net_assets} onChange={handleChange} placeholder="純資産 (千円)" className={inp} />
                </div>
                <input type="number" name="machines" value={formData.machines} onChange={handleChange} placeholder="機械装置 (千)" className={inp} />
                <input type="number" name="other_assets" value={formData.other_assets} onChange={handleChange} placeholder="その他資産 (千)" className={inp} />
              </div>
            )}

            {/* Step 5: 経費 */}
            {step === 5 && (
              <div className="grid grid-cols-2 gap-2">
                <input type="number" name="depreciation" value={formData.depreciation} onChange={handleChange} placeholder="減価償却(資産)" className={inp} />
                <input type="number" name="dep_expense" value={formData.dep_expense} onChange={handleChange} placeholder="減価償却(経費)" className={inp} />
                <input type="number" name="rent" value={formData.rent} onChange={handleChange} placeholder="賃借料(資産)" className={inp} />
                <input type="number" name="rent_expense" value={formData.rent_expense} onChange={handleChange} placeholder="賃借料(経費)" className={inp} />
              </div>
            )}

            {/* Step 6: 信用 */}
            {step === 6 && (
              <div className="space-y-2">
                <select name="grade" value={formData.grade} onChange={handleChange} className={sel}>
                  <option>①1-3 (優良)</option>
                  <option>②4-6 (標準)</option>
                  <option>③要注意以下</option>
                  <option>④無格付</option>
                </select>
                <div className="grid grid-cols-3 gap-2">
                  <input type="number" name="contracts" value={formData.contracts} onChange={handleChange} placeholder="契約件数" className={inp} />
                  <input type="number" name="bank_credit" value={formData.bank_credit} onChange={handleChange} placeholder="銀行与信残" className={inp} />
                  <input type="number" name="lease_credit" value={formData.lease_credit} onChange={handleChange} placeholder="リース与信残" className={inp} />
                </div>
              </div>
            )}

            {/* Step 7: 契約 */}
            {step === 7 && (
              <div className="grid grid-cols-2 gap-2">
                <div className="col-span-2">
                  <input type="number" name="acquisition_cost" value={formData.acquisition_cost} onChange={handleChange} placeholder="取得価格 (千円) ※必須" className={inpReq} required />
                </div>
                <div>
                  <label className={lbl}>契約種類</label>
                  <select name="contract_type" value={formData.contract_type} onChange={handleChange} className={sel}>
                    <option>一般</option><option>自動車</option>
                  </select>
                </div>
                <div>
                  <label className={lbl}>期間(月)</label>
                  <input type="number" name="lease_term" value={formData.lease_term} onChange={handleChange} className={inp} />
                </div>
                <div className="col-span-2">
                  <label className={lbl}>検収年(西暦)</label>
                  <input type="number" name="acceptance_year" value={formData.acceptance_year} onChange={handleChange} className={inp} />
                </div>
              </div>
            )}

            {/* Step 8: 定性(6項目) + パッション */}
            {step === 8 && (
              <div className="space-y-2">
                {(Object.keys(qualOpts) as Array<keyof typeof qualOpts>).map(k => (
                  <div key={k} className="flex flex-col">
                    <label className="text-[10px] font-bold text-slate-400">{qualOpts[k].label}</label>
                    <select name={k} value={formData[k as keyof typeof formData] as string} onChange={handleChange} className="w-full bg-slate-50 border border-slate-200 rounded-md p-1.5 text-xs outline-none">
                      {qualOpts[k].opts.map(o => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
                <div>
                  <label className="text-[10px] font-bold text-slate-400">特記事項・アピールポイント（任意）</label>
                  <textarea name="passion_text" value={formData.passion_text} onChange={handleChange} rows={2} placeholder="担当者コメントがあれば..." className="w-full bg-slate-50 border border-slate-200 rounded-md p-1.5 text-xs outline-none resize-none" />
                </div>
              </div>
            )}

            {/* Step 9: 直感スコア */}
            {step === 9 && (
              <div className="flex flex-col items-center">
                <p className="text-xs text-slate-500 font-bold mb-3">担当者の直感スコア（1:懸念〜5:確信）</p>
                <div className="flex justify-center gap-2">
                  {[1,2,3,4,5].map(v => (
                    <button type="button" key={v} onClick={() => setFormData({...formData, intuition: v})}
                      className={`w-12 h-12 flex items-center justify-center font-black rounded-full border-2 transition-all ${formData.intuition === v ? 'bg-[#E8A838] border-[#E8A838] text-white scale-110' : 'border-slate-200 text-slate-400 bg-white hover:border-[#E8A838]'}`}>
                      {v}
                    </button>
                  ))}
                </div>
              </div>
            )}

          </div>

          <div className="flex gap-2 mt-2">
            <button
              type="button" onClick={goBack} disabled={step === 0}
              className="w-12 h-12 flex items-center justify-center rounded-xl bg-slate-100 text-slate-600 disabled:opacity-30">
              <ArrowLeft className="w-5 h-5" />
            </button>
            <button
              type="submit"
              className="flex-1 h-12 flex items-center justify-center rounded-xl bg-[#1A1A2E] text-white font-bold tracking-wide shadow-[0_4px_0_#0f0f1c] active:shadow-none active:translate-y-1 transition-all">
              {step >= STEPS.length - 1 ? '審査実行 🚀' : '次へ進む'}
            </button>
          </div>
        </form>
        )}

      </div>
    </div>
  );
}
