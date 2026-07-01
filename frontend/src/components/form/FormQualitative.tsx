import React, { useEffect, useState } from 'react';
import { Mic, MicOff, Lightbulb, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { ScoringFormData } from '../../types';
import SliderInput from '../SliderInput';
import { focusNextScreeningNumber, parseHumanNumberInput } from '../../lib/numberInput';

// REV-064: 物件ごとの法定耐用年数 / REV-068: 推奨業種
const ASSET_INFO: Record<string, { usefulLifeYears: number; industryMajor: string }> = {
  '建設機械':           { usefulLifeYears: 6,  industryMajor: 'D 建設業' },
  'IT・OA機器':         { usefulLifeYears: 4,  industryMajor: 'G 情報通信業' },
  '医療機器':           { usefulLifeYears: 6,  industryMajor: 'P 医療・福祉' },
  '車両・運搬車':       { usefulLifeYears: 4,  industryMajor: 'H 運輸業・郵便業' },
  '製造設備・工作機械': { usefulLifeYears: 10, industryMajor: 'E 製造業' },
  'オフィス家具・内装': { usefulLifeYears: 8,  industryMajor: 'R サービス業(他に分類されないもの)' },
  '飲食店設備':         { usefulLifeYears: 8,  industryMajor: 'M 宿泊業・飲食サービス業' },
  '太陽光・省エネ設備': { usefulLifeYears: 17, industryMajor: 'D 建設業' },
};

interface FormQualitativeProps {
  data: ScoringFormData;
  onChange: (name: string, value: string | number | string[]) => void;
}

interface QualitativeItem {
  id: string;
  label: string;
  weight: number;
  options: [number, string][];
}

type SpeechRecognitionConstructor = new () => {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
};

interface SpeechRecognitionEvent {
  results: {
    [index: number]: {
      [index: number]: {
        transcript: string;
      };
    };
  };
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

type SpeechRecognitionWindow = Window & {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
};

const acquisitionQuickValues = [
  { label: '10', value: 10 },
  { label: '30', value: 30 },
  { label: '50', value: 50 },
  { label: '1億', value: 100 },
];

const leaseTermQuickValues = [
  { label: '36ヶ月', value: 36 },
  { label: '48ヶ月', value: 48 },
  { label: '60ヶ月', value: 60 },
  { label: '72ヶ月', value: 72 },
];

const currentYear = new Date().getFullYear();
const acceptanceYearQuickValues = [currentYear - 1, currentYear, currentYear + 1];

const isSpeechSupported = () => {
  if (typeof window === 'undefined') return false;
  const speechWindow = window as SpeechRecognitionWindow;
  return Boolean(speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition);
};

export default function FormQualitative({ data, onChange }: FormQualitativeProps) {
  const [qualItems, setQualItems] = useState<QualitativeItem[]>([]);
  const [assetItems, setAssetItems] = useState<string[]>([]);
  const [isListening, setIsListening] = useState(false);
  const [speechError, setSpeechError] = useState('');
  const [speechSupported, setSpeechSupported] = useState(isSpeechSupported);

  // REV-064/068/050: 物件情報から計算する派生値
  const assetInfo = ASSET_INFO[data.asset_name] ?? null;
  const leaseWarning = (() => {
    const t = data.lease_term;
    if (!t || t <= 0) return null;
    if (t < 12) return 'リース期間は通常12ヶ月以上です';
    if (t > 84) return '84ヶ月（7年）超は長期すぎる場合があります';
    if (assetInfo && t > assetInfo.usefulLifeYears * 12 * 1.2)
      return `リース期間が法定耐用年数（${assetInfo.usefulLifeYears}年）の120%を超過しています`;
    return null;
  })();
  const showIndustrySuggestion = !!(assetInfo && data.industry_major !== assetInfo.industryMajor);
  const assetClarityItems = [
    { key: 'asset_name', label: '物件名', filled: Boolean(data.asset_name) },
    { key: 'asset_detail', label: '型式・仕様', filled: Boolean(data.asset_detail?.trim()) },
    { key: 'asset_purpose', label: '導入目的', filled: Boolean(data.asset_purpose?.trim()) },
    { key: 'asset_location', label: '設置場所', filled: Boolean(data.asset_location?.trim()) },
    { key: 'asset_evidence_level', label: '確認資料', filled: data.asset_evidence_level !== '未確認' },
  ];
  const assetClarityCount = assetClarityItems.filter(item => item.filled).length;

  // マスターデータの取得
  useEffect(() => {
    const fetchMaster = async () => {
      try {
        const [qualRes, assetRes] = await Promise.all([
          fetch(`/api/master/qualitative`),
          fetch(`/api/master/assets`)
        ]);
        
        if (qualRes.ok) {
          const qualData = await qualRes.json();
          setQualItems(qualData.items || []);
        }
        
        if (assetRes.ok) {
          const assetData = await assetRes.json();
          // lease_assets.json の構造に合わせて抽出
          const items = assetData.items?.map((it: { name: string }) => it.name) || [];
          setAssetItems(items);
        }
      } catch (err) {
        console.error("Failed to fetch qual/asset master:", err);
      }
    };
    fetchMaster();
  }, []);
  
  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement | HTMLTextAreaElement | HTMLInputElement>) => {
    onChange(e.target.name, e.target.value);
  };
  
  const handleNumber = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value.trim() === '') {
      onChange(e.target.name, 0);
      return;
    }
    const parsed = parseHumanNumberInput(value);
    if (parsed !== null) onChange(e.target.name, parsed);
  };

  const displayNumber = (value: number | undefined | null): number | "" => (
    Number(value || 0) === 0 ? "" : Number(value)
  );

  const handleNumberFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    e.currentTarget.select();
  };

  const handleNumberKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    focusNextScreeningNumber(e.currentTarget);
  };

  const quickButtonClass = (active: boolean) => (
    `rounded-lg border px-2.5 py-1 text-[11px] font-black transition-colors ${
      active
        ? 'border-indigo-300 bg-indigo-50 text-indigo-700'
        : 'border-slate-200 bg-white text-slate-500 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700'
    }`
  );
  
  const handleSlider = (name: string, value: number) => {
    onChange(name, value);
  }

  const startVoiceInput = () => {
    const speechWindow = window as SpeechRecognitionWindow;
    const Recognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition;

    if (!Recognition) {
      setSpeechError('このブラウザは音声入力に未対応です。');
      setSpeechSupported(false);
      return;
    }

    setSpeechError('');
    setIsListening(true);

    const recognition = new Recognition();
    recognition.lang = 'ja-JP';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event) => {
      const transcript = event.results[0]?.[0]?.transcript?.trim();
      if (!transcript) return;

      const currentMemo = data.passion_text?.trim();
      const nextMemo = currentMemo ? `${currentMemo}\n${transcript}` : transcript;
      onChange('passion_text', nextMemo);
    };

    recognition.onerror = (event) => {
      const message = event.error === 'not-allowed'
        ? 'マイクの使用が許可されませんでした。ブラウザの権限設定を確認してください。'
        : '音声入力でエラーが発生しました。もう一度お試しください。';
      setSpeechError(message);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    try {
      recognition.start();
    } catch {
      setSpeechError('音声入力を開始できませんでした。もう一度お試しください。');
      setIsListening(false);
    }
  };

  return (
    <div className="space-y-6">
      
      {/* 物件情報 */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-indigo-500">🏢</span> リース物件情報・契約条件
        </h3>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-5">
          <div className="space-y-1 lg:col-span-2">
            {/* REV-064: 法定耐用年数バッジ */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-bold text-slate-600 block">対象物件</label>
              {assetInfo && (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 border border-indigo-200">
                  耐用{assetInfo.usefulLifeYears}年
                </span>
              )}
            </div>
            <select name="asset_name" value={data.asset_name} onChange={handleSelect} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 h-[46px]">
              <option value="">（選択してください）</option>
              {assetItems.map(name => <option key={name} value={name}>{name}</option>)}
              {assetItems.length === 0 && (
                <>
                  <option>建設機械</option>
                  <option>IT・OA機器</option>
                  <option>医療機器</option>
                  <option>車両・運搬車</option>
                  <option>製造設備・工作機械</option>
                </>
              )}
            </select>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">物件取得価額 (百万円)</label>
            <input type="text" inputMode="decimal" data-screening-number="true" name="acquisition_cost" value={displayNumber(data.acquisition_cost)} step="0.1" onChange={handleNumber} onFocus={handleNumberFocus} onKeyDown={handleNumberKeyDown} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
            <div className="flex flex-wrap gap-1.5">
              {acquisitionQuickValues.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  onClick={() => onChange('acquisition_cost', item.value)}
                  className={quickButtonClass(Number(data.acquisition_cost || 0) === item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">購入価格または見積額（百万円単位）</p>
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">リース期間 (月)</label>
            <input
              type="text"
              inputMode="decimal"
              data-screening-number="true"
              name="lease_term"
              value={displayNumber(data.lease_term)}
              onChange={handleNumber}
              onFocus={handleNumberFocus}
              onKeyDown={handleNumberKeyDown}
              className={`w-full bg-slate-50 border rounded-lg p-2.5 outline-none focus:ring-2 text-right h-[46px] ${
                leaseWarning ? 'border-amber-400 focus:ring-amber-300' : 'border-slate-300 focus:ring-blue-500'
              }`}
            />
            <div className="flex flex-wrap gap-1.5">
              {leaseTermQuickValues.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  onClick={() => onChange('lease_term', item.value)}
                  className={quickButtonClass(Number(data.lease_term || 0) === item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {/* REV-050: リース期間バリデーション警告 */}
            {leaseWarning && (
              <p className="flex items-center gap-1 text-[10px] font-bold text-amber-600 mt-0.5">
                <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                {leaseWarning}
              </p>
            )}
          </div>
          <div className="space-y-1">
            <label className="text-sm font-bold text-slate-600 block">検収年 (西暦)</label>
            <input type="text" inputMode="numeric" data-screening-number="true" name="acceptance_year" value={displayNumber(data.acceptance_year)} onChange={handleNumber} onFocus={handleNumberFocus} onKeyDown={handleNumberKeyDown} className="w-full bg-slate-50 border border-slate-300 rounded-lg p-2.5 outline-none focus:ring-2 focus:ring-blue-500 text-right h-[46px]" />
            <div className="flex flex-wrap gap-1.5">
              {acceptanceYearQuickValues.map((year) => (
                <button
                  key={year}
                  type="button"
                  onClick={() => onChange('acceptance_year', year)}
                  className={quickButtonClass(Number(data.acceptance_year || 0) === year)}
                >
                  {year}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">物件を受け取った年または予定年（西暦4桁）</p>
          </div>
        </div>

        {/* REV-068: 推奨業種バナー */}
        {showIndustrySuggestion && assetInfo && (
          <div className="flex items-start gap-2 p-3 rounded-xl bg-indigo-50 border border-indigo-200 text-sm">
            <Lightbulb className="w-4 h-4 text-indigo-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="font-bold text-indigo-800 text-[12px]">業種の確認をおすすめします</p>
              <p className="text-[11px] text-indigo-700 mt-0.5">
                <span className="font-bold">{data.asset_name}</span> の推奨業種は
                <span className="font-bold">「{assetInfo.industryMajor}」</span> です（現在: {data.industry_major}）。
              </p>
            </div>
            <button
              type="button"
              onClick={() => onChange('industry_major', assetInfo.industryMajor)}
              className="text-[11px] font-bold px-2.5 py-1 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition whitespace-nowrap"
            >
              適用
            </button>
          </div>
        )}

        <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-black text-slate-800">対象物件の明確化</p>
              <p className="text-[11px] font-bold text-slate-500">稟議で「何を・どこで・何のために使うか」を説明するための確認欄です。</p>
            </div>
            <div className={`rounded-full px-3 py-1 text-[11px] font-black ${
              assetClarityCount >= 4 ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
            }`}>
              明確化 {assetClarityCount}/5
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-600 block">型式・メーカー・仕様</label>
              <input
                type="text"
                name="asset_detail"
                value={data.asset_detail || ''}
                onChange={handleSelect}
                placeholder="例）〇〇製 5軸マシニングセンタ 型式ABC-100"
                className="w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-600 block">設置場所・使用場所</label>
              <input
                type="text"
                name="asset_location"
                value={data.asset_location || ''}
                onChange={handleSelect}
                placeholder="例）本社工場 第2ライン / 新店舗厨房"
                className="w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <div className="space-y-1 md:col-span-2">
              <label className="text-xs font-bold text-slate-600 block">導入目的・用途</label>
              <input
                type="text"
                name="asset_purpose"
                value={data.asset_purpose || ''}
                onChange={handleSelect}
                placeholder="例）受注増対応、外注加工の内製化、配送能力増強"
                className="w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-bold text-slate-600 block">確認資料</label>
              <select
                name="asset_evidence_level"
                value={data.asset_evidence_level || '未確認'}
                onChange={handleSelect}
                className="w-full rounded-lg border border-slate-300 bg-white p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 h-[42px]"
              >
                <option>未確認</option>
                <option>見積あり</option>
                <option>見積・型式確認済</option>
                <option>見積・型式・中古相場確認済</option>
              </select>
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {assetClarityItems.map((item) => (
              <span
                key={item.key}
                className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-black ${
                  item.filled ? 'bg-emerald-100 text-emerald-700' : 'bg-white text-slate-400 border border-slate-200'
                }`}
              >
                {item.filled ? <CheckCircle2 className="h-3 w-3" /> : <span className="h-3 w-3 rounded-full border border-slate-300" />}
                {item.label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* 定性評価（6大項目） */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200">
        <h3 className="text-lg font-bold text-slate-800 border-b border-slate-100 pb-3 mb-4 flex items-center gap-2">
          <span className="text-rose-500">📝</span> 審査定性評価スコアリング
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          {qualItems.map(item => (
            <div key={item.id} className="space-y-1">
              <label className="text-[13px] font-bold text-slate-600 block">
                {item.label} <span className="text-[10px] font-normal text-slate-400">(重み{item.weight}%)</span>
              </label>
              <select 
                name={`qual_corr_${item.id}`} 
                value={data[`qual_corr_${item.id}` as keyof ScoringFormData] as string || "未選択"} 
                onChange={handleSelect} 
                className="w-full bg-slate-50 border border-slate-200 rounded-lg p-2.5 text-sm outline-none focus:border-blue-500 focus:bg-white transition h-[42px]"
              >
                <option value="未選択">未選択</option>
                {item.options.map(([_score, label]) => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </select>
            </div>
          ))}
          
          {qualItems.length === 0 && <p className="text-sm text-slate-400">定性評価項目を読込中...</p>}
        </div>

        <div className="mt-8 border-t border-slate-100 pt-6">
          <label className="text-sm font-bold text-slate-600 block mb-1">🎈 直感スコア (1:懸念あり 〜 5:確信あり)</label>
          <p className="text-[10px] text-slate-400 mb-2">訪問後の現場感覚を数値化。AIの補正係数に反映されます</p>
          <SliderInput label="" name="intuition" value={data.intuition} min={1} max={5} step={1} unit="点" onChange={handleSlider} />
        </div>

        <div className="mt-6 space-y-2">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <label htmlFor="passion_text" className="text-sm font-bold text-slate-600">
              現場メモ
            </label>
            <button
              type="button"
              onClick={startVoiceInput}
              disabled={!speechSupported || isListening}
              className={`inline-flex h-9 items-center justify-center gap-2 rounded-lg px-3 text-xs font-bold transition ${
                !speechSupported
                  ? 'cursor-not-allowed bg-slate-100 text-slate-400'
                  : isListening
                    ? 'bg-rose-100 text-rose-700'
                    : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {speechSupported ? (
                <Mic className="h-4 w-4" aria-hidden="true" />
              ) : (
                <MicOff className="h-4 w-4" aria-hidden="true" />
              )}
              {isListening ? '録音中...' : '音声入力'}
            </button>
          </div>
          <textarea
            id="passion_text"
            name="passion_text"
            value={data.passion_text}
            onChange={handleSelect}
            rows={4}
            placeholder="例: 社長は設備更新で人手不足を補いたい意向。工場は整理されているが、古い機械が多い。投資目的は明確だが、受注先の偏りに少し違和感あり。"
            className="w-full resize-y rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700 outline-none transition focus:border-indigo-400 focus:bg-white focus:ring-2 focus:ring-indigo-100"
          />
          {!speechSupported && (
            <p className="text-xs font-medium text-slate-500">このブラウザは音声入力に未対応です。</p>
          )}
          {speechError && (
            <p className="text-xs font-bold text-rose-600">{speechError}</p>
          )}
        </div>
      </div>

    </div>
  );
}
