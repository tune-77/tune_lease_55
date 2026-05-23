import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify';
import { Activity, AlertTriangle, CheckCircle2, FileText, HelpCircle, MessageSquare, Target, Users } from 'lucide-react';
import type { ScoringFormData } from '@/types';

interface GunshiAdviceProps {
  score: number;
  industry_major: string;
  formData: GunshiFormData;
  onChatLoaded?: (text: string) => void;
}

type GunshiFormData = ScoringFormData;

type ChatMessage = {
  role: 'user' | 'assistant';
  text: string;
  meta?: string;
};

type SimilarCase = {
  id?: number | string;
  name: string;
  industry: string;
  score: number;
  status: string;
  similarity: number;
  equity: number;
  revenue: number;
  conditions: string[];
};

type StrategyCards = {
  headline?: string;
  stance?: string;
  case_facts?: string[];
  risk_cards?: string[];
  today_moves?: string[];
  competitor_moves?: string[];
  questions_to_ask?: string[];
  customer_one_liners?: string[];
  ringi_lines?: string[];
  badges?: string[];
  bayes_factors?: BayesFactor[];
  disclaimer?: string;
};

type BayesFactor = {
  label?: string;
  detail?: string;
  delta_pct?: number;
  direction?: 'base' | 'up' | 'down' | 'flat';
};

type WebHit = {
  domain?: string;
  title?: string;
};

type GunshiChatResponse = {
  reply?: string;
  chat_text?: string;
  saved?: boolean;
  save_reason?: string;
  web_hits?: WebHit[];
  wiki_saved?: boolean;
  weekly_saved?: boolean;
};

type GunshiStreamChunk = {
  type?: 'bayes' | 'phrases' | 'strategy_cards' | 'stream' | 'done';
  prior?: number;
  posterior?: number;
  factors?: BayesFactor[];
  cards?: StrategyCards;
  delta?: string;
};

export default function GunshiAdvice({ score, industry_major, formData, onChatLoaded }: GunshiAdviceProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [humorMode, setHumorMode] = useState<'yanami' | 'standard'>('yanami');
  const [useWeb, setUseWeb] = useState(true);
  const [advisorMode, setAdvisorMode] = useState<'gunshi' | 'chat'>('gunshi');
  const [statusText, setStatusText] = useState('');
  const [similarCases, setSimilarCases] = useState<SimilarCase[]>([]);
  const [similarOpen, setSimilarOpen] = useState(true);
  const [prior, setPrior] = useState<number | null>(null);
  const [posterior, setPosterior] = useState<number | null>(null);
  const [bayesFactors, setBayesFactors] = useState<BayesFactor[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [strategyCards, setStrategyCards] = useState<StrategyCards | null>(null);
  const [strategyOpen, setStrategyOpen] = useState(true);
  const initialFetchKeyRef = useRef<string>("");
  const similarFetchKeyRef = useRef<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, streamingText]);

  const buildStreamPayload = () => {
    const subsidyText = [
      formData.industry_detail,
      formData.passion_text,
      formData.asset_name,
    ].join(" ");
    const equityRatio = Number(formData.total_assets) > 0
      ? (Number(formData.net_assets) / Number(formData.total_assets)) * 100
      : 0;
    return {
      industry_cat: industry_major || "",
      industry_sub: formData.industry_sub || "",
      score,
      resale_eval: "B",
      repeat_count: Number(formData.contracts) || 0,
      subsidy_flag: /補助金|助成金|ものづくり|省力化/.test(subsidyText),
      bank_support: formData.deal_source === "銀行紹介" || formData.main_bank === "メイン先",
      intuition_score: Number(formData.intuition) || 50,
      company_name: formData.company_name || "",
      asset_name: formData.asset_name || "",
      acquisition_cost: Number(formData.acquisition_cost) || 0,
      lease_term: Number(formData.lease_term) || 0,
      contract_type: formData.contract_type || "",
      main_bank: formData.main_bank || "",
      competitor: formData.competitor || "",
      competitor_rate: Number(formData.competitor_rate) || null,
      deal_source: formData.deal_source || "",
      customer_type: formData.customer_type || "",
      nenshu: Number(formData.nenshu) || 0,
      op_profit: Number(formData.op_profit) || 0,
      equity_ratio: equityRatio,
      bank_credit: Number(formData.bank_credit) || 0,
      lease_credit: Number(formData.lease_credit) || 0,
    };
  };

  const buildPayload = (message = "", history: ChatMessage[] = chatHistory) => {
    const subsidyText = [
      formData.industry_detail,
      formData.passion_text,
      formData.asset_name,
    ].join(" ");
    return {
      score,
      industry_major,
      asset_name: formData.asset_name || "",
      resale: "標準",
      repeat_cnt: 1,
      subsidy: /補助金|助成金|ものづくり|省力化/.test(subsidyText),
      bank: formData.deal_source === "銀行紹介" || formData.main_bank === "メイン先",
      intuition: formData.intuition || 50,
      posterior: score > 0 ? score / 100 : 0.5,
      message,
      history,
      humor_style: humorMode === 'yanami' ? 'yanami' : 'standard',
      use_web: useWeb,
      use_obsidian: true,
      mode: advisorMode,
    };
  };

  const buildResponseMeta = (data: GunshiChatResponse) => {
    const metaParts: string[] = [];
    if (data.saved) metaParts.push('Obsidianへ自動保存しました');
    else if (data.save_reason) metaParts.push(`保存なし: ${data.save_reason}`);
    if (Array.isArray(data.web_hits) && data.web_hits.length > 0) {
      const sources = data.web_hits
        .slice(0, 2)
        .map((h) => h.domain || h.title || 'web')
        .filter(Boolean)
        .join(' / ');
      metaParts.push(`Web参照: ${data.web_hits.length}件${sources ? ' (' + sources + ')' : ''}`);
    }
    if (data.wiki_saved) metaParts.push('Wikiへ自動保存しました');
    if (data.weekly_saved) metaParts.push('週次レビューへ保存しました');
    return metaParts.join(' | ');
  };

  const updateStatus = (data: GunshiChatResponse) => {
    if (data.web_hits?.length) setStatusText('回答しました。Web参照あり。');
    else if (data.saved) setStatusText('必要なメモだけObsidianへ保存しました。');
    else setStatusText('回答しました。');
  };

  // 初期フェッチ専用: SSEストリーミング (/api/gunshi/stream)
  const fetchStreamChat = async (displayHistory: ChatMessage[]) => {
    setLoading(true);
    setStreamingText('');
    setStrategyCards(null);
    setPrior(null);
    setPosterior(null);
    setBayesFactors([]);
    setStatusText('AIが考えています...');

    try {
      const payload = buildStreamPayload();
      const response = await fetch('/api/gunshi/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullText = '';
      let finished = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const rawData = line.slice(6).trim();
          if (!rawData) continue;
          try {
            const chunk = JSON.parse(rawData) as GunshiStreamChunk;
            if (chunk.type === 'bayes') {
              setPrior(chunk.prior ?? null);
              setPosterior(chunk.posterior ?? null);
              setBayesFactors(chunk.factors || []);
            } else if (chunk.type === 'strategy_cards') {
              setStrategyCards(chunk.cards || null);
              if (chunk.cards?.bayes_factors?.length) {
                setBayesFactors(chunk.cards.bayes_factors);
              }
            } else if (chunk.type === 'stream' && chunk.delta) {
              fullText += chunk.delta;
              setStreamingText(fullText);
            } else if (chunk.type === 'done') {
              finished = true;
              setChatHistory([...displayHistory, { role: 'assistant', text: fullText }]);
              setStreamingText('');
              setStatusText('回答しました。');
              if (onChatLoaded) onChatLoaded(fullText);
            }
          } catch {
            // Ignore malformed SSE keepalive fragments.
          }
        }
      }

      // done イベントなしで終了した場合もhistoryに追加
      if (!finished && fullText) {
        setChatHistory([...displayHistory, { role: 'assistant', text: fullText }]);
        setStreamingText('');
        setStatusText('回答しました。');
        if (onChatLoaded) onChatLoaded(fullText);
      }
    } catch (err) {
      console.error("Failed to stream gunshi", err);
      const errText = "【通信エラー】軍師からの戦略を受信できませんでした。";
      setChatHistory([...displayHistory, { role: 'assistant', text: errText }]);
      setStreamingText('');
      setStatusText('通信エラーが発生しました。');
      if (onChatLoaded) onChatLoaded(errText);
    } finally {
      setLoading(false);
    }
  };

  // 追加質問: ノンストリーミング（history/message/mode対応）
  const fetchChat = async (
    message = "",
    displayHistory: ChatMessage[] = chatHistory,
    requestHistory: ChatMessage[] = displayHistory
  ) => {
    setLoading(true);
    setStatusText('AIが考えています...');
    try {
      const payload = buildPayload(message, requestHistory);
      const res = await axios.post(`/api/gunshi/chat`, payload);
      const data = res.data;
      const fetchedText = data.reply || data.chat_text || '';
      const meta = buildResponseMeta(data);
      setChatHistory([...displayHistory, { role: 'assistant', text: fetchedText, meta }]);
      updateStatus(data);
      if (onChatLoaded) {
        onChatLoaded(fetchedText);
      }
    } catch (err) {
      console.error("Failed to fetch gunshi chat", err);
      const errText = "【通信エラー】軍師からの戦略を受信できませんでした。";
      setChatHistory([...displayHistory, { role: 'assistant', text: errText }]);
      setStatusText('通信エラーが発生しました。');
      if (onChatLoaded) onChatLoaded(errText);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (score === 0) return;
    const fetchKey = [
      score,
      industry_major,
      formData.industry_sub || "",
      formData.asset_name || "",
      formData.acquisition_cost || 0,
      formData.lease_term || 0,
      formData.contract_type || "",
      formData.main_bank || "",
      formData.competitor || "",
      formData.competitor_rate || "",
      formData.deal_source || "",
      formData.customer_type || "",
      formData.nenshu || 0,
      formData.op_profit || 0,
      formData.net_assets || 0,
      formData.total_assets || 0,
      formData.bank_credit || 0,
      formData.lease_credit || 0,
    ].join(":");
    if (initialFetchKeyRef.current === fetchKey) return;
    initialFetchKeyRef.current = fetchKey;

    const initialQuestion = `この案件（スコア ${score.toFixed(1)}点、${industry_major || "指定なし"}）の稟議を通すための「逆転戦略」を教えてくれ！`;
    const nextHistory: ChatMessage[] = [{ role: 'user', text: initialQuestion }];
    setChatHistory(nextHistory);
    fetchStreamChat(nextHistory);
  }, [score, industry_major, formData]);

  useEffect(() => {
    if (score === 0) return;
    const sigKey = [
      formData.nenshu,
      formData.op_profit,
      formData.net_assets,
      formData.total_assets,
      formData.bank_credit,
      formData.lease_credit,
      formData.industry_sub,
      industry_major,
    ].join(":");
    if (similarFetchKeyRef.current === sigKey) return;
    similarFetchKeyRef.current = sigKey;

    (async () => {
      try {
        const res = await axios.post(`/api/similar/inline`, {
          nenshu: Number(formData.nenshu) || 0,
          op_profit: Number(formData.op_profit) || 0,
          equity_ratio: Number(formData.total_assets) > 0
            ? (Number(formData.net_assets) / Number(formData.total_assets)) * 100
            : 0,
          bank_credit: Number(formData.bank_credit) || 0,
          lease_credit: Number(formData.lease_credit) || 0,
          industry_sub: formData.industry_sub || "",
          industry_major: industry_major || "",
          max_count: 3,
        });
        setSimilarCases(res.data?.similar_cases || []);
      } catch (err) {
        console.error("Failed to fetch similar cases", err);
        setSimilarCases([]);
      }
    })();
  }, [score, industry_major, formData]);

  const handleSubmit = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) return;

    const nextHistory: ChatMessage[] = [...chatHistory, { role: 'user', text: trimmedQuestion }];
    setQuestion("");
    setChatHistory(nextHistory);
    await fetchChat(trimmedQuestion, nextHistory, chatHistory);
  };

  const renderMarkdown = (text: string) => {
    let parsedText = text;
    parsedText = parsedText.replace(/### (.*?)(\n|$)/g, '<h4 class="font-bold text-base text-amber-700 mt-5 border-b border-amber-200 pb-1 mb-2">$1</h4>\n');
    parsedText = parsedText.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-slate-800">$1</strong>');
    return parsedText;
  };

  const renderActionList = (items: string[] | undefined, tone: 'amber' | 'red' | 'blue' | 'emerald' | 'slate' = 'slate') => {
    const toneClass = {
      amber: 'border-amber-200 bg-amber-50 text-amber-900',
      red: 'border-red-200 bg-red-50 text-red-900',
      blue: 'border-blue-200 bg-blue-50 text-blue-900',
      emerald: 'border-emerald-200 bg-emerald-50 text-emerald-900',
      slate: 'border-slate-200 bg-slate-50 text-slate-700',
    }[tone];
    return (
      <div className="space-y-1.5">
        {(items || []).slice(0, 4).map((item, i) => (
          <div key={`${tone}-${i}`} className={`rounded-lg border px-2.5 py-2 text-[11px] leading-4 font-bold ${toneClass}`}>
            {item}
          </div>
        ))}
      </div>
    );
  };

  const renderBayesFactors = () => {
    if (!bayesFactors.length) return null;
    const factorClass = (direction?: BayesFactor['direction']) => {
      if (direction === 'up') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
      if (direction === 'down') return 'border-red-200 bg-red-50 text-red-800';
      if (direction === 'base') return 'border-blue-200 bg-blue-50 text-blue-800';
      return 'border-slate-200 bg-slate-50 text-slate-700';
    };
    const deltaLabel = (factor: BayesFactor) => {
      if (factor.direction === 'base') return '基準';
      const value = Number(factor.delta_pct || 0);
      if (value > 0) return `+${value.toFixed(1)}pt`;
      if (value < 0) return `${value.toFixed(1)}pt`;
      return '±0pt';
    };

    return (
      <div className="mt-3 grid grid-cols-1 gap-1.5">
        {bayesFactors.slice(0, 7).map((factor, i) => (
          <div key={`${factor.label || 'factor'}-${i}`} className={`rounded-lg border px-2.5 py-2 ${factorClass(factor.direction)}`}>
            <div className="flex items-start justify-between gap-2">
              <span className="text-[11px] font-black leading-4">{factor.label || '要因'}</span>
              <span className="shrink-0 text-[10px] font-black">{deltaLabel(factor)}</span>
            </div>
            {factor.detail && (
              <div className="mt-0.5 text-[10px] leading-4 font-medium opacity-80">
                {factor.detail}
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const priorPct = prior !== null ? Math.round(prior * 100) : null;
  const posteriorPct = posterior !== null ? Math.round(posterior * 100) : null;

  return (
    <div className="sticky top-24 h-[calc(100vh-8rem)] bg-[#f8fafc] rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 flex flex-col overflow-hidden">

      {/* ヘッダー */}
      <div className="bg-gradient-to-r from-[#172554] to-[#1e3a8a] text-white p-4 shrink-0 shadow-md z-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-amber-500 border-2 border-white flex justify-center items-center font-black text-xl shadow-inner overflow-hidden">
            <span className="text-2xl mt-1">🏯</span>
          </div>
          <div>
            <h3 className="font-bold text-sm tracking-wide">審査軍師 (Gemini 連動型)</h3>
            <p className="text-[10px] text-blue-200 font-medium">BNベースの戦略提案AI</p>
          </div>
        </div>
      </div>

      <div className="bg-white border-b border-slate-100 px-4 py-3 shrink-0">
        <div className="flex flex-wrap items-center gap-2 mb-2">
          <span className="text-[11px] font-bold text-slate-500 mr-1">モード</span>
          <button
            type="button"
            onClick={() => setAdvisorMode('gunshi')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              advisorMode === 'gunshi'
                ? 'bg-amber-500 text-white border-amber-500 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
            title="案件向け戦略アドバイス（軍師ロジック）"
          >
            🏯 戦略
          </button>
          <button
            type="button"
            onClick={() => setAdvisorMode('chat')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              advisorMode === 'chat'
                ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
            title="Flask版AIチャット相当の自由相談（Web/Obsidian連動）"
          >
            💬 相談
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-bold text-slate-500 mr-1">口調</span>
          <button
            type="button"
            onClick={() => setHumorMode('standard')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              humorMode === 'standard'
                ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            📊 標準
          </button>
          <button
            type="button"
            onClick={() => setHumorMode('yanami')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              humorMode === 'yanami'
                ? 'bg-orange-500 text-white border-orange-500 shadow-sm'
                : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            🎤 八奈見
          </button>
          <label className="ml-auto inline-flex items-center gap-2 text-[12px] font-bold text-slate-600 cursor-pointer">
            <input
              type="checkbox"
              checked={useWeb}
              onChange={(e) => setUseWeb(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
            />
            ネット参照
          </label>
        </div>

        {/* ベイズ推定ゲージ */}
        {priorPct !== null && posteriorPct !== null && (
          <div className="mt-3 pt-3 border-t border-slate-100">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold text-slate-500">📊 ベイズ更新</span>
              <span className="text-[11px] font-bold text-slate-700">
                事前 {priorPct}%
                <span className="text-slate-400 mx-1">→</span>
                <span className={
                  posteriorPct >= 60
                    ? 'text-emerald-600'
                    : posteriorPct >= 40
                    ? 'text-amber-600'
                    : 'text-red-600'
                }>
                  事後 {posteriorPct}%
                </span>
              </span>
            </div>
            <div className="relative h-3 bg-slate-100 rounded-full overflow-hidden">
              {/* 事前確率マーカー */}
              <div
                className="absolute top-0 h-full w-0.5 bg-slate-400 z-10"
                style={{ left: `${priorPct}%` }}
              />
              {/* 事後確率バー (アニメーション付き) */}
              <div
                className={`h-full rounded-full transition-all duration-1000 ease-out ${
                  posteriorPct >= 60
                    ? 'bg-gradient-to-r from-emerald-400 to-emerald-500'
                    : posteriorPct >= 40
                    ? 'bg-gradient-to-r from-amber-400 to-amber-500'
                    : 'bg-gradient-to-r from-red-400 to-red-500'
                }`}
                style={{ width: `${posteriorPct}%` }}
              />
            </div>
            <div className="flex justify-between mt-0.5">
              <span className="text-[9px] text-slate-400">0%</span>
              <span className="text-[9px] text-slate-400">50%</span>
              <span className="text-[9px] text-slate-400">100%</span>
            </div>
            {renderBayesFactors()}
          </div>
        )}
      </div>

      {/* チャットエリア */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <div className="text-center my-2 mb-6">
          <span className="text-[10px] font-bold text-slate-400 bg-slate-200 px-3 py-1 rounded-full">ダッシュボード連携セッション開始</span>
        </div>

        {score > 0 && similarCases.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm">
            <button
              type="button"
              onClick={() => setSimilarOpen(v => !v)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-left"
            >
              <div className="flex items-center gap-2">
                <span className="text-base">📚</span>
                <span className="text-xs font-bold text-slate-700">類似過去案件 ({similarCases.length}件)</span>
                <span className="text-[10px] text-slate-400 font-medium">成約・承認済みのみ</span>
              </div>
              <span className="text-xs text-slate-400">{similarOpen ? '▲' : '▼'}</span>
            </button>
            {similarOpen && (
              <div className="px-3 pb-3 space-y-2">
                {similarCases.map((c, i) => {
                  const isSuccess = c.status.includes('成約') || c.status.includes('承認');
                  return (
                    <div key={c.id ?? i} className="border border-slate-100 rounded-lg p-2.5 bg-slate-50/50">
                      <div className="flex items-center justify-between gap-2 mb-1.5">
                        <div className="text-xs font-bold text-slate-800 truncate">{c.name}</div>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full shrink-0 ${
                          isSuccess ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-200 text-slate-600'
                        }`}>
                          {c.status}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-[10px] text-slate-600 mb-1">
                        <span className="font-medium">{c.industry || '業種未設定'}</span>
                        <span>スコア <span className="font-bold text-slate-800">{Number(c.score).toFixed(1)}</span></span>
                        <span>自己資本 <span className="font-bold text-slate-800">{c.equity}%</span></span>
                        <span className="ml-auto text-amber-600 font-bold">類似度 {c.similarity}%</span>
                      </div>
                      {c.conditions && c.conditions.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1.5">
                          {c.conditions.slice(0, 4).map((cond, j) => (
                            <span key={j} className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-700">
                              {cond}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {strategyCards && (
          <div className="bg-white rounded-xl border border-amber-200 shadow-sm overflow-hidden">
            <button
              type="button"
              onClick={() => setStrategyOpen(v => !v)}
              className="w-full flex items-center justify-between px-4 py-3 text-left bg-gradient-to-r from-amber-50 to-white"
            >
              <div>
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-amber-600" />
                  <span className="text-xs font-black text-slate-800">案件作戦盤</span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500 text-white">
                    {strategyCards.stance || '作戦整理'}
                  </span>
                </div>
                <div className="mt-1 text-[11px] text-slate-500 font-bold line-clamp-1">
                  {strategyCards.headline || 'この案件の今日やること'}
                </div>
              </div>
              <span className="text-xs text-slate-400">{strategyOpen ? '▲' : '▼'}</span>
            </button>

            {strategyOpen && (
              <div className="p-3 space-y-3">
                {strategyCards.badges && strategyCards.badges.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {strategyCards.badges.map((badge, i) => (
                      <span key={i} className="text-[10px] font-bold px-2 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                        {badge}
                      </span>
                    ))}
                  </div>
                )}

                {strategyCards.case_facts && strategyCards.case_facts.length > 0 && (
                  <div className="grid grid-cols-1 gap-1.5">
                    {strategyCards.case_facts.slice(0, 6).map((fact, i) => (
                      <div key={i} className="rounded-md bg-slate-50 border border-slate-100 px-2.5 py-1.5 text-[10px] font-bold text-slate-500">
                        {fact}
                      </div>
                    ))}
                  </div>
                )}

                <div className="grid grid-cols-1 gap-3">
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      <h4 className="text-xs font-black text-emerald-900">今日やる3手</h4>
                    </div>
                    {renderActionList(strategyCards.today_moves, 'emerald')}
                  </div>

                  <div className="rounded-xl border border-red-200 bg-red-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      <h4 className="text-xs font-black text-red-900">審査部のツッコミ予測</h4>
                    </div>
                    {renderActionList(strategyCards.risk_cards, 'red')}
                  </div>

                  <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Users className="w-4 h-4 text-blue-600" />
                      <h4 className="text-xs font-black text-blue-900">競合に負けない動き</h4>
                    </div>
                    {renderActionList(strategyCards.competitor_moves, 'blue')}
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <HelpCircle className="w-4 h-4 text-slate-600" />
                      <h4 className="text-xs font-black text-slate-800">顧客に聞くこと</h4>
                    </div>
                    {renderActionList(strategyCards.questions_to_ask, 'slate')}
                  </div>

                  <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="w-4 h-4 text-amber-600" />
                      <h4 className="text-xs font-black text-amber-900">顧客向け一言・稟議メモ</h4>
                    </div>
                    {renderActionList([
                      ...(strategyCards.customer_one_liners || []).slice(0, 2),
                      ...(strategyCards.ringi_lines || []).slice(0, 2),
                    ], 'amber')}
                  </div>
                </div>

                {strategyCards.disclaimer && (
                  <div className="text-[10px] text-slate-400 leading-4 px-1">
                    {strategyCards.disclaimer}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {chatHistory.map((chat, index) => (
          chat.role === 'user' ? (
            <div key={`${chat.role}-${index}`} className="flex gap-3 flex-row-reverse animate-in fade-in slide-in-from-right-4 duration-300">
              <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex justify-center items-center font-bold text-xs shadow-sm shrink-0">
                You
              </div>
              <div className="bg-blue-600 text-white p-3 rounded-2xl rounded-tr-none shadow-sm max-w-[85%] text-sm whitespace-pre-wrap">
                {chat.text}
              </div>
            </div>
          ) : (
            <div key={`${chat.role}-${index}`} className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="w-8 h-8 rounded-full bg-amber-500 border-2 border-white text-white flex justify-center items-center font-black text-sm shadow-md shrink-0">
                🏯
              </div>
              <div className="w-full">
                <div
                  className="bg-white p-4 rounded-2xl rounded-tl-none shadow border border-amber-200 text-slate-700 leading-7 font-medium whitespace-pre-wrap text-[13px] sm:text-sm w-full prose prose-slate"
                  dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(renderMarkdown(chat.text)) }}
                />
                {chat.meta && (
                  <div className="mt-1.5 text-[11px] text-slate-500 leading-4 px-1">
                    {chat.meta}
                  </div>
                )}
              </div>
            </div>
          )
        ))}

        {/* ストリーミング中のリアルタイムテキスト表示 */}
        {streamingText && (
          <div className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className="w-8 h-8 rounded-full bg-amber-500 border-2 border-white text-white flex justify-center items-center font-black text-sm shadow-md shrink-0">
              🏯
            </div>
            <div className="w-full">
              <div
                className="bg-white p-4 rounded-2xl rounded-tl-none shadow border border-amber-200 text-slate-700 leading-7 font-medium whitespace-pre-wrap text-[13px] sm:text-sm w-full prose prose-slate"
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(renderMarkdown(streamingText)) }}
              />
              <span className="inline-block w-0.5 h-4 bg-amber-500 ml-1 animate-pulse" />
            </div>
          </div>
        )}

        {/* 最初のチャンク待ちスピナー */}
        {loading && !streamingText && (
          <div className="flex gap-3">
             <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex justify-center items-center font-bold text-xs shadow-sm shrink-0">
               🏯
             </div>
             <div className="bg-white p-4 rounded-2xl rounded-tl-none shadow border border-slate-200 flex flex-col gap-3 min-w-[200px]">
               <Activity className="w-5 h-5 animate-spin text-amber-500" />
               <span className="text-xs font-bold text-slate-400">軍師が直近のデータを分析し、<br/>戦略を練り上げています...<br/>（Gemini API 通信中）</span>
             </div>
          </div>
        )}

        {score === 0 && chatHistory.length === 0 && (
           <div className="text-center text-sm text-slate-400 mt-10">案件戦略・業界動向・一般相談を自由に入力できます</div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="p-4 bg-white border-t border-slate-100 shrink-0">
        <div className="space-y-2">
          {statusText && (
            <div className="text-[11px] text-slate-500">
              {statusText}
            </div>
          )}
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!loading && question.trim()) {
                  handleSubmit();
                }
              }
            }}
            placeholder={advisorMode === 'gunshi' ? '軍師にこの案件の戦略・条件・落としどころを問う' : '業界動向・他社事例・自由な相談もOK（Flask AIチャット相当）'}
            rows={3}
            className="w-full bg-slate-50 border border-slate-200 rounded-xl py-3 px-4 text-sm outline-none resize-none text-slate-700 placeholder:text-slate-400"
          />
          <div className="flex items-center justify-between gap-3">
            <div className="text-[11px] text-slate-500">
              Enter で送信、Shift+Enter で改行
            </div>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading || !question.trim()}
              className="inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-xl font-bold disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              <MessageSquare className="w-4 h-4" />
              問う
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
