import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import DOMPurify from 'dompurify';
import { Activity, AlertTriangle, Bot, CheckCircle2, FileText, HelpCircle, Loader2, PenLine, Target, Users } from 'lucide-react';
import type { ScoringFormData } from '@/types';

interface GunshiAdviceProps {
  score: number;
  modelDecision: string;
  industry_major: string;
  formData: GunshiFormData;
  estatContext?: Record<string, unknown> | null;
  onChatLoaded?: (text: string) => void;
  highlightCompanies?: PastCompanyHighlight[];
}

type GunshiFormData = ScoringFormData;

type PastCompanyHighlight = {
  name: string;
  label: '類似案件' | '過去レビュー' | '反面教師';
};

type ChatMessage = {
  role: 'user' | 'assistant';
  text: string;
  meta?: string;
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
  disclaimer?: string;
};

type GunshiStreamChunk = {
  type?: 'bayes' | 'phrases' | 'strategy_cards' | 'stream' | 'done' | 'tool_call' | 'tool_result';
  cards?: StrategyCards;
  delta?: string;
  tool?: string;
};

const TOOL_LABELS: Record<string, string> = {
  get_industry_benchmark: '業種ベンチマーク照合',
  assess_risk_level: 'リスクレベル評価',
};

type HumorMode = 'yanami' | 'standard' | 'yukikaze';

type YukikazeStatus = {
  level: 'STANDBY' | 'CLEAR' | 'WARNING' | 'ALERT' | 'CRITICAL';
  tag: string;
  line: string;
  levelClass: string;
  frameClass: string;
  blinkClass: string;
};

const HUMOR_MODE_STORAGE_KEY = 'lease-gunshi-humor-mode';
const TACTICAL_DIFFICULT_CASE_LINE = 'GOOD LUCK.';

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const highlightCompaniesInHtml = (html: string, companies: PastCompanyHighlight[] = [], yukikaze = false) => {
  const highlights = Array.from(new Map(companies.map((item) => [item.name.trim(), item])).values())
    .filter((item) => item.name.length >= 2)
    .sort((a, b) => b.name.length - a.name.length);
  if (!highlights.length) return html;
  const names = highlights.map((item) => item.name);
  const labelByName = new Map(highlights.map((item) => [item.name, item.label]));
  const className = yukikaze
    ? 'inline-flex items-center gap-1 rounded bg-red-950 px-1.5 py-0.5 font-black text-amber-100 ring-1 ring-red-500'
    : 'inline-flex items-center gap-1 rounded bg-cyan-50 px-1.5 py-0.5 font-black text-cyan-800 ring-1 ring-cyan-200';
  const labelClass = yukikaze
    ? 'rounded bg-black px-1 text-[10px] font-black text-red-200'
    : 'rounded bg-white px-1 text-[10px] font-black text-cyan-600';
  const pattern = new RegExp(`(${names.map(escapeRegExp).join('|')})`, 'g');
  return html.replace(pattern, (match) => `<span class="${className}">${match}<span class="${labelClass}">${labelByName.get(match) || ''}</span></span>`);
};

const normalizeDecision = (value: string, score: number) => {
  const text = String(value || '').replace('条件付き', '条件付');
  if (text.includes('否決') || text.includes('否認')) return '否決';
  if (text.includes('条件付') || text.includes('要審議') || text.includes('要確認') || text.includes('ボーダー')) return '条件付';
  if (text.includes('承認') || text.includes('良決')) return '承認';
  if (score >= 80) return '承認';
  if (score >= 50) return '条件付';
  return '否決';
};

const getInitialHumorMode = (): HumorMode => {
  if (typeof window === 'undefined') return 'yanami';
  const stored = window.localStorage.getItem(HUMOR_MODE_STORAGE_KEY);
  if (stored === 'standard' || stored === 'yanami' || stored === 'yukikaze') return stored;
  return 'yanami';
};

const getYukikazeStatus = (score: number): YukikazeStatus => {
  if (score <= 0) {
    return {
      level: 'STANDBY',
      tag: '[DATA LINK: STANDBY]',
      line: 'AURION CORE ONLINE. Tactical console awaiting pilot vector input.',
      levelClass: 'text-slate-300 border-slate-600 bg-slate-900',
      frameClass: 'border-slate-700',
      blinkClass: '',
    };
  }
  if (score >= 80) {
    return {
      level: 'CLEAR',
      tag: '[TACTICAL STATUS: CLEAR]',
      line: 'No anomaly signature detected. Financial waveform is stable. Maintain current approval vector.',
      levelClass: 'text-emerald-300 border-emerald-500/60 bg-emerald-950/50',
      frameClass: 'border-emerald-500/40',
      blinkClass: '',
    };
  }
  if (score >= 60) {
    return {
      level: 'WARNING',
      tag: '[TACTICAL WARNING: SIGNAL DEFLECTION]',
      line: `Minor distortion detected. Approval route remains open. Pilot visual confirmation is recommended. ${TACTICAL_DIFFICULT_CASE_LINE}`,
      levelClass: 'text-amber-200 border-amber-400 bg-amber-950/60',
      frameClass: 'border-amber-400/70',
      blinkClass: 'animate-[pulse_1.8s_ease-in-out_infinite]',
    };
  }
  if (score >= 40) {
    return {
      level: 'ALERT',
      tag: '[ALERT: ANOMALY CONTACT]',
      line: `Hostile inconsistency detected in financial telemetry. Autopilot judgment restricted. Handing control to pilot. ${TACTICAL_DIFFICULT_CASE_LINE}`,
      levelClass: 'text-red-200 border-red-500 bg-red-950/70',
      frameClass: 'border-red-500/80',
      blinkClass: 'animate-[pulse_1.1s_ease-in-out_infinite]',
    };
  }
  return {
    level: 'CRITICAL',
    tag: '[CRITICAL: MANUAL OVERRIDE REQUIRED]',
    line: `I identify the enemy. You decide whether to engage. Rejection logic dominates. Manual override required. ${TACTICAL_DIFFICULT_CASE_LINE}`,
    levelClass: 'text-red-100 border-red-400 bg-red-900/80',
    frameClass: 'border-red-400',
    blinkClass: 'animate-[pulse_0.65s_ease-in-out_infinite]',
  };
};

export default function GunshiAdvice({ score, modelDecision, industry_major, formData, estatContext, onChatLoaded, highlightCompanies = [] }: GunshiAdviceProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [humorMode, setHumorMode] = useState<HumorMode>(getInitialHumorMode);
  const [yukikazeBooting, setYukikazeBooting] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [streamingText, setStreamingText] = useState('');
  const [toolSteps, setToolSteps] = useState<{tool: string; done: boolean}[]>([]);
  const [strategyCards, setStrategyCards] = useState<StrategyCards | null>(null);
  const [strategyOpen, setStrategyOpen] = useState(false);
  const [humanDecision, setHumanDecision] = useState('');
  const [judgmentChangeReason, setJudgmentChangeReason] = useState('');
  const [judgmentSaving, setJudgmentSaving] = useState(false);
  const [judgmentStatus, setJudgmentStatus] = useState<'success' | 'error' | ''>('');
  const initialFetchKeyRef = useRef<string>("");
  const feedbackCaseIdRef = useRef(`gunshi-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const normalizedModelDecision = normalizeDecision(modelDecision, score);
  const initialStrategyQuestion = score > 0
    ? `この案件（スコア ${score.toFixed(1)}点、${industry_major || "指定なし"}）の稟議を通すための「逆転戦略」を教えてくれ！`
    : "";

  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [chatHistory, streamingText]);

  useEffect(() => {
    setHumanDecision(normalizedModelDecision);
    setJudgmentChangeReason('');
    setJudgmentStatus('');
    feedbackCaseIdRef.current = `gunshi-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }, [normalizedModelDecision, score, formData.company_no]);

  const handleHumorModeChange = (mode: HumorMode) => {
    setHumorMode(mode);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(HUMOR_MODE_STORAGE_KEY, mode);
    }
    if (mode === 'yukikaze') {
      setYukikazeBooting(true);
      window.setTimeout(() => setYukikazeBooting(false), 2200);
    } else {
      setYukikazeBooting(false);
    }
  };

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
      humor_style: humorMode,
      score,
      resale_eval: "B",
      repeat_count: Number(formData.contracts) || 0,
      subsidy_flag: /補助金|助成金|ものづくり|省力化/.test(subsidyText),
      bank_support: formData.deal_source === "銀行紹介" || formData.main_bank === "メイン先",
      intuition_score: Number(formData.intuition) || 50,
      estat_context: estatContext || null,
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

  // 初期フェッチ専用: SSEストリーミング (/api/gunshi/stream)
  const fetchStreamChat = async (displayHistory: ChatMessage[]) => {
    setLoading(true);
    setStreamingText('');
    setStrategyCards(null);
    setToolSteps([]);
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
            if (chunk.type === 'tool_call' && chunk.tool) {
              setToolSteps(prev => [...prev, { tool: chunk.tool!, done: false }]);
            } else if (chunk.type === 'tool_result' && chunk.tool) {
              setToolSteps(prev => prev.map(s => s.tool === chunk.tool ? { ...s, done: true } : s));
            } else if (chunk.type === 'strategy_cards') {
              setStrategyCards(chunk.cards || null);
              setStrategyOpen(Boolean(chunk.cards));
            } else if (chunk.type === 'stream' && chunk.delta) {
              // 初回自動フェッチの回答本文は非表示にする（案件作戦盤カードのみ表示）。
              fullText += chunk.delta;
            } else if (chunk.type === 'done') {
              finished = true;
              setStatusText('回答しました。');
              if (onChatLoaded) onChatLoaded(fullText);
            }
          } catch {
            // Ignore malformed SSE keepalive fragments.
          }
        }
      }

      // done イベントなしで終了した場合も同様に本文は表示しない
      if (!finished && fullText) {
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
      humorMode,
      JSON.stringify(estatContext || null),
    ].join(":");
    if (initialFetchKeyRef.current === fetchKey) return;
    initialFetchKeyRef.current = fetchKey;

    const nextHistory: ChatMessage[] = [{ role: 'user', text: initialStrategyQuestion }];
    setChatHistory(nextHistory);
    fetchStreamChat(nextHistory);
  }, [score, industry_major, formData, estatContext, initialStrategyQuestion, humorMode]);

  const handleOpenShionChat = () => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem('lease-gunshi-context', JSON.stringify({
      score,
      hantei: normalizedModelDecision,
      company_name: formData.company_name || '',
      asset_name: formData.asset_name || '',
      asset_location: formData.asset_location || '',
      industry_sub: formData.industry_sub || '',
      industry_major,
      sales_dept: formData.sales_dept || '',
      case_id: formData.company_no || formData.company_name || '',
    }));
    window.location.href = '/chat';
  };

  const handleRecordJudgmentChange = async () => {
    if (
      judgmentSaving ||
      humanDecision === normalizedModelDecision ||
      judgmentChangeReason.trim().length < 5
    ) return;

    setJudgmentSaving(true);
    setJudgmentStatus('');
    const latestAssistantReply = [...chatHistory]
      .reverse()
      .find((message) => message.role === 'assistant')?.text || '';
    try {
      await axios.post('/api/judgment-feedback', {
        case_id: feedbackCaseIdRef.current,
        model_decision: normalizedModelDecision,
        human_decision: humanDecision,
        reason: judgmentChangeReason.trim(),
        source: 'gunshi_chat',
        score,
        input_snapshot: {
          industry_major,
          industry_sub: formData.industry_sub,
          grade: formData.grade,
          customer_type: formData.customer_type,
          nenshu: formData.nenshu,
          op_profit: formData.op_profit,
          ord_profit: formData.ord_profit,
          net_income: formData.net_income,
          net_assets: formData.net_assets,
          total_assets: formData.total_assets,
          bank_credit: formData.bank_credit,
          lease_credit: formData.lease_credit,
          contracts: formData.contracts,
          asset_name: formData.asset_name,
          acquisition_cost: formData.acquisition_cost,
          lease_term: formData.lease_term,
        },
        evidence_snapshot: {
          strategy_headline: strategyCards?.headline || '',
          strategy_stance: strategyCards?.stance || '',
          risk_cards: strategyCards?.risk_cards || [],
          today_moves: strategyCards?.today_moves || [],
          latest_gunshi_reply: latestAssistantReply.slice(0, 2000),
        },
      });
      setJudgmentStatus('success');
    } catch (error) {
      console.error('Failed to record judgment feedback', error);
      setJudgmentStatus('error');
    } finally {
      setJudgmentSaving(false);
    }
  };

  const renderMarkdown = (text: string, yukikaze = false) => {
    let parsedText = text;
    const headingClass = yukikaze
      ? 'font-bold text-base text-yellow-300 mt-5 border-b border-yellow-500/50 pb-1 mb-2'
      : 'font-bold text-base text-amber-700 mt-5 border-b border-amber-200 pb-1 mb-2';
    const strongClass = yukikaze
      ? 'font-black text-yellow-300'
      : 'font-bold text-slate-800';
    parsedText = parsedText.replace(/### (.*?)(\n|$)/g, `<h4 class="${headingClass}">$1</h4>\n`);
    parsedText = parsedText.replace(/\*\*(.*?)\*\*/g, `<strong class="${strongClass}">$1</strong>`);
    return parsedText;
  };

  const renderDatalinkTranscript = (text: string) => {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    const html: string[] = [];

    lines.forEach((rawLine) => {
      const line = rawLine.trim();
      if (!line) {
        html.push('<div class="h-1"></div>');
        return;
      }

      const txMatch = line.match(/^TX:\s*(.*)$/i);
      const rxMatch = line.match(/^RX:\s*(.*)$/i);
      const signalMatch = line.match(/^SIGNAL:\s*(.*)$/i);
      const pilotTaskMatch = line.match(/^PILOT TASK:\s*(.*)$/i);
      const vectorMatch = line.match(/^VECTOR:\s*(.*)$/i);
      const datalinkMatch = line.match(/^DATALINK LOG:\s*(.*)$/i);

      if (txMatch) {
        html.push(
          `<div class="text-[10px] uppercase tracking-[0.28em] text-red-300">TX</div>` +
          `<div class="mt-0.5 rounded-lg border border-red-900 bg-black/90 px-3 py-2 text-[12px] leading-5 text-amber-100 whitespace-pre-wrap">${DOMPurify.sanitize(txMatch[1])}</div>`
        );
        html.push(
          '<div class="my-2 flex items-center gap-2 text-[10px] uppercase tracking-[0.32em] text-red-400/80">' +
          '<span class="inline-flex items-center gap-1"><span class="animate-pulse">...</span><span>LINK DELAY</span></span>' +
          '</div>'
        );
        return;
      }

      if (rxMatch) {
        html.push(
          `<div class="text-[10px] uppercase tracking-[0.28em] text-amber-300">RX</div>` +
          `<div class="mt-0.5 rounded-lg border border-amber-700 bg-amber-950/80 px-3 py-2 text-[12px] leading-5 text-amber-50 whitespace-pre-wrap">${DOMPurify.sanitize(rxMatch[1])}</div>`
        );
        return;
      }

      if (datalinkMatch) {
        html.push(
          `<div class="text-[10px] uppercase tracking-[0.28em] text-red-300">${DOMPurify.sanitize(line)}</div>`
        );
        return;
      }

      if (signalMatch) {
        html.push(`<div class="rounded-md border border-red-950 bg-black/70 px-3 py-2 text-[11px] font-bold text-red-200">${DOMPurify.sanitize(`SIGNAL: ${signalMatch[1]}`)}</div>`);
        return;
      }

      if (pilotTaskMatch) {
        html.push(`<div class="rounded-md border border-amber-900 bg-black/60 px-3 py-2 text-[11px] font-bold text-amber-100">${DOMPurify.sanitize(`PILOT TASK: ${pilotTaskMatch[1]}`)}</div>`);
        return;
      }

      if (vectorMatch) {
        html.push(`<div class="rounded-md border border-emerald-900 bg-black/60 px-3 py-2 text-[11px] font-bold text-emerald-100">${DOMPurify.sanitize(`VECTOR: ${vectorMatch[1]}`)}</div>`);
        return;
      }

      html.push(`<div class="whitespace-pre-wrap">${DOMPurify.sanitize(line)}</div>`);
    });

    return html.join('');
  };

  const renderAssistantText = (text: string, yukikaze = false) => {
    const looksLikeDatalink = yukikaze && /^(TX:|RX:|DATALINK LOG:|SIGNAL:|PILOT TASK:|VECTOR:)/im.test(text);
    if (looksLikeDatalink) {
      return renderDatalinkTranscript(text);
    }
    return renderMarkdown(text, yukikaze);
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

  const isYukikaze = humorMode === 'yukikaze';
  const yukikazeStatus = getYukikazeStatus(score);
  const isDifficultYukikazeCase = isYukikaze && ['WARNING', 'ALERT', 'CRITICAL'].includes(yukikazeStatus.level);
  const boardCopy = isYukikaze
    ? {
        headerTitle: 'YK // TACTICAL MODE',
        headerSubtitle: 'TACTICAL LEASE SCORING AI',
        queryLabel: 'PILOT QUERY',
        boardTitle: 'TACTICAL BOARD',
        stanceFallback: 'SORTIE PLAN',
        headlineFallback: 'VECTOR: immediate action route locked',
        todayMoves: 'SORTIE TASKS // NEXT 3',
        riskCards: 'REVIEW INTERCEPT POINTS',
        competitorMoves: 'HOSTILE BID COUNTER-VECTOR',
        questionsToAsk: 'PILOT QUERY CHECKPOINTS',
        ringiLines: 'RINGI TRANSMISSION LOG',
      }
    : humorMode === 'yanami'
      ? {
          headerTitle: 'つん子の作戦掲示板',
          headerSubtitle: '稟議を通すための面倒ごと、先に並べといたで',
          queryLabel: 'つん子へのお題',
          boardTitle: 'つん子の作戦掲示板',
          stanceFallback: '通すための段取り',
          headlineFallback: 'この案件、今日どこから片づけるか',
          todayMoves: '今日まず片づける3手',
          riskCards: '審査部に刺されそうな穴',
          competitorMoves: '競合に持ってかれる前の一手',
          questionsToAsk: 'お客さんに今聞いとくこと',
          ringiLines: '稟議に残す一言、あとで泣かない用',
        }
      : {
          headerTitle: '案件作戦盤',
          headerSubtitle: '紫苑レビューを補助する質問・稟議作戦',
          queryLabel: '今回の問い',
          boardTitle: '案件作戦盤',
          stanceFallback: '作戦整理',
          headlineFallback: 'この案件の今日やること',
          todayMoves: '今日やる3手',
          riskCards: '審査部のツッコミ予測',
          competitorMoves: '競合に負けない動き',
          questionsToAsk: '顧客に聞くこと',
          ringiLines: '顧客向け一言・稟議メモ',
        };
  const panelClass = isYukikaze
    ? `2xl:sticky 2xl:top-16 h-[calc(100vh-3rem)] min-h-[900px] bg-[#050505] rounded-2xl shadow-2xl shadow-red-950/40 border ${yukikazeStatus.frameClass} flex flex-col overflow-hidden text-amber-50`
    : '2xl:sticky 2xl:top-16 h-[calc(100vh-3rem)] min-h-[900px] bg-[#f8fafc] rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 flex flex-col overflow-hidden';
  const headerClass = isYukikaze
    ? `bg-black text-amber-100 px-4 py-3 shrink-0 shadow-md z-10 flex items-center justify-between border-b ${yukikazeStatus.frameClass}`
    : 'bg-gradient-to-r from-[#172554] to-[#1e3a8a] text-white p-4 shrink-0 shadow-md z-10 flex items-center justify-between';
  const controlClass = isYukikaze
    ? 'bg-[#080808] border-b border-red-900/50 px-4 py-2 shrink-0 font-mono'
    : 'bg-white border-b border-slate-100 px-4 py-3 shrink-0';
  const chatAreaClass = isYukikaze
    ? 'flex-1 min-h-[520px] overflow-y-auto p-5 space-y-6 bg-[radial-gradient(circle_at_top,rgba(127,29,29,0.18),transparent_32%),#050505]'
    : 'flex-1 min-h-[520px] overflow-y-auto p-5 space-y-6';
  const footerClass = isYukikaze
    ? 'p-3 bg-black border-t border-red-900/60 shrink-0 font-mono'
    : 'p-3 bg-white border-t border-slate-100 shrink-0';

  return (
    <div className={panelClass}>

      {/* ヘッダー */}
      <div className={headerClass}>
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-full border-2 flex justify-center items-center font-black text-xl shadow-inner overflow-hidden ${
            isYukikaze ? `bg-red-950 border-red-500 text-red-200 ${yukikazeStatus.blinkClass}` : 'bg-amber-500 border-white'
          }`}>
            <span className="text-2xl mt-1">{isYukikaze ? 'YK' : '🏯'}</span>
          </div>
          <div>
            <h3 className={`font-bold text-sm tracking-wide ${isYukikaze ? 'font-mono text-amber-100' : ''}`}>
              {boardCopy.headerTitle}
            </h3>
            <p className={`text-[10px] font-medium ${isYukikaze ? 'text-red-300 font-mono tracking-widest' : 'text-blue-200'}`}>
              {boardCopy.headerSubtitle}
            </p>
          </div>
        </div>
      </div>

      <div className={controlClass}>
        {isYukikaze && (
          <div className={`mb-2 rounded-lg border bg-black/80 px-3 py-2 shadow-lg shadow-red-950/30 ${yukikazeStatus.frameClass}`}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-[9px] font-black tracking-[0.24em] text-red-400">SYSTEM MODE</div>
                <div className="mt-0.5 text-[11px] font-black tracking-[0.18em] text-amber-100">YK // TACTICAL MODE</div>
              </div>
              <div className={`rounded-md border px-2 py-1 text-[10px] font-black tracking-widest ${yukikazeStatus.levelClass} ${yukikazeStatus.blinkClass}`}>
                {yukikazeStatus.level}
              </div>
            </div>
            <div className={`mt-2 rounded-md border px-2.5 py-1.5 ${yukikazeStatus.levelClass} ${yukikazeStatus.blinkClass}`}>
              <div className="text-[9px] font-black tracking-widest">{yukikazeStatus.tag}</div>
              <div className="mt-0.5 line-clamp-2 text-[11px] leading-4 font-bold">{yukikazeBooting ? 'LINKING AURION CORE... PILOT AUTHENTICATION: CONFIRMED. ANOMALY DETECTION PROTOCOL: ACTIVE.' : yukikazeStatus.line}</div>
            </div>
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[11px] font-bold mr-1 ${isYukikaze ? 'text-red-300 tracking-widest' : 'text-slate-500'}`}>口調</span>
          <button
            type="button"
            onClick={() => handleHumorModeChange('standard')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              humorMode === 'standard'
                ? 'bg-blue-600 text-white border-blue-600 shadow-sm'
                : isYukikaze ? 'bg-black text-slate-400 border-red-950 hover:border-red-700' : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            📊 標準
          </button>
          <button
            type="button"
            onClick={() => handleHumorModeChange('yanami')}
            className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${
              humorMode === 'yanami'
                ? 'bg-orange-500 text-white border-orange-500 shadow-sm'
                : isYukikaze ? 'bg-black text-slate-400 border-red-950 hover:border-red-700' : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
            }`}
          >
            🎤 つん子
          </button>
          <button
            type="button"
            onClick={() => handleHumorModeChange('yukikaze')}
            className={`px-3 py-1.5 rounded-full text-xs font-black border transition ${
              isYukikaze
                ? 'bg-red-700 text-amber-100 border-red-400 shadow-lg shadow-red-900/40'
                : 'bg-slate-950 text-red-300 border-red-800 hover:border-red-500 hover:text-amber-100'
            }`}
            title="YK tactical console"
          >
            ⚡ ENGAGE YK
          </button>
        </div>

      </div>

      {/* チャットエリア */}
      <div ref={chatScrollRef} className={chatAreaClass}>
        <div className="text-center my-2 mb-6">
          <span className={`text-[10px] font-bold px-3 py-1 rounded-full ${isYukikaze ? 'text-red-300 bg-black border border-red-950 font-mono tracking-widest' : 'text-slate-400 bg-slate-200'}`}>
            {isYukikaze ? 'TACTICAL SESSION LINKED' : 'ダッシュボード連携セッション開始'}
          </span>
        </div>

        {initialStrategyQuestion && (
          <div className={`rounded-xl border shadow-sm px-4 py-3 ${isYukikaze ? 'bg-black/85 border-red-900 text-amber-50 font-mono' : 'bg-blue-50 border-blue-200'}`}>
            <div className={`text-[10px] font-black mb-1 ${isYukikaze ? 'text-red-300 tracking-widest' : 'text-blue-700'}`}>
              {boardCopy.queryLabel}
            </div>
            <div className={`text-sm font-bold leading-6 ${isYukikaze ? 'text-amber-100' : 'text-slate-800'}`}>
              {initialStrategyQuestion}
            </div>
          </div>
        )}

        {strategyCards && (
          <div className={`rounded-xl border shadow-sm overflow-hidden ${isYukikaze ? `bg-black/85 ${yukikazeStatus.frameClass} text-amber-50 font-mono` : 'bg-white border-amber-200'}`}>
            <button
              type="button"
              onClick={() => setStrategyOpen(v => !v)}
              className={`w-full flex items-center justify-between px-4 py-3 text-left ${isYukikaze ? 'bg-gradient-to-r from-red-950/70 to-black' : 'bg-gradient-to-r from-amber-50 to-white'}`}
            >
              <div>
                <div className="flex items-center gap-2">
                  <Target className={`w-4 h-4 ${isYukikaze ? 'text-red-300' : 'text-amber-600'}`} />
                  <span className={`text-xs font-black ${isYukikaze ? 'text-amber-100 tracking-widest' : 'text-slate-800'}`}>{boardCopy.boardTitle}</span>
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${isYukikaze ? `border ${yukikazeStatus.levelClass} ${yukikazeStatus.blinkClass}` : 'bg-amber-500 text-white'}`}>
                    {strategyCards.stance || boardCopy.stanceFallback}
                  </span>
                </div>
                <div className={`mt-1 text-[11px] font-bold line-clamp-1 ${isYukikaze ? 'text-red-300' : 'text-slate-500'}`}>
                  {strategyCards.headline || boardCopy.headlineFallback}
                </div>
              </div>
              <span className={`text-xs ${isYukikaze ? 'text-red-300' : 'text-slate-400'}`}>{strategyOpen ? '▲' : '▼'}</span>
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
                      <h4 className="text-xs font-black text-emerald-900">{boardCopy.todayMoves}</h4>
                    </div>
                    {renderActionList(strategyCards.today_moves, 'emerald')}
                  </div>

                  <div className="rounded-xl border border-red-200 bg-red-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      <h4 className="text-xs font-black text-red-900">{boardCopy.riskCards}</h4>
                    </div>
                    {renderActionList(strategyCards.risk_cards, 'red')}
                  </div>

                  <div className="rounded-xl border border-blue-200 bg-blue-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Users className="w-4 h-4 text-blue-600" />
                      <h4 className="text-xs font-black text-blue-900">{boardCopy.competitorMoves}</h4>
                    </div>
                    {renderActionList(strategyCards.competitor_moves, 'blue')}
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <HelpCircle className="w-4 h-4 text-slate-600" />
                      <h4 className="text-xs font-black text-slate-800">{boardCopy.questionsToAsk}</h4>
                    </div>
                    {renderActionList(strategyCards.questions_to_ask, 'slate')}
                  </div>

                  <div className="rounded-xl border border-amber-200 bg-amber-50/60 p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <FileText className="w-4 h-4 text-amber-600" />
                      <h4 className="text-xs font-black text-amber-900">{boardCopy.ringiLines}</h4>
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

        {chatHistory.map((chat, index) => {
          if (chat.role === 'user' && chat.text === initialStrategyQuestion) {
            return null;
          }
          return chat.role === 'user' ? (
            <div key={`${chat.role}-${index}`} className="flex gap-3 flex-row-reverse animate-in fade-in slide-in-from-right-4 duration-300">
              <div className={`w-8 h-8 rounded-full flex justify-center items-center font-bold text-xs shadow-sm shrink-0 ${isYukikaze ? 'bg-amber-900 text-amber-100 border border-amber-500 font-mono' : 'bg-blue-600 text-white'}`}>
                You
              </div>
              <div className={`p-3 rounded-2xl rounded-tr-none shadow-sm max-w-[85%] text-sm whitespace-pre-wrap ${isYukikaze ? 'bg-amber-950/70 border border-amber-700 text-amber-50 font-mono' : 'bg-blue-600 text-white'}`}>
                {chat.text}
              </div>
            </div>
          ) : (
            <div key={`${chat.role}-${index}`} className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className={`w-8 h-8 rounded-full border-2 flex justify-center items-center font-black text-sm shadow-md shrink-0 ${isYukikaze ? `bg-red-950 border-red-500 text-red-200 font-mono ${yukikazeStatus.blinkClass}` : 'bg-amber-500 border-white text-white'}`}>
                {isYukikaze ? 'YK' : '🏯'}
              </div>
              <div className="w-full">
                <div
                  className={`max-w-none min-h-[360px] p-6 rounded-2xl rounded-tl-none shadow border leading-8 font-medium whitespace-pre-wrap text-sm sm:text-[15px] w-full prose ${isYukikaze ? 'bg-black/90 border-red-900 text-amber-50 prose-invert font-mono' : 'bg-white border-amber-200 text-slate-700 prose-slate'}`}
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightCompaniesInHtml(renderAssistantText(chat.text, isYukikaze), highlightCompanies, isYukikaze)) }}
                />
                {chat.meta && (
                  <div className={`mt-1.5 text-[11px] leading-4 px-1 ${isYukikaze ? 'text-red-300 font-mono' : 'text-slate-500'}`}>
                    {chat.meta}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* 紫苑 ADK ツールステップ表示 */}
        {toolSteps.length > 0 && (
          <div className="mx-1 mb-2 rounded-xl border border-violet-200 bg-violet-50 p-3 text-xs">
            <p className="mb-2 font-semibold text-violet-700">🤖 紫苑が審査中...</p>
            <div className="space-y-1">
              {toolSteps.map((step, i) => (
                <div key={i} className="flex items-center gap-2">
                  {step.done ? (
                    <span className="text-emerald-500">✅</span>
                  ) : (
                    <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-violet-400 border-t-transparent" />
                  )}
                  <span className={step.done ? 'text-slate-600' : 'font-medium text-violet-700'}>
                    {TOOL_LABELS[step.tool] ?? step.tool}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ストリーミング中のリアルタイムテキスト表示 */}
        {streamingText && (
          <div className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className={`w-8 h-8 rounded-full border-2 flex justify-center items-center font-black text-sm shadow-md shrink-0 ${isYukikaze ? `bg-red-950 border-red-500 text-red-200 font-mono ${yukikazeStatus.blinkClass}` : 'bg-amber-500 border-white text-white'}`}>
              {isYukikaze ? 'YK' : '🏯'}
            </div>
            <div className="w-full">
              <div
                className={`max-w-none min-h-[360px] p-6 rounded-2xl rounded-tl-none shadow border leading-8 font-medium whitespace-pre-wrap text-sm sm:text-[15px] w-full prose ${isYukikaze ? 'bg-black/90 border-red-900 text-amber-50 prose-invert font-mono' : 'bg-white border-amber-200 text-slate-700 prose-slate'}`}
                dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightCompaniesInHtml(renderAssistantText(streamingText, isYukikaze), highlightCompanies, isYukikaze)) }}
              />
              <span className={`inline-block w-0.5 h-4 ml-1 animate-pulse ${isYukikaze ? 'bg-red-500' : 'bg-amber-500'}`} />
            </div>
          </div>
        )}

        {/* 最初のチャンク待ちスピナー */}
        {loading && !streamingText && (
          <div className="flex gap-3">
             <div className={`w-8 h-8 rounded-full flex justify-center items-center font-bold text-xs shadow-sm shrink-0 ${isYukikaze ? `bg-red-950 border border-red-500 text-red-200 font-mono ${yukikazeStatus.blinkClass}` : 'bg-amber-500 text-white'}`}>
               {isYukikaze ? 'YK' : '🏯'}
             </div>
             <div className={`p-4 rounded-2xl rounded-tl-none shadow border flex flex-col gap-3 min-w-[200px] ${isYukikaze ? 'bg-black border-red-900 font-mono' : 'bg-white border-slate-200'}`}>
               <Activity className={`w-5 h-5 animate-spin ${isYukikaze ? 'text-red-400' : 'text-amber-500'}`} />
               <span className={`text-xs font-bold ${isYukikaze ? 'text-red-300' : 'text-slate-400'}`}>{isYukikaze ? <>YK is reading telemetry...<br/>Anomaly signature analysis running...</> : <>軍師が直近のデータを分析し、<br/>戦略を練り上げています...<br/>（Gemini API 通信中）</>}</span>
             </div>
          </div>
        )}

        {score === 0 && chatHistory.length === 0 && (
           <div className={`text-center text-sm mt-10 ${isYukikaze ? 'text-red-300 font-mono' : 'text-slate-400'}`}>{isYukikaze ? 'Awaiting pilot query. Engagement protocol is armed.' : '案件戦略・業界動向・一般相談を自由に入力できます'}</div>
        )}
      </div>

      <div className={footerClass}>
        <div className="space-y-2">
          {score > 0 && (
            <div className={`border-b pb-3 ${isYukikaze ? 'border-red-900/60' : 'border-slate-200'}`}>
              <div className="mb-2">
                <div className={`text-xs font-black ${isYukikaze ? 'text-amber-100' : 'text-slate-800'}`}>
                  担当者の最終判断を記録
                </div>
                <div className={`mt-0.5 text-[10px] ${isYukikaze ? 'text-red-300' : 'text-slate-500'}`}>
                  AI判断を変更する場合だけ入力
                </div>
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-[0.8fr_0.8fr_1.6fr_auto] sm:items-end">
                <label className="min-w-0">
                  <span className={`text-[10px] font-bold ${isYukikaze ? 'text-red-300' : 'text-slate-600'}`}>AI判断</span>
                  <input
                    value={normalizedModelDecision}
                    readOnly
                    className={`mt-1 h-9 w-full rounded-md border px-2.5 text-xs font-bold ${
                      isYukikaze
                        ? 'border-red-900 bg-black text-amber-100'
                        : 'border-slate-200 bg-slate-100 text-slate-700'
                    }`}
                  />
                </label>
                <label className="min-w-0">
                  <span className={`text-[10px] font-bold ${isYukikaze ? 'text-red-300' : 'text-slate-600'}`}>担当者判断</span>
                  <select
                    value={humanDecision}
                    onChange={(event) => {
                      setHumanDecision(event.target.value);
                      setJudgmentStatus('');
                    }}
                    className={`mt-1 h-9 w-full rounded-md border px-2.5 text-xs font-bold ${
                      isYukikaze
                        ? 'border-red-800 bg-black text-amber-100'
                        : 'border-slate-300 bg-white text-slate-800'
                    }`}
                  >
                    <option value="承認">承認</option>
                    <option value="条件付">条件付</option>
                    <option value="否決">否決</option>
                  </select>
                </label>
                <label className="min-w-0">
                  <span className={`text-[10px] font-bold ${isYukikaze ? 'text-red-300' : 'text-slate-600'}`}>変更理由</span>
                  <input
                    value={judgmentChangeReason}
                    onChange={(event) => {
                      setJudgmentChangeReason(event.target.value);
                      setJudgmentStatus('');
                    }}
                    placeholder="AI判断を変更した理由"
                    className={`mt-1 h-9 w-full rounded-md border px-2.5 text-xs outline-none ${
                      isYukikaze
                        ? 'border-red-800 bg-black text-amber-100 placeholder:text-red-900'
                        : 'border-slate-300 bg-white text-slate-800 placeholder:text-slate-400'
                    }`}
                  />
                </label>
                <button
                  type="button"
                  onClick={handleRecordJudgmentChange}
                  disabled={
                    judgmentSaving ||
                    humanDecision === normalizedModelDecision ||
                    judgmentChangeReason.trim().length < 5
                  }
                  className={`inline-flex h-9 items-center justify-center gap-1.5 rounded-md border px-3 text-xs font-black disabled:cursor-not-allowed disabled:opacity-40 ${
                    isYukikaze
                      ? 'border-red-600 bg-red-950 text-amber-100 hover:bg-red-900'
                      : 'border-slate-300 bg-white text-slate-800 hover:bg-slate-100'
                  }`}
                >
                  {judgmentSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PenLine className="h-3.5 w-3.5" />}
                  {judgmentSaving ? '記録中' : '変更を記録'}
                </button>
              </div>
              {judgmentStatus === 'success' && (
                <div className={`mt-2 text-[11px] font-bold ${isYukikaze ? 'text-emerald-300' : 'text-emerald-700'}`}>
                  モデル改善候補として記録しました。
                </div>
              )}
              {judgmentStatus === 'error' && (
                <div className={`mt-2 text-[11px] font-bold ${isYukikaze ? 'text-red-300' : 'text-rose-700'}`}>
                  記録に失敗しました。
                </div>
              )}
            </div>
          )}
          {statusText && (
            <div className={`text-[11px] ${isYukikaze ? 'text-red-300' : 'text-slate-500'}`}>
              {isYukikaze ? `SYSTEM: ${statusText}` : statusText}
            </div>
          )}
          <div className="flex items-center justify-between gap-3">
            <div className={`text-[11px] ${isYukikaze ? 'text-red-300' : 'text-slate-500'}`}>
              {isYukikaze ? (
                <span className={yukikazeStatus.level === 'CRITICAL' ? yukikazeStatus.blinkClass : ''}>
                  {isDifficultYukikazeCase
                    ? `${TACTICAL_DIFFICULT_CASE_LINE} I identify the enemy. You decide whether to engage.`
                    : 'I identify the enemy. You decide whether to engage.'}
                </span>
              ) : (
                '深掘りや記憶参照は紫苑チャットへ渡します'
              )}
            </div>
            <button
              type="button"
              onClick={handleOpenShionChat}
              className={`inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-bold disabled:cursor-not-allowed ${isYukikaze ? 'bg-red-800 text-amber-100 border border-red-500 hover:bg-red-700 disabled:bg-red-950 disabled:text-red-900' : 'bg-blue-600 text-white disabled:bg-slate-300'}`}
            >
              <Bot className="w-4 h-4" />
              {isYukikaze ? 'OPEN SHION LINK' : '紫苑に相談'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
