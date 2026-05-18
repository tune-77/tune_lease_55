import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Activity, MessageSquare } from 'lucide-react';

interface GunshiAdviceProps {
  score: number;
  pd_percent: number;
  industry_major: string;
  formData: any;
  onChatLoaded?: (text: string) => void;
}

type ChatMessage = {
  role: 'user' | 'assistant';
  text: string;
  meta?: string;
};

export default function GunshiAdvice({ score, pd_percent, industry_major, formData, onChatLoaded }: GunshiAdviceProps) {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [humorMode, setHumorMode] = useState<'yanami' | 'standard'>('yanami');
  const [useWeb, setUseWeb] = useState(true);
  const [advisorMode, setAdvisorMode] = useState<'gunshi' | 'chat'>('gunshi');
  const [statusText, setStatusText] = useState('');
  const initialFetchKeyRef = useRef<string>("");

  const buildPayload = (message = "", history: ChatMessage[] = chatHistory) => {
    const subsidyText = [
      formData.industry_detail,
      formData.passion_text,
      formData.asset_name,
    ].join(" ");
    return {
      score,
      pd_percent,
      industry_major,
      asset_name: formData.asset_name || "",
      resale: "標準",
      repeat_cnt: 1,
      subsidy: /補助金|助成金|ものづくり|省力化/.test(subsidyText),
      bank: formData.deal_source === "銀行紹介" || formData.main_bank === "メイン先",
      intuition: formData.intuition || 50,
      posterior: 0.5,
      message,
      history,
      humor_style: humorMode === 'yanami' ? 'yanami' : 'standard',
      use_web: useWeb,
      use_obsidian: true,
      mode: advisorMode,
    };
  };

  const buildResponseMeta = (data: any) => {
    const metaParts: string[] = [];
    if (data.saved) metaParts.push('Obsidianへ自動保存しました');
    else if (data.save_reason) metaParts.push(`保存なし: ${data.save_reason}`);
    if (Array.isArray(data.web_hits) && data.web_hits.length > 0) {
      const sources = data.web_hits
        .slice(0, 2)
        .map((h: any) => h.domain || h.title || 'web')
        .filter(Boolean)
        .join(' / ');
      metaParts.push(`Web参照: ${data.web_hits.length}件${sources ? ' (' + sources + ')' : ''}`);
    }
    if (data.wiki_saved) metaParts.push('Wikiへ自動保存しました');
    if (data.weekly_saved) metaParts.push('週次レビューへ保存しました');
    return metaParts.join(' | ');
  };

  const updateStatus = (data: any) => {
    if (data.web_hits?.length) setStatusText('回答しました。Web参照あり。');
    else if (data.saved) setStatusText('必要なメモだけObsidianへ保存しました。');
    else setStatusText('回答しました。');
  };

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
    const fetchKey = `${score}:${pd_percent}:${industry_major}:${formData.asset_name || ""}`;
    if (initialFetchKeyRef.current === fetchKey) return;
    initialFetchKeyRef.current = fetchKey;

    const initialQuestion = `この案件（スコア ${score.toFixed(1)}点、${industry_major || "指定なし"}）の稟議を通すための「逆転戦略」を教えてくれ！`;
    const nextHistory: ChatMessage[] = [{ role: 'user', text: initialQuestion }];
    setChatHistory(nextHistory);
    fetchChat("", nextHistory, []);
  }, [score, pd_percent, industry_major, formData]);

  const handleSubmit = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) return;

    const nextHistory: ChatMessage[] = [...chatHistory, { role: 'user', text: trimmedQuestion }];
    setQuestion("");
    setChatHistory(nextHistory);
    await fetchChat(trimmedQuestion, nextHistory, chatHistory);
  };

  // マークダウンの簡易パース (### 見出し、**太字** をリッチに変換)
  const renderMarkdown = (text: string) => {
    let parsedText = text;
    // ### Heading 3
    parsedText = parsedText.replace(/### (.*?)(\n|$)/g, '<h4 class="font-bold text-base text-amber-700 mt-5 border-b border-amber-200 pb-1 mb-2">$1</h4>\n');
    // **bold**
    parsedText = parsedText.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-slate-800">$1</strong>');
    return parsedText;
  };

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
      </div>
      
      {/* チャットエリア */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <div className="text-center my-2 mb-6">
          <span className="text-[10px] font-bold text-slate-400 bg-slate-200 px-3 py-1 rounded-full">ダッシュボード連携セッション開始</span>
        </div>

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
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(chat.text) }}
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

        {loading && (
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
