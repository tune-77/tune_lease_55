import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE } from '../../lib/api';
import { Activity, ShieldCheck, MessageSquare } from 'lucide-react';

interface GunshiAdviceProps {
  score: number;
  pd_percent: number;
  industry_major: string;
  formData: any;
  onChatLoaded?: (text: string) => void;
}

export default function GunshiAdvice({ score, pd_percent, industry_major, formData, onChatLoaded }: GunshiAdviceProps) {
  const [chatText, setChatText] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (score === 0) return;
    
    const fetchChat = async () => {
      setLoading(true);
      try {
        const payload = {
          score,
          pd_percent,
          industry_major,
          asset_name: formData.asset_name || "",
          resale: "標準",
          repeat_cnt: 1,
          subsidy: false,
          bank: false,
          intuition: formData.shinsa_intuition || 50,
          posterior: 0.5
        };
        const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/api/gunshi/chat`, payload);
        const fetchedText = res.data.chat_text;
        setChatText(fetchedText);
        if (onChatLoaded) {
          onChatLoaded(fetchedText);
        }
      } catch (err) {
        console.error("Failed to fetch gunshi chat", err);
        const errText = "【通信エラー】軍師からの戦略を受信できませんでした。";
        setChatText(errText);
        if (onChatLoaded) onChatLoaded(errText);
      } finally {
        setLoading(false);
      }
    };
    
    fetchChat();
  }, [score, pd_percent, industry_major, formData]);

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
      
      {/* チャットエリア */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        <div className="text-center my-2 mb-6">
          <span className="text-[10px] font-bold text-slate-400 bg-slate-200 px-3 py-1 rounded-full">ダッシュボード連携セッション開始</span>
        </div>

        {score > 0 && !loading && (
           <div className="flex gap-3 flex-row-reverse animate-in fade-in slide-in-from-right-4 duration-300">
             <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex justify-center items-center font-bold text-xs shadow-sm shrink-0">
               You
             </div>
             <div className="bg-blue-600 text-white p-3 rounded-2xl rounded-tr-none shadow-sm max-w-[85%] text-sm">
               この案件（スコア {score.toFixed(1)}点、{industry_major || "指定なし"}）の稟議を通すための「逆転戦略」を教えてくれ！
             </div>
           </div>
        )}

        {loading ? (
          <div className="flex gap-3">
             <div className="w-8 h-8 rounded-full bg-amber-500 text-white flex justify-center items-center font-bold text-xs shadow-sm shrink-0">
               🏯
             </div>
             <div className="bg-white p-4 rounded-2xl rounded-tl-none shadow border border-slate-200 flex flex-col gap-3 min-w-[200px]">
               <Activity className="w-5 h-5 animate-spin text-amber-500" />
               <span className="text-xs font-bold text-slate-400">軍師が直近のデータを分析し、<br/>戦略を練り上げています...<br/>（Gemini API 通信中）</span>
             </div>
          </div>
        ) : (
          chatText && (
            <div className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="w-8 h-8 rounded-full bg-amber-500 border-2 border-white text-white flex justify-center items-center font-black text-sm shadow-md shrink-0">
                🏯
              </div>
              {/* === StreamlitのMarkdown表示のような美しいスタイルで出力する箱 === */}
              <div 
                className="bg-white p-4 rounded-2xl rounded-tl-none shadow border border-amber-200 text-slate-700 leading-7 font-medium whitespace-pre-wrap text-[13px] sm:text-sm w-full prose prose-slate"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(chatText) }}
              />
            </div>
          )
        )}
        
        {score === 0 && (
           <div className="text-center text-sm text-slate-400 mt-10">数値を入力し「審査エンジンを実行」してください</div>
        )}
      </div>

      <div className="p-4 bg-white border-t border-slate-100 shrink-0">
        <div className="relative">
          <input 
            type="text" 
            placeholder="※自動生成モード（個別質問は未実装です）" 
            disabled
            className="w-full bg-slate-50 border border-slate-200 rounded-xl py-3 pl-4 pr-12 text-sm outline-none cursor-not-allowed text-slate-400"
          />
          <button disabled className="absolute flex items-center justify-center right-2 top-2 w-8 h-8 bg-blue-600 text-white rounded-xl disabled:bg-slate-300">
            <MessageSquare className="w-4 h-4 ml-0.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
