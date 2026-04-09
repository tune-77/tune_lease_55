"use client";
import React, { useState, useEffect, useRef } from 'react';

const YANAMI_BOT_MESSAGES = [
  "システム稼働中。いつでもサポートします！",
  "[AIのぼやき] たまには温かいお茶でも飲みたいですね...",
  "[AIのぼやき] 次長、また稟議のハンコ渋ってましたよ...",
  "[AIのぼやき] 最近、運送業のリース申請が多い気がしますね。",
  "[AIのぼやき] DSCRが1.0割ってる案件は、やっぱりヒヤヒヤします...",
  "[AIのぼやき] キャッシュフロー計算書って、嘘がつけないから好きです。",
  "[AIのぼやき] 減価償却費を足し戻す瞬間が、一番テンション上がります！",
  "[AIのぼやき] 競合のXリースさん、最近金利下げてきてますよね...負けられません！"
];

export default function FloatingMebuki() {
  const [mebukiState, setMebukiState] = useState<'guide' | 'approve' | 'challenge' | 'reject'>('guide');
  const [message, setMessage] = useState("システム稼働中。いつでもサポートします！");
  const [isVisible, setIsVisible] = useState(true);
  const eventOverrideRef = useRef<boolean>(false);

  // カスタムイベントでめぶきの状態を制御する
  useEffect(() => {
    const handleMebukiEvent = (e: any) => {
      const { type, text } = e.detail;
      setMebukiState(type);
      setMessage(text);
      setIsVisible(true);
      eventOverrideRef.current = true; // イベントで上書きされたら、しばらくランダムぼやきを停止
      
      // 15秒後にランダムフラグを解除（またぼやけるようになる）
      setTimeout(() => {
        eventOverrideRef.current = false;
      }, 15000);
    };
    
    window.addEventListener('mebuki-action', handleMebukiEvent);
    
    // ランダムなぼやき（低頻度：約5分に1回）
    const boyakiInterval = setInterval(() => {
      if (!eventOverrideRef.current) {
        const randomIndex = Math.floor(Math.random() * YANAMI_BOT_MESSAGES.length);
        setMessage(YANAMI_BOT_MESSAGES[randomIndex]);
        setMebukiState('guide');
        setIsVisible(true);
      }
    }, 300000); // 300秒

    return () => {
      window.removeEventListener('mebuki-action', handleMebukiEvent);
      clearInterval(boyakiInterval);
    };
  }, []);

  return (
    <div className="fixed bottom-6 right-6 z-50 flex items-end justify-end pointer-events-none">
      {/* 吹き出し */}
      <div 
        className={`bg-white text-slate-800 p-4 rounded-2xl rounded-br-none shadow-2xl border-2 border-amber-200 text-sm font-bold leading-relaxed w-56 mb-6 mr-2 pointer-events-auto transition-all duration-300 transform origin-bottom-right whitespace-pre-wrap ${isVisible ? 'scale-100 opacity-100' : 'scale-75 opacity-0'}`}
      >
        {message}
      </div>
      
      {/* めぶきちゃん画像 */}
      <div 
        className="relative w-32 h-32 pointer-events-auto cursor-pointer hover:scale-105 transition-transform drop-shadow-2xl"
        onClick={() => setIsVisible(!isVisible)}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img 
          src={`/mebuki/${mebukiState}.png`} 
          alt="めぶきちゃん" 
          className="w-full h-full object-cover rounded-full border-4 border-white shadow-lg bg-emerald-100"
        />
        
        {/* オンラインバッジ */}
        <div className="absolute bottom-2 right-2 w-5 h-5 bg-green-500 rounded-full border-2 border-white shadow-sm skeleton-pulse flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-full opacity-60"></div>
        </div>
      </div>
    </div>
  );
}

// ユーティリティ関数（どこからでも呼べる）
export const triggerMebuki = (type: 'guide' | 'approve' | 'challenge' | 'reject', text: string) => {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('mebuki-action', { detail: { type, text } }));
  }
};
