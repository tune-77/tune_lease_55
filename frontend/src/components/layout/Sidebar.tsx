"use client";

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Home, MessageSquare, ClipboardCheck, FileText, Bot, Zap, Factory, 
  PenTool, Users, Wrench, PencilRuler, History, ScrollText, 
  PieChart, LineChart, Target, Settings, Calendar, Share2, Network, 
  Eye, BarChart3, TrendingUp, Globe, ChevronDown, ChevronRight, Building,
  PanelLeftClose, PanelLeftOpen, X
} from 'lucide-react';
import { useSidebar } from '@/context/SidebarContext';

export default function Sidebar() {
  const pathname = usePathname();
  const { isCollapsed, toggleSidebar, isMobileOpen, toggleMobile } = useSidebar();

  const menuGroups = [
    {
      title: '審査メイン',
      items: [
        { name: 'ホーム', href: '/home', icon: Home, color: 'text-blue-400' },
        { name: 'リースくん (スマホUI)', href: '/lease-kun', icon: MessageSquare, color: 'text-amber-400' },
        { name: '審査・分析', href: '/', icon: ClipboardCheck, color: 'text-emerald-400' },
        { name: '審査レポート', href: '/report', icon: FileText, color: 'text-indigo-400' },
        { name: 'バッチ審査', href: '/batch', icon: Zap, color: 'text-yellow-400' },
        { name: '物件ファイナンス審査', href: '/finance', icon: Factory, color: 'text-stone-400' },
      ]
    },
    {
      title: '管理・エージェント',
      items: [
        { name: '汎用エージェントハブ', href: '/agent-hub', icon: Bot, color: 'text-violet-400' },
        { name: '結果登録 (成約/失注)', href: '/register', icon: PenTool, color: 'text-rose-400' },
      ]
    },
    {
      title: '高度分析・グラフ',
      items: [
        { name: '履歴分析・ダッシュボード', href: '/history-dash', icon: PieChart, color: 'text-sky-400' },
        { name: '3期財務分析', href: '/financial', icon: BarChart3, color: 'text-rose-500' },
        { name: 'TimesFM 時系列予測', href: '/timesfm', icon: TrendingUp, color: 'text-emerald-500' },
        { name: '競合関係グラフ', href: '/competitor', icon: Share2, color: 'text-orange-400' },
        { name: '案件類似ネットワーク', href: '/similar', icon: Network, color: 'text-teal-400' },
        { name: 'ビジュアルインサイト', href: '/visual', icon: Eye, color: 'text-blue-300' },
        { name: '文明年代記', href: '/chronicle', icon: Globe, color: 'text-purple-500' },
      ]
    },
    {
      title: '係数分析・ログ',
      items: [
        { name: '定性要因分析 (50件〜)', href: '/qualitative', icon: Target, color: 'text-pink-400' },
        { name: '定量要因分析 (50件〜)', href: '/quantitative', icon: LineChart, color: 'text-red-400' },
        { name: '係数分析・更新 (β)', href: '/coef-analysis', icon: Wrench, color: 'text-slate-400' },
        { name: '係数入力（事前係数）', href: '/coef-input', icon: PencilRuler, color: 'text-slate-400' },
        { name: '係数変更履歴', href: '/coef-history', icon: History, color: 'text-slate-400' },
        { name: 'アプリログ', href: '/logs', icon: ScrollText, color: 'text-slate-500' },
      ]
    },
    {
      title: '設定・マスタ',
      items: [
        { name: '審査ルール設定', href: '/rules', icon: Settings, color: 'text-slate-300' },
        { name: '基準金利マスタ', href: '/interest', icon: Calendar, color: 'text-slate-300' },
        { name: '企業番号設定', href: '/corporate-number', icon: Building, color: 'text-blue-300' },
      ]
    }
  ];

  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({
    '審査メイン': true,
    '管理・エージェント': false,
    '高度分析・グラフ': true,
    '係数分析・ログ': false,
    '設定・マスタ': false,
  });

  const toggleGroup = (title: string) => {
    if (isCollapsed) return;
    setOpenGroups(prev => ({ ...prev, [title]: !prev[title] }));
  };

  return (
    <>
      {/* モバイル用背景オーバーレイ */}
      {isMobileOpen && (
        <div 
          className="lg:hidden fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-50 transition-all duration-300"
          onClick={toggleMobile}
        />
      )}

      <aside className={`
        fixed left-0 top-0 h-screen bg-slate-900 border-r border-slate-800 text-slate-300 flex flex-col z-[60] shadow-2xl transition-all duration-300 ease-in-out
        ${isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        ${isCollapsed ? 'w-20' : 'w-64'}
      `}>
        {/* ロゴ・アプリ名 */}
        <div className={`h-16 flex-shrink-0 flex items-center bg-slate-950 border-b border-slate-800 transition-all duration-300 ${isCollapsed ? 'px-4 justify-center' : 'px-5'}`}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/30 flex-shrink-0">
            <ClipboardCheck className="w-5 h-5 text-white" />
          </div>
          {!isCollapsed && (
            <div className="ml-3 overflow-hidden whitespace-nowrap">
              <span className="font-black text-white text-sm tracking-widest block">LEASING</span>
              <span className="text-[10px] text-slate-400 font-bold tracking-tighter">ASSISTANT AI</span>
            </div>
          )}
          
          {/* デスクトップ用トグルボタン */}
          <button 
            onClick={toggleSidebar}
            className={`hidden lg:flex absolute -right-3 top-20 w-6 h-6 bg-slate-800 border border-slate-700 rounded-full items-center justify-center shadow-lg text-slate-400 hover:text-white transition-all z-10 hover:scale-110 active:scale-95`}
          >
            {isCollapsed ? <PanelLeftOpen className="w-3 h-3" /> : <PanelLeftClose className="w-3 h-3" />}
          </button>
        </div>

        {/* モバイル用閉じるボタン */}
        {isMobileOpen && (
          <button 
            onClick={toggleMobile}
            className="lg:hidden absolute top-4 right-4 p-2 text-slate-400 hover:text-white"
          >
            <X className="w-6 h-6" />
          </button>
        )}

        {/* メニューリスト */}
        <nav className="flex-1 overflow-y-auto py-6 px-3 space-y-5 scrollbar-hide">
          {menuGroups.map((group) => (
            <div key={group.title} className="mb-2">
              {!isCollapsed && (
                <button 
                  onClick={() => toggleGroup(group.title)}
                  className="w-full flex items-center justify-between text-[11px] font-black text-slate-500 uppercase tracking-widest px-3 mb-3 hover:text-slate-300 transition-colors group"
                >
                  <span>{group.title}</span>
                  {openGroups[group.title] ? <ChevronDown className="w-3 h-3 text-slate-600" /> : <ChevronRight className="w-3 h-3" />}
                </button>
              )}
              {isCollapsed && (
                <div className="h-px bg-slate-800 w-full mb-4 mx-auto" />
              )}
              
              <div className={`space-y-1 ${isCollapsed ? 'flex flex-col items-center' : ''}`}>
                {(isCollapsed || openGroups[group.title]) && group.items.map((item) => {
                  const isActive = pathname === item.href;
                  return (
                    <Link 
                      key={item.href}
                      href={item.href}
                      onClick={() => { if (isMobileOpen) toggleMobile(); }}
                      className={`
                        flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 group relative
                        ${isActive 
                           ? 'bg-blue-600/10 text-blue-400 border border-blue-500/20 shadow-inner' 
                           : 'hover:bg-slate-800 hover:text-white text-slate-400 border border-transparent'}
                        ${isCollapsed ? 'justify-center w-12 h-12' : 'w-full'}
                      `}
                      title={isCollapsed ? item.name : undefined}
                    >
                      <item.icon className={`w-5 h-5 flex-shrink-0 transition-transform group-hover:scale-110 ${isActive ? 'text-blue-400' : item.color}`} />
                      {!isCollapsed && <span className="tracking-tight">{item.name}</span>}
                      {isActive && !isCollapsed && (
                        <div className="absolute right-3 w-1.5 h-1.5 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.8)]" />
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* 下部設定エリア */}
        <div className={`p-4 bg-slate-950/50 border-t border-slate-800 transition-all ${isCollapsed ? 'items-center flex justify-center' : ''}`}>
           <div className={`flex items-center gap-3 p-2 bg-slate-900 rounded-2xl border border-slate-800 cursor-pointer hover:border-slate-700 transition-all ${isCollapsed ? 'justify-center w-12 h-12' : 'px-3 w-full'}`}>
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-[10px] font-black text-white shrink-0">
                USER
              </div>
              {!isCollapsed && (
                <div className="overflow-hidden">
                  <p className="text-xs font-black text-slate-200 truncate">Kobayashi Isao</p>
                  <p className="text-[10px] font-bold text-slate-500">Premium Plan</p>
                </div>
              )}
           </div>
        </div>
      </aside>
    </>
  );
}
