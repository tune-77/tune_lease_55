"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Home, MessageSquare, ClipboardCheck, FileText, Zap, Factory,
  PenTool, Users, Calendar, Share2, Network,
  Eye, ChevronDown, ChevronRight, ChevronLeft,
  X, Menu, Table2, Swords, MessageCircle,
  BarChart2, BookOpen, Gift, HelpCircle, Megaphone, Calculator,
  LifeBuoy, ClipboardList, Brain,
  Orbit, Sparkles, Search
} from 'lucide-react';
import { useSidebar } from '@/context/SidebarContext';
import ThemeSelector from '@/components/layout/ThemeSelector';

export default function Sidebar() {
  const pathname = usePathname();
  const { isCollapsed, toggleSidebar, isMobileOpen, toggleMobile } = useSidebar();
  const hideMobileEdgeToggle = pathname === '/multi-shion-demo';
  const [isCloudRunHost, setIsCloudRunHost] = useState(false);
  const hideResearchOrgan =
    process.env.NEXT_PUBLIC_HIDE_RESEARCH_ORGAN === "1" || isCloudRunHost;

  useEffect(() => {
    setIsCloudRunHost(window.location.hostname.endsWith(".run.app"));
  }, []);

  const menuGroups = [
    {
      title: '審査ワークフロー',
      defaultOpen: true,
      items: [
        { name: '紫苑コンシェルジュ', href: '/', icon: Home, color: 'text-violet-400' },
        { name: '審査・分析', href: '/screening', icon: ClipboardCheck, color: 'text-emerald-400' },
        { name: 'ダッシュボード', href: '/home', icon: BarChart2, color: 'text-blue-400' },
        { name: '💬 AIチャット', href: '/chat', icon: MessageCircle, color: 'text-cyan-400' },
        { name: '審査レポート', href: '/report', icon: FileText, color: 'text-indigo-400' },
        { name: 'バッチ審査', href: '/batch', icon: Zap, color: 'text-yellow-400' },
        { name: '稟議書・見積依頼書', href: '/ringi', icon: FileText, color: 'text-indigo-400' },
        { name: '結果登録 (成約/失注)', href: '/register', icon: PenTool, color: 'text-rose-400' },
        { name: '過去案件一覧', href: '/cases', icon: Table2, color: 'text-cyan-400' },
      ]
    },
    {
      title: '🌸 紫苑 AI',
      defaultOpen: false,
      items: [
        { name: '🚀 ハッカソンデモ', href: '/demo', icon: Sparkles, color: 'text-yellow-300' },
        { name: 'リース知性体との対話', href: '/lease-intelligence', icon: Brain, color: 'text-violet-400' },
        { name: 'リースくん (スマホUI)', href: '/lease-kun', icon: MessageSquare, color: 'text-amber-400' },
        { name: 'マルチエージェント討論', href: '/debate', icon: Swords, color: 'text-violet-500' },
        ...(!hideResearchOrgan
          ? [{ name: '外部調査器官', href: '/research-organ', icon: Search, color: 'text-sky-300' }]
          : []),
        { name: 'システム概要', href: '/system-overview', icon: Orbit, color: 'text-fuchsia-400' },
      ]
    },
    {
      title: '分析・グラフ',
      defaultOpen: false,
      items: [
        { name: '営業部別分析', href: '/department', icon: Users, color: 'text-emerald-400' },
        { name: '業種別成約率', href: '/industry-winrate', icon: BarChart2, color: 'text-blue-400' },
        { name: '競合関係グラフ', href: '/competitor', icon: Share2, color: 'text-orange-400' },
        { name: '知識宇宙マップ', href: '/knowledge-space', icon: Network, color: 'text-cyan-300' },
        { name: 'ビジュアルインサイト', href: '/visual', icon: Eye, color: 'text-blue-300' },
      ]
    },
    {
      title: '参照・ナレッジ',
      defaultOpen: false,
      items: [
        { name: '法定耐用年数一覧', href: '/useful-life', icon: BookOpen, color: 'text-blue-400' },
        { name: '業種別リース物件例', href: '/industry-assets', icon: Factory, color: 'text-slate-500' },
        { name: '残価設定ガイドライン', href: '/residual-guide', icon: Calculator, color: 'text-purple-400' },
        { name: '営業向け説明ガイド', href: '/sales-guide', icon: Megaphone, color: 'text-blue-500' },
        { name: 'リース/融資/現金 比較', href: '/simulator', icon: Calculator, color: 'text-blue-400' },
        { name: 'FAQ', href: '/faq', icon: HelpCircle, color: 'text-slate-400' },
        { name: 'システム機能一覧', href: '/help', icon: LifeBuoy, color: 'text-blue-300' },
        { name: '改善ログ', href: '/improvement-log', icon: ClipboardList, color: 'text-slate-400' },
      ]
    },
    {
      title: '設定・マスタ',
      defaultOpen: false,
      items: [
        { name: '基準金利マスタ', href: '/interest', icon: Calendar, color: 'text-slate-300' },
        { name: '補助金情報', href: '/subsidy', icon: Gift, color: 'text-emerald-300' },
      ]
    }
  ];

  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(
    Object.fromEntries(menuGroups.map(g => [g.title, g.defaultOpen]))
  );

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
                      {!isCollapsed && <span className="min-w-0 truncate tracking-tight">{item.name}</span>}
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
        <div className={`p-4 bg-slate-950/50 border-t border-slate-800 space-y-3 transition-all ${isCollapsed ? 'flex flex-col items-center' : ''}`}>
          {/* テーマ選択 */}
          <ThemeSelector collapsed={isCollapsed} />

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

      {/* デスクトップ用 画面左端固定トグルボタン */}
      <button
        onClick={toggleSidebar}
        aria-label={isCollapsed ? 'サイドバーを開く' : 'サイドバーを閉じる'}
        className="hidden lg:flex fixed left-0 top-[58vh] -translate-y-1/2 z-[70] w-10 h-20 bg-slate-800 border border-slate-700 border-l-0 rounded-r-2xl items-center justify-center shadow-lg text-slate-400 hover:text-white hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 transition-all active:scale-95"
      >
        {isCollapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
      </button>

      {/* モバイル用 画面右端固定トグルボタン */}
      {!hideMobileEdgeToggle && (
        <button
          onClick={toggleMobile}
          aria-label={isMobileOpen ? 'サイドバーを閉じる' : 'サイドバーを開く'}
          className="lg:hidden fixed right-0 top-[58vh] -translate-y-1/2 z-[70] w-14 h-20 bg-slate-800 border border-slate-700 border-r-0 rounded-l-2xl flex items-center justify-center shadow-lg text-slate-400 hover:text-white hover:bg-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 transition-all active:scale-95"
        >
          {isMobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      )}
    </>
  );
}
