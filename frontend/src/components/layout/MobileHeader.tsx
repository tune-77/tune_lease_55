"use client";

import React from "react";
import { Menu, ClipboardCheck, X } from "lucide-react";
import { useSidebar } from "@/context/SidebarContext";

export default function MobileHeader() {
  const { toggleMobile } = useSidebar();

  return (
    <header className="lg:hidden h-16 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 sticky top-0 z-40">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <ClipboardCheck className="w-5 h-5 text-white" />
        </div>
        <span className="font-bold text-white text-xs tracking-wider">リース・アシスタント</span>
      </div>
      
      <button 
        onClick={toggleMobile}
        className="p-2 text-slate-400 hover:text-white transition-colors"
      >
        <Menu className="w-6 h-6" />
      </button>
    </header>
  );
}
