"use client";

import React from "react";
import { ClipboardCheck } from "lucide-react";

export default function MobileHeader() {
  return (
    <header className="lg:hidden h-16 bg-slate-900 border-b border-slate-800 flex items-center px-4 sticky top-0 z-40 pt-[env(safe-area-inset-top)]">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <ClipboardCheck className="w-5 h-5 text-white" />
        </div>
        <span className="font-bold text-white text-xs tracking-wider">リース・アシスタント</span>
      </div>
    </header>
  );
}
