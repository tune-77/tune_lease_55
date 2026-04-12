"use client";

import React from "react";
import { useSidebar } from "@/context/SidebarContext";

export default function ContentWrapper({ children }: { children: React.ReactNode }) {
  const { isCollapsed } = useSidebar();

  const marginClass = isCollapsed ? 'lg:ml-20' : 'lg:ml-64';

  return (
    <main className={`flex-1 min-h-screen relative transition-all duration-300 overflow-x-hidden pb-[env(safe-area-inset-bottom)] ${marginClass}`}>
      {children}
    </main>
  );
}
