"use client";

import React from "react";
import { useSidebar } from "@/context/SidebarContext";

export default function ContentWrapper({ children }: { children: React.ReactNode }) {
  const { isCollapsed } = useSidebar();

  return (
    <main className={`
      flex-1 min-h-screen relative transition-all duration-300
      lg:ml-${isCollapsed ? '20' : '64'}
      w-full overflow-x-hidden
    `}>
      {children}
    </main>
  );
}
