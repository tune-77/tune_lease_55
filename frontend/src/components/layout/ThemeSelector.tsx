"use client";

import React from 'react';
import { Palette } from 'lucide-react';
import { useTheme, THEMES, ThemeId } from '@/context/ThemeContext';

type Props = {
  collapsed?: boolean;
};

export default function ThemeSelector({ collapsed = false }: Props) {
  const { theme, setTheme } = useTheme();

  if (collapsed) {
    return (
      <div className="flex justify-center">
        <Palette size={18} className="text-slate-500" />
      </div>
    );
  }

  return (
    <div className="px-1">
      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2 px-1">
        背景テーマ
      </p>
      <div className="flex items-center gap-2 px-1">
        {THEMES.map((t) => (
          <button
            key={t.id}
            onClick={() => setTheme(t.id as ThemeId)}
            aria-label={t.label}
            title={t.label}
            className="w-6 h-6 rounded-full flex-shrink-0 transition-transform hover:scale-110 active:scale-95"
            style={{
              backgroundColor: t.bgColor,
              border: theme === t.id
                ? `2px solid ${t.accentColor}`
                : '2px solid #475569',
              boxShadow: theme === t.id
                ? `0 0 0 2px ${t.accentColor}50`
                : 'none',
            }}
          />
        ))}
      </div>
    </div>
  );
}
