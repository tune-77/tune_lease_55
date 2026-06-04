"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Palette } from 'lucide-react';
import { useTheme, THEMES, ThemeId } from '@/context/ThemeContext';

export default function ThemeSelector() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const isDark = theme === 'dark';

  return (
    <div ref={ref} className="fixed bottom-24 right-4 z-40">
      {open && (
        <div
          className={`absolute bottom-14 right-0 rounded-xl shadow-xl border p-3 w-44 ${
            isDark
              ? 'bg-slate-800 border-slate-700 text-slate-100'
              : 'bg-white border-slate-200 text-slate-800'
          }`}
        >
          <p className={`text-xs font-semibold mb-2 px-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            背景テーマ
          </p>
          <div className="flex flex-col gap-1">
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => { setTheme(t.id as ThemeId); setOpen(false); }}
                className={`flex items-center gap-2 w-full rounded-lg px-2 py-1.5 text-sm transition-colors ${
                  theme === t.id
                    ? isDark
                      ? 'bg-slate-700 font-semibold'
                      : 'bg-slate-100 font-semibold'
                    : isDark
                      ? 'hover:bg-slate-700/60'
                      : 'hover:bg-slate-50'
                }`}
              >
                <span
                  className="inline-block w-5 h-5 rounded-full border-2 flex-shrink-0"
                  style={{
                    backgroundColor: t.bgColor,
                    borderColor: theme === t.id ? t.accentColor : '#cbd5e1',
                    boxShadow: theme === t.id ? `0 0 0 2px ${t.accentColor}40` : 'none',
                  }}
                />
                <span>{t.label}</span>
                {theme === t.id && (
                  <span className="ml-auto text-xs" style={{ color: t.accentColor }}>✓</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={() => setOpen(!open)}
        aria-label="テーマを変更"
        className={`w-11 h-11 rounded-full shadow-lg flex items-center justify-center transition-all ${
          open
            ? 'scale-95 opacity-90'
            : 'hover:scale-105'
        } ${
          isDark
            ? 'bg-slate-700 text-slate-200 hover:bg-slate-600'
            : 'bg-white text-slate-600 hover:bg-slate-50'
        } border ${isDark ? 'border-slate-600' : 'border-slate-200'}`}
      >
        <Palette size={20} />
      </button>
    </div>
  );
}
