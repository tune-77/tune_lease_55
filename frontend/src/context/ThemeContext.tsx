"use client";

import React, { createContext, useContext, useState, useEffect } from 'react';

export type ThemeId = 'light' | 'dark' | 'warm' | 'cool' | 'forest';

export interface ThemeOption {
  id: ThemeId;
  label: string;
  bgColor: string;
  accentColor: string;
}

export const THEMES: ThemeOption[] = [
  { id: 'light',  label: 'ライト',    bgColor: '#f8fafc', accentColor: '#3b82f6' },
  { id: 'dark',   label: 'ダーク',    bgColor: '#0f172a', accentColor: '#60a5fa' },
  { id: 'warm',   label: 'ウォーム',  bgColor: '#fdf6e3', accentColor: '#f59e0b' },
  { id: 'cool',   label: 'クール',    bgColor: '#e8f4f8', accentColor: '#0891b2' },
  { id: 'forest', label: 'フォレスト',bgColor: '#f0f7f0', accentColor: '#16a34a' },
];

const STORAGE_KEY = 'lease-ai-theme';

interface ThemeContextType {
  theme: ThemeId;
  setTheme: (id: ThemeId) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<ThemeId>('light');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as ThemeId | null;
    const initial = saved && THEMES.some(t => t.id === saved) ? saved : 'light';
    setThemeState(initial);
    document.documentElement.setAttribute('data-theme', initial);
  }, []);

  const setTheme = (id: ThemeId) => {
    setThemeState(id);
    localStorage.setItem(STORAGE_KEY, id);
    document.documentElement.setAttribute('data-theme', id);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (ctx === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return ctx;
}
