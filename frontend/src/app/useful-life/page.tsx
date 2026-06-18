"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api";
import { BookOpen, Search } from "lucide-react";

type EquipmentItem = {
  name: string;
  years: number;
  note?: string;
};

type Category = {
  name: string;
  items: EquipmentItem[];
};

type UsefulLifeData = {
  description?: string;
  nta_useful_life_url?: string;
  categories: Category[];
};

const CAT_ICONS: Record<string, string> = {
  "建設・土木機械": "🏗️",
  "製造・加工機械": "⚙️",
  "情報通信・オフィス": "💻",
  "運輸・車両": "🚛",
  "空調・電気・設備": "🌡️",
  "宿泊・飲食・小売": "🍳",
  "農業・林業・漁業": "🌾",
  "医療・福祉": "🏥",
};

const YEAR_STYLE = (years: number): { bg: string; text: string; border: string } => {
  if (years <= 4) return { bg: "#eff6ff", text: "#2563eb", border: "#93c5fd" };
  if (years <= 6) return { bg: "#f0fdf4", text: "#16a34a", border: "#86efac" };
  if (years <= 10) return { bg: "#fff7ed", text: "#d97706", border: "#fcd34d" };
  return { bg: "#fef2f2", text: "#dc2626", border: "#fca5a5" };
};

export default function UsefulLifePage() {
  const [data, setData] = useState<UsefulLifeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [maxYears, setMaxYears] = useState(20);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.get<UsefulLifeData>("/api/asset/useful-life-all");
      setData(res.data);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const allItems = useMemo(() => {
    if (!data) return [];
    return data.categories.flatMap((cat) =>
      cat.items.map((it) => ({ ...it, category: cat.name })),
    );
  }, [data]);

  const filtered = useMemo(() => {
    return allItems.filter(
      (it) =>
        it.years <= maxYears &&
        (search === "" || it.name.toLowerCase().includes(search.toLowerCase()) ||
          it.category.toLowerCase().includes(search.toLowerCase())),
    );
  }, [allItems, search, maxYears]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <span className="text-slate-500">読み込み中...</span>
      </div>
    );
  }

  if (!data) {
    return <div className="p-8 text-red-600">データを取得できませんでした。</div>;
  }

  const isFiltering = search !== "" || maxYears < 20;

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* ヘッダー */}
      <div className="flex items-start gap-3">
        <BookOpen className="text-blue-600 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">法定耐用年数 一覧</h1>
          <p className="text-sm text-slate-500">
            国税庁 耐用年数表（令和5年）ベース。
            {data.nta_useful_life_url && (
              <a href={data.nta_useful_life_url} target="_blank" rel="noreferrer" className="ml-1 text-blue-500 underline">
                国税庁サイト →
              </a>
            )}
          </p>
        </div>
      </div>

      {/* 検索・フィルタ */}
      <div className="flex flex-wrap gap-3 items-center bg-slate-50 rounded-xl p-3">
        <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-3 py-1.5 flex-1 min-w-48">
          <Search size={14} className="text-slate-400" />
          <input
            type="text"
            placeholder="品目・カテゴリで検索（例: サーバー）"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="text-sm outline-none w-full text-slate-700 placeholder-slate-400"
          />
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-600">
          <span className="whitespace-nowrap">耐用年数</span>
          {[5, 8, 10, 20].map((n) => (
            <button
              key={n}
              onClick={() => setMaxYears(n)}
              className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                maxYears === n
                  ? "bg-blue-600 text-white"
                  : "bg-white border border-slate-300 text-slate-600 hover:bg-blue-50"
              }`}
            >
              {n}年以下
            </button>
          ))}
        </div>
        <span className="text-xs text-slate-400">{filtered.length} 件</span>
      </div>

      {/* コンテンツ */}
      {isFiltering ? (
        /* 検索結果フラットテーブル */
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 text-slate-600 text-xs border-b border-slate-200">
                <th className="text-left px-4 py-2">カテゴリ</th>
                <th className="text-left px-4 py-2">品目</th>
                <th className="text-center px-3 py-2">耐用年数</th>
                <th className="text-left px-3 py-2">備考</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={4} className="text-center py-8 text-slate-400">
                    条件に合う品目がありません
                  </td>
                </tr>
              ) : (
                filtered
                  .sort((a, b) => a.years - b.years)
                  .map((it, i) => {
                    const s = YEAR_STYLE(it.years);
                    const icon = CAT_ICONS[it.category] ?? "📦";
                    return (
                      <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                        <td className="px-4 py-2 text-slate-500 text-xs">{icon} {it.category}</td>
                        <td className="px-4 py-2 font-medium text-slate-800">{it.name}</td>
                        <td className="px-3 py-2 text-center">
                          <span
                            className="inline-block px-2 py-0.5 rounded-full text-xs font-bold"
                            style={{ background: s.bg, color: s.text, border: `1px solid ${s.border}` }}
                          >
                            {it.years}年
                          </span>
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-400">{it.note ?? ""}</td>
                      </tr>
                    );
                  })
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* カテゴリ別カード */
        <div className="space-y-4">
          {data.categories.map((cat) => {
            const icon = CAT_ICONS[cat.name] ?? "📦";
            const items = cat.items.filter((it) => it.years <= maxYears);
            if (items.length === 0) return null;
            return (
              <div key={cat.name} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-3 bg-slate-50 border-b border-slate-200">
                  <span className="text-lg">{icon}</span>
                  <h2 className="font-semibold text-slate-700">{cat.name}</h2>
                  <span className="ml-auto text-xs text-slate-400">{items.length} 品目</span>
                </div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-slate-500 border-b border-slate-100">
                      <th className="text-left px-4 py-2">品目</th>
                      <th className="text-center px-3 py-2 w-24">耐用年数</th>
                      <th className="text-left px-3 py-2">備考</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((it, i) => {
                      const s = YEAR_STYLE(it.years);
                      return (
                        <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
                          <td className="px-4 py-2 font-medium text-slate-800">{it.name}</td>
                          <td className="px-3 py-2 text-center">
                            <span
                              className="inline-block px-2 py-0.5 rounded-full text-xs font-bold"
                              style={{ background: s.bg, color: s.text, border: `1px solid ${s.border}` }}
                            >
                              {it.years}年
                            </span>
                          </td>
                          <td className="px-3 py-2 text-xs text-slate-400">{it.note ?? ""}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      )}

      {/* リース期間目安 */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <p className="text-sm font-semibold text-blue-800 mb-2">💡 リース期間の設定目安</p>
        <table className="w-full text-xs text-slate-700">
          <thead>
            <tr className="text-blue-700 font-semibold">
              <th className="text-left py-1">法定耐用年数</th>
              <th className="text-left py-1">推奨リース期間</th>
              <th className="text-left py-1">備考</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["4年（PC・POSレジ等）", "2〜4年", "耐用年数内に収める"],
              ["5年（トラック・サーバー等）", "3〜5年", "60〜100%が標準"],
              ["6〜8年（建設機械・エアコン等）", "4〜6年", "耐用年数の60〜80%が安全圏"],
              ["10年以上（工作機械等）", "5〜8年", "耐用年数の50〜70%推奨"],
            ].map(([life, term, note]) => (
              <tr key={life} className="border-t border-blue-100">
                <td className="py-1 font-medium">{life}</td>
                <td className="py-1">{term}</td>
                <td className="py-1 text-slate-500">{note}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-blue-700">
          ⚠️ リース期間が耐用年数の <strong>70%超</strong> になると満了時の残余価値が低下し、物件スコアに影響します。
        </p>
      </div>
    </div>
  );
}
