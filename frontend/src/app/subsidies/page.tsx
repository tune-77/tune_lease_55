"use client";

import React, { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { Gift, Search, ExternalLink, RefreshCw } from "lucide-react";

type Subsidy = {
  id: number;
  name: string;
  max_amount: number;
  industry_codes: string;
  asset_keywords: string;
  deadline: string;
  url: string;
  notes: string;
  active: number;
};

const AMOUNT_COLOR = (amount: number) => {
  if (amount >= 5000) return { bg: "#fef2f2", text: "#dc2626", border: "#fca5a5" };
  if (amount >= 1000) return { bg: "#fff7ed", text: "#d97706", border: "#fcd34d" };
  return { bg: "#f0fdf4", text: "#16a34a", border: "#86efac" };
};

export default function SubsidiesPage() {
  const [subsidies, setSubsidies] = useState<Subsidy[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filtered, setFiltered] = useState<Subsidy[]>([]);

  const fetchData = useCallback(async (q = "") => {
    setLoading(true);
    try {
      const res = await axios.get<Subsidy[]>("/api/subsidies", { params: q ? { q } : {} });
      setSubsidies(res.data);
      setFiltered(res.data);
    } catch {
      setSubsidies([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  useEffect(() => {
    const q = search.toLowerCase();
    setFiltered(
      q
        ? subsidies.filter(
            (s) =>
              s.name.toLowerCase().includes(q) ||
              (s.asset_keywords || "").toLowerCase().includes(q) ||
              (s.notes || "").toLowerCase().includes(q),
          )
        : subsidies,
    );
  }, [search, subsidies]);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* ヘッダー */}
      <div className="flex items-start gap-3">
        <Gift className="text-amber-500 mt-1" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">補助金情報</h1>
          <p className="text-sm text-slate-500">リース対象設備に活用できる主要補助金の一覧。審査時の参考にご活用ください。</p>
        </div>
        <button
          onClick={() => fetchData(search)}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm text-slate-700"
        >
          <RefreshCw size={14} /> 更新
        </button>
      </div>

      {/* 検索 */}
      <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-xl px-3 py-2 shadow-sm">
        <Search size={14} className="text-slate-400" />
        <input
          type="text"
          placeholder="設備名・キーワードで絞り込み（例: サーバー、農業、省エネ）"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="text-sm outline-none w-full text-slate-700 placeholder-slate-400"
        />
        {search && (
          <button onClick={() => setSearch("")} className="text-slate-400 hover:text-slate-600 text-xs">✕</button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="animate-spin text-amber-400" size={24} />
          <span className="ml-2 text-slate-500">読み込み中...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-12 text-slate-400">該当する補助金がありません</div>
      ) : (
        <div className="space-y-4">
          {filtered.map((s) => {
            const c = AMOUNT_COLOR(s.max_amount);
            const keywords = (s.asset_keywords || "").split(",").map((k) => k.trim()).filter(Boolean);
            return (
              <div key={s.id} className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="flex items-center gap-3 px-4 py-3 bg-slate-50 border-b border-slate-200">
                  <h2 className="font-semibold text-slate-800 flex-1">{s.name}</h2>
                  <span
                    className="text-sm font-bold px-2.5 py-0.5 rounded-full"
                    style={{ background: c.bg, color: c.text, border: `1px solid ${c.border}` }}
                  >
                    最大 {s.max_amount.toLocaleString()}万円
                  </span>
                </div>
                <div className="px-4 py-3 space-y-2">
                  <p className="text-sm text-slate-600">{s.notes}</p>
                  {keywords.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {keywords.map((kw) => (
                        <span
                          key={kw}
                          className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs cursor-pointer hover:bg-amber-50 hover:text-amber-700"
                          onClick={() => setSearch(kw)}
                        >
                          {kw}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="flex items-center justify-between text-xs text-slate-400 pt-1">
                    <span>申請期限: <strong className="text-slate-600">{s.deadline || "要確認"}</strong></span>
                    {s.url && (
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1 text-blue-500 hover:text-blue-700 font-medium"
                      >
                        公式サイト <ExternalLink size={11} />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
        <p className="font-semibold mb-1">💡 補助金とリースの組み合わせ</p>
        <ul className="list-disc list-inside space-y-1 text-xs text-amber-700">
          <li>補助金採択後にリース契約を結ぶ場合、補助対象設備の所有権がリース会社にある点に注意</li>
          <li>補助金公募要領で「リース可」と明記されているか必ず確認する</li>
          <li>ものづくり補助金・IT導入補助金はリース併用実績あり（要事務局確認）</li>
          <li>補助金の審査スコアへの影響は <strong>+2〜5pt</strong> 程度（案件内容による）</li>
        </ul>
      </div>
    </div>
  );
}
