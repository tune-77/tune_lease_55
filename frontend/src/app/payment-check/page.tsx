"use client";

import React, { useCallback, useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import { AlertTriangle, CheckCircle2, RefreshCw, XCircle, CreditCard } from "lucide-react";

type Alert = {
  id: number;
  contract_id: string;
  check_date: string;
  payment_status: string;
  overdue_amount: number;
  screening_score: number | null;
  original_score: number | null;
  notes: string;
  industry_sub: string | null;
  severity: "warning" | "critical";
  message: string;
};

type Summary = {
  normal: number;
  overdue: number;
  default: number;
  completed: number;
};

type AlertData = {
  alerts: Alert[];
  summary: Summary;
  total: number;
};

const SEVERITY_STYLE = {
  critical: {
    bg: "bg-red-50",
    border: "border-red-300",
    icon: <XCircle size={18} className="text-red-500 shrink-0 mt-0.5" />,
    badge: "bg-red-100 text-red-700",
    label: "デフォルト",
  },
  warning: {
    bg: "bg-amber-50",
    border: "border-amber-300",
    icon: <AlertTriangle size={18} className="text-amber-500 shrink-0 mt-0.5" />,
    badge: "bg-amber-100 text-amber-700",
    label: "延滞",
  },
};

export default function PaymentCheckPage() {
  const [data, setData] = useState<AlertData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.get<AlertData>("/api/payment/alerts");
      setData(res.data);
    } catch {
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <CreditCard className="text-slate-600" size={26} />
        <div>
          <h1 className="text-2xl font-bold text-slate-800">支払い状況アラート</h1>
          <p className="text-sm text-slate-500">延滞・デフォルト案件を自動検出します。</p>
        </div>
        <button
          onClick={fetchData}
          className="ml-auto flex items-center gap-1 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-sm text-slate-700"
        >
          <RefreshCw size={14} /> 更新
        </button>
      </div>

      {/* サマリー */}
      {data && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "正常", value: data.summary.normal, color: "text-green-600", bg: "bg-green-50", icon: <CheckCircle2 size={18} className="text-green-500" /> },
            { label: "延滞", value: data.summary.overdue, color: "text-amber-600", bg: "bg-amber-50", icon: <AlertTriangle size={18} className="text-amber-500" /> },
            { label: "デフォルト", value: data.summary.default, color: "text-red-600", bg: "bg-red-50", icon: <XCircle size={18} className="text-red-500" /> },
            { label: "完済", value: data.summary.completed, color: "text-slate-500", bg: "bg-slate-50", icon: <CheckCircle2 size={18} className="text-slate-400" /> },
          ].map(({ label, value, color, bg, icon }) => (
            <div key={label} className={`rounded-xl border border-slate-200 ${bg} p-4 flex items-center gap-3 shadow-sm`}>
              {icon}
              <div>
                <p className="text-xs text-slate-500">{label}</p>
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="animate-spin text-slate-400" size={24} />
        </div>
      ) : !data ? (
        <div className="p-8 text-center text-red-600">データの取得に失敗しました</div>
      ) : data.alerts.length === 0 ? (
        <div className="bg-green-50 border border-green-200 rounded-xl p-8 text-center">
          <CheckCircle2 size={36} className="text-green-400 mx-auto mb-2" />
          <p className="font-semibold text-green-700">
            {data.total === 0 ? "支払い履歴がまだ登録されていません" : "現在、延滞・デフォルト案件はありません"}
          </p>
          <p className="text-xs text-green-600 mt-1">
            {data.total === 0
              ? "Streamlit の「支払状況登録」から実績を記録してください"
              : `登録済み ${data.total} 件すべて正常または完済です`}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-700">
            ⚠️ 要対応案件 <span className="text-red-600">{data.alerts.length} 件</span>
          </p>
          {data.alerts.map((alert) => {
            const style = SEVERITY_STYLE[alert.severity];
            return (
              <div
                key={alert.id}
                className={`rounded-xl border ${style.border} ${style.bg} p-4`}
              >
                <div className="flex items-start gap-3">
                  {style.icon}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-bold text-slate-800 text-sm">契約ID: {alert.contract_id}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-bold ${style.badge}`}>{style.label}</span>
                      {alert.industry_sub && (
                        <span className="text-xs text-slate-500">{alert.industry_sub}</span>
                      )}
                    </div>
                    <p className="text-sm text-slate-700 mt-1">{alert.message}</p>
                    <div className="flex gap-4 mt-2 text-xs text-slate-500">
                      <span>確認日: <strong>{alert.check_date}</strong></span>
                      {alert.screening_score && <span>審査スコア: <strong>{alert.screening_score}pt</strong></span>}
                      {alert.original_score && <span>原スコア: <strong>{alert.original_score}pt</strong></span>}
                    </div>
                    {alert.notes && (
                      <p className="mt-1.5 text-xs text-slate-500 bg-white/60 rounded px-2 py-1">{alert.notes}</p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500">
        <p className="font-semibold text-slate-600 mb-1">💡 使い方</p>
        <p>支払い履歴は Streamlit の <strong>「💳 支払状況登録」</strong> ページから登録してください。延滞・デフォルトが記録されると、このページに自動でアラートが表示されます。</p>
      </div>
    </div>
  );
}
