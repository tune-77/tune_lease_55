"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Cloud,
  Database,
  GitBranch,
  History,
  Lock,
  Orbit,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { apiClient } from "@/lib/api";

type DeletionAuditItem = {
  case_id: string;
  parent_table: string;
  status: "matched" | "deleted" | "not_found" | string;
};

type DeletionAuditEvent = {
  event_id: string;
  occurred_at: string;
  route: string;
  reason: string;
  requested_count: number;
  matched_count: number;
  deleted_count: number;
  affected_screening_count: number;
  status: "started" | "completed" | "partial" | "no_match" | string;
  items: DeletionAuditItem[];
};

type DeletionAuditResponse = {
  total: number;
  limit: number;
  offset: number;
  events: DeletionAuditEvent[];
};

type DeletionAuditFilters = {
  status: string;
  dateFrom: string;
  dateTo: string;
};

const emptyAuditFilters: DeletionAuditFilters = { status: "", dateFrom: "", dateTo: "" };

const operationCards = [
  {
    title: "構成を確認する",
    href: "/system-overview",
    detail: "紫苑を中心に、審査AI・Obsidian・判断資産・改善ループがどう接続されているかを見る。",
    icon: Orbit,
    tone: "from-indigo-500 to-violet-600",
  },
  {
    title: "運用サイクルを見る",
    href: "/devops",
    detail: "Cloud Run、Cloud Build、デモDB分離、検疫、昇格までの運用ループを確認する。",
    icon: GitBranch,
    tone: "from-emerald-500 to-teal-600",
  },
  {
    title: "記憶システムを点検する",
    href: "/shion-memory-system",
    detail: "判断資産候補、記憶レビュー、Memory Engineering の状態を確認する。",
    icon: Database,
    tone: "from-sky-500 to-cyan-600",
  },
];

const runtimeSteps = [
  {
    title: "Cloud Run / local runtime",
    detail: "API・Web・スコアリング・チャットを動かす実行環境。Cloud Run版はデモDBで本体DBを守る。",
    icon: Cloud,
  },
  {
    title: "Brain / judgment assets",
    detail: "Obsidian / Markdown Vault 側に判断資産・違和感・改善ログを保持し、正本は直接書き換えない。",
    icon: Database,
  },
  {
    title: "Review / quarantine / promote",
    detail: "改善候補や帰還データは人間レビューを通し、採用・修正採用・保留・却下を残してから昇格する。",
    icon: ShieldCheck,
  },
];

const successMetrics = [
  "変更後30日間の /operations 訪問数",
  "統合画面から /system-overview・/devops・/shion-memory-system へ進んだ回数",
  "改善ログ上の関連情報閲覧数と、success_metric の事後変化",
];

export default function OperationsPage() {
  const [deletionAudit, setDeletionAudit] = useState<DeletionAuditResponse | null>(null);
  const [deletionAuditError, setDeletionAuditError] = useState("");
  const [deletionAuditLoading, setDeletionAuditLoading] = useState(true);
  const [auditFilters, setAuditFilters] = useState<DeletionAuditFilters>(emptyAuditFilters);
  const [appliedAuditFilters, setAppliedAuditFilters] = useState<DeletionAuditFilters>(emptyAuditFilters);

  const loadDeletionAudit = useCallback(async (filters: DeletionAuditFilters) => {
    setDeletionAuditLoading(true);
    setDeletionAuditError("");
    try {
      const response = await apiClient.get<DeletionAuditResponse>("/api/admin/deletion-audit", {
        params: {
          limit: 50,
          offset: 0,
          status: filters.status || undefined,
          date_from: filters.dateFrom || undefined,
          date_to: filters.dateTo || undefined,
        },
      });
      setDeletionAudit(response.data);
      setAppliedAuditFilters(filters);
    } catch {
      setDeletionAuditError("削除監査ログを取得できませんでした。");
    } finally {
      setDeletionAuditLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDeletionAudit(emptyAuditFilters);
  }, [loadDeletionAudit]);

  const auditFilterActive = Boolean(
    appliedAuditFilters.status || appliedAuditFilters.dateFrom || appliedAuditFilters.dateTo,
  );

  const formatAuditTime = (value: string) => {
    if (!value) return "—";
    const normalized = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
    const parsed = new Date(normalized);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("ja-JP");
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-7xl px-5 py-10 md:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-fuchsia-200 bg-fuchsia-50 px-3 py-1 text-xs font-black text-fuchsia-800">
                <Settings2 className="h-4 w-4" />
                システム管理 / 運用情報
              </div>
              <h1 className="mt-5 text-3xl font-black tracking-tight text-slate-950 md:text-5xl">
                低頻度の管理画面を、ここで一度見渡す
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-8 text-slate-600">
                システム概要とDevOpsサイクルを個別に探すのではなく、運用で見るべき情報を一画面にまとめます。
                詳細が必要な時だけ、下のカードから元画面へ進みます。
              </p>
            </div>
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm font-bold leading-7 text-emerald-950 lg:w-[360px]">
              <div className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-emerald-700">
                <CheckCircle2 className="h-4 w-4" />
                採用中の仮説
              </div>
              <p className="mt-2">
                個別アクセスが少ない管理系情報を統合し、到達性と情報閲覧数の増加を30日で確認します。
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-4 px-5 py-8 md:px-8 lg:grid-cols-3">
        {operationCards.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="group rounded-lg border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-fuchsia-200 hover:shadow-md"
          >
            <div className={`inline-flex h-11 w-11 items-center justify-center rounded-lg bg-gradient-to-br ${card.tone} text-white shadow-sm`}>
              <card.icon className="h-5 w-5" />
            </div>
            <div className="mt-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-black text-slate-950">{card.title}</h2>
              <ArrowRight className="h-4 w-4 text-slate-400 transition group-hover:text-fuchsia-600" />
            </div>
            <p className="mt-2 text-sm font-bold leading-7 text-slate-600">{card.detail}</p>
          </Link>
        ))}
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 pb-8 md:px-8 lg:grid-cols-[1fr_0.8fr]">
        <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-fuchsia-600" />
            <h2 className="text-2xl font-black">運用で見るべき中核だけ</h2>
          </div>
          <div className="mt-5 space-y-3">
            {runtimeSteps.map((step) => (
              <div key={step.title} className="flex gap-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white text-slate-700 shadow-sm">
                  <step.icon className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-sm font-black text-slate-950">{step.title}</div>
                  <p className="mt-1 text-xs font-bold leading-6 text-slate-600">{step.detail}</p>
                </div>
              </div>
            ))}
          </div>
          <Link
            href="/cloudrun-return-review"
            className="mt-5 inline-flex items-center gap-2 rounded-lg border border-teal-200 bg-teal-50 px-4 py-3 text-sm font-black text-teal-800 hover:bg-teal-100"
          >
            帰還データ検疫を開く
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>

        <aside className="space-y-5">
          <section className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-5 w-5 text-emerald-600" />
              <h2 className="text-lg font-black">効き方の追跡</h2>
            </div>
            <ul className="mt-4 space-y-3">
              {successMetrics.map((metric) => (
                <li key={metric} className="flex gap-2 text-sm font-bold leading-7 text-slate-600">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                  <span>{metric}</span>
                </li>
              ))}
            </ul>
          </section>

          <section className="rounded-lg border border-violet-200 bg-violet-50 p-5">
            <div className="flex gap-3">
              <Lock className="mt-0.5 h-5 w-5 shrink-0 text-violet-700" />
              <p className="text-sm font-bold leading-7 text-violet-950">
                統合は入口を変えるだけです。詳細情報は削除せず、必要な人が深掘りできるよう元画面を残します。
              </p>
            </div>
          </section>
        </aside>
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-12 md:px-8">
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <div className="flex flex-col gap-4 border-b border-slate-200 px-5 py-5 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-2">
                <History className="h-5 w-5 text-rose-600" />
                <h2 className="text-xl font-black text-slate-950">案件削除監査</h2>
                {deletionAudit && (
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-black text-slate-700">
                    {auditFilterActive ? `該当${deletionAudit.total}件` : `全${deletionAudit.total}件`}
                  </span>
                )}
              </div>
              <p className="mt-2 text-sm font-bold leading-6 text-slate-600">
                削除経路・理由・対象案件ID・関連審査記録を確認する読み取り専用ログです。
              </p>
            </div>
            <button
              type="button"
              onClick={() => void loadDeletionAudit(appliedAuditFilters)}
              disabled={deletionAuditLoading}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-black text-slate-700 hover:bg-slate-50 disabled:cursor-wait disabled:opacity-60"
            >
              <RefreshCw className={`h-4 w-4 ${deletionAuditLoading ? "animate-spin" : ""}`} />
              再読込
            </button>
          </div>

          <form
            className="grid gap-3 border-b border-slate-200 bg-slate-50/70 px-5 py-4 md:grid-cols-[1fr_1fr_1fr_auto] md:items-end"
            onSubmit={(event) => {
              event.preventDefault();
              void loadDeletionAudit(auditFilters);
            }}
          >
            <label className="text-xs font-black text-slate-700">
              状態
              <select
                value={auditFilters.status}
                onChange={(event) => setAuditFilters((current) => ({ ...current, status: event.target.value }))}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-900 outline-none focus:border-fuchsia-400"
              >
                <option value="">すべて</option>
                <option value="completed">completed（削除完了）</option>
                <option value="no_match">no_match（対象なし）</option>
                <option value="partial">partial（一部）</option>
                <option value="started">started（処理中）</option>
              </select>
            </label>
            <label className="text-xs font-black text-slate-700">
              開始日
              <input
                type="date"
                value={auditFilters.dateFrom}
                max={auditFilters.dateTo || undefined}
                onChange={(event) => setAuditFilters((current) => ({ ...current, dateFrom: event.target.value }))}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-900 outline-none focus:border-fuchsia-400"
              />
            </label>
            <label className="text-xs font-black text-slate-700">
              終了日
              <input
                type="date"
                value={auditFilters.dateTo}
                min={auditFilters.dateFrom || undefined}
                onChange={(event) => setAuditFilters((current) => ({ ...current, dateTo: event.target.value }))}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-900 outline-none focus:border-fuchsia-400"
              />
            </label>
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={deletionAuditLoading}
                className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-black text-white hover:bg-slate-800 disabled:cursor-wait disabled:opacity-60"
              >
                絞り込む
              </button>
              <button
                type="button"
                disabled={deletionAuditLoading || (!auditFilterActive && !auditFilters.status && !auditFilters.dateFrom && !auditFilters.dateTo)}
                onClick={() => {
                  setAuditFilters(emptyAuditFilters);
                  void loadDeletionAudit(emptyAuditFilters);
                }}
                className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-black text-slate-700 hover:bg-slate-50 disabled:opacity-40"
              >
                解除
              </button>
            </div>
          </form>

          {deletionAuditError ? (
            <div className="m-5 flex items-center gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm font-bold text-rose-900">
              <AlertTriangle className="h-5 w-5 shrink-0" />
              {deletionAuditError}
            </div>
          ) : deletionAuditLoading && !deletionAudit ? (
            <div className="p-8 text-center text-sm font-bold text-slate-500">監査ログを読み込み中...</div>
          ) : !deletionAudit?.events.length ? (
            <div className="p-8 text-center">
              <ShieldCheck className="mx-auto h-8 w-8 text-emerald-500" />
              <p className="mt-3 text-sm font-black text-slate-800">
                {auditFilterActive ? "条件に一致する削除イベントはありません" : "導入後の削除イベントはまだありません"}
              </p>
              <p className="mt-1 text-xs font-bold text-slate-500">
                {auditFilterActive ? "条件を変えるか、フィルターを解除してください。" : "過去の削除履歴は推測で補完していません。"}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                <thead className="bg-slate-50 text-xs font-black uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-5 py-3">日時 / 状態</th>
                    <th className="px-5 py-3">経路 / 理由</th>
                    <th className="px-5 py-3">件数</th>
                    <th className="px-5 py-3">対象案件ID</th>
                    <th className="px-5 py-3">イベントID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 bg-white">
                  {deletionAudit.events.map((event) => (
                    <tr key={event.event_id} className="align-top hover:bg-slate-50/70">
                      <td className="whitespace-nowrap px-5 py-4">
                        <div className="font-black text-slate-900">{formatAuditTime(event.occurred_at)}</div>
                        <span className={`mt-2 inline-flex rounded-full px-2 py-1 text-[11px] font-black ${
                          event.status === "completed"
                            ? "bg-emerald-100 text-emerald-800"
                            : event.status === "no_match"
                              ? "bg-slate-100 text-slate-700"
                              : "bg-amber-100 text-amber-900"
                        }`}>
                          {event.status}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <div className="font-black text-slate-900">{event.route}</div>
                        <div className="mt-1 text-xs font-bold text-slate-500">{event.reason || "—"}</div>
                      </td>
                      <td className="whitespace-nowrap px-5 py-4 text-xs font-bold leading-6 text-slate-600">
                        <div>要求 {event.requested_count} / 一致 {event.matched_count}</div>
                        <div>削除 {event.deleted_count} / 審査記録 {event.affected_screening_count}</div>
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex max-w-md flex-wrap gap-1.5">
                          {event.items.map((item) => (
                            <span
                              key={`${event.event_id}-${item.case_id}`}
                              title={`${item.parent_table}: ${item.status}`}
                              className={`rounded-md border px-2 py-1 font-mono text-[11px] font-bold ${
                                item.status === "deleted"
                                  ? "border-rose-200 bg-rose-50 text-rose-800"
                                  : "border-slate-200 bg-slate-50 text-slate-700"
                              }`}
                            >
                              {item.case_id}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="max-w-52 break-all px-5 py-4 font-mono text-[11px] font-bold text-slate-500">
                        {event.event_id}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
