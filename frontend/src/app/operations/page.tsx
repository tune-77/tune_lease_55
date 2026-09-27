"use client";

import Link from "next/link";
import { Fragment, useCallback, useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronDown,
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
import { apiClient, getApiErrorDetail } from "@/lib/api";

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

type DeletionAuditItem = {
  case_id: string;
  parent_table: string;
  status: string;
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
  status: string;
  metadata: Record<string, unknown>;
  items: DeletionAuditItem[];
};

type DeletionAuditResponse = {
  total: number;
  limit: number;
  offset: number;
  filters: { status: string; date_from: string; date_to: string };
  events: DeletionAuditEvent[];
};

type DeletionAuditFilters = {
  status: string;
  dateFrom: string;
  dateTo: string;
};

const emptyAuditFilters: DeletionAuditFilters = { status: "", dateFrom: "", dateTo: "" };

// status は削除の結末を表す。partial は「印は付けたのに消えなかった行がある」= 孤児リスクが
// 現実化した状態なので、目で拾えるよう赤にする。
const auditStatusStyles: Record<string, { label: string; tone: string }> = {
  completed: { label: "完了", tone: "border-emerald-200 bg-emerald-50 text-emerald-800" },
  partial: { label: "一部のみ削除", tone: "border-rose-300 bg-rose-50 text-rose-800" },
  no_match: { label: "対象なし", tone: "border-slate-200 bg-slate-100 text-slate-700" },
  started: { label: "未完了", tone: "border-amber-200 bg-amber-50 text-amber-800" },
};

const auditStatusOptions = [
  { value: "", label: "すべて" },
  { value: "completed", label: "完了" },
  { value: "partial", label: "一部のみ削除" },
  { value: "no_match", label: "対象なし" },
  { value: "started", label: "未完了" },
];

const formatAuditTime = (value: string) => {
  if (!value) return "—";
  // 監査ログは UTC naive（"YYYY-MM-DD HH:MM:SS"）で入ることがあるため Z を補って解釈する
  const normalized = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("ja-JP");
};

// partial は「一致した案件の一部が消えなかった」= 孤児リスクが現実化した状態なので、
// 1件でも混ざっていれば必ず警告する（件数で閾値を設けると見逃す側に倒れる）。
function describeAuditConcern(events: DeletionAuditEvent[]): string | null {
  const partials = events.filter((event) => event.status === "partial");
  if (partials.length === 0) return null;
  const affected = partials.reduce((total, event) => total + event.affected_screening_count, 0);
  return `一部のみ削除が${partials.length}件あります。印を付けたのに消えなかった案件があり、審査記録${affected}件が孤児になっている可能性があります。scripts/audit_case_deletion_integrity.py で確認してください。`;
}

function DeletionAuditPanel() {
  const [audit, setAudit] = useState<DeletionAuditResponse | null>(null);
  const [auditError, setAuditError] = useState("");
  const [auditLoading, setAuditLoading] = useState(true);
  const [filters, setFilters] = useState<DeletionAuditFilters>(emptyAuditFilters);
  const [appliedFilters, setAppliedFilters] = useState<DeletionAuditFilters>(emptyAuditFilters);
  const [expandedEventId, setExpandedEventId] = useState("");

  // 取得本体。同期部分では setState しない（effect から呼ぶため。
  // react-hooks/set-state-in-effect はeffect本体の同期的なsetStateを警告する）
  const fetchDeletionAudit = useCallback(async (next: DeletionAuditFilters) => {
    try {
      const response = await apiClient.get<DeletionAuditResponse>("/api/admin/deletion-audit", {
        params: {
          limit: 50,
          offset: 0,
          status: next.status || undefined,
          date_from: next.dateFrom || undefined,
          date_to: next.dateTo || undefined,
        },
      });
      setAudit(response.data);
      setAppliedFilters(next);
      setAuditError("");
    } catch (error) {
      setAuditError(getApiErrorDetail(error, "削除監査ログを取得できませんでした。"));
    } finally {
      setAuditLoading(false);
    }
  }, []);

  // ボタン操作用。初回はstateの初期値が読み込み中なので、ここだけ明示的に立てる
  const loadDeletionAudit = useCallback(
    (next: DeletionAuditFilters) => {
      setAuditLoading(true);
      setAuditError("");
      void fetchDeletionAudit(next);
    },
    [fetchDeletionAudit],
  );

  useEffect(() => {
    // マウント時に一度だけ取得する。setStateはawait後のみだがルールは間接到達も警告するため抑制する
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void fetchDeletionAudit(emptyAuditFilters);
  }, [fetchDeletionAudit]);

  const events = audit?.events ?? [];
  const filterActive = Boolean(appliedFilters.status || appliedFilters.dateFrom || appliedFilters.dateTo);
  const concern = describeAuditConcern(events);

  return (
    <section className="mx-auto max-w-7xl px-5 pb-10 md:px-8">
      <div className="rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <History className="h-6 w-6 text-rose-600" />
              <h2 className="text-2xl font-black">案件削除の監査ログ</h2>
            </div>
            <p className="mt-2 max-w-2xl text-sm font-bold leading-7 text-slate-600">
              どの経路でどの案件が消えたかを読み取り専用で確認します。ここから削除や修復は行いません。
            </p>
          </div>
          <button
            type="button"
            onClick={() => loadDeletionAudit(filters)}
            disabled={auditLoading}
            className="inline-flex shrink-0 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-2 text-sm font-black text-slate-700 hover:bg-slate-100 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${auditLoading ? "animate-spin" : ""}`} />
            再読み込み
          </button>
        </div>

        <div className="mt-5 flex flex-wrap items-end gap-3 rounded-lg border border-slate-200 bg-slate-50 p-4">
          <label className="flex flex-col gap-1 text-xs font-black text-slate-600">
            結末
            <select
              value={filters.status}
              onChange={(event) => setFilters({ ...filters, status: event.target.value })}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-900"
            >
              {auditStatusOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs font-black text-slate-600">
            開始日
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(event) => setFilters({ ...filters, dateFrom: event.target.value })}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-900"
            />
          </label>
          <label className="flex flex-col gap-1 text-xs font-black text-slate-600">
            終了日
            <input
              type="date"
              value={filters.dateTo}
              onChange={(event) => setFilters({ ...filters, dateTo: event.target.value })}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-bold text-slate-900"
            />
          </label>
          <button
            type="button"
            onClick={() => loadDeletionAudit(filters)}
            disabled={auditLoading}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-black text-white hover:bg-slate-800 disabled:opacity-50"
          >
            絞り込む
          </button>
          {filterActive && (
            <button
              type="button"
              onClick={() => {
                setFilters(emptyAuditFilters);
                loadDeletionAudit(emptyAuditFilters);
              }}
              disabled={auditLoading}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-black text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            >
              条件をクリア
            </button>
          )}
        </div>

        {concern && (
          <div className="mt-4 flex gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-rose-600" />
            <p className="text-sm font-bold leading-7 text-rose-950">{concern}</p>
          </div>
        )}

        {auditError && (
          <div className="mt-4 flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
            <p className="text-sm font-bold leading-7 text-amber-950">{auditError}</p>
          </div>
        )}

        {auditLoading && (
          <p className="mt-4 text-sm font-bold text-slate-500">削除監査ログを読み込み中…</p>
        )}

        {!auditLoading && !auditError && events.length === 0 && (
          <p className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm font-bold leading-7 text-slate-600">
            {filterActive
              ? "この条件に一致する削除イベントはありません。"
              : "記録された削除イベントはまだありません。"}
          </p>
        )}

        {!auditLoading && events.length > 0 && (
          <>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[860px] border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-xs font-black uppercase tracking-wider text-slate-500">
                    <th className="py-2 pr-3">発生時刻</th>
                    <th className="py-2 pr-3">経路</th>
                    <th className="py-2 pr-3">理由</th>
                    <th className="py-2 pr-3 text-right">指定</th>
                    <th className="py-2 pr-3 text-right">一致</th>
                    <th className="py-2 pr-3 text-right">削除</th>
                    <th className="py-2 pr-3 text-right">審査記録</th>
                    <th className="py-2 pr-3">結末</th>
                    <th className="py-2" />
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => {
                    const style = auditStatusStyles[event.status] ?? {
                      label: event.status || "不明",
                      tone: "border-slate-200 bg-slate-100 text-slate-700",
                    };
                    const expanded = expandedEventId === event.event_id;
                    return (
                      <Fragment key={event.event_id}>
                        <tr className="border-b border-slate-100 font-bold text-slate-700">
                          <td className="py-2 pr-3 whitespace-nowrap">{formatAuditTime(event.occurred_at)}</td>
                          <td className="py-2 pr-3 font-mono text-xs">{event.route}</td>
                          <td className="py-2 pr-3 text-xs">{event.reason}</td>
                          <td className="py-2 pr-3 text-right tabular-nums">{event.requested_count}</td>
                          <td className="py-2 pr-3 text-right tabular-nums">{event.matched_count}</td>
                          <td className="py-2 pr-3 text-right tabular-nums">{event.deleted_count}</td>
                          <td className="py-2 pr-3 text-right tabular-nums">{event.affected_screening_count}</td>
                          <td className="py-2 pr-3">
                            <span className={`inline-flex rounded-full border px-2 py-0.5 text-xs font-black ${style.tone}`}>
                              {style.label}
                            </span>
                          </td>
                          <td className="py-2">
                            <button
                              type="button"
                              onClick={() => setExpandedEventId(expanded ? "" : event.event_id)}
                              className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs font-black text-slate-600 hover:bg-slate-50"
                            >
                              {expanded ? "閉じる" : `案件 ${event.items.length}件`}
                              <ChevronDown className={`h-3 w-3 transition ${expanded ? "rotate-180" : ""}`} />
                            </button>
                          </td>
                        </tr>
                        {expanded && (
                          <tr className="border-b border-slate-100 bg-slate-50">
                            <td colSpan={9} className="px-3 py-3">
                              {event.items.length === 0 ? (
                                <p className="text-xs font-bold text-slate-500">対象案件の記録はありません。</p>
                              ) : (
                                <ul className="space-y-1">
                                  {event.items.map((item) => (
                                    <li
                                      key={`${event.event_id}-${item.case_id}`}
                                      className="flex flex-wrap items-center gap-2 text-xs font-bold text-slate-600"
                                    >
                                      <span className="font-mono text-slate-900">{item.case_id}</span>
                                      <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5">
                                        {item.parent_table}
                                      </span>
                                      <span
                                        className={
                                          item.status === "deleted"
                                            ? "text-emerald-700"
                                            : "text-rose-700"
                                        }
                                      >
                                        {item.status === "deleted" ? "削除済み" : item.status}
                                      </span>
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs font-bold text-slate-500">
              {audit ? `全 ${audit.total} 件のうち最新 ${events.length} 件を表示` : ""}
            </p>
          </>
        )}
      </div>
    </section>
  );
}

export default function OperationsPage() {
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

      <DeletionAuditPanel />
    </main>
  );
}
