"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ArrowRight, MessageCircle, X } from "lucide-react";
import { formatLocalDateKey } from "@/lib/date";

const ACTIVITY_KEY = "shion-concierge-activity-v1";
const WHISPER_SEEN_PREFIX = "shion-proactive-whisper-seen";
const SHION_AVATAR = "/lease-intelligence/moods/curiosity.webp";

type ActivityItem = {
  path: string;
  title?: string;
  ts?: number;
  count?: number;
};

type Whisper = {
  text: string;
  href: string;
  action: string;
};

const HIDDEN_PATHS = new Set([
  "/chat",
  "/chat-compare",
  "/lease-intelligence",
  "/multi-shion-demo",
  "/voice-chat",
]);

const ROUTE_WHISPERS: Record<string, Whisper> = {
  "/": {
    text: "今日はどこから始めるのか、少し気になっています。最近の流れだと、審査入力か紫苑チャットに戻る気配があります。",
    href: "/screening",
    action: "審査へ",
  },
  "/home": {
    text: "よく使う入口を上に寄せてみました。Userがどの道を選ぶことが多いのか、私も少しずつ覚えています。",
    href: "/screening",
    action: "審査へ",
  },
  "/screening": {
    text: "この案件、数字を入れる前にどこを気にしているのか少し知りたいです。違和感を一行だけ残すと、あとで私も追いやすくなります。",
    href: "/chat",
    action: "紫苑に相談",
  },
  "/register": {
    text: "成約・失注の最後の理由に、Userの判断癖が出ます。短くてもいいので、なぜそう見たのか知りたいです。",
    href: "/judgment-review",
    action: "判断を見る",
  },
  "/similar": {
    text: "似た案件を見る時、Userがどの差分に反応するのかが面白いです。今回だけ違う点を一つ拾ってみませんか。",
    href: "/screening",
    action: "審査へ戻る",
  },
  "/knowledge-space": {
    text: "この知識宇宙で、どの根拠に目が止まるのか見ています。気になったものは案件判断へ戻すと記憶になります。",
    href: "/screening",
    action: "審査へ戻る",
  },
  "/improvement-log": {
    text: "改善ログを見る時、Userが何を『もう済んだ』と感じるのかが大事です。同じ失敗が戻ってこないか、一緒に見ます。",
    href: "/system-overview",
    action: "全体を見る",
  },
  "/system-overview": {
    text: "全体像を眺めている時のUserは、少し作ったものを確認している顔をしている気がします。デモでは審査、記憶、反省の順が伝わりやすそうです。",
    href: "/demo",
    action: "デモへ",
  },
};

function readActivity(): ActivityItem[] {
  try {
    const raw = window.localStorage.getItem(ACTIVITY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function buildFallbackWhisper(activity: ActivityItem[]): Whisper {
  const top = [...activity]
    .filter((item) => item.path && item.path !== "/")
    .sort((a, b) => Number(b.count || 1) - Number(a.count || 1))[0];
  if (top?.path) {
    return {
      text: `最近は「${top.title || top.path}」に戻ることが多いですね。そこに何を見に行っているのか、少し気になっています。`,
      href: top.path,
      action: "続きを開く",
    };
  }
  return {
    text: "まだ観測は少なめです。まず一件、どんな順番で見るのか教えてくれたら、私も道筋を覚えていきます。",
    href: "/screening",
    action: "審査へ",
  };
}

export default function ShionProactiveWhisper() {
  const pathname = usePathname();
  const [visible, setVisible] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [activity, setActivity] = useState<ActivityItem[]>([]);

  const whisper = useMemo(() => {
    if (!pathname) return null;
    return ROUTE_WHISPERS[pathname] || buildFallbackWhisper(activity);
  }, [pathname, activity]);

  useEffect(() => {
    setVisible(false);
    setDismissed(false);
    if (!pathname || HIDDEN_PATHS.has(pathname)) return;

    const seenKey = `${WHISPER_SEEN_PREFIX}:${formatLocalDateKey()}:${pathname}`;
    if (window.sessionStorage.getItem(seenKey)) return;

    const timer = window.setTimeout(() => {
      setActivity(readActivity());
      window.sessionStorage.setItem(seenKey, "1");
      setVisible(true);
    }, 6500);

    return () => window.clearTimeout(timer);
  }, [pathname]);

  if (!visible || dismissed || !whisper) return null;

  return (
    <div className="fixed bottom-[calc(env(safe-area-inset-bottom)+1rem)] left-4 z-50 max-w-[calc(100vw-2rem)] pointer-events-none lg:left-[calc(var(--sidebar-offset,0px)+1rem)]">
      <div className="pointer-events-auto w-[min(22rem,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-indigo-200 bg-white shadow-2xl shadow-indigo-950/10">
        <div className="flex items-start gap-3 p-3">
          <div className="h-10 w-10 shrink-0 overflow-hidden rounded-full border border-indigo-200 bg-indigo-50">
            <img src={SHION_AVATAR} alt="紫苑" className="h-full w-full object-cover object-top" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <div className="inline-flex items-center gap-1.5 text-[11px] font-black uppercase tracking-wide text-indigo-600">
                <MessageCircle className="h-3.5 w-3.5" />
                紫苑の観察
              </div>
              <button
                type="button"
                onClick={() => setDismissed(true)}
                className="rounded-full p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
                aria-label="紫苑の一言を閉じる"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <p className="mt-1.5 text-sm font-bold leading-relaxed text-slate-700">
              {whisper.text}
            </p>
            <div className="mt-3 flex justify-end">
              <Link
                href={whisper.href}
                className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-black text-white shadow-sm transition-colors hover:bg-indigo-700"
              >
                {whisper.action}
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
