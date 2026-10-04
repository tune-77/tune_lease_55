"use client";

// REV-462: 対話の歌メッセージに付く ▶/⏸ プレーヤー（自動再生できなかった時の案内つき）。
import { Pause, Play } from "lucide-react";

type Props = {
  playing: boolean;
  autoplayBlocked: boolean;
  onToggle: () => void;
};

export default function DialogueSongPlayer({ playing, autoplayBlocked, onToggle }: Props) {
  return (
    <div className="mt-2 flex items-center gap-2 rounded-lg border border-violet-200 bg-white px-2 py-1.5">
      <button
        type="button"
        onClick={onToggle}
        aria-label={playing ? "歌を一時停止" : "歌を再生"}
        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-violet-600 text-white hover:bg-violet-700"
      >
        {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      </button>
      <span className="text-[11px] text-slate-500">
        {playing
          ? "紫苑が歌っています…"
          : autoplayBlocked
          ? "自動再生できませんでした。▶ を押すと歌います"
          : "▶ でもう一度聴く"}
      </span>
    </div>
  );
}
