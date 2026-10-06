"use client";

// REV-460: 紫苑の「歌って」ボタン。API が作った wav（VOICEVOX 歌唱）を再生する。
import { useEffect, useState } from "react";
import { Music, Loader2 } from "lucide-react";
import { apiClient } from "@/lib/api";
import ShionIllustration from "@/components/chat/ShionIllustration";

type Props = {
  userId: string;
  theme: string;
  disabled?: boolean;
};

type Song = { title: string; lyrics: string; url: string; credit: string };

type SingResponse = {
  title: string;
  lyrics: string;
  audio_base64: string;
  mime_type: string;
  credit: string;
};

function errorMessage(err: unknown): string {
  const res = (err as { response?: { status?: number; data?: { detail?: unknown } } })?.response;
  if (res?.status === 404) return "歌唱機能は無効です";
  return typeof res?.data?.detail === "string" ? res.data.detail : "歌えませんでした";
}

export default function ShionSingButton({ userId, theme, disabled }: Props) {
  const [loading, setLoading] = useState(false);
  const [song, setSong] = useState<Song | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => {
    if (song) URL.revokeObjectURL(song.url);
  }, [song]);

  const sing = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await apiClient.post<SingResponse>("/api/shion/voice/sing", {
        user_id: userId,
        theme: theme.trim() || "今日の気分",
      });
      const bin = atob(data.audio_base64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const url = URL.createObjectURL(new Blob([bytes], { type: data.mime_type }));
      setSong({ title: data.title, lyrics: data.lyrics, url, credit: data.credit });
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const close = () => {
    setSong(null);
    setError(null);
  };

  return (
    <>
      <button
        type="button"
        onClick={() => void sing()}
        disabled={disabled || loading}
        title="紫苑に歌ってもらう（入力欄の内容がテーマ）"
        className="w-10 h-10 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 bg-violet-600 text-white hover:bg-violet-700 disabled:bg-slate-300"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Music className="w-4 h-4" />}
      </button>
      {(loading || song || error) && (
        <div className="fixed bottom-24 right-4 z-50 w-80 rounded-2xl border border-slate-200 bg-white shadow-xl">
          <div className="px-4 py-2 border-b border-slate-100 text-sm font-medium text-slate-700">
            {loading ? "紫苑が歌を準備中…（数十秒）" : song ? `♪ ${song.title}` : "歌唱"}
          </div>
          {/* 合成を待つ間だけ過去イラストを1枚。歌が届いたら消える（REV-488） */}
          {loading && <ShionIllustration mode="random" className="px-4 pt-2" />}
          {error && <p className="px-4 py-2 text-xs text-rose-600">{error}</p>}
          {song && (
            <div className="px-4 py-2 space-y-2">
              <audio key={song.url} src={song.url} controls autoPlay className="w-full" />
              <p className="text-xs leading-relaxed text-slate-700">{song.lyrics}</p>
              <p className="text-[10px] text-slate-400">{song.credit}</p>
            </div>
          )}
          {!loading && (
            <button type="button" onClick={close} className="w-full py-2 text-xs text-slate-500 hover:text-slate-700">
              閉じる
            </button>
          )}
        </div>
      )}
    </>
  );
}
