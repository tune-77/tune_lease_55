"use client";

// REV-462: 対話画面の「歌って」→ 歌唱API → 再生。曲は message.id ごとの blob URL（再読み込みで消える）。
import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";
import { songBlobUrl, unlockAudio, type SingResponse } from "@/lib/shionSing";

export type DialogueSong = { url: string; autoplayBlocked: boolean };

export function useDialogueSinging() {
  const [singing, setSinging] = useState(false);
  const [songs, setSongs] = useState<Record<number, DialogueSong>>({});
  const [playingSongId, setPlayingSongId] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlsRef = useRef<string[]>([]);

  const getAudio = useCallback(() => {
    if (!audioRef.current) {
      const audio = new Audio();
      audio.onpause = () => setPlayingSongId(null);
      audio.onended = () => setPlayingSongId(null);
      audioRef.current = audio;
    }
    return audioRef.current;
  }, []);

  useEffect(() => () => {
    audioRef.current?.pause();
    urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
  }, []);

  // 送信タップと同じ同期区間で呼ぶこと（最初の await より前に audio を解錠する）
  const sing = useCallback(
    async (text: string, songId: number): Promise<SingResponse> => {
      const audio = getAudio();
      unlockAudio(audio);
      if (typeof window !== "undefined") window.speechSynthesis?.cancel();
      setSinging(true);
      try {
        const { data } = await apiClient.post<SingResponse>(
          "/api/shion/voice/sing",
          { message: text },
          { timeout: 240_000 },
        );
        const url = songBlobUrl(data);
        urlsRef.current.push(url);
        audio.src = url;
        let autoplayBlocked = false;
        try {
          await audio.play();
          setPlayingSongId(songId);
        } catch {
          autoplayBlocked = true; // iPhone 等で自動再生できなければ ▶ で再生してもらう
        }
        setSongs((prev) => ({ ...prev, [songId]: { url, autoplayBlocked } }));
        return data;
      } finally {
        setSinging(false);
      }
    },
    [getAudio],
  );

  const toggleSong = useCallback(
    (songId: number) => {
      const song = songs[songId];
      if (!song) return;
      const audio = getAudio();
      if (playingSongId === songId && !audio.paused) {
        audio.pause();
        return;
      }
      if (audio.src !== song.url) audio.src = song.url;
      audio.play().then(() => setPlayingSongId(songId)).catch(() => setPlayingSongId(null));
    },
    [songs, playingSongId, getAudio],
  );

  return { singing, songs, playingSongId, sing, toggleSong };
}
