"use client";

// REV-463: 紫苑の読み上げを VOICEVOX（歌唱と同じ声）に揃える。
// 1文ずつ /api/shion/voice/tts で合成し、再生中に次の文を先読みする。
// 無効（404）・エンジン停止（503）・再生拒否のときは、残りをブラウザ標準の読み上げへ回す。
import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api";
import { unlockAudio } from "@/lib/shionSing";

const MAX_CHUNK = 120;
const MAX_CHUNKS = 12;

const cleanForSpeech = (text: string) =>
  (text || "")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/^\s*(?:[-・*]|\d+\.)\s+/gm, "")
    .replace(/[*#`>|_~]/g, "");

export const splitForTts = (text: string): string[] => {
  const units = cleanForSpeech(text).match(/[^。！？!?\n]+[。！？!?]*/g) || [];
  const chunks: string[] = [];
  let current = "";
  for (const raw of units) {
    const unit = raw.trim();
    if (!unit) continue;
    // 最初の1文は単独で送り、読み始めまでの待ちを短くする
    if (current && (chunks.length === 0 || current.length + unit.length > MAX_CHUNK)) {
      chunks.push(current);
      current = "";
    }
    for (let i = 0; i < unit.length; i += MAX_CHUNK) {
      const piece = unit.slice(i, i + MAX_CHUNK);
      if (current && current.length + piece.length > MAX_CHUNK) {
        chunks.push(current);
        current = "";
      }
      current += piece;
    }
  }
  if (current) chunks.push(current);
  return chunks.slice(0, MAX_CHUNKS);
};

type Clip = { url: string; credit: string };

export function useShionSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const [credit, setCredit] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const generationRef = useRef(0);

  const getAudio = useCallback(() => {
    if (!audioRef.current) audioRef.current = new Audio();
    return audioRef.current;
  }, []);

  const stop = useCallback(() => {
    generationRef.current += 1;
    audioRef.current?.pause();
    setSpeaking(false);
  }, []);

  useEffect(() => () => {
    generationRef.current += 1;
    audioRef.current?.pause();
  }, []);

  // 送信タップと同じ同期区間で呼ぶ（iPhone は後から play() できるよう要素を解錠しておく必要がある）
  const unlock = useCallback(() => unlockAudio(getAudio()), [getAudio]);

  const speak = useCallback(
    async (text: string, fallback: (rest: string) => void) => {
      const generation = ++generationRef.current;
      const audio = getAudio();
      audio.pause();
      const chunks = splitForTts(text);
      if (!chunks.length) return;

      const fetchClip = (chunk: string): Promise<Clip> =>
        apiClient
          .post<Blob>("/api/shion/voice/tts", { text: chunk }, { responseType: "blob", timeout: 60_000 })
          .then((res) => ({
            url: URL.createObjectURL(res.data),
            credit: decodeURIComponent(String(res.headers["x-voice-credit"] || "")),
          }));

      const playClip = (url: string) =>
        new Promise<void>((resolve, reject) => {
          audio.onended = () => resolve();
          audio.onpause = () => resolve(); // stop() による中断
          audio.src = url;
          audio.play().catch(reject);
        });

      let next: Promise<Clip> | null = fetchClip(chunks[0]);
      for (let i = 0; i < chunks.length; i++) {
        let clip: Clip;
        try {
          clip = await (next as Promise<Clip>);
        } catch {
          if (generationRef.current === generation) {
            setSpeaking(false);
            fallback(chunks.slice(i).join(""));
          }
          return;
        }
        next = i + 1 < chunks.length ? fetchClip(chunks[i + 1]) : null;
        next?.catch(() => undefined); // 中断時の未処理拒否を防ぐ（本処理は次ループで拾う）
        if (generationRef.current !== generation) {
          URL.revokeObjectURL(clip.url);
          next?.then((c) => URL.revokeObjectURL(c.url)).catch(() => undefined);
          return;
        }
        if (clip.credit) setCredit(clip.credit);
        setSpeaking(true);
        try {
          await playClip(clip.url);
        } catch {
          // 自動再生の拒否など。残りはブラウザ読み上げへ
          if (generationRef.current === generation) {
            setSpeaking(false);
            fallback(chunks.slice(i).join(""));
          }
          return;
        } finally {
          URL.revokeObjectURL(clip.url);
        }
        if (generationRef.current !== generation) return;
      }
      setSpeaking(false);
    },
    [getAudio],
  );

  return { speak, stop, unlock, speaking, credit };
}
