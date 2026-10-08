"use client";

// REV-504: マイクの音声入力で「話し終わったら自動で送信」する切り替え（/chat・紫苑対話室で共通）。
// 既定はオフ（数字の聞き間違いがあるので、今どおり確認してから送信）。オンでも認識した文字を入力欄に
// 出したうえで、AUTO_SEND_DELAY_MS のあいだ「送信まで○秒・取り消し」を見せてから送る。設定はブラウザに記憶。
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";

const STORAGE_KEY = "shion-voice-auto-send";
const CHANGE_EVENT = "shion-voice-auto-send-change";

function subscribe(onChange: () => void): () => void {
  window.addEventListener("storage", onChange);
  window.addEventListener(CHANGE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(CHANGE_EVENT, onChange);
  };
}
export const AUTO_SEND_DELAY_MS = 2000;

function readSetting(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function useVoiceAutoSend() {
  // サーバー描画では常にオフ。ブラウザでは保存した設定を読む（effect で state を書き換えない）
  const autoSend = useSyncExternalStore(subscribe, readSetting, () => false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sendRef = useRef<(() => void) | null>(null);

  const clear = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    sendRef.current = null;
    setCountdown(null);
  }, []);

  useEffect(() => clear, [clear]);

  const setAutoSend = useCallback(
    (value: boolean) => {
      if (!value) clear();
      try {
        window.localStorage.setItem(STORAGE_KEY, value ? "1" : "0");
      } catch {
        // 保存できない環境（プライベートブラウズ等）では切り替えられない
      }
      window.dispatchEvent(new Event(CHANGE_EVENT));
    },
    [clear],
  );

  /** 認識が終わった時に呼ぶ。オンなら数秒後に send を実行し、オフなら何もしない（今どおり手で送信）。 */
  const schedule = useCallback(
    (send: () => void) => {
      if (!autoSend) return;
      clear();
      sendRef.current = send;
      const startedAt = Date.now();
      setCountdown(Math.ceil(AUTO_SEND_DELAY_MS / 1000));
      timerRef.current = setInterval(() => {
        const left = AUTO_SEND_DELAY_MS - (Date.now() - startedAt);
        if (left <= 0) {
          const fn = sendRef.current;
          clear();
          fn?.();
        } else {
          setCountdown(Math.ceil(left / 1000));
        }
      }, 200);
    },
    [autoSend, clear],
  );

  return { autoSend, setAutoSend, countdown, schedule, cancel: clear };
}
