"use client";

// REV-421: 紫苑リアルタイム音声通話（試作）。
// ブラウザ → Gemini Live へ直接接続し、Cloud Run は使い切りトークン発行・想起・文字起こし保存だけを担う。
import { useEffect, useRef, useState } from "react";
import { GoogleGenAI, Modality, type LiveServerMessage, type Session } from "@google/genai";
import { Phone, PhoneOff, Loader2 } from "lucide-react";
import { apiClient } from "@/lib/api";
import { VoicevoxSpeaker } from "@/lib/voicevoxSpeaker";

type Props = {
  userId: string;
  disabled?: boolean;
  // REV-504: 保存した通話の会話を渡す（対話室が画面に並べるため）。引数を使わない呼び出し元はそのままでよい
  onEnded?: (turns: VoiceCallTurn[]) => void;
};

export type VoiceCallTurn = { role: "user" | "model"; text: string };
type Turn = VoiceCallTurn;
type Status = "idle" | "connecting" | "live" | "ending";
// REV-530: 通話の声。kore=Gemini Live の声、himari=文字起こしを VOICEVOX（冥鳴ひまり）で読み上げ
type VoiceEngine = "kore" | "himari";
const VOICE_KEY = "shion-voice-engine";

// 通話の残り時間の基準（コンポーネント外で時刻を取る）
function deadlineAfter(seconds: number): number {
  return Date.now() + seconds * 1000;
}

function storedVoice(): VoiceEngine | null {
  try {
    const value = typeof window !== "undefined" ? window.localStorage.getItem(VOICE_KEY) : null;
    return value === "kore" || value === "himari" ? value : null;
  } catch {
    return null;
  }
}


function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function pcm16ToFloat(b64: string): Float32Array<ArrayBuffer> {
  const bin = atob(b64);
  const view = new DataView(new ArrayBuffer(bin.length));
  for (let i = 0; i < bin.length; i++) view.setUint8(i, bin.charCodeAt(i));
  const out = new Float32Array(new ArrayBuffer((bin.length >> 1) * 4));
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
  return out;
}

function errorMessage(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (err instanceof DOMException && err.name === "NotAllowedError") return "マイクの使用が許可されていません";
  return "通話を開始できませんでした";
}

export default function ShionVoiceCall({ userId, disabled, onEnded }: Props) {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);

  const sessionRef = useRef<Session | null>(null);
  const micCtxRef = useRef<AudioContext | null>(null);
  const outCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const playingRef = useRef<AudioBufferSourceNode[]>([]);
  const playAtRef = useRef(0);
  const turnsRef = useRef<Turn[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const endingRef = useRef(false);
  const [voice, setVoice] = useState<VoiceEngine>(() => storedVoice() ?? "kore");
  const [himariAvailable, setHimariAvailable] = useState(false);
  const voiceRef = useRef<VoiceEngine>(voice);
  const speakerRef = useRef<VoicevoxSpeaker | null>(null);

  const switchVoice = (next: VoiceEngine) => {
    voiceRef.current = next;
    setVoice(next);
    try {
      window.localStorage.setItem(VOICE_KEY, next);
    } catch {
      /* 保存できなくても今回の通話は切り替える */
    }
    // 切替時は鳴っている側を止める（次の発話から新しい声）
    stopPlayback();
    speakerRef.current?.stop();
  };

  const appendTurn = (role: Turn["role"], text: string) => {
    const list = turnsRef.current;
    const last = list[list.length - 1];
    // 文字起こしは断片で届くので、同じ話者の連続は1ターンにまとめる
    if (last && last.role === role) last.text += text;
    else list.push({ role, text });
    setTurns([...list]);
  };

  const stopPlayback = () => {
    playingRef.current.forEach((s) => {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    });
    playingRef.current = [];
    playAtRef.current = 0;
  };

  const playChunk = (b64: string) => {
    const ctx = outCtxRef.current;
    if (!ctx) return;
    const samples = pcm16ToFloat(b64);
    const buffer = ctx.createBuffer(1, samples.length, 24000);
    buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime, playAtRef.current);
    src.start(startAt);
    playAtRef.current = startAt + buffer.duration;
    playingRef.current.push(src);
    src.onended = () => {
      playingRef.current = playingRef.current.filter((s) => s !== src);
    };
  };

  const handleToolCall = async (msg: LiveServerMessage) => {
    const calls = msg.toolCall?.functionCalls ?? [];
    const functionResponses = await Promise.all(
      calls.map(async (call) => {
        // REV-531: 記憶・Obsidian・判断資産・審査試算などの読み取り専用ツールをサーバーで実行する
        let result = "ツールを実行できませんでした";
        try {
          result = (await apiClient.post("/api/shion/voice/tool", { name: call.name, args: call.args ?? {} })).data.result;
        } catch {
          /* 失敗はそのまま紫苑に伝える */
        }
        return { id: call.id, name: call.name, response: { result } };
      }),
    );
    sessionRef.current?.sendToolResponse({ functionResponses });
  };

  const handleMessage = (msg: LiveServerMessage) => {
    if (msg.toolCall) void handleToolCall(msg);
    const content = msg.serverContent;
    if (!content) return;
    const himari = voiceRef.current === "himari";
    if (content.interrupted) {
      stopPlayback();
      speakerRef.current?.stop();
    }
    if (content.inputTranscription?.text) {
      appendTurn("user", content.inputTranscription.text);
      // ひまりは Live の再生より遅れて鳴るので、Live のターンが終わった後の割り込みはこちらで止める
      if (himari && speakerRef.current?.speaking) speakerRef.current.stop();
    }
    if (content.outputTranscription?.text) {
      appendTurn("model", content.outputTranscription.text);
      if (himari) speakerRef.current?.push(content.outputTranscription.text);
    }
    if (!himari) {
      for (const part of content.modelTurn?.parts ?? []) {
        if (part.inlineData?.data) playChunk(part.inlineData.data);
      }
    }
    if (content.turnComplete && himari) speakerRef.current?.endTurn();
  };

  const endCall = async () => {
    if (endingRef.current) return;
    endingRef.current = true;
    setStatus("ending");
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    sessionRef.current?.close();
    sessionRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    stopPlayback();
    speakerRef.current?.stop();
    speakerRef.current = null;
    await micCtxRef.current?.close().catch(() => undefined);
    await outCtxRef.current?.close().catch(() => undefined);
    micCtxRef.current = null;
    outCtxRef.current = null;
    const saved = turnsRef.current.filter((t) => t.text.trim());
    if (saved.length > 0) {
      try {
        await apiClient.post("/api/shion/voice/transcript", { user_id: userId, turns: saved.slice(-200) });
        onEnded?.(saved);
      } catch {
        setError("文字起こしの保存に失敗しました");
      }
    }
    setStatus("idle");
    endingRef.current = false;
  };

  const startCall = async () => {
    setError(null);
    setStatus("connecting");
    turnsRef.current = [];
    setTurns([]);
    // iOS Safari はタップ直後（最初の await より前）に作って resume しないと suspended のまま無音になる。
    // マイク側は端末既定レートのまま使い、実レートを mimeType で伝える（Live API 側で再サンプルされる）。
    const outCtx = new AudioContext({ sampleRate: 24000 });
    const micCtx = new AudioContext();
    outCtxRef.current = outCtx;
    micCtxRef.current = micCtx;
    speakerRef.current = new VoicevoxSpeaker(outCtx);
    void outCtx.resume();
    void micCtx.resume();
    try {
      // マイク許可を先に取る（拒否時に1日の発行枠を消費しないため）
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      streamRef.current = stream;
      const { data } = await apiClient.post("/api/shion/voice/session", { user_id: userId });
      setHimariAvailable(Boolean(data.himari_available));
      // 画面で選んだ声を優先。未選択なら SHION_VOICE_ENGINE の既定。ひまりが使えなければ Kore
      const chosen: VoiceEngine = storedVoice() ?? (data.default_voice === "himari" ? "himari" : "kore");
      const effective: VoiceEngine = chosen === "himari" && data.himari_available ? "himari" : "kore";
      voiceRef.current = effective;
      setVoice(effective);

      const ai = new GoogleGenAI({ apiKey: data.token, httpOptions: { apiVersion: "v1alpha" } });
      const session = await ai.live.connect({
        model: data.model,
        config: { responseModalities: [Modality.AUDIO] },
        callbacks: {
          onmessage: handleMessage,
          onerror: () => setError("通話中にエラーが発生しました"),
          onclose: () => void endCall(),
        },
      });
      sessionRef.current = session;

      await micCtx.audioWorklet.addModule("/pcm-capture-worklet.js");
      const node = new AudioWorkletNode(micCtx, "pcm-capture");
      node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
        sessionRef.current?.sendRealtimeInput({ audio: { data: toBase64(e.data), mimeType: `audio/pcm;rate=${micCtx.sampleRate}` } });
      };
      micCtx.createMediaStreamSource(stream).connect(node);

      // 上限はトークン失効でサーバー側が強制する。ここは表示と、切れる前に自分から切るため。
      const endsAt = deadlineAfter(data.max_seconds);
      setRemaining(data.max_seconds);
      timerRef.current = setInterval(() => {
        const left = Math.max(0, Math.round((endsAt - Date.now()) / 1000));
        setRemaining(left);
        if (left === 0) void endCall();
      }, 1000);
      setStatus("live");
    } catch (err) {
      setError(errorMessage(err));
      await endCall();
    }
  };

  useEffect(() => () => void endCall(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const active = status !== "idle";
  const mmss = `${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, "0")}`;

  return (
    <>
      <button
        type="button"
        onClick={active ? () => void endCall() : () => void startCall()}
        disabled={disabled && !active}
        title={active ? "通話を終了" : "紫苑と音声通話（試作）"}
        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors flex-shrink-0 ${
          active ? "bg-rose-600 text-white" : "bg-emerald-600 text-white hover:bg-emerald-700 disabled:bg-slate-300"
        }`}
      >
        {status === "connecting" || status === "ending" ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : active ? (
          <PhoneOff className="w-4 h-4" />
        ) : (
          <Phone className="w-4 h-4" />
        )}
      </button>
      {(active || error) && (
        <div className="fixed bottom-24 right-4 z-50 w-80 max-h-96 flex flex-col rounded-2xl border border-slate-200 bg-white shadow-xl">
          <div className="flex items-center justify-between px-4 py-2 border-b border-slate-100 text-sm">
            <span className="font-medium text-slate-700">
              {status === "connecting" ? "接続中…" : status === "live" ? "紫苑と通話中" : status === "ending" ? "保存中…" : "音声通話"}
            </span>
            {status === "live" && <span className="tabular-nums text-slate-500">残り {mmss}</span>}
          </div>
          {status === "live" && himariAvailable && (
            <div className="flex items-center gap-1 px-4 py-1.5 border-b border-slate-100 text-[11px] text-slate-500">
              <span>声:</span>
              {(["kore", "himari"] as const).map((engine) => (
                <button
                  key={engine}
                  type="button"
                  onClick={() => switchVoice(engine)}
                  className={`rounded-md px-2 py-0.5 ${voice === engine ? "bg-emerald-600 text-white" : "bg-slate-100 hover:bg-slate-200"}`}
                >
                  {engine === "kore" ? "Kore" : "ひまり"}
                </button>
              ))}
              {voice === "himari" && <span className="ml-auto text-[10px] text-slate-400">VOICEVOX:冥鳴ひまり</span>}
            </div>
          )}
          {error && <p className="px-4 py-2 text-xs text-rose-600">{error}</p>}
          <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1 text-xs leading-relaxed">
            {turns.map((t, i) => (
              <p key={i} className={t.role === "user" ? "text-slate-500" : "text-slate-800"}>
                <span className="font-medium">{t.role === "user" ? "あなた" : "紫苑"}：</span>
                {t.text}
              </p>
            ))}
          </div>
          {!active && error && (
            <button type="button" onClick={() => setError(null)} className="m-2 text-xs text-slate-500 hover:text-slate-700">
              閉じる
            </button>
          )}
        </div>
      )}
    </>
  );
}
