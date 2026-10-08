"use client";

// REV-504: 音声入力の「話し終わったら自動送信」の切り替えと、送信までの残り秒・取り消し。
type Props = {
  autoSend: boolean;
  onChange: (value: boolean) => void;
  countdown: number | null;
  onCancel: () => void;
};

export default function VoiceAutoSendControl({ autoSend, onChange, countdown, onCancel }: Props) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-slate-500">
      <label className="inline-flex items-center gap-1 cursor-pointer select-none" title="音声入力が終わったら、認識した文字を見せてから自動で送信します（数字は聞き間違いに注意）">
        <input
          type="checkbox"
          className="h-3 w-3 accent-blue-600"
          checked={autoSend}
          onChange={(e) => onChange(e.target.checked)}
        />
        話し終わったら自動送信
      </label>
      {countdown !== null && (
        <span className="inline-flex items-center gap-1 rounded-full bg-blue-50 px-2 py-0.5 text-blue-700">
          送信まで{countdown}秒
          <button type="button" onClick={onCancel} className="underline hover:text-blue-900">
            取り消し
          </button>
        </span>
      )}
    </div>
  );
}
