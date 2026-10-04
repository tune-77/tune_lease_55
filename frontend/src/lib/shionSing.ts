// REV-462: 対話の「歌って」系の発言を歌唱API（/api/shion/voice/sing）へ回すための補助。

// 「歌って」「歌を聞かせて」「一曲お願い」等の依頼形だけを拾う。
// 「歌ってる」「歌ってた」「歌えるといいね」のような依頼でない言い回しは除く。
const SING_REQUEST_RE =
  /(?:歌|うた)って(?![いたるま])|(?:歌|うた)を(?:聞|聴|き)かせて|一曲(?:お願い|おねがい|(?:歌|うた)って)/;

export const isSingRequest = (text: string): boolean => {
  const value = text.trim();
  return value.length > 0 && value.length <= 80 && SING_REQUEST_RE.test(value);
};

export type SingResponse = {
  title: string;
  theme?: string;
  lyrics: string;
  audio_base64: string;
  mime_type: string;
  credit: string;
  remaining_today?: number;
};

export const songBlobUrl = (data: SingResponse): string => {
  const bin = atob(data.audio_base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: data.mime_type }));
};

// iPhone（Safari）は、タップ直後に一度再生した audio 要素でないと後から play() できない。
// 合成は数十秒かかるので、タップの瞬間に無音 wav を再生して要素を「解錠」しておく。
// CSP の media-src は 'self' blob: のみなので data: ではなく blob: で渡す。
const silentWavUrl = (): string => {
  const sampleRate = 8000;
  const samples = 800; // 0.1秒
  const buf = new ArrayBuffer(44 + samples);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate, true);
  v.setUint16(32, 1, true);
  v.setUint16(34, 8, true); // 8bit
  str(36, "data");
  v.setUint32(40, samples, true);
  new Uint8Array(buf, 44).fill(128); // 8bit PCM の無音
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
};

export const unlockAudio = (audio: HTMLAudioElement): void => {
  const url = silentWavUrl();
  audio.src = url;
  audio
    .play()
    .catch(() => {})
    .finally(() => URL.revokeObjectURL(url));
};
