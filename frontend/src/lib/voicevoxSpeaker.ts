// REV-530: 通話の文字起こしを1文ずつ VOICEVOX（/api/shion/voice/tts・冥鳴ひまり）で読み上げる。
// Gemini Live はテキスト出力に対応しないため、音声で受けた返答の文字起こしを使う。
// - 最初の1文だけ読点（、）でも区切り、声が出るまでを短くする
// - 再生中に次の文を先に合成しておく（順番は保つ）
// - stop() で再生と待ち行列を捨てる（割り込み・切替・終了時）
// 再生は呼び出し側がタップ直後に resume した AudioContext を使う（iPhone Safari の自動再生制限対策）。
import { apiClient } from "@/lib/api";

const SENTENCE_END = /[。！？!?\n]/;
const FIRST_CLAUSE_END = /[、，,]/;
const FIRST_CLAUSE_MIN_CHARS = 6;

export class VoicevoxSpeaker {
  private buffer = "";
  private firstOfTurn = true;
  private queue: Promise<AudioBuffer | null>[] = [];
  private current: AudioBufferSourceNode | null = null;
  private generation = 0;
  private draining = false;

  constructor(private readonly ctx: AudioContext, private readonly onSpeakingChange?: (speaking: boolean) => void) {}

  /** 再生中か、合成・再生待ちの文がある */
  get speaking(): boolean {
    return this.current !== null || this.queue.length > 0;
  }

  /** 文字起こしの断片を足す。文がそろった分から合成を始める。 */
  push(text: string): void {
    this.buffer += text;
    this.flush(false);
  }

  /** 紫苑のターンが終わった。残りを読み上げ、次のターンは最初の1文の扱いに戻す。 */
  endTurn(): void {
    this.flush(true);
    this.firstOfTurn = true;
  }

  stop(): void {
    this.generation += 1;
    this.buffer = "";
    this.firstOfTurn = true;
    this.queue = [];
    const current = this.current;
    this.current = null;
    try {
      current?.stop();
    } catch {
      /* already stopped */
    }
    this.onSpeakingChange?.(false);
  }

  private flush(final: boolean): void {
    for (;;) {
      const end = this.buffer.search(SENTENCE_END);
      const clause = this.firstOfTurn ? this.buffer.search(FIRST_CLAUSE_END) : -1;
      const useClause = clause >= FIRST_CLAUSE_MIN_CHARS - 1 && (end < 0 || clause < end);
      const cut = useClause ? clause : end;
      if (cut < 0) break;
      this.enqueue(this.buffer.slice(0, cut + 1));
      this.buffer = this.buffer.slice(cut + 1);
    }
    if (final && this.buffer.trim()) {
      this.enqueue(this.buffer);
      this.buffer = "";
    }
  }

  private enqueue(raw: string): void {
    const text = raw.trim();
    if (!text) return;
    this.firstOfTurn = false;
    this.queue.push(this.synthesize(text, this.generation));
    void this.drain();
  }

  private async synthesize(text: string, generation: number): Promise<AudioBuffer | null> {
    try {
      const { data } = await apiClient.post<ArrayBuffer>(
        "/api/shion/voice/tts",
        { text },
        { responseType: "arraybuffer", timeout: 20_000 },
      );
      if (generation !== this.generation) return null;
      return await this.ctx.decodeAudioData(data);
    } catch {
      return null; // 1文の失敗で通話は止めない（その文だけ飛ばす）
    }
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.length > 0) {
        const head = this.queue[0];
        const generation = this.generation;
        const buffer = await head;
        if (this.queue[0] !== head) continue; // stop() で捨てられた
        this.queue.shift();
        if (buffer && generation === this.generation) await this.play(buffer);
      }
    } finally {
      this.draining = false;
      if (!this.speaking) this.onSpeakingChange?.(false);
    }
  }

  private play(buffer: AudioBuffer): Promise<void> {
    return new Promise((resolve) => {
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.ctx.destination);
      src.onended = () => {
        if (this.current === src) this.current = null;
        resolve();
      };
      this.current = src;
      this.onSpeakingChange?.(true);
      src.start();
    });
  }
}
