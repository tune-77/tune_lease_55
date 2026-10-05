"""REV-469: 紫苑の歌にピアノ伴奏をつける（費用ゼロ・ローカル完結）。

Gemini が楽譜と一緒に返すコード進行を、歌声と同じテンポ・調で MIDI にし、
FluidSynth + GM SoundFont でローカル合成して VOICEVOX の歌声とミックスする。
拍→秒の対応は VOICEVOX へ渡した実フレーム長から作るので、丸め誤差が積もってもずれない。
伴奏を作れない時（ツール未導入・失敗）は None を返し、呼び出し側は歌声だけを返す。

環境変数:
  SHION_SING_ACCOMPANIMENT   "0" で伴奏オフ（既定 "1"）
  SHION_SING_SOUNDFONT       .sf2 のパス（既定 ~/Library/Application Support/tune_lease_55/soundfonts/GeneralUser-GS.sf2）
  SHION_SING_ACCOMP_LEVEL    伴奏の音量（歌声の RMS に対する比。既定 0.5、0〜2）
  SHION_SING_ACCOMP_PROGRAM  GM 音色番号（既定 0=ピアノ。24=ナイロンギター 等）
  SHION_FLUIDSYNTH_BIN       fluidsynth のパス（未指定なら PATH と Homebrew の既定位置を探す）
"""
from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

FRAME_RATE = 93.75  # VOICEVOX 歌唱APIのフレームレート
_BEAT_GRID = 12  # 拍の復元は 1/12 拍単位（16分・3連符を表せる）
_INTRO_BEATS = 4  # 歌い出し前に1小節の前奏
_TAIL_SECONDS = 2.0
_PPQ = 960  # MIDI は 120bpm 固定で秒を直接 tick にする（1秒 = 1920 tick）
_TICKS_PER_SEC = _PPQ * 2
_DEFAULT_SOUNDFONT = Path.home() / "Library/Application Support/tune_lease_55/soundfonts/GeneralUser-GS.sf2"

_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_QUALITIES = {
    "": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "M7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10), "dim": (0, 3, 6), "dim7": (0, 3, 6, 9), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7sus4": (0, 5, 7, 10), "add9": (0, 4, 7, 14),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "9": (0, 4, 7, 10, 14), "m9": (0, 3, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
}
_CHORD_RE = re.compile(
    r"^([A-G])([#b♯♭]?)(" + "|".join(sorted((q for q in _QUALITIES if q), key=len, reverse=True))
    + r")?(?:/([A-G])([#b♯♭]?))?$"
)


def accompaniment_enabled() -> bool:
    return os.environ.get("SHION_SING_ACCOMPANIMENT", "1") != "0"


def _pc(letter: str, accidental: str) -> int:
    return (_NOTE_PC[letter] + {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(accidental, 0)) % 12


def parse_chord(symbol: object) -> tuple[int, tuple[int, ...], int] | None:
    """"Am7" / "C/E" 等を (根音pc, 構成音の音程, ベースpc) にする。読めなければ None。"""
    m = _CHORD_RE.match(str(symbol or "").strip().replace("△", "maj").replace("°", "dim"))
    if not m:
        return None
    root = _pc(m.group(1), m.group(2))
    bass = _pc(m.group(4), m.group(5)) if m.group(4) else root
    return root, _QUALITIES[m.group(3) or ""], bass


def _note_beats(notes: list[dict], bpm: int) -> list[float]:
    """VOICEVOX の各音符のフレーム長から元の拍数を復元する（丸め誤差 ≦0.5 フレームは格子で吸収）。"""
    frames_per_beat = 60 / bpm * FRAME_RATE
    return [max(1, round(n["frame_length"] / frames_per_beat * _BEAT_GRID)) / _BEAT_GRID for n in notes]


class _Timeline:
    """拍位置 → 歌声 wav 上の秒。音符の境界ごとに実フレームへ合わせ、範囲外は既定テンポで外挿する。"""

    def __init__(self, notes: list[dict], bpm: int):
        self.sec_per_beat = 60 / bpm
        body = notes[1:-1]  # 先頭・末尾の休符を除いた部分が楽曲（拍0 = 先頭休符の直後）
        beats = _note_beats(body, bpm)
        frame = notes[0]["frame_length"]
        beat = 0.0
        self.points = [(0.0, frame / FRAME_RATE)]
        for n, b in zip(body, beats):
            frame += n["frame_length"]
            beat += b
            self.points.append((beat, frame / FRAME_RATE))
        self.total_beats = beat
        self.melody = [(start, b, n["key"]) for n, b, (start, _) in zip(body, beats, self.points) if n["key"] is not None]

    def sec(self, beat: float) -> float:
        pts = self.points
        if beat <= pts[0][0]:
            return pts[0][1] + (beat - pts[0][0]) * self.sec_per_beat
        for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
            if beat <= b1:
                return s0 + (s1 - s0) * (beat - b0) / (b1 - b0)
        return pts[-1][1] + (beat - pts[-1][0]) * self.sec_per_beat


def _fit_score(segments: list[tuple[float, float, tuple]], melody: list[tuple], shift: int = 0) -> float:
    """メロディの音（拍数で重みづけ）のうち、鳴っているコードの構成音に含まれる割合。"""
    hit = total = 0.0
    for start, beats, key in melody:
        mid = start + beats / 2
        for s, e, (root, intervals, bass) in segments:
            if s <= mid < e:
                tones = {(root + shift + i) % 12 for i in intervals} | {(bass + shift) % 12}
                hit += beats if key % 12 in tones else 0
                break
        total += beats
    return hit / total if total else 0.0


def _harmonize(melody: list[tuple], total_beats: float) -> list[tuple[float, float, tuple]]:
    """コード進行が使えない時の代わり: 1小節ごとにメロディと最も重なる長・短三和音を選ぶ。"""
    candidates = [(r, q, r) for r in (0, 7, 5, 9, 2, 4, 11, 10, 3, 8, 1, 6) for q in ((0, 4, 7), (0, 3, 7))]
    segments, prev = [], candidates[0]
    bar = 0.0
    while bar < total_beats:
        in_bar = [(s, b, k) for s, b, k in melody if bar <= s + b / 2 < bar + 4]
        if in_bar:
            prev = max(candidates, key=lambda c: (_fit_score([(bar, bar + 4, c)], in_bar), c == prev))
        segments.append((bar, min(bar + 4, total_beats), prev))
        bar += 4
    return segments


def plan_chords(chords: object, timeline: _Timeline) -> list[tuple[float, float, tuple]]:
    """Gemini のコード進行を拍区間に並べ、曲の長さに合わせて繰り返し・切り詰める。

    調がメロディとずれていれば移調し、それでも合わなければメロディから付け直す。
    """
    parsed = []
    for c in chords if isinstance(chords, list) else []:
        if isinstance(c, dict) and (chord := parse_chord(c.get("chord"))):
            try:
                beats = min(8.0, max(0.5, float(c.get("beats") or 4)))
            except (TypeError, ValueError):
                beats = 4.0
            parsed.append((beats, chord))
    total = timeline.total_beats
    segments: list[tuple[float, float, tuple]] = []
    pos = 0.0
    while parsed and pos < total:
        for beats, chord in parsed:
            if pos >= total:
                break
            segments.append((pos, min(pos + beats, total), chord))
            pos += beats
    melody = timeline.melody
    if segments:
        base = _fit_score(segments, melody)
        shift = max(range(12), key=lambda t: _fit_score(segments, melody, t))
        if shift and _fit_score(segments, melody, shift) > base + 0.15:
            logger.info("shion sing: chords transposed by %d to match melody", shift)
            segments = [(s, e, ((r + shift) % 12, iv, (b + shift) % 12)) for s, e, (r, iv, b) in segments]
        if _fit_score(segments, melody) >= 0.35:
            return segments
        logger.info("shion sing: chords do not fit melody, re-harmonizing")
    return _harmonize(melody, total)


def _voicing(root: int, intervals: tuple[int, ...], bass: int) -> tuple[int, list[int]]:
    """歌の音域（MIDI 57〜）とぶつからないよう、ベースは C2〜B2、和音は G3〜F4 あたりに置く。"""
    bass_note = 36 + bass
    tones = sorted({55 + (root + i - 55) % 12 for i in intervals})
    return bass_note, tones


def _var_len(n: int) -> bytes:
    out = [n & 0x7F]
    while n > 0x7F:
        n >>= 7
        out.append((n & 0x7F) | 0x80)
    return bytes(reversed(out))


def build_midi(segments: list[tuple[float, float, tuple]], timeline: _Timeline, program: int) -> tuple[bytes, float]:
    """伴奏の MIDI（SMF type 0）と、歌声の前に足す無音秒数を返す。"""
    intro_start = timeline.sec(-_INTRO_BEATS)
    pad = max(0.0, -intro_start)  # 前奏が 0 秒より前に食い込む分だけ全体を後ろへずらす
    events: list[tuple[int, int, bytes]] = []  # (tick, 順序, データ)

    def at(beat: float) -> int:
        return round((timeline.sec(beat) + pad) * _TICKS_PER_SEC)

    def note(start: float, end: float, key: int, vel: int) -> None:
        events.append((at(start), 1, bytes([0x90, key, vel])))
        events.append((at(end), 0, bytes([0x80, key, 0])))

    first = segments[0][2]
    last = segments[-1][2]
    plan = [(-_INTRO_BEATS, 0.0, first)] + segments
    for s, e, chord in plan:
        bass_note, tones = _voicing(*chord)
        arp = [tones[0], tones[-1], tones[1], tones[-1]]  # 8分音符の分散和音（下・上・中・上）
        t, step = s, 0
        while t < e - 1e-6:  # 鳴らした音はコードの終わりまで伸ばす（ペダル風）
            if step % 4 == 0:
                note(t, e, bass_note, 72 if step == 0 else 60)
            note(t, e, arp[step % 4], 56 if step % 2 == 0 else 46)
            t, step = t + 0.5, step + 1
    end = timeline.total_beats  # 終わりの和音: 歌い終わりから2拍、低音＋和音をそっと置く
    bass_note, tones = _voicing(*last)
    for k in [bass_note, *tones]:
        note(end, end + 2, k, 56)

    events.sort(key=lambda ev: (ev[0], ev[1]))
    track = bytearray()
    track += b"\x00\xff\x51\x03" + (500000).to_bytes(3, "big")  # 120bpm（秒→tick を固定）
    track += b"\x00\xc0" + bytes([program & 0x7F])
    last_tick = 0
    for tick, _, data in events:
        track += _var_len(tick - last_tick) + data
        last_tick = tick
    track += _var_len(round(_TAIL_SECONDS * _TICKS_PER_SEC)) + b"\xff\x2f\x00"  # 残響ぶん待って終わる
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + _PPQ.to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track), pad


def _fluidsynth_bin() -> str | None:
    configured = os.environ.get("SHION_FLUIDSYNTH_BIN", "").strip()
    if configured:
        return configured if Path(configured).exists() else None
    # launchd 配下は PATH が最小限なので Homebrew の既定位置も探す
    return shutil.which("fluidsynth") or next(
        (p for p in ("/opt/homebrew/bin/fluidsynth", "/usr/local/bin/fluidsynth") if Path(p).exists()), None
    )


def _soundfont() -> Path | None:
    path = Path(os.environ.get("SHION_SING_SOUNDFONT", "").strip() or _DEFAULT_SOUNDFONT).expanduser()
    return path if path.is_file() else None


def render_midi(midi: bytes, sample_rate: int) -> bytes:
    fluidsynth, soundfont = _fluidsynth_bin(), _soundfont()
    if not fluidsynth or not soundfont:
        raise FileNotFoundError("fluidsynth or soundfont not found")
    with tempfile.TemporaryDirectory(prefix="shion_sing_") as tmp:
        mid, out = Path(tmp) / "accomp.mid", Path(tmp) / "accomp.wav"
        mid.write_bytes(midi)
        subprocess.run(
            [fluidsynth, "-ni", "-q", "-g", "0.5", "-r", str(sample_rate), "-T", "wav", "-O", "s16",
             "-F", str(out), str(soundfont), str(mid)],
            check=True, capture_output=True, timeout=60,
        )
        return out.read_bytes()


def _read_wav(data: bytes):
    import numpy as np

    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("16-bit wav only")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return rate, pcm.reshape(-1, channels)


def _accomp_level() -> float:
    try:
        return min(2.0, max(0.0, float(os.environ.get("SHION_SING_ACCOMP_LEVEL", "0.5"))))
    except ValueError:
        return 0.5


def mix(vocal_wav: bytes, accomp_wav: bytes, pad_seconds: float, level: float) -> bytes:
    """歌声（モノラル）を中央、伴奏（ステレオ）を歌声の RMS × level に揃えて重ね、16bit ステレオ wav にする。"""
    import numpy as np

    rate, vocal = _read_wav(vocal_wav)
    acc_rate, acc = _read_wav(accomp_wav)
    if acc_rate != rate:
        raise ValueError("sample rate mismatch")
    vocal = vocal.mean(axis=1)
    acc = acc if acc.shape[1] == 2 else np.repeat(acc[:, :1], 2, axis=1)
    pad = round(pad_seconds * rate)
    length = max(pad + len(vocal), len(acc))
    out = np.zeros((length, 2), dtype=np.float32)
    out[pad:pad + len(vocal)] += vocal[:, None]
    voiced = vocal[np.abs(vocal) > 0.01]
    acc_rms = float(np.sqrt(np.mean(acc ** 2))) if acc.size else 0.0
    if voiced.size and acc_rms > 0:
        out[:len(acc)] += acc * (float(np.sqrt(np.mean(voiced ** 2))) * level / acc_rms)
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.95:
        out *= 0.95 / peak
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((out * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def add_accompaniment(vocal_wav: bytes, vv_score: dict, chords: object, bpm: int) -> bytes | None:
    """歌声 wav に伴奏を重ねた wav を返す。オフ・失敗時は None（歌声だけで返す）。"""
    if not accompaniment_enabled():
        return None
    try:
        timeline = _Timeline(vv_score["notes"], bpm)
        segments = plan_chords(chords, timeline)
        program = int(os.environ.get("SHION_SING_ACCOMP_PROGRAM", "0") or 0)
        midi, pad = build_midi(segments, timeline, program)
        with wave.open(io.BytesIO(vocal_wav)) as w:
            rate = w.getframerate()
        return mix(vocal_wav, render_midi(midi, rate), pad, _accomp_level())
    except Exception as exc:
        logger.warning("shion sing: accompaniment skipped (%s)", type(exc).__name__)
        return None
