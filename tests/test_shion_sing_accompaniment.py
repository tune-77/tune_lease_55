"""REV-469: 紫苑の歌の伴奏（コード解析・拍の整列・MIDI・ミックス）。"""
import io
import wave

import numpy as np
import pytest

import api.shion_sing_accompaniment as acc


def _vv(beats_list, bpm=120):
    """VOICEVOX 楽譜（先頭・末尾休符つき）。key は C メジャーの音。"""
    keys = [60, 64, 67, 65, 62, 60, 59, 60]
    notes = [{"key": None, "frame_length": 15, "lyric": ""}]
    for i, b in enumerate(beats_list):
        notes.append({"key": keys[i % len(keys)], "frame_length": max(1, round(b * 60 / bpm * acc.FRAME_RATE)), "lyric": "ら"})
    notes.append({"key": None, "frame_length": 30, "lyric": ""})
    return {"notes": notes}


def _wav(samples, rate=24000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.asarray(samples) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("symbol,expected", [
    ("C", (0, (0, 4, 7), 0)),
    ("Am7", (9, (0, 3, 7, 10), 9)),
    ("F#m", (6, (0, 3, 7), 6)),
    ("Bb", (10, (0, 4, 7), 10)),
    ("C/E", (0, (0, 4, 7), 4)),
    ("Gsus4", (7, (0, 5, 7), 7)),
    ("Cmaj7", (0, (0, 4, 7, 11), 0)),
    ("H", None),
    ("", None),
    (None, None),
])
def test_parse_chord(symbol, expected):
    assert acc.parse_chord(symbol) == expected


def test_timeline_follows_actual_voicevox_frames():
    # 120bpm の1拍 = 46.875 フレーム → 47 に丸められ、64音で約8フレームずれる。境界ごとに実フレームへ合わせる
    vv = _vv([1] * 64)
    tl = acc._Timeline(vv["notes"], 120)
    assert tl.total_beats == 64
    for beat in (0, 1, 32, 64):
        assert tl.sec(beat) == pytest.approx((15 + 47 * beat) / acc.FRAME_RATE)
    assert tl.sec(-4) == pytest.approx(15 / acc.FRAME_RATE - 2.0)  # 前奏は既定テンポで外挿


def test_plan_chords_repeats_progression_to_song_length():
    tl = acc._Timeline(_vv([1] * 16)["notes"], 120)
    seg = acc.plan_chords([{"chord": "C", "beats": 4}, {"chord": "G", "beats": 4}], tl)
    assert [(s, e) for s, e, _ in seg] == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert [c[0] for _, _, c in seg] == [0, 7, 0, 7]


def test_plan_chords_transposes_wrong_key():
    tl = acc._Timeline(_vv([1] * 16)["notes"], 120)
    seg = acc.plan_chords([{"chord": "D", "beats": 4}, {"chord": "A", "beats": 4}], tl)  # C の歌に D の進行
    assert acc._fit_score(seg, tl.melody) >= 0.35
    assert seg[0][2][0] == 0


def test_plan_chords_harmonizes_when_chords_missing():
    tl = acc._Timeline(_vv([1] * 8)["notes"], 120)
    seg = acc.plan_chords("garbage", tl)
    assert [(s, e) for s, e, _ in seg] == [(0, 4), (4, 8)]
    assert acc._fit_score(seg, tl.melody) >= 0.5


def test_build_midi_adds_intro_padding():
    tl = acc._Timeline(_vv([1] * 8)["notes"], 120)
    midi, pad = acc.build_midi(acc.plan_chords([{"chord": "C", "beats": 4}], tl), tl, program=0)
    assert midi.startswith(b"MThd") and b"MTrk" in midi and midi.endswith(b"\xff\x2f\x00")
    assert pad == pytest.approx(2.0 - 15 / acc.FRAME_RATE)  # 1小節(2秒)の前奏ぶん歌声を遅らせる


def test_mix_balances_levels_and_pads_vocal():
    t = np.arange(24000) / 24000
    vocal = 0.4 * np.sin(2 * np.pi * 440 * t)
    accomp = np.repeat((0.05 * np.sin(2 * np.pi * 220 * np.arange(36000) / 24000))[:, None], 2, axis=1)
    out = acc.mix(_wav(vocal), _wav(accomp.reshape(-1), channels=2), pad_seconds=0.5, level=0.5)
    rate, pcm = acc._read_wav(out)
    assert rate == 24000 and pcm.shape == (36000, 2)
    head = pcm[:12000, 0]  # 歌声が入る前（伴奏のみ）
    assert np.sqrt(np.mean(head ** 2)) == pytest.approx(0.4 / np.sqrt(2) * 0.5, rel=0.05)


def test_add_accompaniment_off_flag_and_failures(monkeypatch):
    vv = _vv([1] * 8)
    vocal = _wav(np.zeros(24000))
    monkeypatch.setenv("SHION_SING_ACCOMPANIMENT", "0")
    assert acc.add_accompaniment(vocal, vv, [{"chord": "C", "beats": 4}], 120) is None
    monkeypatch.setenv("SHION_SING_ACCOMPANIMENT", "1")
    monkeypatch.setenv("SHION_FLUIDSYNTH_BIN", "/nonexistent/fluidsynth")
    assert acc.add_accompaniment(vocal, vv, [{"chord": "C", "beats": 4}], 120) is None


def test_add_accompaniment_renders_with_sample_rate(monkeypatch):
    vv = _vv([1] * 8)
    rendered = []

    def fake_render(midi, rate):
        rendered.append(rate)
        return _wav(np.zeros(rate * 3 * 2), rate=rate, channels=2)

    monkeypatch.setenv("SHION_SING_ACCOMPANIMENT", "1")
    monkeypatch.setattr(acc, "render_midi", fake_render)
    out = acc.add_accompaniment(_wav(0.3 * np.ones(48000), rate=48000), vv, [{"chord": "C", "beats": 4}], 120)
    assert rendered == [48000]
    assert acc._read_wav(out)[1].shape[1] == 2
