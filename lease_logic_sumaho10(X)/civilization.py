"""
文明存続探索システム — アルカイアの慟哭 世界観
Civilization Survival Explorer（小説『アルカイアの慟哭』に登場する文明のためのツール）

プロトコル: 「自然（地球）を破壊するものには断固として抵抗し、排除する。」
本ツールは、その観測枠の下で文明が生き残る方法を探るためにある。

- 文明存続の探索: 技術・倫理・自然との調和度から、プロトコル適合/要観測/排除対象を判定
- 50億年スケール再生試行: 多数の文明の再生・栄枯を波でシミュレーション（リタとYOSHITUNEの観測枠）

起動: streamlit run "lease_logic_sumaho10(X)/civilization.py" （リポジトリルートから）
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# Part 1: 文明審査（Judgment）
# ---------------------------------------------------------------------------

CIVILIZATION_METRICS = {
    "technological_level": {
        "label": "技術水準",
        "weight": 30,
        "criteria": {
            "kardashev_type": "カルダシェフスケール（Type 0-III）",
            "ai_development": "AI発展度",
            "space_exploration": "宇宙探査能力",
            "energy_utilization": "エネルギー利用効率",
            "tech_progress": "技術的進歩・革新性（発展の速さ・イノベーション）",
        },
    },
    "ethical_development": {
        "label": "倫理水準",
        "weight": 35,
        "criteria": {
            "violence_index": "暴力指数（低いほど良い）",
            "inequality": "不平等度（低いほど良い）",
            "cooperation": "協調性",
            "wisdom": "知恵・判断力",
        },
    },
    "sustainability": {
        "label": "自然との調和",
        "weight": 35,
        "criteria": {
            "environmental_impact": "環境負荷（低いほど自然を破壊しない）",
            "resource_management": "資源管理能力",
            "population_control": "人口制御",
            "long_term_planning": "長期計画性",
        },
    },
}

JUDGMENT_THRESHOLDS = {
    "preserve": 60,
    "monitor": 40,
    "terminate": 0,
}

# 小説『アルカイアの慟哭』文明興亡データベースの破滅パターン6分類（衰退理由）
DECAY_REASON_LABELS = [
    "技術的破滅",
    "生物学的破滅",
    "環境的破滅",
    "社会的破滅",
    "宇宙的破滅",
    "その他",
]

# 学術的重み（extinction_patterns の count 比率: 1250, 1100, 890, 760, 500, 500 / 5000）
DECAY_REASON_WEIGHTS = {
    "技術的破滅": 0.25,
    "生物学的破滅": 0.22,
    "環境的破滅": 0.178,
    "社会的破滅": 0.152,
    "宇宙的破滅": 0.1,
    "その他": 0.1,
}


def infer_decay_reasons(
    tech_level: float,
    ethical_dev: float,
    sustainability: float,
    **kwargs: float,
) -> List[str]:
    """文明の指標から衰退理由を推定（小説の6分類: 技術的・生物学的・環境的・社会的・宇宙的・その他）。"""
    reasons: List[str] = []
    env_high = kwargs.get("environmental_impact", 50) > 60
    violence_high = kwargs.get("violence_index", 50) > 60
    inequality_high = kwargs.get("inequality", 50) > 60
    space_high = kwargs.get("space_exploration", 50) >= 70
    if sustainability < 35 or env_high:
        reasons.append("環境的破滅")
    if ethical_dev < 40 or violence_high or inequality_high:
        reasons.append("社会的破滅")
    tech_progress = kwargs.get("tech_progress", 50)
    if tech_level >= 70 and (kwargs.get("ai_development", 50) > 70 or kwargs.get("energy_utilization", 50) > 70):
        reasons.append("技術的破滅")
    if tech_progress >= 75 and tech_level >= 60:
        if "技術的破滅" not in reasons:
            reasons.append("技術的破滅")
    if tech_level >= 65 and space_high:
        reasons.append("宇宙的破滅")
    if sustainability < 50 and not env_high:
        reasons.append("生物学的破滅")
    if not reasons:
        reasons.append("その他")
    return reasons


class CivilizationJudgmentSystem:
    """文明存続探索 — プロトコル「自然を破壊するものには抵抗し排除する」に照らした適合度判定"""

    def __init__(self):
        self.civilizations: List[Dict] = []
        self.judgment_history: List[Dict] = []

    def calculate_civilization_score(
        self, tech_level: float, ethical_dev: float, sustainability: float
    ) -> Dict:
        total_score = tech_level * 0.30 + ethical_dev * 0.35 + sustainability * 0.35
        if total_score >= JUDGMENT_THRESHOLDS["preserve"]:
            judgment, action = "preserve", "プロトコル適合"
            description = "自然（地球）を破壊しない文明として観測継続。再生試行として期待が持てる。"
        elif total_score >= JUDGMENT_THRESHOLDS["monitor"]:
            judgment, action = "monitor", "要観測"
            description = "再生試行の成否が不透明。観測を続け、次の確定に備える。"
        else:
            judgment, action = "terminate", "排除・再生失敗"
            description = "プロトコルに照らし排除対象。この文明は再生として失敗と記録される。"
        warnings: List[str] = []
        if tech_level > 80 and ethical_dev < 50:
            warnings.append("⚠️ 高度技術×低倫理 — 自然を破壊する文明へ転じる危険")
        if sustainability < 30:
            warnings.append("⚠️ 持続不可能 — 自滅の可能性。プロトコル発動の対象となりうる。")
        if ethical_dev < 30:
            warnings.append("🚨 倫理崩壊 — 排除の対象。観測枠では再生失敗と確定。")
        return {
            "tech_level": tech_level,
            "ethical_dev": ethical_dev,
            "sustainability": sustainability,
            "total_score": round(total_score, 2),
            "judgment": judgment,
            "action": action,
            "description": description,
            "warnings": warnings,
        }

    def assess_civilization(
        self,
        name: str,
        planet: str,
        age: int,
        population: int,
        kardashev_type: float,
        **kwargs: float,
    ) -> Dict:
        tech_level_raw = min(
            100,
            kardashev_type * 25
            + kwargs.get("ai_development", 50) * 0.5
            + kwargs.get("space_exploration", 50) * 0.25
            + kwargs.get("energy_utilization", 50) * 0.25,
        )
        tech_progress = kwargs.get("tech_progress", 50)
        tech_level = 0.75 * tech_level_raw + 0.25 * tech_progress
        violence = 100 - kwargs.get("violence_index", 50)
        inequality = 100 - kwargs.get("inequality", 50)
        cooperation = kwargs.get("cooperation", 50)
        wisdom = kwargs.get("wisdom", 50)
        ethical_dev = violence * 0.3 + inequality * 0.2 + cooperation * 0.3 + wisdom * 0.2
        env_impact = 100 - kwargs.get("environmental_impact", 50)
        resource_mgmt = kwargs.get("resource_management", 50)
        pop_control = kwargs.get("population_control", 50)
        long_term = kwargs.get("long_term_planning", 50)
        sustainability = env_impact * 0.3 + resource_mgmt * 0.3 + pop_control * 0.2 + long_term * 0.2
        result = self.calculate_civilization_score(tech_level, ethical_dev, sustainability)
        result["tech_progress"] = tech_progress
        result["decay_reasons"] = infer_decay_reasons(
            tech_level, ethical_dev, sustainability,
            environmental_impact=kwargs.get("environmental_impact", 50),
            violence_index=kwargs.get("violence_index", 50),
            inequality=kwargs.get("inequality", 50),
            space_exploration=kwargs.get("space_exploration", 50),
            tech_progress=tech_progress,
        )
        result.update({
            "name": name,
            "planet": planet,
            "age": age,
            "population": population,
            "kardashev_type": kardashev_type,
            "assessed_at": datetime.now().isoformat(),
        })
        self.civilizations.append(result)
        self.judgment_history.append({
            "timestamp": datetime.now().isoformat(),
            "civilization": name,
            "judgment": result["judgment"],
            "score": result["total_score"],
        })
        return result

    def export_to_json(self, output_path: str = "civilization_data.json") -> str:
        out = {
            "generated_at": datetime.now().isoformat(),
            "total_civilizations": len(self.civilizations),
            "data": self.civilizations,
            "summary": {
                "preserve": sum(1 for c in self.civilizations if c["judgment"] == "preserve"),
                "monitor": sum(1 for c in self.civilizations if c["judgment"] == "monitor"),
                "terminate": sum(1 for c in self.civilizations if c["judgment"] == "terminate"),
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return output_path


# ---------------------------------------------------------------------------
# Part 2: 文明シミュレーション（Simulation）
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sympy import primerange

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=False):
        return iterable

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 日本語ラベル: japanize_matplotlib があれば使用（IPAexGothic は環境により読めないことがあるため使用しない）
    try:
        import japanize_matplotlib  # noqa: F401
    except ImportError:
        pass
    plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の化け防止
except ImportError:
    plt = None

PLANET_DATA_SIM = {
    "水星": {"radius_au": 0.39, "terraforming_difficulty": 9},
    "金星": {"radius_au": 0.72, "terraforming_difficulty": 10},
    "地球": {"radius_au": 1.00, "terraforming_difficulty": 0},
    "火星": {"radius_au": 1.52, "terraforming_difficulty": 4},
    "木星": {"radius_au": 5.20, "terraforming_difficulty": 8},
    "土星": {"radius_au": 9.58, "terraforming_difficulty": 8},
    "天王星": {"radius_au": 19.22, "terraforming_difficulty": 9},
    "海王星": {"radius_au": 30.10, "terraforming_difficulty": 9},
}


def generate_and_judge_random_civilizations(
    num_civilizations: int,
    planet_names: List[str],
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """再生試行数分のパラメータをランダムに生成し、文明存続の探索で判定して結果のリストを返す。"""
    if seed is not None:
        np.random.seed(seed)
    if not planet_names or num_civilizations <= 0:
        return []
    system = CivilizationJudgmentSystem()
    results: List[Dict[str, Any]] = []
    for i in range(num_civilizations):
        name = f"文明_{i + 1}"
        planet = str(np.random.choice(planet_names))
        age = int(np.random.uniform(100, 500_000))
        population = max(1, int(np.random.uniform(1e6, 1e12)))
        kardashev = float(np.random.uniform(0.0, 3.0))
        ai_dev = int(np.random.uniform(0, 100))
        space_exp = int(np.random.uniform(0, 100))
        energy_util = int(np.random.uniform(0, 100))
        tech_progress = int(np.random.uniform(0, 100))
        violence = int(np.random.uniform(0, 100))
        inequality = int(np.random.uniform(0, 100))
        cooperation = int(np.random.uniform(0, 100))
        wisdom = int(np.random.uniform(0, 100))
        env_impact = int(np.random.uniform(0, 100))
        resource_mgmt = int(np.random.uniform(0, 100))
        pop_control = int(np.random.uniform(0, 100))
        long_term = int(np.random.uniform(0, 100))
        r = system.assess_civilization(
            name=name,
            planet=planet,
            age=age,
            population=population,
            kardashev_type=kardashev,
            ai_development=ai_dev,
            space_exploration=space_exp,
            energy_utilization=energy_util,
            tech_progress=tech_progress,
            violence_index=violence,
            inequality=inequality,
            cooperation=cooperation,
            wisdom=wisdom,
            environmental_impact=env_impact,
            resource_management=resource_mgmt,
            population_control=pop_control,
            long_term_planning=long_term,
        )
        results.append(r)
    return results


def _sim_difficulty_factor(difficulty: np.ndarray, max_difficulty: float, inverse: bool = False) -> np.ndarray:
    difficulty = np.asarray(difficulty)
    if max_difficulty <= 0:
        return np.ones_like(difficulty, dtype=float)
    scaled = difficulty / max_difficulty
    if inverse:
        return 1 + scaled * 1.0
    return np.maximum(0.1, 1 - scaled * 0.8)


def civilization_lifecycle_wave(
    t: np.ndarray,
    birth_time: float,
    lifetime: float,
    base_frequency: float,
    max_amplitude: float,
    phase: float,
    decay_rate_within_life: float,
) -> np.ndarray:
    """単一文明の栄枯衰退を表す波"""
    wave = np.zeros_like(t, dtype=float)
    active = (t >= birth_time) & (t < birth_time + lifetime)
    t_active = t[active] - birth_time
    if len(t_active) > 0:
        amp = max_amplitude * np.exp(-t_active * decay_rate_within_life)
        wave[active] = amp * np.sin(2 * np.pi * base_frequency * t_active + phase)
    return wave


def run_large_scale_civilization_simulation(
    t: np.ndarray,
    num_civilizations: int,
    freq_range: tuple,
    amp_range: tuple,
    phase_range: tuple,
    lifetime_range: tuple,
    decay_rate_within_life_range: tuple,
    nonlinear_weight: float,
    lambda_global_decay: float,
    escape_chance: float,
    planet_data: Dict[str, Dict[str, float]],
    bulk_judgment_data: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """多数文明の栄枯衰退シミュレーション。合成波とFFT結果を返す。
    bulk_judgment_data が渡された場合は、文明存続の探索で保存した判定結果を元に衰退理由・惑星を設定する。"""
    min_freq, max_freq = freq_range
    min_amp, max_amp = amp_range
    min_phase, max_phase = phase_range
    min_lifetime, max_lifetime = lifetime_range
    min_decay, max_decay = decay_rate_within_life_range

    planet_names = list(planet_data.keys())
    planet_radii = np.array([planet_data[p]["radius_au"] for p in planet_names])
    planet_diff = np.array([planet_data[p]["terraforming_difficulty"] for p in planet_names])
    N_t = len(t)
    num_planets = len(planet_names)
    max_difficulty = float(np.max(planet_diff)) if num_planets > 0 else 0

    if len(t) < 2:
        return None

    reason_names = DECAY_REASON_LABELS

    use_bulk = (
        bulk_judgment_data is not None
        and len(bulk_judgment_data) >= num_civilizations
        and num_planets > 0
    )
    if use_bulk:
        bulk = bulk_judgment_data[:num_civilizations]
        civilization_planets = np.array([d.get("planet", planet_names[0]) for d in bulk])
        default_reason = reason_names[0]
        decay_reasons = np.array([
            (d.get("decay_reasons") or [default_reason])[0]
            if (d.get("decay_reasons") or [default_reason])[0] in reason_names
            else default_reason
            for d in bulk
        ])
        planet_to_idx = {p: i for i, p in enumerate(planet_names)}
        civ_planet_idx = np.array([
            planet_to_idx.get(civilization_planets[i], 0)
            for i in range(num_civilizations)
        ])
        civilization_difficulties = planet_diff[civ_planet_idx]
    else:
        birth_upper = max(t[0], t[-1] - max_lifetime)
        birth_times = np.random.uniform(t[0], birth_upper, num_civilizations)
        birth_probs = 1.0 / (planet_diff + 1)
        birth_probs /= np.sum(birth_probs)
        civ_planet_idx = np.random.choice(num_planets, size=num_civilizations, p=birth_probs)
        civilization_planets = np.array([planet_names[i] for i in civ_planet_idx])
        civilization_difficulties = planet_diff[civ_planet_idx]
        total_r = sum(DECAY_REASON_WEIGHTS.values())
        decay_ratios = {k: v / total_r for k, v in DECAY_REASON_WEIGHTS.items()}
        weights = np.array([decay_ratios[r] for r in reason_names])
        decay_reasons = np.random.choice(reason_names, size=num_civilizations, p=weights)

    birth_upper = max(t[0], t[-1] - max_lifetime)
    birth_times = np.random.uniform(t[0], birth_upper, num_civilizations)

    lifetimes = np.zeros(num_civilizations)
    decay_rates = np.zeros(num_civilizations)
    for reason in reason_names:
        idx = np.where(decay_reasons == reason)[0]
        count = len(idx)
        if count == 0:
            continue
        if reason == "技術的破滅":
            lifetimes[idx] = np.random.uniform(min_lifetime * 0.1, max_lifetime * 0.5, count)
            decay_rates[idx] = np.random.uniform(max_decay * 2, max_decay * 10, count)
        elif reason == "生物学的破滅":
            lifetimes[idx] = np.random.uniform(min_lifetime * 0.3, max_lifetime * 0.7, count)
            decay_rates[idx] = np.random.uniform(min_decay, max_decay * 3, count)
        elif reason == "環境的破滅":
            lifetimes[idx] = np.random.uniform(min_lifetime * 0.2, max_lifetime * 0.5, count)
            decay_rates[idx] = np.random.uniform(min_decay, max_decay * 3, count)
        elif reason == "社会的破滅":
            lifetimes[idx] = np.random.uniform(min_lifetime * 0.05, min_lifetime * 0.3, count)
            decay_rates[idx] = np.random.uniform(max_decay * 5, max_decay * 20, count)
        elif reason == "宇宙的破滅":
            lifetimes[idx] = np.random.uniform(max_lifetime * 0.8, max_lifetime * 2.0, count)
            decay_rates[idx] = np.random.uniform(min_decay * 0.05, min_decay * 0.5, count)
        else:
            lifetimes[idx] = np.random.uniform(min_lifetime, max_lifetime, count)
            decay_rates[idx] = np.random.uniform(min_decay, max_decay, count)

    p = np.random.permutation(num_civilizations)
    lifetimes = lifetimes[p]
    decay_rates = decay_rates[p]
    decay_reasons = decay_reasons[p]

    n_civ = len(decay_reasons)
    decay_reason_ratios = {r: float(np.sum(decay_reasons == r) / n_civ) for r in reason_names}

    if max_difficulty > 0:
        lf = _sim_difficulty_factor(civilization_difficulties, max_difficulty, inverse=False)
        df = _sim_difficulty_factor(civilization_difficulties, max_difficulty, inverse=True)
        lifetimes = np.clip(lifetimes * lf, min_lifetime * 0.1, max_lifetime * 3.0)
        decay_rates = np.clip(decay_rates * df, min_decay * 0.1, max_decay * 20.0)

    base_freqs = np.random.uniform(min_freq, max_freq, num_civilizations)
    max_amps = np.random.uniform(min_amp, max_amp, num_civilizations)
    phases = np.random.uniform(min_phase, max_phase, num_civilizations)

    dist_matrix = np.abs(planet_radii[:, None] - planet_radii[None, :])
    max_dist = np.max(dist_matrix) if num_planets > 1 else 0
    conflict_risk = np.zeros(num_civilizations)
    if num_planets > 1:
        for i in range(num_civilizations):
            idx = civ_planet_idx[i]
            min_d = np.min([dist_matrix[idx, j] for j in range(num_planets) if j != idx])
            conflict_risk[i] = max(0, 1 - min_d / max_dist)
    conflict_idx = np.where(decay_reasons == "社会的破滅")[0]
    for i in conflict_idx:
        lifetimes[i] *= 1 - conflict_risk[i] * 0.5
        decay_rates[i] *= 1 + conflict_risk[i] * 0.5

    combined = np.zeros(N_t, dtype=float)
    for i in tqdm(range(num_civilizations), desc="文明波生成", leave=False):
        w = civilization_lifecycle_wave(
            t, birth_times[i], lifetimes[i], base_freqs[i], max_amps[i], phases[i], decay_rates[i]
        )
        if decay_reasons[i] == "宇宙的破滅":
            w[t >= birth_times[i] + lifetimes[i]] = 0
        combined += w
    if nonlinear_weight > 0:
        combined = np.tanh(combined * nonlinear_weight)
    final_wave = combined * np.exp(-lambda_global_decay * t)

    if N_t < 2:
        return {
            "time": t, "final_wave": final_wave, "frequencies": np.array([]),
            "amplitude_spectrum": np.array([]), "birth_times": birth_times, "lifetimes": lifetimes,
            "base_freqs": base_freqs, "max_amps": max_amps, "phases": phases,
            "decay_rates_within_life": decay_rates, "civilization_planets": civilization_planets,
            "civilization_conflict_risk": conflict_risk, "decay_reasons": decay_reasons,
            "decay_reason_ratios": decay_reason_ratios, "civilization_difficulties": civilization_difficulties,
        }
    T_samp = t[1] - t[0] if len(t) > 1 else 1.0
    yf = fft(final_wave)
    xf = fftfreq(N_t, T_samp)
    amplitude_spectrum = 2.0 / N_t * np.abs(yf[0 : N_t // 2])
    frequencies = xf[0 : N_t // 2]
    return {
        "time": t, "final_wave": final_wave, "frequencies": frequencies,
        "amplitude_spectrum": amplitude_spectrum, "birth_times": birth_times, "lifetimes": lifetimes,
        "base_freqs": base_freqs, "max_amps": max_amps, "phases": phases,
        "decay_rates_within_life": decay_rates, "civilization_planets": civilization_planets,
        "civilization_conflict_risk": conflict_risk, "decay_reasons": decay_reasons,
        "decay_reason_ratios": decay_reason_ratios, "civilization_difficulties": civilization_difficulties,
    }


def run_simulation_visualization(
    simulation_results: Optional[Dict],
    T_sim: float,
    save_dir: Optional[str] = None,
) -> List[str]:
    """シミュレーション結果を図として保存し、保存したパスのリストを返す。plt が無い場合は空リスト。"""
    if plt is None or simulation_results is None:
        return []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []

    t = simulation_results["time"]
    final_wave = simulation_results["final_wave"]
    frequencies = simulation_results["frequencies"]
    amplitude_spectrum = simulation_results["amplitude_spectrum"]
    birth_times = simulation_results["birth_times"]
    lifetimes = simulation_results["lifetimes"]
    decay_rates = simulation_results["decay_rates_within_life"]
    civilization_planets = simulation_results["civilization_planets"]
    decay_reasons = simulation_results["decay_reasons"]
    civilization_difficulties = simulation_results["civilization_difficulties"]

    def _save(name: str, fig):
        if save_dir:
            p = os.path.join(save_dir, f"{name}.png")
            fig.savefig(p, dpi=120, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(t / 1e9, final_wave)
    ax.set_title(f"{T_sim/1e9:.0f}億年スケール - {len(birth_times)} 文明の合成波動")
    ax.set_xlabel("時間 (億年)")
    ax.set_ylabel("合成活動レベル")
    ax.grid(True)
    _save("civilization_wave_time", fig)

    if len(frequencies) > 0 and len(amplitude_spectrum) > 0:
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(frequencies, amplitude_spectrum)
        ax.set_xlim(0, 1e-7 * 2)
        ax.set_title("FFT振幅スペクトル")
        ax.set_xlabel("周波数 (1/年)")
        ax.set_ylabel("振幅")
        ax.grid(True)
        _save("civilization_fft_spectrum", fig)

    if len(birth_times) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(birth_times / 1e9, bins=50, color="skyblue", edgecolor="black")
        ax1.set_title("文明の発生時間分布")
        ax1.set_xlabel("発生時間 (億年)")
        ax1.set_ylabel("文明数")
        ax1.grid(True)
        ax2.hist(lifetimes, bins=50, color="lightcoral", edgecolor="black")
        ax2.set_title("文明の寿命分布")
        ax2.set_xlabel("寿命 (年)")
        ax2.set_ylabel("文明数")
        ax2.set_yscale("log")
        ax2.grid(True)
        plt.tight_layout()
        _save("civilization_birth_lifetime_hist", fig)

    if len(decay_reasons) > 0:
        counts = pd.Series(decay_reasons).value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
        ax.set_title("文明の衰退理由の割合")
        ax.axis("equal")
        _save("civilization_decay_reasons_pie", fig)

    if len(civilization_planets) > 0:
        planet_counts = pd.Series(civilization_planets).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(planet_counts)), planet_counts.values, tick_label=planet_counts.index)
        ax.set_title("惑星ごとの文明発生数")
        ax.set_xlabel("惑星")
        ax.set_ylabel("文明数")
        plt.xticks(rotation=45)
        plt.tight_layout()
        _save("civilization_planets_bar", fig)

    return paths


# ---------------------------------------------------------------------------
# Part 3: Streamlit UI（文明審査 + 文明シミュレーション）
# ---------------------------------------------------------------------------

import streamlit as st

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESERVE_LINE = JUDGMENT_THRESHOLDS["preserve"]
MONITOR_LINE = JUDGMENT_THRESHOLDS["monitor"]

ARCAIA_LORE_PATH = os.path.join(_SCRIPT_DIR, "arcaia_lore.json")
EARTH_CIVILIZATIONS_PATH = os.path.join(_SCRIPT_DIR, "earth_civilizations.json")
TECH_LEVEL_TABLE_PATH = os.path.join(_SCRIPT_DIR, "technology_level_table.json")
SOLAR_SYSTEM_HABITABILITY_PATH = os.path.join(_SCRIPT_DIR, "solar_system_habitability.json")

# 宇宙歴の選択肢（年）: 表示名と経過年
COSMIC_TIME_OPTIONS = [
    ("現在", 0),
    ("5億年後", 500_000_000),
    ("10億年後", 1_000_000_000),
    ("20億年後", 2_000_000_000),
    ("30億年後", 3_000_000_000),
    ("50億年後（太陽系の崩壊）", 5_000_000_000),
    ("70億年後（白色矮星期）", 7_000_000_000),
]


def _load_solar_system_habitability() -> Dict[str, Any]:
    """太陽系の崩壊と生息域のタイムラインを読み込む。"""
    if not os.path.isfile(SOLAR_SYSTEM_HABITABILITY_PATH):
        return {}
    try:
        with open(SOLAR_SYSTEM_HABITABILITY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_habitability_at_years(years_from_now: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """経過年に対応する太陽系イベント（生息域）を返す。"""
    events = data.get("events") or []
    if not events:
        return None
    chosen = events[0]
    for ev in events:
        if ev.get("years_from_now", 0) <= years_from_now:
            chosen = ev
    return chosen


def _load_technology_level_table() -> Dict[str, Any]:
    """技術レベル表（人類誕生〜100億年後）を読み込む。"""
    if not os.path.isfile(TECH_LEVEL_TABLE_PATH):
        return {}
    try:
        with open(TECH_LEVEL_TABLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_tech_level_row(tech_level: float, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """技術水準スコア（0-100）に対応する技術レベル表の行を返す。"""
    levels = table.get("levels") or []
    if not levels:
        return None
    tech_level = max(0, min(100, tech_level))
    for i, row in enumerate(levels):
        lo = row.get("score_min", 0)
        hi = row.get("score_max", 100)
        is_last = i == len(levels) - 1
        if lo <= tech_level < hi or (is_last and lo <= tech_level <= hi):
            return row
    return levels[-1]


def _load_arcaia_lore() -> Dict[str, Any]:
    """アルカイアの慟哭 文明興亡データベースを読み込む。"""
    if not os.path.isfile(ARCAIA_LORE_PATH):
        return {}
    try:
        with open(ARCAIA_LORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_earth_civilizations() -> Dict[str, Any]:
    """地球上の勃興文明データベースを読み込む。"""
    if not os.path.isfile(EARTH_CIVILIZATIONS_PATH):
        return {}
    try:
        with open(EARTH_CIVILIZATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_year_range(year_range: str) -> tuple:
    """year_range 文字列を (start_year, end_year) にパース。BCは負の数。パース失敗時は (None, None)。"""
    import re
    if not year_range or not isinstance(year_range, str):
        return (None, None)
    parts = year_range.replace("頃", "").replace("年代", "").strip().split("-")
    if len(parts) < 2:
        return (None, None)
    def to_year(s):
        s = s.strip()
        m = re.search(r"(\d+)", s)
        if not m:
            return None
        n = int(m.group(1))
        if "BC" in s.upper() or "紀元前" in s:
            return -n
        return n
    start, end = to_year(parts[0]), to_year(parts[1])
    return (start, end)


def _get_similar_earth_civilizations(
    tech_level: float,
    ethical_dev: float,
    sustainability: float,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """地球上の勃興文明データベースから、技術・倫理・持続可能性が近い文明を返す。審査結果の参照用。"""
    data = _load_earth_civilizations()
    civs = data.get("civilizations") or []
    scored = []
    for c in civs:
        t = c.get("tech_level")
        e = c.get("ethical_dev")
        s = c.get("sustainability")
        if t is None or e is None or s is None:
            continue
        dist = np.sqrt((t - tech_level) ** 2 + (e - ethical_dev) ** 2 + (s - sustainability) ** 2)
        total = t * 0.30 + e * 0.35 + s * 0.35
        scored.append({
            "name": c.get("name", "不明"),
            "collapse_reason": c.get("collapse_reason", ""),
            "decay_reasons": c.get("decay_reasons", []),
            "tech_level": t,
            "ethical_dev": e,
            "sustainability": s,
            "total_score": total,
            "distance": float(dist),
        })
    scored.sort(key=lambda x: x["distance"])
    return scored[:k]


def _reset_judgment_inputs():
    for k, v in {
        "civ_name": "未命名文明", "civ_planet": "惑星名", "civ_age": 10000,
        "civ_population": 1_000_000_000, "civ_kardashev": 0.73,
        "civ_ai_development": 50, "civ_space_exploration": 50, "civ_energy_utilization": 50,
        "civ_tech_progress": 50,
        "civ_violence_index": 50, "civ_inequality": 50, "civ_cooperation": 50, "civ_wisdom": 50,
        "civ_environmental_impact": 50, "civ_resource_management": 50,
        "civ_population_control": 50, "civ_long_term_planning": 50,
        "num_civilizations": 2000,
    }.items():
        st.session_state[k] = v
    for k in ("last_result", "last_submitted_inputs"):
        st.session_state.pop(k, None)


st.set_page_config(page_title="文明存続探索 — アルカイアの慟哭", page_icon="🌍", layout="wide")
st.markdown("""
<style>
div[data-baseweb="slider"] { min-width: min(100%, 320px) !important; }
.stSlider label p { font-size: 18px !important; font-weight: bold !important; }
[data-testid="stMetric"] { padding: 0.6rem !important; border-radius: 10px !important; border-left: 4px solid #1e3a5f !important; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🌍 文明存続探索システム")
st.sidebar.caption("小説『アルカイアの慟哭』世界観 — 文明が生き残る方法を探るツール。")
st.sidebar.markdown("**プロトコル**")
st.sidebar.markdown("*「自然（地球）を破壊するものには断固として抵抗し、排除する。」*")

# 世界観（arcaia_lore.json）
_lore = _load_arcaia_lore()
if _lore:
    _meta = _lore.get("meta", {})
    _arcaia = _lore.get("arcaia", {})
    st.sidebar.markdown("---")
    st.sidebar.markdown("**観測者**")
    st.sidebar.caption(f"**{_arcaia.get('name', 'アルカイア')}** — {_arcaia.get('type', '超知性AI')}")
    _comp = _arcaia.get("companion", {})
    if _comp:
        st.sidebar.caption(f"相棒: {_comp.get('name', 'リタ')}（{_comp.get('species', '犬')}）")
    st.sidebar.caption(_arcaia.get("philosophy", ""))
    st.sidebar.caption(f"観測期間: {_meta.get('observation_period', '50億年')} / 総サイクル: {_meta.get('total_cycles', 5000)}")

# メインモード: 文明存続の探索 / 50億年スケール再生試行 / 文明興亡データベース
if "main_mode" not in st.session_state:
    st.session_state["main_mode"] = "📋 文明存続の探索"
if st.session_state.get("_main_mode_request"):
    st.session_state["main_mode_widget"] = st.session_state.pop("_main_mode_request")
main_mode = st.sidebar.radio(
    "メインモード",
    ["📋 文明存続の探索", "🌐 50億年スケール再生試行", "📚 文明興亡データベース"],
    key="main_mode_widget",
)
st.session_state["main_mode"] = main_mode

st.sidebar.markdown("---")
st.sidebar.markdown("**観測枠での判定**")
st.sidebar.caption(f"- {PRESERVE_LINE}点以上: プロトコル適合")
st.sidebar.caption(f"- {MONITOR_LINE}〜{PRESERVE_LINE}点: 要観測")
st.sidebar.caption(f"- {MONITOR_LINE}点未満: 排除・再生失敗")
st.sidebar.caption("※閾値は地球上の勃興文明データベースの実例を参考に設定。観測結果では類似した歴史上の文明を参照表示。")

_tech_table_sidebar = _load_technology_level_table()
if _tech_table_sidebar.get("levels"):
    with st.sidebar.expander("📐 技術レベル表（人類誕生〜100億年後）", expanded=False):
        st.caption(_tech_table_sidebar.get("meta", {}).get("usage", ""))
        for row in _tech_table_sidebar["levels"]:
            st.markdown(f"**Lv.{row.get('level')}** {row.get('name', '')} — {row.get('period', '')}")
            st.caption(f"プラス: {' / '.join(row.get('plus', [])[:2])}{'…' if len(row.get('plus', [])) > 2 else ''}")
            st.caption(f"マイナス: {' / '.join(row.get('minus', [])[:2])}{'…' if len(row.get('minus', [])) > 2 else ''}")

st.title("🌍 文明存続探索システム")
st.caption("プロトコルに照らし、文明が生き残る方法を探る。")

# ========== 文明存続の探索 ==========
if main_mode == "📋 文明存続の探索":
    if "nav_mode_civ" not in st.session_state:
        st.session_state["nav_mode_civ"] = "📝 パラメータ入力"
    if st.session_state.get("_nav_request"):
        st.session_state["nav_mode_civ_widget"] = st.session_state.pop("_nav_request")
    if st.session_state.pop("_jump_to_analysis_civ", False):
        st.session_state["nav_mode_civ"] = "📊 観測結果"
        st.session_state["nav_mode_civ_widget"] = "📊 観測結果"
    nav_mode = st.radio("表示", ["📝 パラメータ入力", "📊 観測結果"], horizontal=True, key="nav_mode_civ_widget")
    st.session_state["nav_mode_civ"] = nav_mode

    if nav_mode == "📝 パラメータ入力":
        st.header("📝 再生文明のパラメータ入力")
        st.caption("観測枠に載せる文明の指標を入力し、プロトコル適合度を探索する。")
        if st.button("🆕 新しく入力する"):
            _reset_judgment_inputs()
            st.rerun()
        last_inp = st.session_state.get("last_submitted_inputs") or {}
        def _v(k, default): return last_inp.get(k, st.session_state.get(k, default))

        # 再生試行数（文明の数）— 50億年スケール再生試行でもこの値を使う
        num_civ_shared = st.number_input(
            "再生試行数（文明の数）",
            min_value=100,
            max_value=20000,
            value=int(_v("num_civilizations", 2000)),
            step=500,
            key="civ_num_civilizations",
            help="50億年スケール再生試行のシミュレーションでもこの値が使われます。",
        )
        st.session_state["num_civilizations"] = num_civ_shared

        st.markdown("**一括生成・判定・保存**")
        if st.button("🔄 再生試行数分のパラメータをランダム生成して判定し、データを保存", key="bulk_gen_judge"):
            with st.spinner(f"{num_civ_shared:,} 文明をランダム生成・判定中..."):
                bulk = generate_and_judge_random_civilizations(
                    num_civ_shared,
                    list(PLANET_DATA_SIM.keys()),
                    seed=None,
                )
            if bulk:
                st.session_state["bulk_judgment_data"] = bulk
                out_path = os.path.join(_SCRIPT_DIR, "bulk_judgment_data.json")
                out = {
                    "generated_at": datetime.now().isoformat(),
                    "num_civilizations": len(bulk),
                    "data": bulk,
                    "summary": {
                        "preserve": sum(1 for c in bulk if c.get("judgment") == "preserve"),
                        "monitor": sum(1 for c in bulk if c.get("judgment") == "monitor"),
                        "terminate": sum(1 for c in bulk if c.get("judgment") == "terminate"),
                    },
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                st.success(f"判定データ {len(bulk):,} 件を保存しました。（{out_path}）")
            else:
                st.warning("生成できませんでした。再生試行数と惑星データを確認してください。")

        if st.session_state.get("bulk_judgment_data"):
            st.caption(f"保存済み: 文明存続の探索データ {len(st.session_state['bulk_judgment_data']):,} 件 — 50億年スケール再生試行でこのデータを元に計算します。")

        with st.form("civ_form"):
            st.info("必須: 文明名称・惑星名・人口（1以上）。未入力項目は50で計算。自然を破壊しない文明かどうかの目安として観測する。")
            with st.expander("📌 文明の基本情報（再生試行のラベル）", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    name = st.text_input("文明名称", value=_v("civ_name", "未命名文明"), key="civ_name_in")
                    planet = st.text_input("惑星名", value=_v("civ_planet", "惑星名"), key="civ_planet_in")
                with c2:
                    age = st.number_input("文明年齢（年）", 0, 1_000_000, int(_v("civ_age", 10000)), 1000, key="civ_age_in")
                    population = st.number_input("人口", 0, 10**15, int(_v("civ_population", 1_000_000_000)), 100_000_000, key="civ_pop_in")
                cosmic_time_label = st.selectbox(
                    "想定する宇宙歴（太陽系の経過年）",
                    options=[opt[0] for opt in COSMIC_TIME_OPTIONS],
                    index=0,
                    key="cosmic_time_select",
                    help="太陽系の崩壊は約50億年後。10億年後には地球は居住不可とされる。この時点での人類の生息域を判定に使う。",
                )
                cosmic_time_years = next(y for l, y in COSMIC_TIME_OPTIONS if l == cosmic_time_label)
                kardashev = st.slider("カルダシェフ (Type 0〜III)", 0.0, 3.0, float(_v("civ_kardashev", 0.73)), 0.01, key="civ_kard_in")
            with st.expander("🔬 技術水準（30%）— 現在の水準と技術的進歩を審査に反映", expanded=True):
                t1, t2, t3, t4 = st.columns(4)
                with t1: ai_dev = st.slider("AI発展度", 0, 100, int(_v("civ_ai_development", 50)), key="civ_ai")
                with t2: space_exp = st.slider("宇宙探査能力", 0, 100, int(_v("civ_space_exploration", 50)), key="civ_space")
                with t3: energy_util = st.slider("エネルギー利用効率", 0, 100, int(_v("civ_energy_utilization", 50)), key="civ_energy")
                with t4: tech_progress = st.slider("技術的進歩・革新性", 0, 100, int(_v("civ_tech_progress", 50)), key="civ_tech_progress", help="発展の速さ・イノベーション能力。高すぎる進歩は技術的破滅リスクにもつながる。")
            with st.expander("⚖️ 倫理水準（35%）", expanded=True):
                e1, e2, e3, e4 = st.columns(4)
                with e1: violence = st.slider("暴力指数(低いほど良い)", 0, 100, int(_v("civ_violence_index", 50)), key="civ_violence")
                with e2: inequality = st.slider("不平等度(低いほど良い)", 0, 100, int(_v("civ_inequality", 50)), key="civ_inequality")
                with e3: cooperation = st.slider("協調性", 0, 100, int(_v("civ_cooperation", 50)), key="civ_cooperation")
                with e4: wisdom = st.slider("知恵・判断力", 0, 100, int(_v("civ_wisdom", 50)), key="civ_wisdom")
            with st.expander("🌍 自然との調和（35%）— 自然を破壊しないか", expanded=True):
                s1, s2, s3, s4 = st.columns(4)
                with s1: env_impact = st.slider("環境負荷(低いほど良い)", 0, 100, int(_v("civ_environmental_impact", 50)), key="civ_env")
                with s2: resource_mgmt = st.slider("資源管理能力", 0, 100, int(_v("civ_resource_management", 50)), key="civ_resource")
                with s3: pop_control = st.slider("人口制御", 0, 100, int(_v("civ_population_control", 50)), key="civ_pop_ctrl")
                with s4: long_term = st.slider("長期計画性", 0, 100, int(_v("civ_long_term_planning", 50)), key="civ_long_term")
            submitted = st.form_submit_button("観測・適合度判定", type="primary")

        if submitted and name and name.strip() and planet and planet.strip() and population >= 1:
            system = CivilizationJudgmentSystem()
            result = system.assess_civilization(
                name=name.strip(), planet=planet.strip(), age=int(age), population=int(population),
                kardashev_type=float(kardashev), ai_development=ai_dev, space_exploration=space_exp,
                energy_utilization=energy_util, tech_progress=tech_progress,
                violence_index=violence, inequality=inequality,
                cooperation=cooperation, wisdom=wisdom, environmental_impact=env_impact,
                resource_management=resource_mgmt, population_control=pop_control, long_term_planning=long_term,
            )
            st.session_state["last_result"] = {
                "score": result["total_score"], "hantei": result["action"],
                "tech_level": result["tech_level"], "ethical_dev": result["ethical_dev"],
                "sustainability": result["sustainability"], "judgment": result["judgment"],
                "action": result["action"], "description": result["description"], "warnings": result["warnings"],
                "name": result["name"], "planet": result["planet"], "age": result["age"],
                "population": result["population"], "kardashev_type": result["kardashev_type"],
                "decay_reasons": result.get("decay_reasons", [DECAY_REASON_LABELS[-1]]),
                "tech_progress": result.get("tech_progress", 50),
                "cosmic_time_years": cosmic_time_years,
                "cosmic_time_label": cosmic_time_label,
            }
            st.session_state["last_submitted_inputs"] = {
                "civ_name": name, "civ_planet": planet, "civ_age": age, "civ_population": population,
                "civ_kardashev": kardashev, "civ_ai_development": ai_dev, "civ_space_exploration": space_exp,
                "civ_energy_utilization": energy_util, "civ_tech_progress": tech_progress,
                "civ_violence_index": violence, "civ_inequality": inequality,
                "civ_cooperation": cooperation, "civ_wisdom": wisdom,
                "civ_environmental_impact": env_impact, "civ_resource_management": resource_mgmt,
                "civ_population_control": pop_control, "civ_long_term_planning": long_term,
                "num_civilizations": st.session_state.get("num_civilizations", 2000),
                "cosmic_time_years": cosmic_time_years,
                "cosmic_time_label": cosmic_time_label,
            }
            st.session_state["_jump_to_analysis_civ"] = True
            st.rerun()
        elif submitted:
            if not (name and name.strip()): st.error("文明名称を入力してください。")
            if not (planet and planet.strip()): st.error("惑星名を入力してください。")
            if population < 1: st.error("人口は1以上を入力してください。")

    if nav_mode == "📊 観測結果":
        if "last_result" not in st.session_state:
            st.info("「再生文明のパラメータ入力」でデータを入れ、「観測・適合度判定」を押してください。")
        else:
            res = st.session_state["last_result"]
            st.markdown(f"### 📊 観測結果 — {res.get('name', '文明')}")
            st.caption("プロトコル「自然（地球）を破壊するものには断固として抵抗し、排除する。」に照らした適合度。")
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1: st.metric("適合度スコア", f"{res['score']:.1f} / 100")
            with k2: st.metric("観測枠での判定", res["hantei"])
            with k3: st.metric("技術水準", f"{res.get('tech_level', 0):.1f}")
            with k4: st.metric("技術的進歩", f"{res.get('tech_progress', 50):.0f}")
            with k5: st.metric("倫理水準", f"{res.get('ethical_dev', 0):.1f}")
            with k6: st.metric("自然との調和", f"{res.get('sustainability', 0):.1f}")
            for w in (res.get("warnings") or []):
                st.warning(w)
            st.success(res["description"])
            decay_reasons = res.get("decay_reasons") or []
            if decay_reasons:
                st.markdown("**衰退理由（この文明の指標から推定）:** " + " / ".join(decay_reasons))

            solar_data = _load_solar_system_habitability()
            cosmic_years = res.get("cosmic_time_years", 0)
            habitability_ev = _get_habitability_at_years(cosmic_years, solar_data) if solar_data else None
            planet_name = (res.get("planet") or "地球").strip()
            if habitability_ev:
                st.markdown("---")
                st.markdown("**🌍 人類の生息域判定（太陽系の崩壊タイムラインに基づく）**")
                st.caption(f"想定時点: **{res.get('cosmic_time_label', '現在')}**（太陽系の経過年 {cosmic_years:,} 年）")
                st.markdown(habitability_ev.get("description", ""))
                hab = habitability_ev.get("habitability") or {}
                planet_status = hab.get(planet_name) or hab.get("地球") or "—"
                if any(x in (planet_status or "") for x in ("不可", "消滅", "限界")):
                    st.error(f"**{planet_name}**: {planet_status} — この時点では選択惑星は人類の生息域として持続不可能。移住・脱出が前提。")
                else:
                    st.success(f"**{planet_name}**: {planet_status}")
                st.markdown("**各惑星の状態（この時点）**")
                for p, status in hab.items():
                    st.caption(f"• {p}: {status}")
                if habitability_ev.get("summary"):
                    st.info(habitability_ev["summary"])
                if habitability_ev.get("unhabitable_note"):
                    st.warning(habitability_ev["unhabitable_note"])

            tech_table = _load_technology_level_table()
            tech_row = _get_tech_level_row(res.get("tech_level", 50), tech_table) if tech_table else None
            if tech_row:
                st.markdown("---")
                st.markdown("**📐 想定技術レベル（技術レベル表より）**")
                st.markdown(f"**レベル {tech_row.get('level', '')}**: {tech_row.get('name', '')} — *{tech_row.get('period', '')}*")
                st.caption(tech_row.get("description", ""))
                col_plus, col_minus = st.columns(2)
                with col_plus:
                    st.markdown("**プラス面**")
                    for p in tech_row.get("plus", []):
                        st.write(f"• {p}")
                with col_minus:
                    st.markdown("**マイナス面（審査上の留意点）**")
                    for m in tech_row.get("minus", []):
                        st.write(f"• {m}")
                risk_note = tech_row.get("minus", [])
                if risk_note and res.get("ethical_dev", 50) < 50:
                    st.warning(f"この技術レベルで想定されるリスクの一つとして「{risk_note[0]}」等が挙がる。倫理水準とのバランスに注意。")

            similar = _get_similar_earth_civilizations(
                res.get("tech_level", 50),
                res.get("ethical_dev", 50),
                res.get("sustainability", 50),
                k=5,
            )
            if similar:
                from collections import Counter as _C
                agg = _C()
                for s in similar:
                    for r in s.get("decay_reasons") or []:
                        agg[r] += 1
                top_reasons = [r for r, _ in agg.most_common(3)] if agg else []
                with st.expander("📚 歴史上の類似文明（地球上の勃興文明データベースより）", expanded=True):
                    st.caption("技術・倫理・自然との調和のスコアが近い地球上の文明。いずれも何らかの形で終焉しているため、参照として活用できる。")
                    if top_reasons:
                        st.info(f"**類似文明で多かった破滅パターン:** 「{'」「'.join(top_reasons)}」 — データベースに基づく補足参考。")
                    for s in similar:
                        st.markdown(f"**{s['name']}** — 適合度目安 {s['total_score']:.0f}点（距離: {s['distance']:.0f}）")
                        st.caption(f"崩壊理由: {s['collapse_reason'][:80]}{'…' if len(s['collapse_reason']) > 80 else ''}")
                        if s["decay_reasons"]:
                            st.caption(f"破滅パターン: {' / '.join(s['decay_reasons'])}")
                        st.write("")

            st.divider()
            st.caption(f"名称: {res.get('name')} | 惑星: {res.get('planet')} | 文明年齢: {res.get('age', 0):,}年 | 人口: {res.get('population', 0):,} | カルダシェフ Type {res.get('kardashev_type')}")
            if st.button("📝 パラメータ入力に戻る", key="switch_input"):
                st.session_state["_nav_request"] = "📝 パラメータ入力"
                st.rerun()

# ========== 50億年スケール再生試行 ==========
if main_mode == "🌐 50億年スケール再生試行":
    st.header("🌐 50億年スケール再生試行")
    st.caption("アルカイアによる無数の再生試行を、時間軸上の波としてシミュレーションする。リタとYOSHITUNEの観測枠で、文明の栄枯・スペクトルを可視化する。実行に数十秒かかることがある。")
    # 文明存続の探索で入力した「再生試行数（文明の数）」を使う（未入力時は2000）
    num_civ = int(st.session_state.get("num_civilizations") or st.session_state.get("last_submitted_inputs", {}).get("num_civilizations", 2000))
    bulk_data = st.session_state.get("bulk_judgment_data")
    use_bulk = bulk_data is not None and len(bulk_data) >= num_civ
    if use_bulk:
        st.info(f"**使用する文明数: {num_civ:,}** — 文明存続の探索で保存した判定データを元に50億年スケール再生試行を計算します。")
    else:
        st.info(f"**使用する文明数: {num_civ:,}**（「文明存続の探索」で「再生試行数分をランダム生成して判定・保存」を実行すると、そのデータを元に計算します。）")
    if st.button("🚀 再生試行シミュレーション実行", type="primary", key="run_sim"):
        T_sim = 5e9
        N_steps = 10000
        t = np.linspace(0, T_sim, N_steps)
        with st.spinner("50億年スケールで再生試行を計算中..."):
            np.random.seed(42)
            res = run_large_scale_civilization_simulation(
                t=t, num_civilizations=int(num_civ),
                freq_range=(1e-9, 1e-7), amp_range=(0.5, 5.0), phase_range=(0, 2 * np.pi),
                lifetime_range=(1e5, 5e7), decay_rate_within_life_range=(1e-8, 1e-6),
                nonlinear_weight=0.5, lambda_global_decay=5e-10, escape_chance=0.1,
                planet_data=PLANET_DATA_SIM,
                bulk_judgment_data=bulk_data if use_bulk else None,
            )
        if res is None:
            st.error("シミュレーションに失敗しました。")
        else:
            fig_dir = os.path.join(_SCRIPT_DIR, "civilization_figures")
            paths = run_simulation_visualization(res, T_sim, save_dir=fig_dir)
            st.session_state["sim_result_paths"] = paths
            st.session_state["sim_result"] = res
            st.session_state["sim_used_bulk_data"] = use_bulk
            st.success(f"観測完了。{len(paths)} 枚の図を保存した。" + ("（文明存続の探索の保存データを元に計算）" if use_bulk else ""))
            st.rerun()

    if st.session_state.get("sim_result_paths"):
        st.subheader("観測結果（可視化）")
        sim_res = st.session_state.get("sim_result")
        if sim_res and "decay_reasons" in sim_res:
            decay_reasons = sim_res["decay_reasons"]
            if len(decay_reasons) > 0:
                counts = pd.Series(decay_reasons).value_counts()
                st.markdown("**衰退理由の内訳（各文明ごとに判定した結果の合計）**")
                if st.session_state.get("sim_used_bulk_data"):
                    st.caption(f"全 {len(decay_reasons):,} 文明 — 文明存続の探索でランダム生成・判定して保存したデータを元に集計しています。")
                else:
                    st.caption(f"全 {len(decay_reasons):,} 文明 — 破滅パターン6分類（技術的・生物学的・環境的・社会的・宇宙的・その他）の重みに基づきランダムに割り当て、集計しています。")
                for reason, count in counts.items():
                    pct = 100.0 * count / len(decay_reasons)
                    st.write(f"- **{reason}**: {count:,} 文明 ({pct:.1f}%)")
                st.divider()
        for p in st.session_state["sim_result_paths"]:
            if os.path.isfile(p):
                st.image(p, use_container_width=True)
        if st.button("結果をクリア", key="clear_sim"):
            st.session_state.pop("sim_result_paths", None)
            st.session_state.pop("sim_result", None)
            st.session_state.pop("sim_used_bulk_data", None)
            st.rerun()

# ========== 文明興亡データベース（arcaia_lore / earth_civilizations / technology_level_table） ==========
if main_mode == "📚 文明興亡データベース":
    _db_lore = _load_arcaia_lore()
    _earth_db = _load_earth_civilizations()
    _tech_table_db = _load_technology_level_table()
    has_arcaia = _db_lore and "cycles" in _db_lore
    has_earth = _earth_db and "civilizations" in _earth_db
    has_tech_table = _tech_table_db and "levels" in _tech_table_db

    if not has_arcaia and not has_earth and not has_tech_table:
        st.header("📚 文明興亡データベース")
        st.info("`arcaia_lore.json`、`earth_civilizations.json`、または `technology_level_table.json` を同じフォルダに配置すると、データを閲覧できます。")
    else:
        db_source = st.radio(
            "データソース",
            [x for x in [
                ("📐 技術レベル表（人類〜100億年後）", "tech_table") if has_tech_table else None,
                ("🌍 地球上の勃興文明", "earth") if has_earth else None,
                ("🔮 アルカイアの観測（50億年サイクル）", "arcaia") if has_arcaia else None,
            ] if x is not None],
            format_func=lambda x: x[0],
            key="db_source_select",
            horizontal=True,
        )
        db_source = db_source[1] if isinstance(db_source, tuple) else db_source

        if db_source == "tech_table" and has_tech_table:
            _meta_t = _tech_table_db.get("meta", {})
            st.header(f"📐 {_meta_t.get('title', '技術レベル表')}")
            st.caption(_meta_t.get("scope", "") + " — " + (_meta_t.get("usage", "")))
            st.caption("審査の技術水準スコア（0-100）に応じて、観測結果で該当レベルを参照表示し、プラス面・マイナス面を審査に活かす。")
            st.divider()
            for row in _tech_table_db["levels"]:
                with st.expander(f"**レベル {row.get('level')}**: {row.get('name', '')} — {row.get('period', '')}", expanded=(row.get("level", 0) in (0, 7, 8, 16))):
                    st.markdown(row.get("description", ""))
                    st.markdown("**スコア範囲:** " + str(row.get("score_min", "")) + " 〜 " + str(row.get("score_max", "")))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**プラス面**")
                        for p in row.get("plus", []):
                            st.write(f"• {p}")
                    with c2:
                        st.markdown("**マイナス面（審査上の留意点）**")
                        for m in row.get("minus", []):
                            st.write(f"• {m}")

        elif db_source == "earth" and has_earth:
            _meta_db = _earth_db.get("meta", {})
            st.header(f"📚 {_meta_db.get('title', '地球上の勃興文明データベース')}")
            st.caption(_meta_db.get("description", ""))
            civs = _earth_db["civilizations"]
            st.caption(f"**{len(civs)}** 件の文明を、技術・倫理・自然との調和・破滅パターン6分類に当てはめて整理。")

            # 集計グラフ
            if plt is not None and civs:
                st.subheader("📊 集計グラフ")
                names_short = []
                tech_vals, eth_vals, sus_vals = [], [], []
                for c in civs:
                    t = c.get("tech_level")
                    e = c.get("ethical_dev")
                    s = c.get("sustainability")
                    if t is not None and e is not None and s is not None:
                        tech_vals.append(t)
                        eth_vals.append(e)
                        sus_vals.append(s)
                        n = c.get("name", "不明")
                        names_short.append(n[:10] + "…" if len(n) > 10 else n)
                if names_short:
                    x = np.arange(len(names_short))
                    w = 0.25
                    fig1, ax1 = plt.subplots(figsize=(14, 5))
                    ax1.bar(x - w, tech_vals, width=w, label="技術", color="steelblue")
                    ax1.bar(x, eth_vals, width=w, label="倫理", color="seagreen")
                    ax1.bar(x + w, sus_vals, width=w, label="自然との調和", color="darkorange")
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(names_short, rotation=45, ha="right")
                    ax1.set_ylabel("スコア (0-100)")
                    ax1.set_title("文明ごとの技術・倫理・自然との調和")
                    ax1.legend(loc="upper right")
                    ax1.set_ylim(0, 105)
                    ax1.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)

                from collections import Counter
                decay_counts = Counter()
                for c in civs:
                    for r in c.get("decay_reasons") or []:
                        decay_counts[r] += 1
                if decay_counts:
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    reasons = list(decay_counts.keys())
                    counts = [decay_counts[r] for r in reasons]
                    bars = ax2.barh(reasons, counts, color="coral", alpha=0.8)
                    ax2.set_xlabel("該当件数（1文明が複数該当可）")
                    ax2.set_title("破滅パターン（6分類）別 該当件数")
                    ax2.grid(axis="x", alpha=0.3)
                    for b, v in zip(bars, counts):
                        ax2.text(v + 0.1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                total_scores = []
                for c in civs:
                    t, e, s = c.get("tech_level"), c.get("ethical_dev"), c.get("sustainability")
                    if t is not None and e is not None and s is not None:
                        total_scores.append(t * 0.30 + e * 0.35 + s * 0.35)
                if total_scores:
                    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
                    ax3.hist(total_scores, bins=10, range=(0, 100), color="teal", alpha=0.7, edgecolor="black")
                    ax3.axvline(60, color="green", linestyle="--", label="適合 60点以上")
                    ax3.axvline(40, color="orange", linestyle="--", label="要観測 40点以上")
                    ax3.set_xlabel("適合度スコア（技術30%+倫理35%+自然35%）")
                    ax3.set_ylabel("文明数")
                    ax3.set_title("適合度スコアの分布")
                    ax3.legend()
                    ax3.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)

                # 地球文明の年表（タイムライン図 + 年代順一覧）
                st.subheader("📅 地球文明の年表")
                timeline_data = []
                for c in civs:
                    start, end = _parse_year_range(c.get("year_range", ""))
                    if start is not None and end is not None:
                        timeline_data.append({
                            "name": c.get("name", "不明"),
                            "name_short": (c.get("name", "不明")[:12] + "…") if len(c.get("name", "")) > 12 else c.get("name", "不明"),
                            "start": start,
                            "end": end,
                            "year_range": c.get("year_range", ""),
                        })
                if timeline_data:
                    timeline_data.sort(key=lambda x: x["start"])
                    names_tl = [t["name_short"] for t in timeline_data]
                    starts = np.array([t["start"] for t in timeline_data])
                    ends = np.array([t["end"] for t in timeline_data])
                    y_pos = np.arange(len(names_tl))
                    fig4, ax4 = plt.subplots(figsize=(12, max(6, len(names_tl) * 0.35)))
                    ax4.barh(y_pos, ends - starts, left=starts, height=0.6, color="steelblue", alpha=0.8, edgecolor="navy")
                    ax4.set_yticks(y_pos)
                    ax4.set_yticklabels(names_tl, fontsize=9)
                    ax4.set_xlabel("年（BC=紀元前）")
                    ax4.set_title("地球文明の年代範囲")
                    ax4.axvline(0, color="gray", linestyle="-", linewidth=0.8)
                    ax4.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)

                # 年表（年代順テーブル）
                if timeline_data:
                    st.markdown("**年代順一覧**")
                    table_rows = []
                    for t in timeline_data:
                        s, e = t["start"], t["end"]
                        bc = "BC" if s < 0 else "AD"
                        start_str = f"{abs(s)}BC" if s < 0 else f"{s}AD"
                        end_str = f"{abs(e)}BC" if e < 0 else f"{e}AD"
                        table_rows.append({"文明名": t["name"], "開始": start_str, "終了": end_str, "期間（目安）": t["year_range"]})
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            st.divider()
            for c in civs:
                name = c.get("name", "不明")
                region = c.get("region", "")
                year_range = c.get("year_range", "")
                tech = c.get("technology_level", "")
                collapse = c.get("collapse_reason", "")
                decay = c.get("decay_reasons", [])
                tech_level = c.get("tech_level")
                ethical_dev = c.get("ethical_dev")
                sustainability = c.get("sustainability")
                event = c.get("event") or {}
                ext_method = event.get("extinction_method", collapse or "—")
                duration = c.get("duration_years", event.get("duration", ""))
                lessons = c.get("lessons", [])
                process = event.get("process", [])
                label = f"{name} — {ext_method}"
                if decay:
                    label += f" 【{' / '.join(decay)}】"
                with st.expander(label, expanded=False):
                    st.markdown(f"**地域** {region} ・ **期間** {year_range}")
                    if duration:
                        st.caption(f"持続: {duration}")
                    st.markdown(f"**技術水準** {tech}")
                    if tech_level is not None and ethical_dev is not None and sustainability is not None:
                        total = tech_level * 0.30 + ethical_dev * 0.35 + sustainability * 0.35
                        st.caption(f"本システムでの目安: 技術 {tech_level} / 倫理 {ethical_dev} / 自然との調和 {sustainability} → 適合度 {total:.0f}点")
                    st.markdown("**崩壊理由**")
                    st.write(collapse)
                    if decay:
                        st.markdown("**破滅パターン（6分類）** " + " / ".join(decay))
                    if process:
                        st.markdown("**経緯**")
                        for line in process:
                            st.write(f"- {line}")
                    if lessons:
                        st.markdown("**教訓**")
                        for l in lessons:
                            st.write(f"- {l}")

        else:
            _meta_db = _db_lore.get("meta", {})
            st.header(f"📚 {_meta_db.get('title', '文明興亡データベース')}")
            st.caption(f"観測者: {_meta_db.get('observer', 'アルカイア')} / 観測期間: {_meta_db.get('observation_period', '50億年')} / 総サイクル: {_meta_db.get('total_cycles', 5000)}")
            cycles_list = _db_lore["cycles"]
            extinction_patterns = _db_lore.get("extinction_patterns", {})
            if extinction_patterns:
                with st.expander("📊 破滅パターン別件数", expanded=False):
                    for name, data in extinction_patterns.items():
                        if isinstance(data, dict) and "count" in data:
                            exs = data.get("examples", [])
                            st.write(f"**{name}**: {data['count']} 件 — 例: {', '.join(exs[:4])}{'...' if len(exs) > 4 else ''}")
            st.divider()
            for c in cycles_list:
                cycle_num = c.get("cycle", 0)
                civ_name = c.get("civilization_name", "不明")
                year_range = c.get("year_range", "")
                tech = c.get("technology_level", "")
                event = c.get("event") or {}
                ext_method = event.get("extinction_method", "—")
                duration = event.get("duration", "")
                lessons = c.get("lessons", [])
                arcaia_note = c.get("arcaia_note") or c.get("arcaia_conclusion", "")
                with st.expander(f"サイクル {cycle_num}: {civ_name} — {ext_method}", expanded=(cycle_num in (1, 5000))):
                    st.markdown(f"**期間** {year_range} ・ **技術** {tech}")
                    if duration:
                        st.caption(f"持続: {duration}")
                    process = event.get("process", [])
                    if process:
                        st.markdown("**経緯**")
                        for line in process:
                            st.write(f"- {line}")
                    if lessons:
                        st.markdown("**教訓**")
                        for l in lessons:
                            st.write(f"- {l}")
                    if arcaia_note:
                        st.caption(f"*アルカイアの一言: {arcaia_note}*")
