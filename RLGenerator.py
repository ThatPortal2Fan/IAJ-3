"""
RLGenerator.py - RL level generator for the SMB PCG benchmark (Level 2).

Approach: a PCGRL-style *editing agent* ("narrow" representation) trained with
Maskable PPO.
  * An episode = `n_passes` sweeps over a level of `level_width` slices.
  * At each step the agent overwrites the slice under the pointer, then the
    pointer moves one slice to the right (wrapping around).
  * Reward = change in a cheap proxy score, plus a terminal bonus:
        proxy = w_valid * static_quality  -  w_ctl * control_error
    - static_quality mirrors the *non-A* half* of MarioProblem.quality()
      (tubes, empty ratio, row noise, floating enemies), computed exactly and
      fast with numpy, so the reward is dense.
    - control_error compares estimated enemies/coins/jumps with the target.
  * The control target is part of the observation (goal-conditioned policy).
  * The real benchmark (A*) is only used for evaluation, calibration and,
    optionally, as a rare terminal bonus (A* is slow).

Reuses slice tables / helpers from NonRLGenerator.py (same folder).

Usage:
    python RLGenerator.py inspect [--ref-dir my_refs]  # benchmark params + references + smoke test
    python RLGenerator.py train --steps 1000000 --run runs/v1
    python RLGenerator.py plot  --run runs/v1
    python RLGenerator.py generate --run runs/v1 --n 50 --out renders_rl
    python RLGenerator.py generate --run runs/v1 --n 50 --out renders_rl \
                                   --ctl-enemies 10 --ctl-coins 8 --ctl-jumps 20
    python RLGenerator.py eval  --run runs/v1 --n 20    # REAL benchmark metrics (slow: A*)
    python RLGenerator.py calibrate --levels renders    # proxy counts vs A* counts
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pcg_benchmark
from pcg_benchmark.probs.utils import get_range_reward   # same helper MarioProblem.quality uses

try:                                                      # only needed for training
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:                                       # keeps `inspect`/`generate` importable
    BaseCallback = object

# Reuse everything that already works in the non-RL generator.
from NonRLGenerator import (
    SLICES, SLICE_INFO, HEIGHT, EMPTY_ID, NOISE, ENEMY, SOLID,
    KILLABLE_COUNT, MAX_HEIGHT_DIFF, MAX_GAP_RUN,
    _count_jumps, _coin_reachable_in_slice, save_levels,
    pipe_step, can_close,
)

N_SLICES = len(SLICES)

ROOT = Path(__file__).resolve().parent          # project root = folder containing this file
REF_DIR = ROOT / "reflevels"                    # default reference-level folder
REF_SUFFIXES = (".txt", ".lvl")                 # ASCII levels only (ignores stray files)
CTL_KEYS = ("enemies", "coins", "jumps")   # order used in target vectors / observations


# --------------------------------------------------------------------------- #
# Config  (everything that ends up in the report's hyperparameter table)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- benchmark
    bench_name: str = "smb-v0"       # must match the env you evaluate with (see evaluator.py note)
    level_width: Optional[int] = None    # filled from the benchmark by resolve()
    ctl_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # resolve()
    cerror: Dict[str, int] = field(default_factory=dict)                      # resolve()
    empty_thr: float = 0.5           # resolve()
    fenemies_thr: float = 0.1        # resolve()

    # --- representation ("narrow": pointer moves automatically)
    window: int = 8                  # slices visible on each side of the pointer
    n_passes: int = 2                # sweeps over the level per episode
    slice_encoding: str = "tiles+features"

    # --- initial level: a noisy window of a reference level from REF_DIR (reflevels/ by default)
    ref_noise: float = 0.3           # P(slice is replaced by a random one) at episode start
    #   1 = pure noise. The path is selected by the --ref-dir CLI flag.

    # --- reward
    w_valid: float = 1.0             # weight of static_quality in [0,1]
    w_ctl: float = 1.0               # weight of control error
    ctl_gated: bool = True           # scale control term by static_quality (benchmark gives
    #                                  controlability 0 to statically-invalid levels)
    noise_gated: bool = False        # benchmark only counts noise once empty&tube are fine;
    #                                  False = always count it (denser signal for training)
    ctl_tol: float = 0.05            # terminal bonus if control_error <= this (and valid)
    terminal_bonus: float = 5.0
    reward_scale: float = 10.0       # multiplies the shaping reward (deltas are tiny)
    w_div: float = 0.0               # terminal bonus * mean Hamming distance to recent levels
    div_buffer: int = 32
    bench_terminal_prob: float = 0.0 # P(episode ends with a real A* evaluation). A* is SLOW.
    w_bench: float = 5.0

    # --- action masking
    mask_floating_enemies: bool = True
    mask_height_diff: bool = True
    mask_gap_run: bool = True

    # --- PPO
    total_steps: int = 1_000_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    net_arch: Tuple[int, ...] = (256, 256)
    seed: int = 0

    # --- periodic *real* benchmark evaluation during training
    eval_every: int = 200_000
    eval_n: int = 10

    # --- generation
    gen_attempts: int = 1            # best-of-k rollouts per target (k=1: plain policy)

    def resolve(self, bench: Optional["BenchmarkAdapter"] = None) -> "Config":
        """Fill width / control ranges / tolerances / thresholds from the benchmark."""
        bench = bench or BenchmarkAdapter(self.bench_name)
        if self.level_width is None:
            self.level_width = bench.width
        if not self.ctl_ranges:
            self.ctl_ranges = {k: (0.0, float(bench.ctl_max[k])) for k in CTL_KEYS}
        if not self.cerror:
            self.cerror = {k: int(bench.cerror[k]) for k in CTL_KEYS}
        self.empty_thr, self.fenemies_thr = bench.empty_thr, bench.fenemies_thr
        return self

    @property
    def widths(self) -> np.ndarray:
        return np.array([self.ctl_ranges[k][1] - self.ctl_ranges[k][0] for k in CTL_KEYS],
                        dtype=np.float32)

    @property
    def tols(self) -> np.ndarray:
        return np.array([self.cerror[k] for k in CTL_KEYS], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Pre-computed slice tables (all numpy, so every step is fast)
# --------------------------------------------------------------------------- #
CHARS = sorted(set("".join(SLICES)))
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
SLICE_CHARS = np.array([[CHAR_IDX[c] for c in s] for s in SLICES], dtype=np.int16)   # (N, H)
IS_PIPE = np.array([[c == "t" for c in s] for s in SLICES], dtype=np.int8)           # (N, H)

ENEMY_TOTAL = np.zeros(N_SLICES)
ENEMY_FLOAT = np.zeros(N_SLICES)
for _sid, _s in enumerate(SLICES):
    for _y, _c in enumerate(_s):
        if _c in ENEMY:
            ENEMY_TOTAL[_sid] += 1
            if _y + 1 >= HEIGHT or _s[_y + 1] not in SOLID:
                ENEMY_FLOAT[_sid] += 1

KILL = np.array([KILLABLE_COUNT[s] for s in range(N_SLICES)])
FLOATING = np.array([SLICE_INFO[s].floor_y is None or SLICE_INFO[s].floor_y < HEIGHT - 2
                     for s in range(N_SLICES)])
FLOATING_ENEMY = np.array([SLICE_INFO[s].has_enemy and not SLICE_INFO[s].enemy_grounded
                           for s in range(N_SLICES)])
TOP_Y = np.array([-1 if SLICE_INFO[s].top_solid_y is None else SLICE_INFO[s].top_solid_y
                  for s in range(N_SLICES)])
JUMPS = np.array([[_count_jumps(a, b) for b in range(N_SLICES)] for a in range(N_SLICES)],
                 dtype=np.int16)
REACH = np.array([[_coin_reachable_in_slice(SLICES[s], y) for y in range(HEIGHT)]
                  for s in range(N_SLICES)], dtype=bool)
COIN_ROWS = np.array([[c == "o" for c in s] for s in SLICES], dtype=bool)


def _features_matrix() -> np.ndarray:
    """9 hand-made numbers per slice (uses SLICE_INFO)."""
    rows = []
    for sid in range(N_SLICES):
        i = SLICE_INFO[sid]
        rows.append([
            0.0 if i.floor_y is None else (i.floor_y + 1) / HEIGHT,
            0.0 if i.top_solid_y is None else (i.top_solid_y + 1) / HEIGHT,
            float(i.floor_y is None), float(i.has_pipe), float(i.has_enemy),
            float(i.has_enemy and not i.enemy_grounded),
            min(KILL[sid], 3) / 3.0, min(int(COIN_ROWS[sid].sum()), 3) / 3.0,
            float(sid == EMPTY_ID),
        ])
    return np.asarray(rows, dtype=np.float32)


def _tiles_matrix() -> np.ndarray:
    """One-hot tile type for every row of the slice: (N, H * n_chars)."""
    out = np.zeros((N_SLICES, HEIGHT, len(CHARS)), dtype=np.float32)
    for sid in range(N_SLICES):
        out[sid, np.arange(HEIGHT), SLICE_CHARS[sid]] = 1.0
    return out.reshape(N_SLICES, -1)


@lru_cache(maxsize=None)
def build_slice_encoding(spec: str) -> np.ndarray:
    """(N_SLICES, D) matrix; row i is how slice i is shown to the policy."""
    mats = []
    for part in [p for p in spec.split("+") if p]:
        if part == "features":
            mats.append(_features_matrix())
        elif part == "onehot":
            mats.append(np.eye(N_SLICES, dtype=np.float32))
        elif part == "tiles":
            mats.append(_tiles_matrix())
        else:
            raise ValueError(f"unknown slice encoding part: {part!r}")
    return np.concatenate(mats, axis=1)


# --------------------------------------------------------------------------- #
# Proxy score: static quality + estimated control counts
# --------------------------------------------------------------------------- #
def static_terms(level, cfg: Config) -> Dict:
    """Non-A* part of MarioProblem.info()/quality(), vectorised.

    Same quantities and the same get_range_reward calls as problem.py, so
    `valid` means "the benchmark would run A* on this level".
    """
    lv = np.asarray(level, dtype=np.int64)
    n = len(lv)

    # tubes: number of odd-length horizontal runs of 't' (per row)
    pad = np.zeros((HEIGHT, n + 2), dtype=np.int8)
    pad[:, 1:-1] = IS_PIPE[lv].T
    d = np.diff(pad, axis=1)
    _, sx = np.nonzero(d == 1)
    _, ex = np.nonzero(d == -1)
    tube = int(((ex - sx) % 2).sum())

    # row noise above the allowed amount (noise.txt)
    G = SLICE_CHARS[lv]
    changes = (G[1:] != G[:-1]).sum(axis=0) / max(1, n - 1)
    allowed = np.where(NOISE > 0, np.maximum(NOISE, 2.0 / max(1, n - 1)), 0.0)
    excess = np.clip(changes - allowed, 0.0, None)

    empty = float((lv == EMPTY_ID).mean())
    total_en = ENEMY_TOTAL[lv].sum()
    fenemies = float(ENEMY_FLOAT[lv].sum() / total_en) if total_en > 0 else 0.0

    tube_r = get_range_reward(tube, 0, 0, 0, 10)
    empty_r = get_range_reward(empty, 0, cfg.empty_thr, 1)
    fenem_r = get_range_reward(fenemies, 0, 0, cfg.fenemies_thr, 1)
    noise_r = 0.0
    if (not cfg.noise_gated) or (empty_r >= 1 and tube_r >= 1):
        for e in excess:
            noise_r += get_range_reward(float(e), 0, 0, 0, 1)
    noise_r /= HEIGHT                                   # benchmark divides by 16

    valid = bool(tube == 0 and empty > cfg.empty_thr and excess.sum() <= 1e-9
                 and fenemies < cfg.fenemies_thr)
    return {
        "valid": valid,
        "static_q": float((tube_r + empty_r + noise_r + fenem_r) / 4.0),   # in [0, 1]
        "tube": tube, "empty": empty, "fenemies": fenemies,
        "noise_excess": float(excess.sum()),
    }


def estimate_counts(level) -> np.ndarray:
    """Estimated [enemies, coins, jumps] for a whole level (vectorised).

    Same estimators as NonRLGenerator (killable enemies, reachable coin tiles, jump
    opportunities) but computed for the full level at once. They only approximate
    what the benchmark's A* agent really does -> use `calibrate` to measure the gap.
    """
    lv = np.asarray(level, dtype=np.int64)
    n = len(lv)
    enemies = KILL[lv].sum()
    jumps = JUMPS[EMPTY_ID, lv[0]] + JUMPS[lv[:-1], lv[1:]].sum()

    R = REACH[lv]                                       # coin row reachable from own slice
    Rp = np.concatenate([np.repeat(REACH[EMPTY_ID][None, :], MAX_GAP_RUN, axis=0), R], axis=0)
    near = np.zeros_like(R)
    for k in range(1, MAX_GAP_RUN + 1):                 # ... or from the previous 4 slices
        near |= Rp[MAX_GAP_RUN - k: MAX_GAP_RUN - k + n]
    coins = (COIN_ROWS[lv] & (R | near)).sum()
    return np.array([enemies, coins, jumps], dtype=np.float32)


def control_error(counts: np.ndarray, targets: np.ndarray, cfg: Config) -> float:
    """Mean range-normalised distance, with a dead zone equal to the benchmark's
    tolerance (inside +-cerror the benchmark gives full controllability)."""
    dist = np.maximum(0.0, np.abs(counts - targets) - cfg.tols)
    return float(np.mean(dist / cfg.widths))


def targets_to_control(t) -> Dict[str, int]:
    return {k: int(round(float(t[i]))) for i, k in enumerate(CTL_KEYS)}


# --------------------------------------------------------------------------- #
# Benchmark adapter (the *real* quality / diversity / controllability)
# --------------------------------------------------------------------------- #
def _find_problem(env):
    for attr in ("_problem", "problem", "_game", "game"):    # same trick as in_depth_evaluator.py
        p = getattr(env, attr, None)
        if p is not None:
            return p
    return None


class BenchmarkAdapter:
    """Thin wrapper so the rest of the code never touches pcg_benchmark directly.

    Verified against your evaluator scripts:  env.evaluate(levels, controls) ->
    (quality, diversity, controlability, details, infos).  Per-level numbers come from
    the MarioProblem methods (problem.py): info(), quality(info), controlability(info, c).
    """

    def __init__(self, name: str = "smb-v0"):
        self.name = name
        self.env = pcg_benchmark.make(name)
        self.problem = _find_problem(self.env)
        p = self.problem
        self.width = int(getattr(p, "_width", 150))
        self.empty_thr = float(getattr(p, "_empty", 0.5))
        self.fenemies_thr = float(getattr(p, "_fenemies", 0.1))
        # control ranges / tolerance exactly as in problem.py (fallback: same formulas)
        self.ctl_max = {"enemies": max(2, int(self.width / 7)),
                        "jumps": max(2, int(self.width / 5)),
                        "coins": max(2, int(self.width / 10))}
        self.cerror = {k: int(0.1 * v) for k, v in self.ctl_max.items()}
        try:
            space = p._control_space._value
            for k in CTL_KEYS:
                self.ctl_max[k] = int(space[k]._max_value)
            if getattr(p, "_cerror", None):
                self.cerror = {k: int(p._cerror[k]) for k in CTL_KEYS}
        except Exception:
            pass    # keep the formula values

    def describe(self):
        print(f"benchmark: {self.name}   problem found: {self.problem is not None}")
        print(f"width={self.width}  empty>{self.empty_thr}  fenemies<{self.fenemies_thr}")
        print(f"control max: {self.ctl_max}   tolerance (cerror): {self.cerror}")
        print("content_space:", self.env.content_space)
        print("control_space:", self.env.control_space)
        try:
            import pcg_benchmark.probs.smb as smb_pkg
            print("files in", Path(smb_pkg.__file__).parent, ":",
                  sorted(p.name for p in Path(smb_pkg.__file__).parent.iterdir()))
        except Exception:
            pass

    def evaluate(self, levels: List[List[int]], controls: List[Dict[str, int]]):
        levels = [[int(x) for x in lv] for lv in levels]
        return self.env.evaluate(levels, controls)    # q, d, c, details, infos

    def score_level(self, level, control: Dict[str, int]) -> Tuple[float, float]:
        """Real (A*-based) quality and controllability of ONE level. Slow."""
        info = self.problem.info(np.asarray(level, dtype=np.int64))
        return float(self.problem.quality(info)), float(self.problem.controlability(info, control))

    def render(self, levels: List[List[int]]):
        return self.env.render(levels)


# --------------------------------------------------------------------------- #
# Reference levels (plain files in REF_DIR)
# --------------------------------------------------------------------------- #
_SLICE_LOOKUP = {s: i for i, s in enumerate(SLICES)}

def _ascii_to_ids(text: str) -> Tuple[List[int], int]:
    """ASCII level (HEIGHT rows) -> slice ids (exact matches only)."""
    rows = [r.rstrip() for r in text.splitlines() if r.strip()]
    if len(rows) != HEIGHT:
        raise ValueError(f"expected {HEIGHT} rows, got {len(rows)}")
    
    w = max(len(r) for r in rows)
    rows = [r.ljust(w, r[-1]) for r in rows]
    lo, hi = 0, w
    
    # Strip benchmark start/end padding if present
    if w > 6 and rows[HEIGHT - 3][1] == "M" and rows[HEIGHT - 3][w - 2] == "F":
        lo, hi = 3, w - 3
        
    ids = []
    for x in range(lo, hi):
        col = "".join(rows[y][x] for y in range(HEIGHT))
        if col in _SLICE_LOOKUP:
            ids.append(_SLICE_LOOKUP[col])
        else:
            raise ValueError(f"Unknown slice column encountered: {col}")
    return ids, 0


def load_reference_levels(path: str, suffixes: Optional[Tuple[str, ...]] = None,
                          names: Optional[List[str]] = None) -> List[List[int]]:
    """Load levels (slice-id lists) from a file or a directory.

    Supported: .txt/.lvl (ASCII), .json (list of ints, or list of lists), .npy.
    `suffixes` restricts which files are read; if `names` is given, the source file name of
    every loaded level is appended to it.
    """
    p = Path(path)
    files = sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]
    if suffixes:
        files = [f for f in files if f.suffix.lower() in suffixes]
    levels: List[List[int]] = []

    def add(lv_list, fname):
        levels.extend(lv_list)
        if names is not None:
            names.extend([fname] * len(lv_list))

    for f in files:
        suf = f.suffix.lower()
        try:
            if suf == ".json":
                data = json.loads(f.read_text())
                if isinstance(data, dict):
                    data = data.get("levels", data.get("content"))
                if data and isinstance(data[0], (int, float)):
                    data = [data]
                if not (data and isinstance(data[0], list)):
                    continue                           # e.g. targets.json
                add([[int(x) for x in lv] for lv in data], f.name)
            elif suf == ".npy":
                arr = np.load(f)
                arr = arr[None] if arr.ndim == 1 else arr
                add([[int(x) for x in lv] for lv in arr], f.name)
            elif suf in (".txt", ".lvl"):
                ids, approx = _ascii_to_ids(f.read_text())
                if approx:
                    print(f"  {f.name}: {approx}/{len(ids)} columns not in slices.txt -> nearest slice")
                add([ids], f.name)
        except Exception as e:                         # skip files we cannot parse
            print(f"  skipping {f.name}: {e}")
    return levels


_REF_CACHE: Optional[Tuple[List[List[int]], List[str]]] = None


def set_reference_dir(path: str) -> None:
    """Set the reference-level folder and clear the loaded-level cache."""
    global REF_DIR, _REF_CACHE
    REF_DIR = Path(path).expanduser().resolve()
    _REF_CACHE = None


def get_reference_levels(refresh: bool = False) -> List[List[int]]:
    """The reference levels from REF_DIR (loaded once per process, then cached)."""
    return _load_refs(refresh)[0]


def get_reference_names() -> List[str]:
    return _load_refs(False)[1]


def _load_refs(refresh: bool) -> Tuple[List[List[int]], List[str]]:
    global _REF_CACHE
    if _REF_CACHE is None or refresh:
        if not REF_DIR.is_dir():
            raise FileNotFoundError(
                f"Reference level folder not found: {REF_DIR}\n"
                f"Create it next to RLGenerator.py and put the ASCII level files (.txt) in it.")
        names: List[str] = []
        levels = load_reference_levels(str(REF_DIR), suffixes=REF_SUFFIXES, names=names)
        keep = [i for i, lv in enumerate(levels) if len(lv) >= 2]
        levels, names = [levels[i] for i in keep], [names[i] for i in keep]
        if not levels:
            raise FileNotFoundError(
                f"No usable reference levels ({'/'.join(REF_SUFFIXES)}, {HEIGHT} rows each) in {REF_DIR}")
        print(f"Loaded {len(levels)} reference level(s) from {REF_DIR}")
        _REF_CACHE = (levels, names)
    return _REF_CACHE


def _fit_to_width(level: List[int], width: int, rng) -> List[int]:
    if len(level) > width:
        s = int(rng.integers(0, len(level) - width + 1))
        return level[s:s + width]
    return level + [EMPTY_ID] * (width - len(level))


def describe_references(cfg: Config):
    """Print what the reference levels look like to the static proxy (used by `inspect`)."""
    refs, names = get_reference_levels(), get_reference_names()
    rng = np.random.default_rng(0)
    print(f"reference levels in {REF_DIR}  (level_width={cfg.level_width}):")
    for lv, nm in zip(refs, names):
        win = _fit_to_width(lv, cfg.level_width, rng)
        st = static_terms(win, cfg)
        note = "" if len(lv) >= cfg.level_width else f"  (shorter than width: padded with {cfg.level_width - len(lv)} empty slices)"
        print(f"  {nm:24s} {len(lv):4d} slices  static_q={st['static_q']:.3f} valid={st['valid']} "
              f"empty={st['empty']:.2f} tube={st['tube']} fenemies={st['fenemies']:.2f}{note}")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class SMBEditEnv(gym.Env):
    """Pointer sweeps over the level; the action is the slice to place there ("narrow")."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, bench: Optional[BenchmarkAdapter] = None,
                 reference_levels: Optional[List[List[int]]] = None):
        super().__init__()
        if cfg.level_width is None or not cfg.ctl_ranges:
            raise ValueError("Config not resolved: call cfg.resolve() first")
        self.cfg = cfg
        self.bench = bench
        self.width = cfg.level_width
        self.max_steps = cfg.n_passes * self.width
        self.enc = build_slice_encoding(cfg.slice_encoding)

        self.refs = reference_levels if reference_levels else get_reference_levels()

        D = self.enc.shape[1]
        self.action_space = spaces.Discrete(N_SLICES)
        # window + (target, counts, gap) + (progress, position) + 5 global static terms
        self.obs_dim = (2 * cfg.window + 1) * (D + 1) + 9 + 2 + 5
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

        self.level = np.zeros(self.width, dtype=np.int64)
        self.targets = np.zeros(3, dtype=np.float32)
        self.counts = np.zeros(3, dtype=np.float32)
        self.parts: Dict = {}
        self.pos = self.t = 0
        self.prev_score = 0.0
        self.recent: deque = deque(maxlen=cfg.div_buffer)

    # ---- helpers ---------------------------------------------------------- #
    def _init_level(self) -> np.ndarray:
        """Random reference level -> random window of `width` slices -> random-slice noise."""
        rng = self.np_random
        ref = self.refs[int(rng.integers(0, len(self.refs)))]
        lv = np.array(_fit_to_width(list(ref), self.width, rng), dtype=np.int64)
        noisy = rng.random(self.width) < self.cfg.ref_noise
        lv[noisy] = rng.integers(0, N_SLICES, int(noisy.sum()))
        return lv

    def _sample_targets(self) -> np.ndarray:
        return np.array([self.np_random.integers(int(lo), int(hi) + 1)
                         for lo, hi in (self.cfg.ctl_ranges[k] for k in CTL_KEYS)],
                        dtype=np.float32)

    def _score(self, level) -> Tuple[float, Dict]:
        cfg = self.cfg
        parts = static_terms(level, cfg)
        self.counts = estimate_counts(level)
        parts["ctl_err"] = control_error(self.counts, self.targets, cfg)
        gate = parts["static_q"] if cfg.ctl_gated else 1.0
        score = cfg.w_valid * parts["static_q"] - cfg.w_ctl * gate * parts["ctl_err"]
        self.parts = parts
        return score, parts

    def _terminal_reward(self, parts: Dict) -> float:
        cfg, bonus = self.cfg, 0.0
        if parts["valid"] and parts["ctl_err"] <= cfg.ctl_tol:
            bonus += cfg.terminal_bonus
        if cfg.w_div > 0 and self.recent:
            bonus += cfg.w_div * float(np.mean([np.mean(self.level != r) for r in self.recent]))
        self.recent.append(self.level.copy())
        if (self.bench is not None and cfg.bench_terminal_prob > 0 and parts["valid"]
                and self.np_random.random() < cfg.bench_terminal_prob):
            q, c = self.bench.score_level(self.level, targets_to_control(self.targets))
            bonus += cfg.w_bench * (q + c) / 2.0
        return bonus

    def _get_obs(self) -> np.ndarray:
        W, D = self.cfg.window, self.enc.shape[1]
        idx = np.arange(self.pos - W, self.pos + W + 1)
        inside = (idx >= 0) & (idx < self.width)
        win = np.zeros((2 * W + 1, D + 1), dtype=np.float32)
        win[inside, :D] = self.enc[self.level[idx[inside]]]
        win[inside, D] = 1.0                                   # "inside the level" flag

        w = self.cfg.widths
        p = self.parts
        glob = np.array([p["static_q"], p["empty"], min(p["tube"], 10) / 10.0,
                         min(p["fenemies"] / max(self.cfg.fenemies_thr, 1e-6), 3.0) / 3.0,
                         min(p["noise_excess"], 1.0)], dtype=np.float32)
        misc = np.array([self.t / self.max_steps, self.pos / self.width], dtype=np.float32)
        return np.concatenate([win.ravel(), self.targets / w, self.counts / w,
                               (self.targets - self.counts) / w, misc, glob]).astype(np.float32)

    # ---- gym API ---------------------------------------------------------- #
    def action_masks(self) -> np.ndarray:
        """Boolean mask of allowed slices at the current pointer (for MaskablePPO)."""
        cfg, mask = self.cfg, np.ones(N_SLICES, dtype=bool)
        
        if cfg.mask_floating_enemies:
            mask &= ~FLOATING_ENEMY
        # Look left
        left_enemy = (self.pos > 0) and (ENEMY_TOTAL[self.level[self.pos - 1]] > 0)
        # Look right (only on pass 2+ when the right slice is actually locked in)
        right_enemy = (self.t >= self.width and self.pos < self.width - 1) and (ENEMY_TOTAL[self.level[self.pos + 1]] > 0)
        
        if left_enemy or right_enemy:
            mask &= ~(ENEMY_TOTAL > 0)
        
        if cfg.mask_height_diff:
            last_top = -1
            for p in range(self.pos - 1, -1, -1):
                if TOP_Y[self.level[p]] >= 0:
                    last_top = TOP_Y[self.level[p]]
                    break
                    
            if last_top < 0:
                last_top = TOP_Y[EMPTY_ID]
                
            if last_top >= 0:
                mask &= ~((TOP_Y >= 0) & (np.abs(TOP_Y - last_top) > MAX_HEIGHT_DIFF))
        
        if cfg.mask_gap_run:
            run_left = 0
            for p in range(self.pos - 1, -1, -1):
                if not FLOATING[self.level[p]]: break
                run_left += 1
            run_right = 0
            if self.t >= self.width:
                for p in range(self.pos + 1, self.width):
                    if not FLOATING[self.level[p]]: break
                    run_right += 1
            if run_left + 1 + run_right > MAX_GAP_RUN:
                mask &= ~FLOATING

        in_pipe = np.zeros(HEIGHT, dtype=bool)
        just_closed = False
        
        for p in range(self.pos):
            was_open = bool(in_pipe.any())
            ok, next_pipe = pipe_step(in_pipe, self.level[p])
            
            if not ok:
                # The starting level had a broken pipe from random noise.
                # Reset the tracker so it doesn't crash the env.
                in_pipe = np.zeros(HEIGHT, dtype=bool)
                just_closed = False
            else:
                in_pipe = next_pipe
                now_closed = not in_pipe.any()
                just_closed = was_open and now_closed and SLICE_INFO[self.level[p]].has_pipe

        # Check if we are about to bump into a pipe on the right (during pass 2)
        pipe_ahead = (self.t >= self.width and self.pos < self.width - 1 
                      and SLICE_INFO[self.level[self.pos + 1]].has_pipe)

        # Mask invalid pipe actions
        for sid in range(N_SLICES):
            if not mask[sid]: continue
            
            is_pipe_slice = SLICE_INFO[sid].has_pipe
            if is_pipe_slice:
                # Prevent adjacent pipes: don't start one if one just closed, 
                # or if we are outside a pipe and about to crash into one.
                if just_closed or (not in_pipe.any() and pipe_ahead):
                    mask[sid] = False
                    continue
                
            ok, nxt = pipe_step(in_pipe, sid)
            if not ok:
                mask[sid] = False
            elif self.pos >= self.width - 4:
                if nxt.any(): mask[sid] = False
            elif not can_close(nxt):
                mask[sid] = False
                
        required_empty = int(np.floor(cfg.empty_thr * self.width)) + 1
        
        # Only enforce the hard mask during the final editing pass
        if self.t >= self.width:
            # Slices behind the pointer are permanently locked in for this episode
            locked_empty = (self.level[:self.pos] == EMPTY_ID).sum()
            remaining_to_lock = self.width - self.pos
            
            # Mathematical panic (Evaluated per slice option)
            for sid in range(N_SLICES):
                if not mask[sid]: continue
                
                is_empty = (sid == EMPTY_ID)
                ok, nxt = pipe_step(in_pipe, sid)
                in_pipe_after = bool(nxt.any()) if ok else False
                
                max_future = (remaining_to_lock - 1) - (1 if in_pipe_after else 0)
                
                if locked_empty + int(is_empty) + max_future < required_empty:
                    mask[sid] = False
                    
            # Anti-desert during final pass
            # We add a 0.05 buffer so the agent doesn't ride the absolute minimum threshold.
            pacing_thr = cfg.empty_thr + 0.05
            if self.pos > 5 and not in_pipe.any():
                if locked_empty / self.pos < pacing_thr:
                    if mask[EMPTY_ID]:
                        mask[:] = False
                        mask[EMPTY_ID] = True

        # Calculate the absolute allowed changes per row based on width and noise.txt
        allowed = np.where(NOISE > 0, np.maximum(NOISE, 2.0 / max(1, self.width - 1)), 0.0)
        ceiling = np.floor(allowed * (self.width - 1) + 1e-9).astype(np.int32)
        
        G = SLICE_CHARS[self.level]
        base_changes = (G[1:] != G[:-1]).sum(axis=0)
        
        # Subtract the current transitions involving the pointer's slice
        if self.pos > 0:
            base_changes -= (G[self.pos] != G[self.pos - 1])
        if self.pos < self.width - 1:
            base_changes -= (G[self.pos + 1] != G[self.pos])
            
        # Calculate what the new transitions would be for every possible slice
        new_trans = np.zeros((N_SLICES, HEIGHT), dtype=np.int32)
        if self.pos > 0:
            new_trans += (SLICE_CHARS != G[self.pos - 1])
        if self.pos < self.width - 1:
            new_trans += (SLICE_CHARS != G[self.pos + 1])
            
        total_changes = base_changes + new_trans
        
        # If the level is already too noisy to fix in a single step (due to the random 
        # initialization), the effective ceiling forces the agent to minimize noise.
        effective_ceiling = np.maximum(ceiling, base_changes + new_trans.min(axis=0))
        
        # Mask out any slice that pushes the noise above the effective ceiling for ANY row
        mask &= ~(total_changes > effective_ceiling).any(axis=1)

        if not mask.any():
            mask[EMPTY_ID] = True
            
        return mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        self.level = self._init_level()
        tg = options.get("targets")
        self.targets = np.asarray(tg, dtype=np.float32) if tg is not None else self._sample_targets()
        self.pos = self.t = 0
        self.prev_score, _ = self._score(self.level)
        return self._get_obs(), {}

    def step(self, action):
        self.level[self.pos] = int(action)
        score, parts = self._score(self.level)
        reward = (score - self.prev_score) * self.cfg.reward_scale    # potential-based shaping
        self.prev_score = score

        self.pos = (self.pos + 1) % self.width
        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps

        info: Dict = {}
        if truncated:
            reward += self._terminal_reward(parts)
            info.update(valid=float(parts["valid"]), ctl_err=parts["ctl_err"],
                        static_q=parts["static_q"])                   # read by Monitor
        return self._get_obs(), float(reward), terminated, truncated, info


# --------------------------------------------------------------------------- #
# Generation + evaluation
# --------------------------------------------------------------------------- #
def _sample_targets(cfg: Config, n: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.array([rng.integers(int(cfg.ctl_ranges[k][0]), int(cfg.ctl_ranges[k][1]) + 1)
                      for k in CTL_KEYS], dtype=np.float32) for _ in range(n)]


def generate_levels(model, cfg: Config, targets_list: List[np.ndarray],
                    deterministic: bool = False, seed: int = 0) -> List[List[int]]:
    """Run the trained policy once (or best-of-k) per target.
    deterministic=False samples from the policy, which keeps diversity."""
    env = SMBEditEnv(cfg)
    levels = []
    for i, tg in enumerate(targets_list):
        best, best_score = None, -np.inf
        for a in range(max(1, cfg.gen_attempts)):
            obs, _ = env.reset(seed=seed + 1000 * a + i, options={"targets": tg})
            done = False
            while not done:
                action, _ = model.predict(obs, action_masks=env.action_masks(),
                                          deterministic=deterministic)
                obs, _, term, trunc, _ = env.step(int(action))
                done = term or trunc
            if env.prev_score > best_score:
                best, best_score = [int(x) for x in env.level], env.prev_score
        levels.append(best)
    return levels


def evaluate_local(levels: List[List[int]], targets: List[np.ndarray], cfg: Config) -> Dict[str, float]:
    """Cheap sanity metrics that don't need the benchmark's A* (NOT the benchmark metrics)."""
    st = [static_terms(l, cfg) for l in levels]
    arr = np.array(levels)
    n = len(arr)
    ham = 0.0
    if n > 1:
        ham = float(np.mean([np.mean(arr[i] != arr[j])
                             for i in range(n) for j in range(i + 1, n)]))
    return {
        "static_valid_rate": float(np.mean([s["valid"] for s in st])),
        "mean_static_q": float(np.mean([s["static_q"] for s in st])),
        "ctl_err_proxy": float(np.mean([control_error(estimate_counts(l), t, cfg)
                                        for l, t in zip(levels, targets)])),
        "hamming_diversity": ham,   # the benchmark's diversity is path-based, not this
    }


def evaluate_benchmark(levels: List[List[int]], targets: List[np.ndarray],
                       bench: BenchmarkAdapter) -> Dict[str, float]:
    """Level 3: the three benchmark dimensions, with each level's own control target.

    The first three numbers are the benchmark's official aggregates (env.evaluate); the
    per-level means come from MarioProblem.quality / controlability directly.
    """
    controls = [targets_to_control(t) for t in targets]
    q, d, c, _details, infos = bench.evaluate(levels, controls)
    out = {"quality": float(q), "diversity": float(d), "controlability": float(c)}
    if bench.problem is not None:
        out["mean_level_quality"] = float(np.mean([bench.problem.quality(i) for i in infos]))
        out["mean_level_controlability"] = float(np.mean(
            [bench.problem.controlability(i, ct) for i, ct in zip(infos, controls)]))
    out["astar_completed_rate"] = float(np.mean([i["complete"] >= 1.0 for i in infos]))
    return out


def calibrate(bench: BenchmarkAdapter, cfg: Config, levels_dir: Path):
    """Compare proxy counts with the counts A* really achieves (on statically-valid levels)."""
    levels = load_reference_levels(str(levels_dir), suffixes=(".json",))
    levels = [lv for lv in levels if len(lv) == cfg.level_width]
    if not levels:
        print(f"No levels of width {cfg.level_width} found in {levels_dir}")
        return
    controls = [{k: 0 for k in CTL_KEYS}] * len(levels)
    _, _, _, _, infos = bench.evaluate(levels, controls)
    rows = []
    for lv, info in zip(levels, infos):
        if info["complete"] <= 0:          # statically invalid: A* never ran
            continue
        proxy = estimate_counts(lv)
        real = np.array([info["enemies"], info["coins"], info["jumps"]], dtype=float)
        rows.append((proxy, real, info["complete"]))
        print(f"proxy e/c/j = {proxy.astype(int).tolist()}  A* e/c/j = {real.astype(int).tolist()}"
              f"  complete={info['complete']:.2f}")
    if not rows:
        print("No statically valid levels -> nothing to calibrate.")
        return
    proxy = np.array([r[0] for r in rows]); real = np.array([r[1] for r in rows])
    print(f"\n{len(rows)}/{len(levels)} levels were statically valid")
    for i, k in enumerate(CTL_KEYS):
        ratio = real[:, i].sum() / max(1.0, proxy[:, i].sum())
        corr = np.corrcoef(proxy[:, i], real[:, i])[0, 1] if len(rows) > 2 else float("nan")
        print(f"  {k:8s} real/proxy = {ratio:.2f}   correlation = {corr:.2f}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class BenchEvalCallback(BaseCallback):
    """Every `every` steps: generate a few levels and score them with the REAL benchmark.
    These are the 'fitness evolution' points for the report (written to eval_log.csv)."""

    def __init__(self, cfg: Config, run_dir: Path):
        super().__init__()
        self.cfg, self.run_dir = cfg, run_dir
        self.next_eval = cfg.eval_every
        self.bench: Optional[BenchmarkAdapter] = None

    def _on_step(self) -> bool:
        if self.cfg.eval_every > 0 and self.num_timesteps >= self.next_eval:
            self.next_eval += self.cfg.eval_every
            self._run_eval()
        return True

    def _run_eval(self):
        if self.bench is None:
            self.bench = BenchmarkAdapter(self.cfg.bench_name)
        targets = _sample_targets(self.cfg, self.cfg.eval_n, seed=self.num_timesteps)
        levels = generate_levels(self.model, self.cfg, targets, seed=self.num_timesteps)
        res = evaluate_benchmark(levels, targets, self.bench)
        res.update(evaluate_local(levels, targets, self.cfg))
        for k, v in res.items():
            self.logger.record(f"bench/{k}", v)
        path = self.run_dir / "eval_log.csv"
        new = not path.exists()
        with path.open("a") as f:
            if new:
                f.write("timesteps," + ",".join(res) + "\n")
            f.write(f"{self.num_timesteps}," + ",".join(f"{v:.5f}" for v in res.values()) + "\n")
        print(f"[eval @ {self.num_timesteps}] " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))


def make_env_fn(cfg: Config, rank: int, run_dir: Path):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        bench = BenchmarkAdapter(cfg.bench_name) if cfg.bench_terminal_prob > 0 else None
        env = SMBEditEnv(cfg, bench)
        env = Monitor(env, str(run_dir / f"monitor_{rank}"),
                      info_keywords=("valid", "ctl_err", "static_q"))
        env.reset(seed=cfg.seed + rank)
        return env
    return _init


def train(cfg: Config, run_dir: Path):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))   # for the report

    vec = SubprocVecEnv([make_env_fn(cfg, i, run_dir) for i in range(cfg.n_envs)])
    model = MaskablePPO(
        "MlpPolicy", vec,
        learning_rate=cfg.learning_rate, n_steps=cfg.n_steps, batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range, ent_coef=cfg.ent_coef,
        policy_kwargs=dict(net_arch=list(cfg.net_arch)),
        seed=cfg.seed, verbose=1, tensorboard_log=str(run_dir / "tb"),
    )
    callbacks = [CheckpointCallback(save_freq=max(1, 100_000 // cfg.n_envs),
                                    save_path=str(run_dir / "checkpoints"))]
    if cfg.eval_every > 0:
        callbacks.append(BenchEvalCallback(cfg, run_dir))
    model.learn(total_timesteps=cfg.total_steps, callback=CallbackList(callbacks))
    model.save(str(run_dir / "final_model"))   # submit this ("trained brain")
    print(f"Saved model to {run_dir/'final_model'}.zip")


def plot_training(run_dir: Path, window: int = 100):
    """Training curves (Monitor CSVs) + real-benchmark points (eval_log.csv)."""
    import pandas as pd
    import matplotlib.pyplot as plt

    frames = [pd.read_csv(p, skiprows=1) for p in sorted(run_dir.glob("monitor_*.csv"))]
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()                  # episode lengths summed over all envs
    cols = [("r", "episode reward"), ("static_q", "static quality (proxy)"),
            ("valid", "fraction statically valid"), ("ctl_err", "control error (proxy)")]
    ev_path = run_dir / "eval_log.csv"
    ev = pd.read_csv(ev_path) if ev_path.exists() else None
    ncols = 4 + (1 if ev is not None else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4))
    for ax, (col, title) in zip(axes, cols):
        ax.plot(df["steps"], df[col].rolling(window, min_periods=1).mean())
        ax.set_title(title); ax.set_xlabel("env steps")
    if ev is not None:
        for col in ["quality", "diversity", "controlability"]:
            axes[-1].plot(ev["timesteps"], ev[col], marker="o", label=col)
        axes[-1].set_title("real benchmark (periodic eval)"); axes[-1].set_xlabel("env steps")
        axes[-1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=150)
    print(f"Wrote {run_dir/'training_curves.png'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_cfg(run_dir: Path) -> Config:
    d = json.loads((run_dir / "config.json").read_text())
    d = {k: v for k, v in d.items() if k in Config.__dataclass_fields__}   # ignore removed fields
    d["net_arch"] = tuple(d["net_arch"])
    d["ctl_ranges"] = {k: tuple(v) for k, v in d["ctl_ranges"].items()}
    return Config(**d)


def smoke_test(cfg: Config):
    """Random-policy episode from the reference levels: checks the env runs, shows reward scale."""
    env = SMBEditEnv(cfg)
    print(f"slices={N_SLICES} encoding={cfg.slice_encoding!r} obs_dim={env.obs_dim} "
          f"steps/episode={env.max_steps}")
    for seed in range(3):
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        start_q = env.parts["static_q"]
        total, done, info = 0.0, False, {}
        while not done:
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            _, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        print(f"random policy, reference start {seed}: start static_q={start_q:.3f}  "
              f"return={total:8.3f}  final={info}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "train", "plot", "generate", "eval", "calibrate"])
    ap.add_argument("--run", type=str, default="runs/v1")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=str, default="renders_rl/v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--bench", type=str, default="smb-v0")
    ap.add_argument("--encoding", type=str, default=None)
    ap.add_argument("--ref-dir", type=str, default="refeasy", help="folder containing reference levels")
    ap.add_argument("--levels", type=str, default="refeasy", help="folder for `calibrate`")
    ap.add_argument("--ctl-enemies", type=int, default=6) #6  10  15
    ap.add_argument("--ctl-coins", type=int, default=2) #2  8  12
    ap.add_argument("--ctl-jumps", type=int, default=15) #15  20  25
    args = ap.parse_args()
    if args.ref_dir is not None:
        set_reference_dir(args.ref_dir)
    run_dir = Path(args.run)

    def fresh_cfg() -> Config:
        cfg = Config(seed=args.seed, bench_name=args.bench)
        if args.encoding:
            cfg.slice_encoding = args.encoding
        if args.steps:
            cfg.total_steps = args.steps
        return cfg.resolve()

    if args.cmd == "inspect":
        cfg = fresh_cfg()
        BenchmarkAdapter(args.bench).describe()
        describe_references(cfg)
        smoke_test(cfg)

    elif args.cmd == "train":
        cfg = fresh_cfg()
        get_reference_levels()          # fail fast (before building 8 envs) if REF_DIR is missing/empty
        train(cfg, run_dir)

    elif args.cmd == "plot":
        plot_training(run_dir)

    elif args.cmd == "calibrate":
        calibrate(BenchmarkAdapter(args.bench), fresh_cfg(), Path(args.levels))

    elif args.cmd in ("generate", "eval"):
        from sb3_contrib import MaskablePPO
        cfg = _load_cfg(run_dir)
        model = MaskablePPO.load(str(run_dir / "final_model"))
        fixed = [args.ctl_enemies, args.ctl_coins, args.ctl_jumps]
        if all(v is not None for v in fixed):
            targets = [np.array(fixed, dtype=np.float32)] * args.n     # same target for all levels
        else:
            targets = _sample_targets(cfg, args.n, args.seed)
        levels = generate_levels(model, cfg, targets, args.deterministic, args.seed)
        print("local:", evaluate_local(levels, targets, cfg))

        if args.cmd == "generate":
            out = Path(args.out)
            save_levels(levels, out, env=pcg_benchmark.make(cfg.bench_name))
            (out / "targets.json").write_text(json.dumps([targets_to_control(t) for t in targets]))
        else:
            res = evaluate_benchmark(levels, targets, BenchmarkAdapter(cfg.bench_name))
            print("benchmark:", json.dumps(res, indent=2))
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "eval.json").write_text(json.dumps(res, indent=2))