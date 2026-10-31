"""
NonRLGenerator.py — Heuristic rule-based generator for the SMB PCG benchmark.

Pipes are ordinary slices. A forward-lookahead invariant guarantees every
opened pipe can be closed by at least one slice in the library, so the
generator can never end up in an unclosable pipe state.

Controllability tracking (enemies / coins / jumps) is *estimate-based*:
we count "opportunities" during generation and cap them at
    target * ctl_overshoot
so that the A* — which misses some fraction of opportunities — ends up
near the benchmark target. Tune --ctl-overshoot per quantity.

Output: JSON (slice ids), TXT (ASCII level) and PNG (rendered),
ready to drop into the external game repo.

Usage:
    python NonRLGenerator.py --inspect
    python NonRLGenerator.py --gen 50 --width 150 --out renders
    python NonRLGenerator.py --gen 50 --ctl-enemies 6 --ctl-coins 2 --ctl-jumps 15 \
                    --ctl-overshoot 2
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pcg_benchmark

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
SLICES_FILE = Path(pcg_benchmark.__file__).parent / "probs" / "smb" / "slices.txt"
NOISE_FILE  = Path(pcg_benchmark.__file__).parent / "probs" / "smb" / "noise.txt"

EMPTY_ID = 0
SOLID    = set("X#tSQ")
GROUND   = set("X#")
ENEMY    = set("ygkr")   # matches Problem._caculate_fenemies
KILLABLE = set("gkr")    # spiky 'y' is excluded — cannot be stomped

MAX_HEIGHT_DIFF       = 4
MAX_GAP_RUN           = 4
HARD_MIN_EMPTY        = 0.52   # benchmark requires > 0.5
SOFT_TARGET_EMPTY     = 0.58
MAX_FLOAT_ENEMY_RATIO = 0.10
MAX_GEN_RETRIES       = 4      # only for empty-budget failures now

# --------------------------------------------------------------------------- #
# Slice classification
# --------------------------------------------------------------------------- #
SLICES = [l.strip() for l in SLICES_FILE.read_text().splitlines() if l.strip()]
NOISE  = np.array([float(v) for v in NOISE_FILE.read_text().strip().split(",")])
HEIGHT = len(SLICES[0])
GROUND_ROW = HEIGHT - 2   # 14

def _is_floating(info):
    return info.floor_y is None or info.floor_y < GROUND_ROW

@dataclass
class SliceInfo:
    sid: int
    s: str
    floor_y: Optional[int]
    top_solid_y: Optional[int]
    has_ground: bool
    has_pipe: bool
    has_enemy: bool
    has_coin: bool
    enemy_grounded: bool

def _classify(sid, s):
    floor_y     = max((y for y in range(len(s)) if s[y] in SOLID), default=None)
    top_solid_y = min((y for y in range(len(s)) if s[y] in SOLID), default=None)
    has_ground  = any(c in GROUND for c in s)
    has_pipe    = "t" in s
    has_enemy   = any(c in ENEMY  for c in s)
    has_coin    = "o" in s
    grounded    = True
    for y in range(len(s)):
        if s[y] in ENEMY:
            if y + 1 >= len(s) or s[y + 1] not in SOLID:
                grounded = False
                break
    return SliceInfo(sid, s, floor_y, top_solid_y, has_ground,
                     has_pipe, has_enemy, has_coin, grounded)

SLICE_INFO = {sid: _classify(sid, s) for sid, s in enumerate(SLICES)}

assert "t" not in SLICES[EMPTY_ID], (
    "Slice 0 contains a pipe tile. Seed/empty logic assumes it doesn't."
)

# --------------------------------------------------------------------------- #
# [CTL] Precomputed controllability helpers
# --------------------------------------------------------------------------- #
KILLABLE_COUNT = {
    sid: sum(1 for c in s if c in KILLABLE)
    for sid, s in enumerate(SLICES)
}

COIN_POSITIONS = {
    sid: [y for y, c in enumerate(s) if c == 'o']
    for sid, s in enumerate(SLICES)
}

def _coin_reachable_in_slice(s: str, cy: int) -> bool:
    """Coin at row `cy` is reachable from a solid tile in the same column
    within MAX_HEIGHT_DIFF rows below it (jump up)."""
    for sy in range(cy, min(HEIGHT, cy + MAX_HEIGHT_DIFF + 1)):
        if s[sy] in SOLID:
            return True
    return False

def _count_reachable_coins(sid, side, left, right, seed_id):
    """Count coins in slice `sid` reachable either from the slice itself or
    from a neighbouring slice within MAX_GAP_RUN horizontal distance."""
    s = SLICES[sid]
    coins = COIN_POSITIONS[sid]
    if not coins:
        return 0

    # Neighbours ordered by spatial closeness (1..MAX_GAP_RUN), across the seed if needed
    if side == 'L':
        near = [left[-i] for i in range(1, min(len(left), MAX_GAP_RUN) + 1)]
        if len(left) < MAX_GAP_RUN:
            near.append(seed_id)
    else:
        near = [right[-i] for i in range(1, min(len(right), MAX_GAP_RUN) + 1)]
        if len(right) < MAX_GAP_RUN:
            near.append(seed_id)

    count = 0
    for cy in coins:
        if _coin_reachable_in_slice(s, cy):
            count += 1
            continue
        for nid in near:
            if _coin_reachable_in_slice(SLICES[nid], cy):
                count += 1
                break
    return count

def _count_jumps(prev_id, new_id):
    """Estimate required jumps between two adjacent slices.
    - Closing a gap (walkable surface reappears) -> +1
    - Upward step on top_solid_y                  -> +1
    """
    prev = SLICE_INFO[prev_id]
    new  = SLICE_INFO[new_id]
    if prev.floor_y is None:
        return 1 if new.floor_y is not None else 0
    if (new.floor_y is not None
            and prev.top_solid_y is not None
            and new.top_solid_y is not None
            and new.top_solid_y < prev.top_solid_y):
        return 1
    return 0

# Feature weight multipliers for weighted slice choice.
# 1.0 = neutral.  >1 boosts that feature.  0 disables the feature boost
# (a slice can still be picked for its other features or its base weight of 1.0).
NATURAL_RATE = {
    "enemy": 1.0,
    "coin":  1.0,
    "jump":  1.0,
}

def _contributes_to(sid, feat, prev_id, side, left, right, seed_id):
    if feat == "enemy":
        return KILLABLE_COUNT[sid] > 0
    if feat == "coin":
        return _count_reachable_coins(sid, side, left, right, seed_id) > 0
    if feat == "jump":
        return _count_jumps(prev_id, sid) > 0
    return False

# --------------------------------------------------------------------------- #
# Pipe state machine + forward-lookahead closer check
# --------------------------------------------------------------------------- #
def pipe_step(in_pipe: np.ndarray, sid: int):
    s = SLICES[sid]
    new_state = in_pipe.copy()
    for y in range(HEIGHT):
        if s[y] == "t":
            new_state[y] = not in_pipe[y]
        elif in_pipe[y]:
            return False, None
    return True, new_state

def _can_close_bytes(in_pipe_bytes: bytes) -> bool:
    state = np.frombuffer(in_pipe_bytes, dtype=bool)
    if not state.any():
        return True
    for sid in range(len(SLICES)):
        s = SLICES[sid]
        ok, nxt = pipe_step(state, sid)
        if not ok or nxt.any():
            continue
        if all((s[y] == "t") == bool(state[y]) for y in range(HEIGHT)):
            return True
    return False

def can_close(in_pipe: np.ndarray) -> bool:
    return _can_close_bytes(in_pipe.tobytes())

# --------------------------------------------------------------------------- #
# Local validation — mirrors MarioProblem.info() preconditions
# --------------------------------------------------------------------------- #
def validate(level_ids):
    n = len(level_ids)
    if n == 0:
        return False, {"reason": "empty level"}

    tube = 0
    for y in range(HEIGHT):
        run = 0
        for i in range(n):
            if SLICES[level_ids[i]][y] == "t":
                run += 1
            else:
                if run % 2 == 1:
                    tube += 1
                run = 0
        if run % 2 == 1:
            tube += 1
    if tube > 0:
        return False, {"reason": "tube", "count": tube}

    empty = sum(1 for c in level_ids if c == EMPTY_ID) / n
    if empty <= 0.5:
        return False, {"reason": "empty", "value": float(empty)}

    changes = np.zeros(HEIGHT, dtype=np.int32)
    for i in range(1, n):
        a, b = SLICES[level_ids[i - 1]], SLICES[level_ids[i]]
        for y in range(HEIGHT):
            if a[y] != b[y]:
                changes[y] += 1
    for y in range(HEIGHT):
        if NOISE[y] == 0:
            if changes[y] > 0:
                return False, {"reason": "noise", "row": y, "changes": int(changes[y])}
        else:
            allowed = max(NOISE[y], 2.0 / (n - 1))
            if changes[y] / (n - 1) > allowed + 1e-9:
                return False, {"reason": "noise", "row": y,
                               "actual": float(changes[y] / (n - 1)),
                               "allowed": float(allowed)}

    total = floating = 0
    for sid in level_ids:
        s = SLICES[sid]
        for y in range(HEIGHT):
            if s[y] in ENEMY:
                total += 1
                if y + 1 >= HEIGHT or s[y + 1] not in SOLID:
                    floating += 1
    if total > 0 and floating * 10 >= total:
        return False, {"reason": "fenemies", "float": floating, "total": total}

    return True, {"empty": float(empty), "tube": 0,
                  "fenemies": (floating / total) if total else 0.0}

# --------------------------------------------------------------------------- #
# Incremental generator state
# --------------------------------------------------------------------------- #
@dataclass
class GenState:
    changes:     np.ndarray
    empty_count: int
    enemy_float: int
    enemy_total: int
    target_w:    int
    current_len:  int
    in_pipe_L:   np.ndarray
    in_pipe_R:   np.ndarray
    just_closed_L: bool
    just_closed_R: bool
    last_top_L: Optional[int]
    last_top_R: Optional[int]
    killable_enemies: int = 0
    reachable_coins:  int = 0
    jumps_required:   int = 0
    target_killable_enemies: int = 0
    target_reachable_coins:  int = 0
    target_jumps:            int = 0

    @property
    def remaining(self):
        return self.target_w - self.current_len

def _noise_ceiling(state, row):
    if NOISE[row] == 0:
        return 0
    allowed = max(NOISE[row], 2.0 / max(1, state.target_w - 1))
    return int(np.floor(allowed * (state.target_w - 1) + 1e-9))

def can_place(new_id, prev_id, state, gap_run, side):
    new, prev = SLICE_INFO[new_id], SLICE_INFO[prev_id]

    just_closed = state.just_closed_L if side == "L" else state.just_closed_R
    if just_closed and new.has_pipe:
        return False
    last_floor = state.last_top_L if side == "L" else state.last_top_R
    if new.top_solid_y is not None and last_floor is not None:
        if abs(new.top_solid_y - last_floor) > MAX_HEIGHT_DIFF:
            return False
    if state.remaining <= 4:
        if new.top_solid_y is None:
            return False
        empty_top = SLICE_INFO[EMPTY_ID].top_solid_y
        if empty_top is not None and abs(new.top_solid_y - empty_top) > MAX_HEIGHT_DIFF:
            return False

    if new.top_solid_y is None and gap_run >= MAX_GAP_RUN:
        return False

    if new.has_enemy and not new.enemy_grounded:
        return False

    in_pipe = state.in_pipe_L if side == "L" else state.in_pipe_R
    ok, new_state = pipe_step(in_pipe, new_id)
    if not ok:
        return False
    if in_pipe.any() and (new_state & ~in_pipe).any():
        return False
    if state.remaining <= 4:
        if new_state.any():
            return False
    elif not can_close(new_state):
        return False

    if _is_floating(new) and gap_run >= MAX_GAP_RUN:
        return False

    empty_slice = SLICE_INFO[EMPTY_ID]
    for y in range(HEIGHT):
        delta = 1 if new.s[y] != prev.s[y] else 0
        current_noise = state.changes[y] + delta
        ceiling = _noise_ceiling(state, y)
        if current_noise > ceiling:
            return False
        if current_noise == ceiling and state.remaining > 4 and new.s[y] != empty_slice.s[y]:
            return False

    return True

def apply_place(state, new_id, prev_id, side, coin_add=0):
    new, prev = SLICE_INFO[new_id], SLICE_INFO[prev_id]
    for y in range(HEIGHT):
        if new.s[y] != prev.s[y]:
            state.changes[y] += 1
    if new_id == EMPTY_ID:
        state.empty_count += 1
    if new.has_enemy:
        state.enemy_total += 1
        if not new.enemy_grounded:
            state.enemy_float += 1

    # [CTL] controllability bookkeeping
    state.killable_enemies += KILLABLE_COUNT[new_id]
    state.reachable_coins  += coin_add
    state.jumps_required   += _count_jumps(prev_id, new_id)

    in_pipe = state.in_pipe_L if side == "L" else state.in_pipe_R
    was_open = bool(in_pipe.any())
    _, nxt = pipe_step(in_pipe, new_id)
    now_closed = not nxt.any()
    closed_now = was_open and now_closed and new.has_pipe
    if side == "L":
        state.in_pipe_L = nxt
        state.just_closed_L = closed_now
        if new.top_solid_y is not None:
            state.last_top_L = new.top_solid_y
    else:
        state.in_pipe_R = nxt
        state.just_closed_R = closed_now
        if new.top_solid_y is not None:
            state.last_top_R = new.top_solid_y
    state.current_len += 1


def _weighted_choice(candidates, prev_id, side, left, right, seed_id,
                     nat_rates, rng):
    """Pick a slice from `candidates` with probability proportional to
    its feature weight:

        weight = 1.0
               + nat_rates["enemy"] * killable_enemies_in_slice
               + nat_rates["coin"]  * reachable_coins_in_slice
               + nat_rates["jump"]  * jumps_required_to_enter
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return int(candidates[0])

    ws = np.empty(len(candidates), dtype=float)
    for i, sid in enumerate(candidates):
        w  = 1.0
        w += nat_rates["enemy"] * KILLABLE_COUNT[sid]
        w += nat_rates["coin"]  * _count_reachable_coins(
            sid, side, left, right, seed_id)
        w += nat_rates["jump"]  * _count_jumps(prev_id, sid)
        ws[i] = w

    total = ws.sum()
    if total <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=ws / total))
# --------------------------------------------------------------------------- #
# Generator — single attempt (pipe-safe by construction)
# --------------------------------------------------------------------------- #
def _generate_once(target_width, seed_id, rng, fail_tally=None,
                   ctl_enemies=None, ctl_coins=None, ctl_jumps=None,
                   ctl_overshoot=1.3, nat_rates=None):
    if nat_rates is None:
        nat_rates = NATURAL_RATE
    left, right = [], []
    left_prev = right_prev = seed_id
    gap_L = gap_R = 0

    # [CTL] resolve targets (midpoint of benchmark range if unspecified) × overshoot
    if ctl_enemies is None:
        ctl_enemies = max(2, int(target_width / 7)) // 2
    if ctl_coins is None:
        ctl_coins = max(2, int(target_width / 10)) // 2
    if ctl_jumps is None:
        ctl_jumps = max(2, int(target_width / 5)) // 2

    state = GenState(
        changes     = np.zeros(HEIGHT, dtype=np.int32),
        empty_count = 1 if seed_id == EMPTY_ID else 0,
        enemy_float = 0,
        enemy_total = 1 if SLICE_INFO[seed_id].has_enemy else 0,
        target_w    = target_width,
        current_len = 1,
        in_pipe_L   = np.zeros(HEIGHT, dtype=bool),
        in_pipe_R   = np.zeros(HEIGHT, dtype=bool),
        just_closed_L = False,
        just_closed_R = False,
        last_top_L = SLICE_INFO[seed_id].top_solid_y,
        last_top_R = SLICE_INFO[seed_id].top_solid_y,
        # [CTL] initial feature counts
        killable_enemies = KILLABLE_COUNT[seed_id],
        reachable_coins  = _count_reachable_coins(seed_id, 'R', [], [], seed_id),
        jumps_required   = 0,
        # [CTL] effective targets with overshoot
        target_killable_enemies = int(ctl_enemies * ctl_overshoot * ctl_overshoot),
        target_reachable_coins  = int(ctl_coins   * ctl_overshoot * ctl_overshoot),
        target_jumps            = int(ctl_jumps   * ctl_overshoot),
    )
    if SLICE_INFO[seed_id].has_enemy and not SLICE_INFO[seed_id].enemy_grounded:
        state.enemy_float = 1

    while len(left) + len(right) + 1 < target_width:
        if len(left) <= len(right):
            side, prev_id, gap_run = "L", left_prev, gap_L
        else:
            side, prev_id, gap_run = "R", right_prev, gap_R

        current_len = len(left) + len(right) + 1
        remaining   = target_width - current_len
        empty_frac  = state.empty_count / current_len

        in_pipe = state.in_pipe_L if side == "L" else state.in_pipe_R
        must_close_pipe = (remaining <= 5) and in_pipe.any()
        last_top = state.last_top_L if side == "L" else state.last_top_R

        if must_close_pipe:
            force_empty = False
        elif gap_run >= MAX_GAP_RUN:
            force_empty = True
        elif empty_frac < HARD_MIN_EMPTY:
            force_empty = True
        elif empty_frac < SOFT_TARGET_EMPTY:
            p = (SOFT_TARGET_EMPTY - empty_frac) / \
                (SOFT_TARGET_EMPTY - HARD_MIN_EMPTY)
            force_empty = rng.random() < p
        else:
            if last_top == 14 and not in_pipe.any() and rng.random() < 0.3:
                force_empty = True
            else:
                force_empty = False

        if state.empty_count + remaining <= HARD_MIN_EMPTY * target_width + 1:
            force_empty = True

        candidates = [
            sid for sid in range(len(SLICES))
            if can_place(sid, prev_id, state, gap_run, side)
        ]
        empty_ok = can_place(EMPTY_ID, prev_id, state, gap_run, side)

        # [CTL] cap controllability features at effective targets.
        # Non-empty slices whose feature contribution would exceed a cap are dropped.
        # If everything gets filtered out, fall back to EMPTY_ID so generation never stalls.
        if candidates:
            filtered = []
            for sid in candidates:
                if state.killable_enemies + KILLABLE_COUNT[sid] > state.target_killable_enemies:
                    continue
                coin_add = _count_reachable_coins(sid, side, left, right, seed_id)
                if state.reachable_coins + coin_add > state.target_reachable_coins:
                    continue
                jump_add = _count_jumps(prev_id, sid)
                if state.jumps_required + jump_add > state.target_jumps:
                    continue
                filtered.append(sid)
            if filtered:
                candidates = filtered
            elif empty_ok:
                candidates = [EMPTY_ID]
            # else: leave candidates as-is (accept overshoot rather than starve)

        if force_empty and not empty_ok and not must_close_pipe:
            if last_top is not None and last_top < 14:
                step_down_candidates = [
                    c for c in candidates
                    if SLICE_INFO[c].top_solid_y is not None and SLICE_INFO[c].top_solid_y > last_top
                ]
                if step_down_candidates:
                    candidates = step_down_candidates
            force_empty = False

        if force_empty:
            if empty_ok:
                new_id = EMPTY_ID
            elif candidates:
                new_id = _weighted_choice(candidates, prev_id, side, left, right,
                                        seed_id, nat_rates, rng)
            else:
                fail_tally["no_valid_candidate"] += 1
                return None
        elif candidates:
            new_id = _weighted_choice(candidates, prev_id, side, left, right,
                                    seed_id, nat_rates, rng)
        else:
            fail_tally["no_valid_candidate"] += 1
            return None

        # [CTL] coin_add must be computed BEFORE the list append so `left`/`right`
        # still refer to the existing neighbours (not the newly placed slice).
        coin_add = _count_reachable_coins(new_id, side, left, right, seed_id)

        if side == "L":
            left.append(new_id);  left_prev  = new_id
        else:
            right.append(new_id); right_prev = new_id
        apply_place(state, new_id, prev_id, side, coin_add=coin_add)

        new_gap = (gap_run + 1) if _is_floating(SLICE_INFO[new_id]) else 0
        if side == "L":
            gap_L = new_gap
        else:
            gap_R = new_gap

    level = list(reversed(left)) + [seed_id] + right

    if state.empty_count / len(level) <= 0.5:
        fail_tally["not_enough_empty"] += 1
        return None
    return level

def generate(target_width, seed_id=EMPTY_ID, rng=None, fail_tally=None,
             ctl_enemies=None, ctl_coins=None, ctl_jumps=None,
             ctl_overshoot=1.3, nat_rates=None):
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        level = _generate_once(target_width, seed_id, rng, fail_tally,
                               ctl_enemies, ctl_coins, ctl_jumps,
                               ctl_overshoot, nat_rates)
        if level is not None:
            return level
    return None

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def to_text(level_ids):
    content = np.pad(np.array(level_ids), 3)
    width = len(content)
    rows = []
    for y in range(HEIGHT):
        row = ""
        for x in range(width):
            if x == 1 and y == HEIGHT - 3:
                row += "M"
            elif x == width - 2 and y == HEIGHT - 3:
                row += "F"
            else:
                row += SLICES[content[x]][y]
        rows.append(row)
    return rows

def save_levels(levels, out_dir: Path, env=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, level in enumerate(levels):
        ids = [int(x) for x in level]

        (out_dir / f"level_{i}.json").write_text(json.dumps(ids))

        rows = to_text(ids)
        (out_dir / f"level_{i}.txt").write_text("\n".join(rows) + "\n")

        if env is not None:
            images = env.render([ids])
            if isinstance(images, (list, tuple)):
                for img in images:
                    if hasattr(img, "save"):
                        img.save(out_dir / f"level_{i}.png")
                        break

    print(f"Wrote {len(levels)} levels (JSON+TXT"
          + ("+PNG" if env is not None else "") + f") to {out_dir}/")

# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def inspect():
    print(f"Slices: {len(SLICES)}   Height: {HEIGHT}")

    print("\nNoise ceiling per row (noise.txt):")
    for y in range(HEIGHT):
        flag = "  <-- HARD (no change allowed)" if NOISE[y] == 0 else ""
        print(f"  row {y:2d}: {NOISE[y]:.3f}{flag}")

    print("\nSlice 0 (counted by benchmark as 'empty'):")
    for y, c in enumerate(SLICES[0]):
        print(f"  y={y:2d}: {c}")

    from collections import Counter
    tags = Counter()
    for sid, info in SLICE_INFO.items():
        t = []
        if info.has_pipe:   t.append("pipe")
        if info.has_enemy:  t.append("enemy" + ("" if info.enemy_grounded else "(FLOAT)"))
        if info.has_coin:   t.append("coin")
        if info.floor_y is None:  t.append("GAP")
        elif not t:               t.append("ground")
        tags[",".join(t)] += 1
    print("\nSlice summary:")
    for k, v in tags.most_common():
        print(f"  {v:3d}  {k}")

    # [CTL] precomputed per-slice controllability contributions
    n_enemy = sum(KILLABLE_COUNT.values())
    n_coin  = sum(len(v) for v in COIN_POSITIONS.values())
    print(f"\n[CTL] Killable enemies in library: {n_enemy}")
    print(f"[CTL] Coin tiles in library:       {n_coin}")

def report(levels, ctl_targets=None):
    valid = 0
    reasons = {}
    for lv in levels:
        if lv is None:
            reasons["null"] = reasons.get("null", 0) + 1
            continue
        ok, r = validate(lv)
        if ok:
            valid += 1
        else:
            key = r.get("reason", "?")
            reasons[key] = reasons.get(key, 0) + 1
    total = len(levels)
    print(f"\nGenerated {total} levels, {valid} valid "
          f"({100*valid/total:.1f}%)")
    if reasons:
        print("Failures:", reasons)
    if ctl_targets:
        print(f"[CTL] Overshoot caps: enemies<={ctl_targets['enemies']}, "
              f"coins<={ctl_targets['coins']}, jumps<={ctl_targets['jumps']}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--gen",   type=int, default=10)
    ap.add_argument("--width", type=int, default=150)
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--out",   type=str, default="renders_nrl")
    ap.add_argument("--no-render", action="store_true",
                    help="skip PNG rendering (faster for big batches)")
    ap.add_argument("--ctl-enemies", type=int, default=6, #6  10  15
                    help="benchmark target for kills (default: midpoint of range)")
    ap.add_argument("--ctl-coins", type=int, default=2, #2  8  12
                    help="benchmark target for coins (default: midpoint of range)")
    ap.add_argument("--ctl-jumps", type=int, default=15, #15  20  25
                    help="benchmark target for jumps (default: midpoint of range)")
    ap.add_argument("--ctl-overshoot", type=float, default=1.5,
                    help="multiply targets by this (A* misses some opportunities)")
    ap.add_argument("--nat-rate-enemy", type=float, default=3.0, #3  10  10
                    help="weight multiplier for enemy-bearing slices. 1.0 = neutral. "
                         "e.g. 3.0 makes a 1-killable slice ~2x as likely to be chosen")
    ap.add_argument("--nat-rate-coin",  type=float, default=1.0, #1  10  10
                    help="weight multiplier for coin-bearing slices. 1.0 = neutral")
    ap.add_argument("--nat-rate-jump",  type=float, default=1.0, #1  10  10
                    help="weight multiplier for jump-inducing slices. 1.0 = neutral") 
    args = ap.parse_args()

    if args.inspect:
        inspect()

    if args.gen >= 0:
        rng = np.random.default_rng(args.seed)
        fail_tally = {"no_valid_candidate": 0, "not_enough_empty": 0}
        nat_rates = {
            "enemy": args.nat_rate_enemy,
            "coin":  args.nat_rate_coin,
            "jump":  args.nat_rate_jump,
        }
        levels = [
            generate(args.width, args.seed, rng, fail_tally,
                        ctl_enemies=args.ctl_enemies,
                        ctl_coins=args.ctl_coins,
                        ctl_jumps=args.ctl_jumps,
                        ctl_overshoot=args.ctl_overshoot,
                        nat_rates=nat_rates)
            for _ in range(args.gen)
        ]

        # Resolve effective targets for reporting
        base_e = args.ctl_enemies if args.ctl_enemies is not None else max(2, args.width // 7) // 2
        base_c = args.ctl_coins   if args.ctl_coins   is not None else max(2, args.width // 10) // 2
        base_j = args.ctl_jumps   if args.ctl_jumps   is not None else max(2, args.width // 5) // 2
        ctl_targets = {
            "enemies": int(base_e * args.ctl_overshoot * args.ctl_overshoot),
            "coins":   int(base_c * args.ctl_overshoot * args.ctl_overshoot),
            "jumps":   int(base_j * args.ctl_overshoot),
        }

        report(levels, ctl_targets)

        print("\nFailure Reasons:")
        for reason, count in fail_tally.items():
            print(f"    {reason}: {count} failures")

        levels = [lv for lv in levels if lv is not None]

        if args.out and levels:
            env = None
            if not args.no_render:
                env = pcg_benchmark.make("smb-v0")
            save_levels(levels, Path(args.out), env=env)