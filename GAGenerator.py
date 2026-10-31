"""
GAGenerator.py — SGA on top of NonRLGenerator.

Genome = tuple of Segment(kind, slices).
  - Decoder is deterministic: concatenate slices, then validate + structural check.
  - No RNG in the decoder — same genome always yields the same level.
  - Initial population seeds from NonRLGenerator outputs.

Notes on the decoder:
  NonRL builds the L side right-to-left, so the final level array has the
  L side in reverse placement order.  can_place is directional, so using
  it during decode would reject valid levels.  Instead decode uses:
    * validate()          — the benchmark's own cheap check
    * _check_structure()  — pairwise height diff, gap run, floating enemies
  These catch the structural issues can_place was preventing, without
  caring about direction.
"""
import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pcg_benchmark

from NonRLGenerator import (
    SLICE_INFO, EMPTY_ID,
    MAX_HEIGHT_DIFF, MAX_GAP_RUN,
    validate, _is_floating,
    generate as nonrl_generate,
    save_levels,
)

# --------------------------------------------------------------------------- #
# GA configuration
# --------------------------------------------------------------------------- #
SHARING_SIGMA = 15
SHARING_COEF  = 0.005
ARCHIVE_SIZE  = 5
ELITES        = 2

GROUNDED_SLICES = [sid for sid, info in SLICE_INFO.items()
                   if not _is_floating(info) and not info.has_pipe]
FLOATING_SLICES = [sid for sid, info in SLICE_INFO.items()
                   if _is_floating(info) and not info.has_pipe]
PIPE_SLICES     = [sid for sid, info in SLICE_INFO.items() if info.has_pipe]


# --------------------------------------------------------------------------- #
# Genome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Segment:
    kind: str
    slices: Tuple[int, ...]


Genome = Tuple[Segment, ...]


def _valid_segment(seg: Segment) -> bool:
    if not seg.slices:
        return False
    if seg.kind == "pipe":
        return (all(SLICE_INFO[s].has_pipe for s in seg.slices)
                and len(seg.slices) % 2 == 0)
    if seg.kind == "gap":
        return all(_is_floating(SLICE_INFO[s]) and not SLICE_INFO[s].has_pipe
                   for s in seg.slices)
    if seg.kind == "ground":
        return all(not _is_floating(SLICE_INFO[s]) and not SLICE_INFO[s].has_pipe
                   for s in seg.slices)
    return False


def genome_seed(genome: Genome) -> int:
    return int(hashlib.sha256(repr(genome).encode()).hexdigest()[:16], 16)


def slice_sequence(genome: Genome):
    return [s for seg in genome for s in seg.slices]


# --------------------------------------------------------------------------- #
# Structural check (direction-agnostic replacement for can_place)
# --------------------------------------------------------------------------- #
def _check_structure(level) -> bool:
    gap_run = 0
    prev_top = None
    for sid in level:
        info = SLICE_INFO[sid]
        # Pairwise height diff (symmetric, so safe in either direction)
        if info.top_solid_y is not None and prev_top is not None:
            if abs(info.top_solid_y - prev_top) > MAX_HEIGHT_DIFF:
                return False
        # Gap run length
        if _is_floating(info):
            gap_run += 1
            if gap_run > MAX_GAP_RUN:
                return False
        else:
            gap_run = 0
        # Floating enemy
        if info.has_enemy and not info.enemy_grounded:
            return False
        # Track last known top for next iteration
        if info.top_solid_y is not None:
            prev_top = info.top_solid_y
    return True


# --------------------------------------------------------------------------- #
# Decoder (deterministic, direction-agnostic)
# --------------------------------------------------------------------------- #
def decode(genome: Genome) -> Optional[list]:
    level = slice_sequence(genome)
    if not level:
        return None
    if not _check_structure(level):
        return None
    ok, _ = validate(level)
    return level if ok else None


# --------------------------------------------------------------------------- #
# Reverse: level -> genome
# --------------------------------------------------------------------------- #
def level_to_genome(level) -> Optional[Genome]:
    if not level:
        return None
    segments = []
    i = 0
    n = len(level)
    while i < n:
        sid = level[i]
        info = SLICE_INFO[sid]
        if info.has_pipe:
            kind = "pipe"
            member = lambda s: SLICE_INFO[s].has_pipe
        elif _is_floating(info):
            kind = "gap"
            member = lambda s: _is_floating(SLICE_INFO[s]) and not SLICE_INFO[s].has_pipe
        else:
            kind = "ground"
            member = lambda s: not _is_floating(SLICE_INFO[s]) and not SLICE_INFO[s].has_pipe

        j = i
        while j < n and member(level[j]):
            j += 1
        seg = Segment(kind, tuple(level[i:j]))
        if not _valid_segment(seg):
            return None
        segments.append(seg)
        i = j
    return tuple(segments)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def _random_ground_segment(rng, length_range=(2, 5)) -> Segment:
    L = int(rng.integers(*length_range, endpoint=True))
    return Segment("ground", tuple(int(rng.choice(GROUNDED_SLICES)) for _ in range(L)))


def _random_gap_segment(rng, length_range=(2, 4)) -> Segment:
    L = int(rng.integers(*length_range, endpoint=True))
    return Segment("gap", tuple(int(rng.choice(FLOATING_SLICES)) for _ in range(L)))


def _random_pipe_segment(rng) -> Optional[Segment]:
    if not PIPE_SLICES:
        return None
    sid = int(rng.choice(PIPE_SLICES))
    return Segment("pipe", (sid, sid))


def _mutate(genome: Genome, rng) -> Genome:
    if not genome:
        return (_random_ground_segment(rng),)
    op = int(rng.integers(0, 9))
    g = list(genome)

    if op == 0:  # change a single slice in a random segment
        i = int(rng.integers(0, len(g)))
        seg = g[i]
        s = list(seg.slices)
        k = int(rng.integers(0, len(s)))
        pool = (PIPE_SLICES if seg.kind == "pipe" else
                FLOATING_SLICES if seg.kind == "gap" else GROUNDED_SLICES)
        s[k] = int(rng.choice(pool))
        if seg.kind == "pipe":
            if len(s) >= 2:
                if k % 2 == 0 and k + 1 < len(s):
                    s[k + 1] = s[k]
                elif k % 2 == 1:
                    s[k - 1] = s[k]
        new_seg = Segment(seg.kind, tuple(s))
        if _valid_segment(new_seg):
            g[i] = new_seg
        return tuple(g)

    if op == 1:  # extend a segment
        i = int(rng.integers(0, len(g)))
        seg = g[i]
        if seg.kind == "pipe":
            sid = int(rng.choice(PIPE_SLICES))
            new_seg = Segment("pipe", seg.slices + (sid, sid))
        elif seg.kind == "gap":
            new_seg = Segment("gap", seg.slices + (int(rng.choice(FLOATING_SLICES)),))
        else:
            new_seg = Segment("ground", seg.slices + (int(rng.choice(GROUNDED_SLICES)),))
        if _valid_segment(new_seg):
            g[i] = new_seg
        return tuple(g)

    if op == 2:  # shrink a segment
        i = int(rng.integers(0, len(g)))
        seg = g[i]
        if seg.kind == "pipe":
            if len(seg.slices) <= 2:
                return tuple(g)
            new_seg = Segment("pipe", seg.slices[:-2])
        else:
            if len(seg.slices) <= 1:
                return tuple(g)
            new_seg = Segment(seg.kind, seg.slices[:-1])
        if _valid_segment(new_seg):
            g[i] = new_seg
        return tuple(g)

    if op == 3:  # insert a new segment
        pos = int(rng.integers(0, len(g) + 1))
        choice = int(rng.integers(0, 3))
        if choice == 0 and PIPE_SLICES:
            new_seg = _random_pipe_segment(rng)
        elif choice == 1 and FLOATING_SLICES:
            new_seg = _random_gap_segment(rng)
        else:
            new_seg = _random_ground_segment(rng)
        if new_seg is not None:
            g.insert(pos, new_seg)
        return tuple(g)

    if op == 4:  # delete a segment (keep at least 2)
        if len(g) <= 2:
            return tuple(g)
        i = int(rng.integers(0, len(g)))
        del g[i]
        return tuple(g)

    if op == 5:  # swap two adjacent segments
        if len(g) < 2:
            return tuple(g)
        i = int(rng.integers(0, len(g) - 1))
        g[i], g[i + 1] = g[i + 1], g[i]
        return tuple(g)

    if op == 6:  # replace a segment with a random one of the same kind
        i = int(rng.integers(0, len(g)))
        kind = g[i].kind
        if kind == "pipe":
            new_seg = _random_pipe_segment(rng)
        elif kind == "gap":
            new_seg = _random_gap_segment(rng)
        else:
            new_seg = _random_ground_segment(rng)
        if new_seg is not None:
            g[i] = new_seg
        return tuple(g)

    if op == 7:  # toggle pipe presence
        if not PIPE_SLICES:
            return tuple(g)
        i = int(rng.integers(0, len(g)))
        seg = g[i]
        if seg.kind == "ground":
            new_seg = _random_pipe_segment(rng)
            if new_seg is not None:
                g[i] = new_seg
        elif seg.kind == "pipe":
            g[i] = _random_ground_segment(rng, length_range=(2, 4))
        return tuple(g)

    # op == 8: duplicate a segment
    i = int(rng.integers(0, len(g)))
    g.insert(i + 1, g[i])
    return tuple(g)


def _crossover(p1: Genome, p2: Genome, rng) -> Genome:
    if not p1 or not p2:
        return p1 or p2
    i = int(rng.integers(1, max(2, len(p1))))
    j = int(rng.integers(1, max(2, len(p2))))
    return p1[:i] + p2[j:]


def _repair_width(genome: Genome, target_w: int, rng) -> Genome:
    """Force exact width.  problem.diversity() requires equal widths across
    the batch — varying widths crash the broadcast at problem.py:172."""
    g = list(genome)
    def width():
        return sum(len(seg.slices) for seg in g)

    # Trim down to target_w
    while width() > target_w and len(g) > 1:
        seg = g[-1]
        if seg.kind != "pipe" and len(seg.slices) > 1:
            g[-1] = Segment(seg.kind, seg.slices[:-1])
        elif seg.kind == "pipe" and len(seg.slices) > 2:
            g[-1] = Segment("pipe", seg.slices[:-2])
        else:
            g.pop()

    # Pad up to target_w
    while width() < target_w:
        need = target_w - width()
        if need >= 2:
            length = min(need, 4)
            g.append(_random_ground_segment(rng, length_range=(length, length)))
        else:
            g.append(Segment("ground", (int(rng.choice(GROUNDED_SLICES)),)))

    return tuple(g)


# --------------------------------------------------------------------------- #
# Fitness + sharing
# --------------------------------------------------------------------------- #
def _hamming(a, b) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def _sharing_penalty(genome: Genome, population, seq_self) -> float:
    penalty = 0.0
    for other in population:
        if other is genome:
            continue
        d = _hamming(seq_self, slice_sequence(other))
        if d < SHARING_SIGMA:
            penalty += 1.0 - d / SHARING_SIGMA
    return penalty


# --------------------------------------------------------------------------- #
# Initial population
# --------------------------------------------------------------------------- #
def _seed_initial_population(pop_n: int, width: int, control: dict, rng,
                             nat_rates: dict) -> list:
    genomes = []
    attempts = 0
    max_attempts = pop_n * 10
    while len(genomes) < pop_n and attempts < max_attempts:
        attempts += 1
        lvl = nonrl_generate(
            width,
            seed_id=EMPTY_ID,
            rng=np.random.default_rng(int(rng.integers(0, 2 ** 31))),
            fail_tally={"no_valid_candidate": 0, "not_enough_empty": 0},
            ctl_enemies=control["enemies"],
            ctl_coins=control["coins"],
            ctl_jumps=control["jumps"],
            ctl_overshoot=1.5,
            nat_rates=nat_rates,
        )
        if lvl is None:
            continue
        g = level_to_genome(lvl)
        if g is None:
            continue
        if decode(g) is None:
            continue
        genomes.append(g)

    if not genomes:
        raise RuntimeError(
            f"Could not seed any genome from NonRLGenerator "
            f"({attempts} attempts)"
        )
    if len(genomes) < pop_n:
        print(f"  warning: seeded only {len(genomes)}/{pop_n} "
              f"(filling remainder with mutations)")
    while len(genomes) < pop_n:
        base = genomes[int(rng.integers(0, len(genomes)))]
        genomes.append(_mutate(base, rng))
    return genomes[:pop_n]


# --------------------------------------------------------------------------- #
# Main SGA loop
# --------------------------------------------------------------------------- #
def run_sga(width: int, pop_n: int, gens: int, seed: int,
            control: dict, out_dir: str, nat_rates: dict):
    rng = np.random.default_rng(seed)
    env = pcg_benchmark.make("smb-v0")
    problem = None
    for attr in ("_problem", "problem", "_game", "game"):
        problem = getattr(env, attr, None)
        if problem is not None:
            break
    if problem is None:
        raise RuntimeError("Could not access benchmark problem")

    cache = {}
    archive = []

    print(f"Seeding {pop_n} genomes from NonRLGenerator…")
    t0 = time.time()
    pop = _seed_initial_population(pop_n, width, control, rng, nat_rates)
    print(f"  seeded in {time.time()-t0:.1f}s")

    for gen in range(gens):
        t_gen = time.time()

        # Decode + evaluate
        infos = []
        for g in pop:
            key = genome_seed(g)
            if key not in cache:
                lvl = decode(g)
                cache[key] = problem.info(lvl) if lvl is not None else None
            infos.append(cache[key])

        valid = [(g, info) for g, info in zip(pop, infos) if info is not None]

        if len(valid) >= 2:
            D_pop = float(np.mean([
                problem.diversity(valid[a][1], valid[b][1])
                for a in range(len(valid))
                for b in range(a + 1, len(valid))
            ]))
        else:
            D_pop = 0.0

        seqs = [slice_sequence(g) for g in pop]
        scores = []
        for i, (g, info) in enumerate(zip(pop, infos)):
            if info is None:
                scores.append(0.0)
                continue
            Q = problem.quality(info)
            C = problem.controlability(info, control)
            s = 0.5 * Q + 0.3 * D_pop + 0.2 * C
            s -= SHARING_COEF * _sharing_penalty(g, pop, seqs[i])
            scores.append(float(s))

        for g, s in zip(pop, scores):
            archive.append((s, g))
        archive.sort(key=lambda x: -x[0])
        archive = archive[:ARCHIVE_SIZE]

        best = max(scores)
        mean = float(np.mean(scores))
        n_bad = sum(1 for i in infos if i is None)
        print(f"gen {gen+1:2d} | best={best:.3f} mean={mean:.3f} "
              f"D={D_pop:.3f} bad={n_bad}/{pop_n} "
              f"({time.time()-t_gen:.1f}s)")

        if gen == gens - 1:
            break

        new_pop = [pop[i] for i in np.argsort(scores)[-ELITES:]]
        while len(new_pop) < pop_n:
            for _attempt in range(20):
                def tournament():
                    idx = rng.choice(len(pop), size=3, replace=False)
                    return pop[max(idx, key=lambda i: scores[i])]
                child = _crossover(tournament(), tournament(), rng)
                child = _mutate(child, rng)
                child = _repair_width(child, width, rng)
                if decode(child) is not None:
                    new_pop.append(child)
                    break
            else:
                new_pop.append(pop[int(rng.integers(0, pop_n))])
        pop = new_pop

    best_score, best_genome = archive[0]
    best_level = decode(best_genome)
    if best_level is None:
        raise RuntimeError("Best genome fails to decode — archive corrupted")
    print(f"\nFinal best: S={best_score:.3f}, length={len(best_level)}")
    print(f"Genome: {best_genome}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_levels([best_level], out, env=env)
    (out / "best_genome.txt").write_text(repr(best_genome))
    (out / "archive_scores.txt").write_text(
        "\n".join(f"{s:.4f}" for s, _ in archive) + "\n"
    )
    print(f"Wrote {out}/level_0.{{json,txt,png}}, best_genome.txt, archive_scores.txt")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=150)
    ap.add_argument("--pop",   type=int, default=15)
    ap.add_argument("--gens",  type=int, default=15)
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--out",   type=str, default="renders_ga")
    ap.add_argument("--ctl-enemies", type=int, default=6)
    ap.add_argument("--ctl-coins",   type=int, default=2)
    ap.add_argument("--ctl-jumps",   type=int, default=15)
    ap.add_argument("--nat-rate-enemy", type=float, default=3.0)
    ap.add_argument("--nat-rate-coin",  type=float, default=1.0)
    ap.add_argument("--nat-rate-jump",  type=float, default=1.0)
    args = ap.parse_args()

    nat_rates = {
        "enemy": args.nat_rate_enemy,
        "coin":  args.nat_rate_coin,
        "jump":  args.nat_rate_jump,
    }
    control = {
        "enemies": args.ctl_enemies,
        "coins":   args.ctl_coins,
        "jumps":   args.ctl_jumps,
    }
    run_sga(args.width, args.pop, args.gens, args.seed, control,
            args.out, nat_rates)