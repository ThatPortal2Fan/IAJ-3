Loaded 1 levels from ''

==============================================================================
Quality:        1.0000  (all levels)
Diversity:      1.0000  (all levels)
Controlability: 0.0000  — all 1 levels completed
==============================================================================
Control target (applied to every level): {'enemies': 15, 'jumps': 30, 'coins': 12}
Control tolerance:                       {'enemies': 2, 'jumps': 3, 'coins': 1}
Reference: empty > 0.5, fenemies < 0.1

Manual per-level controlability (completed only): [np.float64(0.592)]
Manual mean:                     0.5920
Manual min:                      0.5920
  NOTE: raw env.evaluate gave 0.0000 over all levels; the corrected value above (0.0000) excludes incomplete solves.

=== QUALITY per level ===
  #  score | tube empty noise fenem compl | %empty  %fen tube# | issue
----------------------------------------------------------------------
  0  1.000 | 1.00  1.00  1.00  1.00  1.00 |  0.540 0.000     0 | ok
----------------------------------------------------------------------
1/1 fully valid.  Failures: tube=0 empty=0 noise=0 fenemies=0 solver=0

=== CONTROLABILITY per level (completed levels only) ===
  target: {'enemies': 15, 'jumps': 30, 'coins': 12}   plateau = target ± cerror

  # score |    enemies: actual∈[lo,hi]      jumps: actual∈[lo,hi]      coins: actual∈[lo,hi]
--------------------------------------------------------------------------------------------
  0  0.59 |  10 ∈ [13,17] → 0.77     10 ∈ [27,33] → 0.37      7 ∈ [11,13] → 0.64
--------------------------------------------------------------------------------------------
Mean component rewards (n=1 completed): enemies=0.769  jumps=0.370  coins=0.636
Weakest dimension: jumps  →  this is what you need to fix first.
