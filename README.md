# IAJ-3
## Installation
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install git+https://github.com/amidos2006/pcg_benchmark.git
python -m pip install gymnasium stable-baselines3 sb3-contrib matplotlib pandas tensorboard
```

In order to play generated levels:
```sh
git clone https://github.com/Mawiszus/TOAD-GUI.git
pip3 install -r requirements.txt -f "https://download.pytorch.org/whl/torch_stable.html"    # (5GB de espaço cuidado)
sudo apt install python3-tk
```

## Training (RL only)

```sh
python RLGenerator.py train --steps 1000000 --run runs/v1    # steps = number of training steps, run = output directory (separate by model versions)
python RLGenerator.py plot --run runs/v1    # to evaluate the final model
```

## Generation

For Non-RL:

```sh
python NonRLGenerator.py --gen 20 --seed 0 --width 50 --out renders_nrl    # gen = num of levels, seed is the rng seed, width is level width and out is level folder name
```

For RL:

```sh
python RLGenerator.py generate --run runs/v1 --n 50 --out renders_rl/v1 --ref-dir refeasy --seed 0  # n = num of levels, out = output directory, ref-dir = reference levels directory. extra flags can be used, see RLGenerator.py.
```

For SGA:

```sh
python GAGenerator.py --out renders_ga --seed 0  # see more flags in GAGenerator.py, line 503
```

## Playing

Only do this step if you installed TOAD-GUI.

```sh
cd TOAD-GUI
python main.py
```

Then select your level and play.

## Benchmark

```sh
python in_depth_evaluator.py --path renders_nrl  # pick the path of the levels you created.
```

What I suggest:
* Quality	0.50	Levels should be structurally valid and playable.
* Diversity	0.30	We want different levels, not just variations of the same layout.
* Controllability	0.20	Useful, but matching exact numbers of enemies, jumps, and coins is less important than producing good levels.


### Quality: already combines several properties

In the SMB implementation, quality is calculated as:
\[
Q(L)=\frac{T+E+N+G+4P}{8}
\]
where:
- \(T\): tube consistency score.
- \(E\): empty-space score.
- \(N\): horizontal noise score.
- \(G\): floating-enemy score.
- \(P\): A* completion percentage.

### Diversity: it compares how the agent traverses levels
The SMB diversity function compares the locations visited by the A* agent in two levels. It constructs a grid of visited locations and compares the two grids.

The utility get_range_reward() then converts measurements into scores between zero and one using a trapezoidal reward function. Depending on the configured thresholds, a measurement can receive partial credit, full credit, or zero.

One consequence is that two visually different levels may still score poorly on diversity if the agent traverses them in similar ways. Conversely, the benchmark's diversity score doesn't necessarily tell us whether every level looks interesting to a human.

### Controllability: it's about matching requested targets
The SMB controllability metric compares the actual numbers of enemies, jumps, and coins with requested target values. It allows a tolerance around each target.    
