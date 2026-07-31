# Numerical Einstein Equations

This repository implements a first-order Picard iteration for characteristic
initial value problems in the vacuum and Einstein--scalar-field equations.
The double-null convention is

\[
g_4=-4\Omega^2\,du\,dv+\gamma_{AB}(d\theta^A-b^Adu)(d\theta^B-b^Bdu).
\]

The numerical state stores the sphere metric, shift, log lapse, both full
weighted null forms, torsion, both weighted null-lapse coefficients, and—when
present—the scalar field and its three first derivatives. Characteristic data
are immutable, serialized before iteration, and checked by content hash after
every sweep.

## Installation

Python 3.12 is required. From the repository root:

```bash
python -m pip install -e .
python -m pytest -q tests/unit tests/integration
```

## Quick start

Every run uses a validated TOML configuration and a new output directory:

```bash
python -m nee.experiments.exp01_regular_vacuum \
  --config configs/experiments/exp01/smoke.toml --output results/exp01-smoke
```

Replace `smoke.toml` with `standard.toml` for the research configuration.
Experiment 8 also provides `angular-control.toml`. Existing output directories
are rejected unless `--resume` is explicitly supplied.

## Experiments

The public module entrances are:

1. `nee.experiments.exp01_regular_vacuum` — regular Schwarzschild and Kerr.
2. `nee.experiments.exp02_schwarzschild_horizon` — static horizon approach and Kruskal crossing.
3. `nee.experiments.exp03_schwarzschild_interior` — interior approach to radius zero.
4. `nee.experiments.exp04_vacuum_strong_short_pulse` — strong outgoing pulse and zero control.
5. `nee.experiments.exp05_vacuum_crossed_pulses` — crossed characteristic shears.
6. `nee.experiments.exp06_regular_exact_scalar` — Fisher/JNW exact scalar solutions.
7. `nee.experiments.exp07_nonspherical_scalar` — smooth nonspherical scalar data.
8. `nee.experiments.exp08_scalar_trapped_section` — scalar pulse, continuation, and trapped-section audit.

The shared alternative is `nee run expNN --config ... --output ...`.

Smoke configurations complete in seconds to about a minute on a laptop.
Standard configurations range from minutes to many hours and the largest
angular/coordinate cases require several gigabytes of memory. Run the standard
campaigns sequentially unless the machine has ample memory.

## Outputs and method

A successful run writes `resolved-config.toml`, `manifest.json`,
`boundary-data.npz`, `final-state.npz`, `summary.json`, `residual-maps.npz`,
`figures/`, and `run.log`. Raw runs under `results/` are ignored by Git.

The mathematical derivation is in [docs/article/article.tex](docs/article/article.tex).
The state, Picard order, discretization, and reliability rules are summarized
in [docs/algorithms/method.md](docs/algorithms/method.md).

## Citation

Citation metadata are provided in `CITATION.cff`. The project license will be
recorded in `LICENSE` once selected by the copyright holder.
