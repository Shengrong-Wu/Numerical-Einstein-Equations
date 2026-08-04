# Numerical Einstein Equations

This repository implements a first-order Picard iteration for characteristic
initial value problems in the Einstein vacuum and Einstein--massless-scalar
equations. The spacetime is evolved in a double-null gauge represented by the
section metric `gamma`, lapse `Omega`, and angular shift `b`:

```text
g_4 = -4 Omega^2 du dv
      + gamma_AB (dtheta^A - b^A du)(dtheta^B - b^B du)
```

The project is designed for freely prescribed characteristic geometry beyond
spherical symmetry. It includes one reusable numerical package, validated
configurations for eight experiments, exact-solution comparisons, independent
curvature audits, and a search for trapped two-spheres.

The research article is available as [docs/article.pdf](docs/article.pdf).

## Research contributions

The main contributions are:

- A coefficient-complete first-order formulation of the vacuum and
  Einstein--scalar equations suitable for direct numerical evaluation in
  double-null gauge.
- A geometry-guided Picard map whose update order follows the characteristic
  constraint and transport hierarchy.
- Separate immutable characteristic data and mutable interior iterates, with
  trace reimposition and content-hash checks during evolution.
- Legendre--Gauss--Lobatto spectral elements in the two null directions,
  fractional-power and logarithmic coordinate maps, and pole-free angular
  differentiation with spherical-harmonic Galerkin projection.
- Independent four-metric curvature reconstruction that does not reuse the
  equation right-hand sides employed by the Picard sweep.
- Numerical evolution of strong nonspherical characteristic geometry,
  crossed low-regularity data, and nonspherical Einstein--scalar data.
- An angularly controlled numerical candidate for a strictly trapped
  two-sphere. This is a trapped-section result, not yet a coordinate-refined
  apparent-horizon reconstruction.

AI-assisted software development was used for implementation, refactoring,
testing, and documentation. Mathematical conventions, iteration design,
experiment definitions, acceptance criteria, and interpretation are fixed by
the research specification and checked through exact benchmarks, convergence
studies, and independent residuals.

## Numerical experiments

| Experiment | Program entrance | Purpose and retained result |
|---|---|---|
| 1. Regular vacuum benchmarks | `nee.experiments.exp01_regular_vacuum` | Evolves regular Schwarzschild and Kerr data on short and long double-null rectangles. Numerical section metrics are compared directly with the exact metrics, followed by an independent vacuum-curvature audit. |
| 2. Schwarzschild horizon | `nee.experiments.exp02_schwarzschild_horizon` | Approaches the horizon in static coordinates and crosses it in regular Kruskal coordinates on `-1 <= u <= -0.5` and `0 <= v <= 0.5`. The experiment separates coordinate degeneration from genuine curvature error. |
| 3. Schwarzschild interior | `nee.experiments.exp03_schwarzschild_interior` | Evolves entirely inside the horizon toward future boundaries with exact radii from `1` down to `1/32`. The calculation records the increasing metric error and iteration difficulty near the curvature singularity. |
| 4. Strong outgoing vacuum pulse | `nee.experiments.exp04_vacuum_strong_short_pulse` | Prescribes a strong nonspherical trace-free variation of the initial section metric and compares it with an otherwise identical zero control. The protected independent Ricci residual is `2.453e-3` for the pulse and `1.059e-6` for the control. |
| 5. Crossed characteristic shears | `nee.experiments.exp05_vacuum_crossed_pulses` | Prescribes square-root profiles on both initial null hypersurfaces. The shears are finite, while the corresponding transverse curvature is unbounded at the corner. The full rectangle is evolved by slab continuation and audited component by component. |
| 6. Exact Einstein--scalar benchmarks | `nee.experiments.exp06_regular_exact_scalar` | Evolves four Fisher--JNW solutions and the Schwarzschild vacuum limit. On the finest coordinate grid, the maximum section-metric error remains below `5.4e-10` for all four scalar cases. |
| 7. Nonspherical Einstein--scalar data | `nee.experiments.exp07_nonspherical_scalar` | Evolves angularly varying lapse, shift, scalar field, and trace-free metric data with fractional-power behavior at the outgoing corner. The protected independent Einstein--scalar curvature residual decreases under coordinate refinement to a maximum of `3.470e-2`. |
| 8. Scalar-pulse trapped section | `nee.experiments.exp08_scalar_trapped_section` | Evolves a stronger scalar pulse together with anisotropic trace-free metric data. Standard and angular-control runs agree on a finite converged prefix and select the trapped-section candidate `(u, v) = (-0.4696271025, 0.04)`, where the two expansion suprema are `-0.0593348` and `-4.9754530`. |

Experiments without an explicit interior solution are assessed using
independently reconstructed curvature residuals. Low-regularity endpoint
layers and spectral-element interfaces are reported separately from protected
interior statistics; they are not silently replaced or included in a claimed
convergence region.

## Installation

Python 3.12 is required. From the repository root:

```bash
python -m pip install -e .
python -m pytest -q tests/unit tests/integration
```

The runtime dependencies and their retained versions are declared in
`pyproject.toml`.

## Running an experiment

Every experiment has a small smoke configuration and a retained research
configuration. For example:

```bash
python -m nee.experiments.exp01_regular_vacuum \
  --config configs/experiments/exp01/smoke.toml \
  --output results/exp01-smoke
```

Replace `smoke.toml` with `standard.toml` for the research configuration.
Experiment 8 also provides `angular-control.toml`.

The common command-line entrance is:

```bash
nee run exp01 \
  --config configs/experiments/exp01/smoke.toml \
  --output results/exp01-smoke
```

Existing output directories are rejected unless `--resume` is supplied.
Resume mode revalidates the resolved configuration and saved content hashes
before accepting an earlier checkpoint or completed run.

Smoke runs are intended for installation and interface checks. The standard
experiments range from minutes to many hours and the larger nonspherical cases
require several gigabytes of memory. Run standard campaigns sequentially
unless the machine has ample memory.

## Project layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── configs/
│   └── experiments/       Validated smoke and research configurations
├── docs/
│   └── article.pdf        Research article
├── scripts/               Analysis and figure-generation helpers
├── src/
│   └── nee/               Installable numerical package
├── tests/
│   ├── unit/              Geometry, discretization, and identity tests
│   ├── integration/       Public workflow and serialization tests
│   └── regression/        Numerical regression tests
└── results/               Local run output; ignored by Git
```

The main package is divided by responsibility:

- `src/nee/state/`: typed fields, immutable boundary data, Picard iterates,
  and validation.
- `src/nee/geometry/`: sphere geometry, tensor operations, connections, null
  geometry, and curvature reconstruction.
- `src/nee/discretization/`: LGL meshes, double-null grids, spherical
  harmonics, quadrature, projection, and independent overgrids.
- `src/nee/equations/`: vacuum and Einstein--scalar equation providers.
- `src/nee/initial_data/`: characteristic constraints, exact extraction,
  short pulses, crossed data, and scalar-pulse construction.
- `src/nee/solver/`: Picard sweeps, relaxation, stopping rules, and slab
  continuation.
- `src/nee/diagnostics/`: exact comparisons, construction defects,
  independent curvature audits, convergence regions, and trapped-section
  searches.
- `src/nee/exact_solutions/`: Minkowski, Schwarzschild, Kerr, and Fisher--JNW
  data.
- `src/nee/io/`: deterministic artifacts, checkpoints, manifests, and
  provenance.
- `src/nee/experiments/`: the eight public experiment packages and shared
  campaign support.
- `src/nee/numerics/`: retained low-level spectral and evolution kernels used
  by the public components above.

## Run outputs

A completed run writes:

```text
resolved-config.toml
manifest.json
boundary-data.npz
final-state.npz
summary.json
residual-maps.npz
figures/
run.log
```

Raw numerical output under `results/` is intentionally not committed. The
manifest records the resolved configuration, code revision, dependency and
platform information, array schemas, and content hashes needed to audit a
run.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The source code and
documentation are distributed under the MIT License; see `LICENSE`.
