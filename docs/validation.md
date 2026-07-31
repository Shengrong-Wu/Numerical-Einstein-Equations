# Full-run validation record

All commands were run sequentially from the repository root with Python 3.12,
NumPy 2.4.3, SciPy 1.16.3, and Matplotlib 3.10.7 on arm64 macOS. Raw result
directories are intentionally excluded from Git. Peak RSS below normalizes the
platform runtime counter to GiB.

| Run | Exit/status | Wall time (s) | Peak RSS (GiB) | Output |
|---|---:|---:|---:|---:|
| Experiment 1 standard | 0 / completed | 6259.41 | 1.69 | 1.0 GiB |
| Experiment 2 standard | 0 / completed | 1725.25 | 1.28 | 268 MiB |
| Experiment 3 standard | 0 / completed | 150.26 | 0.10 | 1.7 MiB |
| Experiment 4 standard | 0 / completed | 938.21 | 1.74 | 717 MiB |
| Experiment 5 standard | 0 / completed | 3087.59 | 2.37 | 492 MiB |
| Experiment 6 standard | 0 / completed | 3211.11 | 1.49 | 512 MiB |
| Experiment 7 standard | 2 / failed | 2312.45 | 1.83 | 857 MiB |
| Experiment 8 standard | 0 / completed | 1895.46 | 2.36 | 310 MiB |
| Experiment 8 angular control | 0 / completed | 3590.13 | 3.61 | 537 MiB |

Experiment 7 exits nonzero because two extreme stress inputs make the incoming
scalar constraint radicand negative. This is an explicit invalid-data gate; 22
other cases complete.

The finalized Experiment 3 artifact set includes its separately regenerated
and hashed characteristic boundary. An independent repeat preserved 198 arrays
(160,518 values) and 2,205 numerical summary leaves bit for bit; manifest-hash
resume validation also passed.

## Same-grid numerical comparisons

The array gate is rtol=5e-12, atol=5e-13; integers, booleans, masks, shapes,
and statuses are exact. Summary scalars use \(10^{-9}\) relative tolerance, or
explicit \(10^{-12}\) absolute tolerance below \(10^{-12}\).

| Experiment | Compared numerical content | Maximum absolute difference | Outcome |
|---|---:|---:|---|
| 1 | 368 arrays / 184,036,464 values; 46 summaries | 0 | pass |
| 2 | 120 arrays / 50,427,000 values | 0 | pass; 54 corrected local-stencil scalars also exact |
| 3 | 130 arrays / 123,954 values in the 13 common valid cases | 0 | pass; five stabilized cases have no valid same-status reference |
| 4 | 11 settled-state arrays / 11,416,400 values | \(8.88\times10^{-16}\) | state pass; residual-summary exception |
| 5 | 13 arrays and 865 summary scalars | 0 | pass |
| 6 | 336 numerical/exact arrays / 107,353,392 values; 12 summaries | 0 | pass |
| 7 | 308 mapped-state arrays / 131,901,891 values | \(3.52\times10^{-14}\) | state pass; four derivative arrays and residual summaries fail |

Experiment 4's independent component summary differs by at most
\(1.02\times10^{-7}\) absolute and \(5.26\times10^{-5}\) relative. Experiment
7's state drift is at roundoff scale, while coordinate derivatives amplify it
in stored residual maps and summary grids. These gates remain marked failed;
their tolerances were not widened.

## Scientific interpretation

- Experiments 1, 2, 5, and 6 satisfy their exact/replay controls. The static
  near-horizon chart in Experiment 2 remains conditioning-limited.
- Experiment 3 completes after positivity-preserving backtracking, but its
  closest-to-zero cases have large error and residual and support no accuracy
  claim.
- Experiment 4 has a settled strong-pulse state and exact zero-control audit,
  but the derivative-amplified residual-summary replay is an open limitation.
- Experiment 7 passes coordinate/angular refinement of the protected current
  audit, while its two invalid stress probes and diagnostic replay exception
  remain visible.
- Experiment 8 finds overlapping negative-expansion cells at both angular
  bands, but neither continuation settles below \(10^{-4}\), the protected
  residuals remain large, and no coordinate refinement is available. It is not
  a candidate or certificate.
