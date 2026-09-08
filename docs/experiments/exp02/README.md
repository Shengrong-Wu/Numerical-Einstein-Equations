# Experiment 2: Schwarzschild horizon coordinates

Static Schwarzschild charts approach the horizon with offsets
\(10^{-1},10^{-2},10^{-4},10^{-6}\). A Kruskal chart crosses it on the principal
Lambert-W branch. Each case uses 17, 33, and 65 nodes per coordinate, 50 sphere
points, retained angular degree five, and eight Picard sweeps on
\([-1,-1/2]\times[0,1/2]\).

The Kruskal offsets give \(-1/4\le U\le1/4\), \(1\le V\le3/2\), with
\(1.67549\le r\le2.24420\) for mass one. The primary error is the section
metric's Frobenius difference from the exact solution. The supplementary
residual uses first derivatives of weighted connection variables. The
former metric-second-derivative, arbitrary-precision diagnostic is omitted.
Static-chart conditioning must be distinguished from physical curvature
at the regular horizon.

See the regenerated [table](../../article/tables/exp02-rows.tex),
[provenance](../../article/tables/exp02-statistics.json), and
[derivative conventions](../numerical-conventions.md).
