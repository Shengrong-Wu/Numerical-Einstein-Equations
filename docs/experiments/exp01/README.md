# Experiment 1: regular exact vacuum solutions

Schwarzschild and Kerr metrics with mass one are evolved on short and long
double-null rectangles. Kerr uses rotations 0, 0.3, 0.7, and 0.9 and reference
radius four. Coordinate refinements contain 17, 33, and 65 nodes per axis.
Schwarzschild uses 50 sphere points and retained degree five; Kerr angular
controls use 30, 50, and 86 points with retained degrees three, five, and seven.
Every case receives eight Picard sweeps.

The primary statistic is the pointwise Frobenius error of the section metric
against the exact reference, summarized over all coordinate and sphere points.
First-order residuals and primitive/connection consistency are supplementary
checks; a small Picard update alone is not an accuracy certificate.

The regenerated [article table](../../article/tables/exp01-rows.tex) and
[numerical provenance](../../article/tables/exp01-statistics.json) record the
finest-grid results. See the [revision report](../revision-2026-09-08.md) for
comparisons with retained results and the repeated Kerr calculation.
