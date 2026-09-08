# Experiment 5: vacuum crossed characteristic shears

The prescribed shears are
\(\Omega\widehat\chi(-1,v)=v^{1/2}\mathcal T_{X_1}[g]\) and
\(\Omega\widehat{\underline\chi}(u,0)=(u+1)^{1/2}\mathcal T_{X_2}[g]\).
Their derivatives along the initial null faces can diverge at the corner.
The corresponding extreme curvature components are not evaluated to form the
Ricci diagnostic.

The standard experiment uses unit amplitudes, six degree-ten square-root
elements in each coordinate, 300 angular points, and retained/work degrees
8/14. Each slab receives at most 30 sweeps and must settle at tolerance
\(10^{-8}\). Smoke mode uses amplitudes 0.1 and a smaller angular/coordinate grid.

The displayed aggregate sums six weighted sectionwise Ricci norms. The 44
Raychaudhuri defect is reported separately. The protected mask removes the
first and last complete elements and three nodes on both sides of each
remaining interface, leaving 144 sections in the standard experiment.
The open-grid statistics and plot exclude the four outer endpoint lines.
Large unprotected residuals indicate differentiation error, independently of
the limiting behavior of the extreme curvature components.

See the regenerated [table](../../article/tables/exp05-rows.tex),
[provenance](../../article/tables/exp05-statistics.json), and
[revision report](../revision-2026-09-08.md).

![Six-component residual spectrum](results/residual-spectrum.png)
