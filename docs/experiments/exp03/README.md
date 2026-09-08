# Experiment 3: Schwarzschild interior

Six future boundaries, \(\epsilon=2^{-1},\ldots,2^{-6}\), approach the
Schwarzschild singularity on 17, 33, and 49 point grids. The mapped boundary
has exact radius \(r=2M\epsilon\); the singular boundary itself is excluded.
A case receives at most 20 Picard sweeps. Dyadic backtracking preserves a
positive radius when an undamped update would leave the positive-radius region.

The primary error is \(\sqrt2|r_{\rm num}^2-r_{\rm exact}^2|\). Supplementary
residuals use first derivatives of the stored spherical weighted forms.
Finite output from the stabilized near-singularity cases does not establish
convergence or accuracy there. The article reports that limitation explicitly.

See the regenerated [table](../../article/tables/exp03-rows.tex) and
[numerical provenance](../../article/tables/exp03-statistics.json).
