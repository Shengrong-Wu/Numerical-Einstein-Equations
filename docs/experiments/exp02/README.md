# Experiment 2: Schwarzschild horizon coordinates

## Setup

Static Schwarzschild charts approach the horizon with offsets
\(10^{-1},10^{-2},10^{-4},10^{-6}\). A Kruskal chart crosses it on the principal
Lambert-\(W\) branch. Each family is run on 17, 33, and 65 coordinate nodes with
eight Picard sweeps on
\(-1\leq u\leq-1/2,\ 0\leq v\leq1/2\). In shifted Kruskal coordinates this is
\(-1/4\leq U\leq1/4,\ 1\leq V\leq3/2\), covering
\(1.67549\leq r\leq2.24420\) for \(M=1\).

## Diagnostics and result

All 15 enlarged-domain cases completed. On the finest grid, the Kruskal
section-metric error has median \(1.134\times10^{-13}\), mean
\(1.122\times10^{-12}\), and maximum \(2.923\times10^{-11}\). The
35-decimal-digit local stencil audit for the Kruskal sequence decreases

\[
3.99034\times10^{-6}\;\longrightarrow\;1.48083\times10^{-7}
\;\longrightarrow\;7.86204\times10^{-9}.
\]

The near-horizon static residual does not decrease uniformly: the static chart
becomes ill-conditioned as its Omega degenerates. Those values are reported as
a coordinate limitation, not evidence of a physical singularity. The local
arbitrary-precision stencil supersedes an ill-conditioned global-polynomial
diagnostic.
