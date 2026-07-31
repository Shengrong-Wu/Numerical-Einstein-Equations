# Experiment 2: Schwarzschild horizon coordinates

## Setup

Static Schwarzschild charts approach the horizon with offsets
\(10^{-1},10^{-2},10^{-4},10^{-6}\). A Kruskal chart crosses it on the principal
Lambert-\(W\) branch. Each family is run on 17, 33, and 65 coordinate nodes with
eight Picard sweeps.

## Diagnostics and result

All 15 cases completed, and 120 state arrays containing 50,427,000 values were
bitwise identical in the same-grid validation. The 35-decimal-digit local
stencil audit for the Kruskal sequence decreases

\[
6.22449\times10^{-7}\;\longrightarrow\;2.83680\times10^{-8}
\;\longrightarrow\;6.98740\times10^{-9}.
\]

The near-horizon static residual does not decrease uniformly: the static chart
becomes ill-conditioned as its lapse degenerates. Those values are reported as
a coordinate limitation, not evidence of a physical singularity. The local
arbitrary-precision stencil supersedes an ill-conditioned global-polynomial
diagnostic.
