# Experiment 1: regular exact vacuum solutions

## Setup

Schwarzschild and Kerr metrics with mass one are evaluated on short and long
regular double-null domains. The Kerr rotations are \(a=0,0.3,0.7,0.9\), with
reference radius four. Coordinate grids use 17, 33, and 65 nodes; angular
controls use 30, 50, and 86 Fibonacci points with retained degrees 3, 5, and 7.

## Diagnostics and result

Every case records exact g closures, fresh first-order null residuals,
an independently reconstructed four-g Ricci tensor, and a deliberately
mutated missing-Lie-derivative control. All 46 standard cases completed. The
same-grid validation compared 368 state arrays containing 184,036,464 values;
every value was bitwise identical to the retained numerical reference.

The finest regular Schwarzschild short-domain independent masked residual is
\(1.74\times10^{-7}\). Rotating Kerr cases have larger reconstructed residuals,
so the exact g/closure comparison—not a small aggregate residual alone—is
the primary correctness check.
