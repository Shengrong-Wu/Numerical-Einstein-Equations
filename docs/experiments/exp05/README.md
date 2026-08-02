# Experiment 5: vacuum crossed pulses

## Setup

Mild outgoing and incoming shears of amplitude 0.1 are evolved slab by slab on
six degree-ten square-root elements in each null direction. The angular space
uses \(L=8,W=14\) and 300 points. Each slab receives up to 30 sweeps.

## Result

All six slabs settle, with final weighted updates between
\(1.85\times10^{-9}\) and \(8.69\times10^{-9}\). The six terminal updates are
\(1.850,4.897,8.694,4.404,6.984,4.761\times10^{-9}\). The protected
six-component residual including the 44 equation is 0.00508699, and the
minimum `g` eigenvalue is 0.0293816. All immutable incoming and outgoing traces
are restored exactly after each slab sweep, up to a
\(1.78\times10^{-15}\) incoming `Omega_trchi` roundoff error.

![Six-component convergence regions](results/convergence-regions.png)

The gray region is excluded by derivative and interface stencils. Colored
cells are classified without interpolation across spectral-element
interfaces.
