# Experiment 4: vacuum strong short pulse

## Setup

A low-band outgoing shear pulse of unit strength and divisor 3.2 is evolved to
\(v=0.005\) with \(L=10,W=20\), 550 angular points, four square-root coordinate
elements, and six complete Picard sweeps. The identical zero-strength run is a
control.

## Result and limitation

The strong and numerical-zero runs complete all sweeps. Eleven settled-state
arrays containing 11,416,400 values pass the same-grid array gate, with maximum
absolute difference \(8.88\times10^{-16}\). An exact Minkowski component audit
is bitwise identical. The strong-pulse independently masked component sum is
0.0395199919.

The reconstructed component summary differs from the retained summary by up to
\(1.02\times10^{-7}\) absolute and \(5.26\times10^{-5}\) relative because tiny
state perturbations are amplified by coordinate differentiation. The state
array gate passes, but the residual-summary replay gate does not; this is
reported without widening the tolerance.
