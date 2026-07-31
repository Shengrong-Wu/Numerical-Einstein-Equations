# Experiment 3: Schwarzschild interior

## Setup

Six future boundaries, \(\epsilon=2^{-1},\ldots,2^{-6}\), approach \(r=0\)
from inside Schwarzschild on 17, 33, and 49 point grids. The singular boundary
itself is excluded from all masks. When an undamped curved-coordinate Picard
step would leave the positive-radius cone, dyadic backtracking selects the
largest positive, nonincreasing update.

## Result and limitation

All 18 computations produced finite artifacts. Thirteen cases that already
admitted an undamped positive iteration match 130 retained arrays containing
123,954 values bitwise. The stabilization permits the remaining five cases to
finish computationally, but it does not make them scientifically reliable.

At \(\epsilon=1/64\) on the finest grid, the radius relative error is 16.43,
the future-boundary radius error is 0.513, and the independently overgridded
masked residual is \(9.78\times10^5\). No convergence or accuracy claim is made
for those near-singularity cases.
