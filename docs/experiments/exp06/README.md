# Experiment 6: regular exact scalar solutions

## Setup

Regular Fisher/JNW data use \(\sigma=1\) and
\(\nu=0.99,0.8,0.5,0.2\) on 2, 4, and 8 degree-six elements. Each case runs
eight Einstein--scalar Picard sweeps. The separate \(\nu=1\) control must have
vanishing scalar variables and Schwarzschild mass \(1/2\).

## Result

All 12 numerical cases completed and the vacuum-limit control passed. Numerical
and exact state archives contribute 336 mapped arrays and 107,353,392 values;
all are bitwise identical in the same-grid validation, as are all 12 summaries.
The finest \(\nu=0.99\) scalar-field maximum absolute error is
\(1.14\times10^{-14}\), and the sphere-metric maximum absolute error is
\(2.09\times10^{-12}\).
