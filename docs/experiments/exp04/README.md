# Experiment 4: vacuum strong short pulse

## Setup

A low-band outgoing Omega_chih pulse of unit strength and divisor 3.2 is evolved to
\(v=0.005\) with \(L=10,W=20\), 550 angular points, four square-root coordinate
elements, and six complete Picard sweeps. The identical zero-strength run is a
control.

## Result and limitation

The fresh strong and numerical-zero runs complete all six sweeps. Their final
weighted updates are \(1.68823\times10^{-10}\) and
\(2.92684\times10^{-13}\), respectively. Every boundary SDC element is
accepted; the strong-pulse maximum collocation and overgrid defects are
\(6.36\times10^{-16}\) and \(1.47\times10^{-8}\).

On the protected independent four-metric audit, the strong-pulse combined
residual is 0.00245259. The numerical zero control is
\(1.05944\times10^{-6}\) on the broadest protected region and decreases to
\(1.11450\times10^{-7}\) away from the short-pulse endpoint.

Using the evolved metric and area form, the strong datum has
\(\|\widehat\chi(-1,0.005)\|_{L^2}=3.75480\), with pointwise maximum
\(1.26734\). Thus the strong-pulse residual is not the result of an
effectively vanishing perturbation.
