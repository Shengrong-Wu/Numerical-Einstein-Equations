# Experiment 8: scalar trapped-section formation

## Setup

The power-law scalar pulse uses \(\kappa=0.25\), \(\delta=0.1\), scale 0.01,
signed scalar amplitude \(-2\), and quadrupolar shear amplitude 0.1, so
\[
 e_4\phi(-1,v)-e_4\phi(-1,0)
 =-2\left(\frac{v}{v+0.01}\right)^{0.1}.
\]
A 12-element
degree-eight tau domain is extended by one whole element through exact-prefix
prolongation; the transverse direction has three degree-eleven elements. The
standard band is \(L=5,W=10,N=170\), and the control is
\(L=7,W=14,N=300\). Both independently resample the two expansion suprema on
1000 sphere points with a three-node endpoint halo.

## Fresh negative-pulse result

Both production boundary constructions complete with finite arrays. On the
outgoing face, \(e_4\phi\) ranges from \(-2.2627417\) at the corner to
\(-4.2186072\) at \(v=0.04\), while the constructed outgoing expansion remains
positive with minimum \(0.880366\). Thus the initial faces themselves contain
no trapped section.

The standard and angular-control evolutions both fail before completing base
Picard sweep 1. The coupled outgoing Raychaudhuri/metric RK4 march overflows,
and the nonfinite-state guard stops the runs after 53.21 and 95.57 seconds,
respectively. The same failure at \((L,W,N)=(5,10,170)\) and \((7,14,300)\)
shows that it is not removed by the tested angular refinement.

## Limitation

Changing the sign makes the outgoing scalar derivative substantially more
negative and strengthens the focusing source \(-(e_4\phi)^2\). The present
fixed-grid Picard/RK discretization cannot evolve this datum across the base
domain. No continued state, expansion-sign audit, trapped-section candidate,
or certificate is produced. The overflow is consistent with severe focusing
or a double-null coordinate breakdown, but it is not by itself evidence of a
physical trapped surface.
