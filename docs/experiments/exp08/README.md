# Experiment 8: scalar trapped-section formation

## Setup

The power-law scalar pulse uses \(\kappa=0.25\), \(\delta=0.1\), scale 0.01,
signed scalar amplitude \(-2\), and quadrupolar Omega_chih amplitude 0.1, so
\[
 e_4\phi(-1,v)-e_4\phi(-1,0)
 =-2\left(\frac{v}{v+0.01}\right)^{0.1}.
\]
A 12-element degree-eight tau domain is first tested through
\(u=-0.02\); the transverse direction has three degree-eleven elements. The
standard band is \(L=5,W=10,N=170\), and the angular control is
\(L=7,W=14,N=300\). Both independently resample the two expansion suprema on
1000 sphere points with a three-node endpoint halo.

## Fresh negative-pulse result

Both production boundary constructions complete with finite arrays. On the
outgoing face, \(e_4\phi\) ranges from \(-2.2627417\) at the corner to
\(-4.2186072\) at \(v=0.04\), while the constructed outgoing expansion remains
positive with minimum \(0.880366\). Thus the initial faces themselves contain
no trapped section.

The complete 12-element standard and angular-control domains encounter the
same Raychaudhuri pole before completing sweep 1. Each run records that caustic
and then recomputes on the exact four-element prefix ending at
\(u=-0.2714417617\). This retains the original nodes and operation order on
those elements while excluding the later coordinate singularity.

Both prefix runs complete all 14 sweeps. Their final updates are
\(4.09\times10^{-16}\) and \(6.76\times10^{-16}\). Each independent
1000-point sign audit finds 32 trapped grid sections and 23 after the
three-node endpoint halo. The first protected candidate is
\[
 (u,v)=(-0.4696271025,0.04),
\]
where the standard outgoing and incoming expansion suprema are
\(-0.0593348\) and \(-4.9754530\). At the innermost retained corner the
corresponding sign margins are 3.62907 and 10.5650. The angular control has the
same candidate indices and changes these margins by less than
\(1.7\times10^{-5}\).

## Limitation

The full-domain overflow is a double-null caustic and is not itself used as
evidence. The trapped-section claim comes only from the converged finite-prefix
state and its independent angular resampling. Standard and angular-control
sign results agree, but a separate coordinate-refinement certificate has not
yet been run; the summaries therefore label the result a numerical candidate
rather than a refinement certificate.
