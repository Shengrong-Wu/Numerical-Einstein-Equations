# Experiment 7: nonspherical scalar data

## Setup

Globally smooth real-harmonic lapse, shift, shear, and scalar free data are
completed by the Einstein--scalar characteristic constraints on
\([-1,-1/2]\times[0,0.04]\). Coordinate levels use degrees
\((6,9),(8,11),(8,11)\) with 2, 2, and 3 elements. Angular levels use
\((L,W,N)=(6,12,200),(8,16,350),(10,20,550)\). Six sweeps are followed by a
rotated control and stress matrix.

## Result and limitation

The coordinate protected residual decreases
\(28.18\to0.429\to0.0650\), and the angular sequence decreases
\(0.529\to0.429\to0.411\); closure envelopes also decrease. All five refinement
updates finish below \(10^{-6}\). Across the 22 valid cases, 308 mapped state
arrays containing 131,901,891 values pass the array gate, with maximum absolute
difference \(3.52\times10^{-14}\).

Two extreme stress probes have a negative incoming-scalar constraint radicand
and are correctly rejected, so the aggregate command exits 2. Four
derivative/update-map arrays and derivative-amplified residual-summary values
do not pass the replay tolerance. The state/refinement gate passes, but the full
diagnostic replay gate does not.
