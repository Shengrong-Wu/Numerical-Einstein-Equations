# Weighted state and derivative conventions

The evolved forms are X = Omega chi and Xbar = Omega chibar, with
W = Omega omega and Wbar = Omega omegabar. Both traces use the current
sphere metric. The scalar null derivatives are P4 = partial_v phi and
P3 = (partial_u + b) phi.

For a scalar f the product rule gives

    Omega^2 nabla_4 f = partial_v(Omega f) + 2 W (Omega f),
    Omega^2 nabla_3 f = (partial_u + b)(Omega f) + 2 Wbar (Omega f).

Consequently, writing A = tr(X), Abar = tr(Xbar), the characteristic
Raychaudhuri equations for Ric = dphi tensor dphi are

    partial_v A + A^2/2 + 4 W A + |hat(X)|^2 + P4^2 = 0,
    (partial_u + b) Abar + Abar^2/2 + 4 Wbar Abar + |hat(Xbar)|^2 + P3^2 = 0.

Set the matter terms to zero for vacuum. The coefficient 4 contains both
the original connection term and the product-rule contribution. Tensor
transport additionally includes the covariant-derivative connection terms;
it cannot be replaced by componentwise scalar transport.

The lapse definitions are partial_v log(Omega) = -2 W and
(partial_u + b) log(Omega) = -2 Wbar. When the shift is updated, convert the
half-step incoming coefficient by

    Wbar_new = Wbar_half - (b_new - b_old) . grad(log(Omega_new))/2,

with the same angular projection as the sweep. Restore the immutable
characteristic traces after projection and relaxation.

## Diagnostics at rough characteristic data

The independent state audit resamples the complete weighted state and takes
first null derivatives of its connection components. It does not construct
second null derivatives of the metric or compute R_4A4B by differentiating
outgoing shear in v. The trace Raychaudhuri diagnostic does not require that
shear derivative: Ric_44 vanishes for the exact vacuum constraint solution,
while its numerical discretization defect can still be measured separately.

Metric/connection and scalar-gradient consistency are reported separately,
using first derivatives. Thus these are first-order state residuals, whose
interpretation also depends on the consistency checks. Intrinsic Gauss
curvature and sphere differential operators act on the smooth angular data;
the restriction concerns differentiating rough data again in a null direction.

Physical null derivatives need not exist at v=0 for the fractional-power
profiles. No endpoint placeholder from a mapped derivative is an accepted
curvature value. Reported residuals exclude the singular endpoints and state
the additional interface and endpoint masks explicitly. Refinement compares
these same prescribed regions.
