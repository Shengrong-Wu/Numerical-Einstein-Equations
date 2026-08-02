# Numerical method

## State and immutable data

The interior iterate stores the complete weighted first-order state
`(g, b, log Ω, Ωχ, Ωχ̄, ζ, Ωω, Ωω̄, φ, Ωe₃φ, Ωe₄φ, ∇φ)`. Traces and shears are
derived from the current `g`; they are never independent persistent fields.
In particular, the outgoing expansion variable is
`Omega_trchi = tr_g(Omega_chi) = Ω tr χ`. No inverse-Ω expansion is stored or
evolved.
The two characteristic faces live in a separate read-only `BoundaryData`
object. Their serialized hash is verified after every sweep.

## Picard order

One sweep evaluates the outgoing `Omega_chih`/scalar half step, the
`Omega_omegab` and `Omega_omega` coefficients, `log_Omega`, `zeta`, `b`, `g`
and `Omega_chi`, `Omega_chib`, and
incoming scalar reconstruction in that order. Boundary traces are reimposed
before validation. The public function returns a new state and never mutates
its state or boundary arguments.

The outgoing Raychaudhuri and metric equations are advanced directly in the
theory-native variable \(A=\Omega\operatorname{tr}\chi\):
\[
 \partial_v A=-\tfrac12 A^2-4(\Omega\omega)A
 -|\Omega\hat\chi|_g^2-(\Omega e_4\phi)^2,
 \qquad
 \partial_v g=A g+2\Omega\hat\chi .
\]
The scalar term is omitted in vacuum runs.

## Discretization

Coordinates use Legendre--Gauss--Lobatto spectral elements. Composite elements
share interface nodes and declare their differentiation rule explicitly.
Angular fields use equal-area sphere samples with pole-free ambient tangent
frames. Nonlinear expressions are evaluated in a declared work band and
projected into the retained spherical-harmonic band.

## Audits

Construction defects, reconstructed first-order Ricci components, independent
four-g residuals, and exact-solution errors are reported separately.
Reliability masks exclude declared coordinate endpoints, element-interface
stencils, and singular physical derivatives. Experiment 8 additionally
resamples both expansion suprema on an independent 1000-point sphere and
applies a three-node continuation-endpoint halo.
