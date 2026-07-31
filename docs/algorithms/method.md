# Numerical method

## State and immutable data

The interior iterate stores the complete weighted first-order state
`(γ, b, log Ω, Ωχ, Ωχ̄, ζ, Ωω, Ωω̄, φ, Ωe₃φ, Ωe₄φ, ∇φ)`. Traces and shears are
derived from the current metric; they are never independent persistent fields.
The two characteristic faces live in a separate read-only `BoundaryData`
object. Their serialized hash is verified after every sweep.

## Picard order

One sweep evaluates the outgoing shear/scalar half step, lapse coefficients,
lapse, torsion, shift, sphere metric and outgoing form, incoming form, and
incoming scalar reconstruction in that order. Boundary traces are reimposed
before validation. The public function returns a new state and never mutates
its state or boundary arguments.

## Discretization

Coordinates use Legendre--Gauss--Lobatto spectral elements. Composite elements
share interface nodes and declare their differentiation rule explicitly.
Angular fields use equal-area sphere samples with pole-free ambient tangent
frames. Nonlinear expressions are evaluated in a declared work band and
projected into the retained spherical-harmonic band.

## Audits

Construction defects, reconstructed first-order Ricci components, independent
four-metric residuals, and exact-solution errors are reported separately.
Reliability masks exclude declared coordinate endpoints, element-interface
stencils, and singular physical derivatives. Experiment 8 additionally
resamples both expansion suprema on an independent 1000-point sphere and
applies a three-node continuation-endpoint halo.
