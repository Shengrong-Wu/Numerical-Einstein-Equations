# Experiment 3: Schwarzschild interior

The spherical interior solver approaches but never includes the singular
boundary. It uses six distances `2^-1` through `2^-6` and coordinate counts
17, 33, and 49. Errors are reported both absolutely and relative to the exact
interior radius, together with protected-region residuals.
