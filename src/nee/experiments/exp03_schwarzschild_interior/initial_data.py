"""Exact spherical interior data terminated before radius zero."""

def epsilon_sequence() -> tuple[float, ...]:
    return tuple(2.0 ** -power for power in range(1, 7))
