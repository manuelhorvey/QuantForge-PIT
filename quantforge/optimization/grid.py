from __future__ import annotations
from itertools import product


def parameter_grid(grid: dict[str, list]):
    """Deterministic parameter enumeration; log every emitted configuration as a trial."""
    if not grid or any(not values for values in grid.values()):
        raise ValueError("grid and each parameter's values must be nonempty")
    keys = sorted(grid)
    for values in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))
