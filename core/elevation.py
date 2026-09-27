"""
Elevation statistics for a cell.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class ElevationStats:
    """
    Contains summary statistics of elevations in a cell.
    """
    min: float
    max: float
    mean: float
    std: float
    count: int

    @classmethod
    def from_values(cls, z_values: np.ndarray) -> 'ElevationStats':
        if len(z_values) == 0:
            # Empty cell: use NaN or 0? We'll use 0 and count 0.
            return cls(0.0, 0.0, 0.0, 0.0, 0)
        z = np.asarray(z_values)
        return cls(
            min=float(np.min(z)),
            max=float(np.max(z)),
            mean=float(np.mean(z)),
            std=float(np.std(z)),
            count=int(len(z))
        )

    def to_dict(self):
        return {
            'min_z': self.min,
            'max_z': self.max,
            'mean_z': self.mean,
            'std_z': self.std,
            'point_count': self.count
        }