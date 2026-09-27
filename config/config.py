from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class ResolutionBand:
    r_min: float
    r_max: float
    cell_size: float

    def __post_init__(self):
        if self.r_min < 0 or self.r_max <= self.r_min:
            raise ValueError("Invalid band: r_min must be >= 0 and r_max > r_min")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")


# Default SemanticKITTI importance map (higher = more important for refinement)
DEFAULT_IMPORTANCE_MAP: Dict[int, int] = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
    6: 1, 7: 1, 8: 1,          # person, bicyclist, motorcyclist
    9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
    15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
}


@dataclass
class FoveatedConfig:
    # ---- Resolution / coverage ----
    bands: List[ResolutionBand] = field(default_factory=lambda: [
        ResolutionBand(0.0, 10.0, 0.05),
        ResolutionBand(10.0, 25.0, 0.10),
        ResolutionBand(25.0, 50.0, 0.25),
        ResolutionBand(50.0, 100.0, 0.50),
    ])
    max_range: float = 100.0
    sensor_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ---- Aggregation ----
    aggregation: str = "mean"
    min_points_per_cell: int = 1
    use_confidence_weighting: bool = True
    store_histogram: bool = True

    # ---- Traversability / dynamic ----
    obstacle_height_threshold: float = 0.30
    drivable_ratio_threshold: float = 0.50
    dynamic_ratio_threshold: float = 0.50

    # ---- Sentinels ----
    unknown_sentinel: int = 255
    unknown_ratio_sentinel: float = -1.0
    unknown_confidence_sentinel: float = -1.0
    unknown_raw_class_sentinel: int = -1

    # ---- Risk-aware ----
    risk_aware: bool = False
    importance_map: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_IMPORTANCE_MAP))
    min_refinement_size: float = 0.05

    def validate(self):
        for i in range(len(self.bands) - 1):
            if self.bands[i].r_max != self.bands[i + 1].r_min:
                raise ValueError(
                    f"Band {i} r_max ({self.bands[i].r_max}) != "
                    f"Band {i+1} r_min ({self.bands[i+1].r_min})"
                )
        if self.bands[-1].r_max != self.max_range:
            raise ValueError(
                f"Last band r_max ({self.bands[-1].r_max}) != max_range ({self.max_range})"
            )
        base = min(b.cell_size for b in self.bands)
        for b in self.bands:
            ratio = b.cell_size / base
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"Cell size {b.cell_size} is not a multiple of base {base}"
                )


@dataclass
class UniformConfig:
    resolution: float = 0.05
    max_range: float = 100.0
    aggregation: str = "mean"
    min_points_per_cell: int = 1
    use_confidence_weighting: bool = True
    store_histogram: bool = True

    obstacle_height_threshold: float = 0.30
    drivable_ratio_threshold: float = 0.50
    dynamic_ratio_threshold: float = 0.50

    unknown_sentinel: int = 255
    unknown_ratio_sentinel: float = -1.0
    unknown_confidence_sentinel: float = -1.0
    unknown_raw_class_sentinel: int = -1