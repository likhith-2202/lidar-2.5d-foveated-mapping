"""
CellData: represents a single map cell.
All thresholds and sentinels are passed in from config.
"""

import numpy as np
from typing import Dict, Any, Optional
from .elevation import ElevationStats
from .semantics import SemanticStats


class CellData:
    __slots__ = [
        'band_id', 'cell_size', 'ix', 'iy', 'center_x', 'center_y',
        'points_z', 'points_class', 'points_conf',
        'points_drivable', 'points_dynamic', 'points_intensity',
        '_elevation', '_semantics', '_point_count',
        'frame_ids', 'timestamps',
        # configurable fields
        'unknown_sentinel',
        'unknown_ratio_sentinel',
        'dynamic_ratio_threshold',
        'obstacle_height_threshold',
        'drivable_ratio_threshold',
        'use_confidence_weighting',
    ]

    def __init__(self, band_id: int, cell_size: float, ix: int, iy: int,
                 unknown_sentinel: int = 255,
                 unknown_ratio_sentinel: float = -1.0,
                 dynamic_ratio_threshold: float = 0.5,
                 obstacle_height_threshold: float = 0.30,
                 drivable_ratio_threshold: float = 0.50,
                 use_confidence_weighting: bool = True):
        self.band_id = band_id
        self.cell_size = cell_size
        self.ix = ix
        self.iy = iy
        self.center_x = (ix + 0.5) * cell_size
        self.center_y = (iy + 0.5) * cell_size

        self.unknown_sentinel = unknown_sentinel
        self.unknown_ratio_sentinel = unknown_ratio_sentinel
        self.dynamic_ratio_threshold = dynamic_ratio_threshold
        self.obstacle_height_threshold = obstacle_height_threshold
        self.drivable_ratio_threshold = drivable_ratio_threshold
        self.use_confidence_weighting = use_confidence_weighting

        self.points_z = []
        self.points_class = []
        self.points_conf = []
        self.points_drivable = []
        self.points_dynamic = []
        self.points_intensity = []
        self.frame_ids = []
        self.timestamps = []
        self._point_count = 0
        self._elevation = None
        self._semantics = None

    def add_point(self, z: float, class_id: int, confidence: float,
                  drivable: int = None, dynamic: int = None,
                  intensity: float = 0.0,
                  frame_id: int = 0, timestamp: float = 0.0):
        if drivable is None:
            drivable = self.unknown_sentinel
        if dynamic is None:
            dynamic = self.unknown_sentinel
        self.points_z.append(z)
        self.points_class.append(class_id)
        self.points_conf.append(confidence)
        self.points_drivable.append(drivable)
        self.points_dynamic.append(dynamic)
        self.points_intensity.append(intensity)
        self.frame_ids.append(frame_id)
        self.timestamps.append(timestamp)
        self._point_count += 1

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def elevation(self) -> ElevationStats:
        if self._elevation is None:
            self._elevation = ElevationStats.from_values(np.array(self.points_z))
        return self._elevation

    @property
    def semantics(self) -> SemanticStats:
        if self._semantics is None:
            self._semantics = SemanticStats.from_values(
                np.array(self.points_class),
                np.array(self.points_conf),
                use_confidence_weighting=self.use_confidence_weighting,
            )
        return self._semantics

    @property
    def drivable_ratio(self) -> float:
        arr = np.asarray(self.points_drivable)
        known = arr[arr != self.unknown_sentinel]
        if len(known) == 0:
            return self.unknown_ratio_sentinel
        return float(np.mean(known == 1))

    @property
    def dynamic_ratio(self) -> float:
        arr = np.asarray(self.points_dynamic)
        known = arr[arr != self.unknown_sentinel]
        if len(known) == 0:
            return self.unknown_ratio_sentinel
        return float(np.mean(known == 1))

    @property
    def mean_intensity(self) -> float:
        return float(np.mean(self.points_intensity)) if self.points_intensity else 0.0

    def is_dynamic(self) -> Optional[bool]:
        r = self.dynamic_ratio
        if r < 0:
            return None
        return r > self.dynamic_ratio_threshold

    def traversability(self,
                       obstacle_height_threshold: Optional[float] = None,
                       drivable_ratio_threshold: Optional[float] = None) -> str:
        if obstacle_height_threshold is None:
            obstacle_height_threshold = self.obstacle_height_threshold
        if drivable_ratio_threshold is None:
            drivable_ratio_threshold = self.drivable_ratio_threshold

        if self.point_count == 0:
            return 'unknown'
        h = self.elevation.max - self.elevation.min
        if h > obstacle_height_threshold:
            return 'non_drivable'
        r = self.drivable_ratio
        if r < 0:
            return 'unknown'
        return 'drivable' if r >= drivable_ratio_threshold else 'non_drivable'

    def to_dict(self, include_histogram: bool = True,
                obstacle_height_threshold: Optional[float] = None,
                drivable_ratio_threshold: Optional[float] = None) -> Dict[str, Any]:
        elev = self.elevation
        sem = self.semantics
        d = {
            'band_id': self.band_id,
            'cell_size': self.cell_size,
            'ix': self.ix,
            'iy': self.iy,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'point_count': self.point_count,
            'min_z': elev.min,
            'max_z': elev.max,
            'mean_z': elev.mean,
            'std_z': elev.std,
            'obstacle_height': elev.max - elev.min,
            'dominant_class': sem.dominant_class,
            'dominant_confidence': sem.dominant_confidence,
            'occupancy': 'occupied' if self.point_count > 0 else 'unknown',
            'drivable_ratio': self.drivable_ratio,
            'dynamic_ratio': self.dynamic_ratio,
            'is_dynamic': self.is_dynamic(),
            'mean_intensity': self.mean_intensity,
            'traversability': self.traversability(
                obstacle_height_threshold=obstacle_height_threshold,
                drivable_ratio_threshold=drivable_ratio_threshold,
            ),
        }
        if include_histogram and sem.histogram is not None:
            d['class_histogram'] = sem.histogram
        return d

    def __repr__(self):
        return (f"Cell(band={self.band_id}, size={self.cell_size}, "
                f"ix={self.ix}, iy={self.iy}, pts={self.point_count})")