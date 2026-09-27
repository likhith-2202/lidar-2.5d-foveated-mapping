"""
Uniform-resolution 2.5D baseline mapper.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from collections import defaultdict
from .points import normalize_points
from .cell import CellData
from config.config import UniformConfig


class UniformMapper:

    def __init__(self, config: UniformConfig):
        self.config = config
        self.resolution = config.resolution
        self.max_range = config.max_range
        self.cells: Dict[Tuple[int, int], CellData] = {}
        self.statistics = {}

    def _make_cell(self, ix: int, iy: int) -> CellData:
        return CellData(
            band_id=0, cell_size=self.resolution, ix=ix, iy=iy,
            unknown_sentinel=self.config.unknown_sentinel,
            unknown_ratio_sentinel=self.config.unknown_ratio_sentinel,
            dynamic_ratio_threshold=self.config.dynamic_ratio_threshold,
            obstacle_height_threshold=self.config.obstacle_height_threshold,
            drivable_ratio_threshold=self.config.drivable_ratio_threshold,
            use_confidence_weighting=self.config.use_confidence_weighting,
        )

    def build(self, points) -> Dict[Tuple[int, int], CellData]:
        t0 = time.perf_counter()
        pts = normalize_points(
            points,
            unknown_sentinel=self.config.unknown_sentinel,
            unknown_confidence_sentinel=self.config.unknown_confidence_sentinel,
            unknown_raw_class_sentinel=self.config.unknown_raw_class_sentinel,
        )
        t1 = time.perf_counter()

        r = np.sqrt(pts['x'] ** 2 + pts['y'] ** 2)
        mask = r <= self.max_range
        pts = pts[mask]
        t2 = time.perf_counter()

        ix = np.floor(pts['x'] / self.resolution).astype(np.int32)
        iy = np.floor(pts['y'] / self.resolution).astype(np.int32)
        t3 = time.perf_counter()

        self.cells = {}
        cell_raw = defaultdict(list)
        for i in range(len(pts)):
            key = (int(ix[i]), int(iy[i]))
            cell_raw[key].append((
                float(pts['z'][i]),
                int(pts['class_id'][i]),
                float(pts['confidence'][i]),
                int(pts['drivable'][i]),
                int(pts['dynamic'][i]),
                float(pts['intensity'][i]),
                int(pts['frame_id'][i]),
                float(pts['timestamp'][i]),
            ))
        t4 = time.perf_counter()

        for key, data_list in cell_raw.items():
            cell = self._make_cell(key[0], key[1])
            for z, cid, conf, drv, dyn, inten, fid, ts in data_list:
                cell.add_point(z, cid, conf, drv, dyn, inten, fid, ts)
            self.cells[key] = cell
        t5 = time.perf_counter()

        self.statistics = {
            'num_points_in': len(pts),
            'num_points_filtered': len(points) - len(pts) if hasattr(points, '__len__') else 0,
            'num_cells': len(self.cells),
            'preprocess_time_ms': (t1 - t0) * 1000,
            'filter_time_ms': (t2 - t1) * 1000,
            'projection_time_ms': (t3 - t2) * 1000,
            'insertion_time_ms': (t4 - t3) * 1000,
            'aggregation_time_ms': (t5 - t4) * 1000,
            'total_time_ms': (t5 - t0) * 1000,
        }
        return self.cells

    def get_cell(self, ix: int, iy: int) -> Optional[CellData]:
        return self.cells.get((ix, iy))

    def to_list(self) -> list:
        return [
            cell.to_dict(
                include_histogram=self.config.store_histogram,
                obstacle_height_threshold=self.config.obstacle_height_threshold,
                drivable_ratio_threshold=self.config.drivable_ratio_threshold,
            )
            for cell in self.cells.values()
        ]