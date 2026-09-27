"""
Main foveated 2.5D mapper using distance-based resolution bands.
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from .points import normalize_points
from .cell import CellData
from config.config import FoveatedConfig, ResolutionBand


class FoveatedMapper:

    def __init__(self, config: FoveatedConfig):
        self.config = config
        self.config.validate()
        self.bands: List[ResolutionBand] = config.bands
        self.max_range = config.max_range
        self.cells: Dict[Tuple[int, int, int], CellData] = {}
        self.statistics = {}

    def _make_cell(self, band_id: int, ix: int, iy: int) -> CellData:
        return CellData(
            band_id=band_id,
            cell_size=self.bands[band_id].cell_size,
            ix=ix, iy=iy,
            unknown_sentinel=self.config.unknown_sentinel,
            unknown_ratio_sentinel=self.config.unknown_ratio_sentinel,
            dynamic_ratio_threshold=self.config.dynamic_ratio_threshold,
            obstacle_height_threshold=self.config.obstacle_height_threshold,
            drivable_ratio_threshold=self.config.drivable_ratio_threshold,
            use_confidence_weighting=self.config.use_confidence_weighting,
        )

    def build(self, points) -> Dict[Tuple[int, int, int], CellData]:
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
        r = r[mask]
        t2 = time.perf_counter()

        n = len(pts)
        ix = np.empty(n, dtype=np.int32)
        iy = np.empty(n, dtype=np.int32)

        thresholds = [band.r_max for band in self.bands]
        # side='right' so a point exactly at r_max of band i goes into band i+1
        # (half-open interval [r_min, r_max) semantics)
        band_ids = np.searchsorted(thresholds, r, side='right')
        band_ids[band_ids == len(self.bands)] = len(self.bands) - 1

        for bidx, band in enumerate(self.bands):
            mask_band = band_ids == bidx
            if np.any(mask_band):
                ix[mask_band] = np.floor(pts['x'][mask_band] / band.cell_size).astype(np.int32)
                iy[mask_band] = np.floor(pts['y'][mask_band] / band.cell_size).astype(np.int32)
        t3 = time.perf_counter()

        self.cells = {}
        cell_raw = defaultdict(list)
        for i in range(n):
            key = (int(band_ids[i]), int(ix[i]), int(iy[i]))
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
            band_id, ix_cell, iy_cell = key
            cell = self._make_cell(band_id, ix_cell, iy_cell)
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
            'band_assignment_time_ms': (t3 - t2) * 1000,
            'insertion_time_ms': (t4 - t3) * 1000,
            'aggregation_time_ms': (t5 - t4) * 1000,
            'total_time_ms': (t5 - t0) * 1000,
        }
        return self.cells

    def get_cell(self, band_id: int, ix: int, iy: int) -> Optional[CellData]:
        return self.cells.get((band_id, ix, iy))

    def to_list(self) -> list:
        return [
            cell.to_dict(
                include_histogram=self.config.store_histogram,
                obstacle_height_threshold=self.config.obstacle_height_threshold,
                drivable_ratio_threshold=self.config.drivable_ratio_threshold,
            )
            for cell in self.cells.values()
        ]