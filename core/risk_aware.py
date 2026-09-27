"""
Risk-aware foveation extension.
Importance map, refinement size, and thresholds all come from config.
"""

import numpy as np
from typing import Dict
from .foveated_mapper import FoveatedMapper
from .cell import CellData
from .points import normalize_points
from config.config import FoveatedConfig


class RiskAwareMapper:

    def __init__(self, base_config: FoveatedConfig,
                 importance_map: Dict[int, int] = None,
                 min_refinement_size: float = None):
        self.base_config = base_config
        self.base_mapper = FoveatedMapper(base_config)
        self.importance_map = (importance_map
                               if importance_map is not None
                               else base_config.importance_map)
        self.min_refinement_size = (min_refinement_size
                                    if min_refinement_size is not None
                                    else base_config.min_refinement_size)

    def build(self, points):
        base_cells = self.base_mapper.build(points)
        return self._refine_cells(base_cells, points)

    def _make_cell(self, band_id: int, cell_size: float, ix: int, iy: int) -> CellData:
        return CellData(
            band_id=band_id, cell_size=cell_size, ix=ix, iy=iy,
            unknown_sentinel=self.base_config.unknown_sentinel,
            unknown_ratio_sentinel=self.base_config.unknown_ratio_sentinel,
            dynamic_ratio_threshold=self.base_config.dynamic_ratio_threshold,
            obstacle_height_threshold=self.base_config.obstacle_height_threshold,
            drivable_ratio_threshold=self.base_config.drivable_ratio_threshold,
            use_confidence_weighting=self.base_config.use_confidence_weighting,
        )

    def _refine_cells(self, base_cells, points):
        pts = normalize_points(
            points,
            unknown_sentinel=self.base_config.unknown_sentinel,
            unknown_confidence_sentinel=self.base_config.unknown_confidence_sentinel,
            unknown_raw_class_sentinel=self.base_config.unknown_raw_class_sentinel,
        )
        refined_cells = {}

        cells_to_refine = set()
        for key, cell in base_cells.items():
            for cid in cell.points_class:
                if self.importance_map.get(cid, 0) > 0:
                    cells_to_refine.add(key)
                    break

        if not cells_to_refine:
            return base_cells

        r = np.sqrt(pts['x'] ** 2 + pts['y'] ** 2)
        thresholds = [band.r_max for band in self.base_config.bands]
        # side='right' for half-open [r_min, r_max) band semantics
        band_ids = np.searchsorted(thresholds, r, side='right')
        band_ids[band_ids == len(self.base_config.bands)] = len(self.base_config.bands) - 1

        for i in range(len(pts)):
            bidx = int(band_ids[i])
            band = self.base_config.bands[bidx]
            ix_b = int(np.floor(pts['x'][i] / band.cell_size))
            iy_b = int(np.floor(pts['y'][i] / band.cell_size))
            key = (bidx, ix_b, iy_b)

            if key in cells_to_refine:
                sub_size = self.min_refinement_size
                sub_ix = int(np.floor(pts['x'][i] / sub_size))
                sub_iy = int(np.floor(pts['y'][i] / sub_size))
                sub_key = (bidx, sub_ix, sub_iy)
                if sub_key not in refined_cells:
                    refined_cells[sub_key] = self._make_cell(bidx, sub_size, sub_ix, sub_iy)
                refined_cells[sub_key].add_point(
                    float(pts['z'][i]),
                    int(pts['class_id'][i]),
                    float(pts['confidence'][i]),
                    int(pts['drivable'][i]),
                    int(pts['dynamic'][i]),
                    float(pts['intensity'][i]),
                    int(pts['frame_id'][i]),
                    float(pts['timestamp'][i]),
                )
            else:
                if key not in refined_cells:
                    refined_cells[key] = base_cells[key]
        return refined_cells

    def to_list(self) -> list:
        return [
            cell.to_dict(
                include_histogram=self.base_config.store_histogram,
                obstacle_height_threshold=self.base_config.obstacle_height_threshold,
                drivable_ratio_threshold=self.base_config.drivable_ratio_threshold,
            )
            for cell in self.base_mapper.cells.values()
        ]