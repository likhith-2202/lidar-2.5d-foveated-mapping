import numpy as np
from core.cell import CellData
from core.elevation import ElevationStats

def test_elevation_stats():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    stats = ElevationStats.from_values(z)
    assert stats.min == 1.0
    assert stats.max == 4.0
    assert stats.mean == 2.5
    assert stats.count == 4
    assert stats.std > 0

def test_cell_elevation():
    cell = CellData(band_id=0, cell_size=0.5, ix=0, iy=0)
    cell.add_point(1.0, 0, 0.5)
    cell.add_point(2.0, 0, 0.5)
    cell.add_point(3.0, 0, 0.5)
    elev = cell.elevation
    assert elev.min == 1.0
    assert elev.max == 3.0
    assert elev.mean == 2.0
    assert cell.point_count == 3