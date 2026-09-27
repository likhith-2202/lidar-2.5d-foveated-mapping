import numpy as np
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig
from core.points import normalize_points


def test_foveated_bands():
    config = FoveatedConfig()
    mapper = FoveatedMapper(config)
    pts = normalize_points(np.array([
        [5.0, 0.0, 0.0, 1, 0.9],
        [15.0, 0.0, 0.0, 2, 0.8],
        [30.0, 0.0, 0.0, 3, 0.7],
        [75.0, 0.0, 0.0, 4, 0.6],
    ]))
    cells = mapper.build(pts)
    assert len(cells) == 4
    for key, cell in cells.items():
        band_id = key[0]
        expected_size = config.bands[band_id].cell_size
        assert cell.cell_size == expected_size


def test_foveated_boundary_10m():
    config = FoveatedConfig()
    mapper = FoveatedMapper(config)
    pts = normalize_points(np.array([[10.0, 0.0, 0.0, 1, 0.9]]))
    cells = mapper.build(pts)
    assert len(cells) == 1
    key = next(iter(cells))
    assert key[0] == 1


def test_foveated_alignment():
    config = FoveatedConfig()
    base = min(b.cell_size for b in config.bands)
    for b in config.bands:
        assert (b.cell_size / base).is_integer()