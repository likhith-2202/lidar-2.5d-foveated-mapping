import numpy as np
from core.uniform_mapper import UniformMapper
from config.config import UniformConfig
from core.points import normalize_points


def test_uniform_basic():
    config = UniformConfig(resolution=0.5, max_range=10.0)
    mapper = UniformMapper(config)
    pts = normalize_points(np.array([
        [0.1, 0.1, 0.5, 1, 0.9],
        [0.2, 0.2, 1.0, 2, 0.8],
        [1.0, 1.0, 2.0, 1, 0.7],
        [5.0, 5.0, 0.0, 0, 0.5],
        [20.0, 0.0, 0.0, 0, 0.5],
    ]))
    cells = mapper.build(pts)
    assert len(cells) == 3
    cell = mapper.get_cell(0, 0)
    assert cell.point_count == 2
    assert cell.elevation.max == 1.0
    assert cell.elevation.min == 0.5
    assert cell.semantics.dominant_class == 1


def test_uniform_empty():
    config = UniformConfig(resolution=0.1, max_range=1.0)
    mapper = UniformMapper(config)
    pts = normalize_points(np.array([[5.0, 5.0, 0.0, 0, 0.5]]))
    cells = mapper.build(pts)
    assert len(cells) == 0