import numpy as np
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig, ResolutionBand
from core.points import normalize_points


def test_exact_distances():
    bands = [
        ResolutionBand(0, 10, 0.1),
        ResolutionBand(10, 20, 0.2),
        ResolutionBand(20, 30, 0.3),
    ]
    config = FoveatedConfig(bands=bands, max_range=30.0)
    mapper = FoveatedMapper(config)
    test_cases = [
        (5.0, 0),
        (10.0, 1),
        (19.99, 1),
        (20.0, 2),
        (30.0, 2),
        (30.1, -1),
    ]
    for r, expected_band in test_cases:
        pts = normalize_points(np.array([[r, 0.0, 0.0, 0, 0.5]]))
        cells = mapper.build(pts)
        if expected_band == -1:
            assert len(cells) == 0
        else:
            assert len(cells) == 1
            key = next(iter(cells))
            assert key[0] == expected_band


def test_negative_coordinates():
    config = FoveatedConfig()
    mapper = FoveatedMapper(config)
    pts = normalize_points(np.array([
        [-5.0, -5.0, 0.0, 1, 0.9],
        [-15.0, -15.0, 0.0, 2, 0.8],
    ]))
    cells = mapper.build(pts)
    assert len(cells) == 2
    for key in cells:
        assert key[1] < 0 and key[2] < 0