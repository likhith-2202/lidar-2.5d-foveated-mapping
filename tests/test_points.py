import numpy as np
import pytest
from core.points import normalize_points, CANONICAL_DTYPE


def test_normalize_n4():
    pts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    norm = normalize_points(pts)
    assert norm.dtype == CANONICAL_DTYPE
    assert len(norm) == 2
    assert norm['x'][0] == 1.0
    assert norm['class_id'][1] == 8
    assert norm['confidence'][0] == -1.0
    assert norm['drivable'][0] == 255
    assert norm['dynamic'][0] == 255


def test_normalize_n5():
    pts = np.array([[1, 2, 3, 4, 0.9]])
    norm = normalize_points(pts)
    assert norm['confidence'][0] == 0.9


def test_normalize_dict():
    d = {'x': [1, 2], 'y': [3, 4], 'z': [5, 6],
         'class_id': [7, 8], 'confidence': [0.5, 0.6]}
    norm = normalize_points(d)
    assert len(norm) == 2
    assert norm['class_id'][1] == 8


def test_normalize_with_new_fields():
    d = {
        'x': [1, 2], 'y': [3, 4], 'z': [5, 6],
        'class_id': [7, 8], 'confidence': [0.9, 0.8],
        'drivable': [1, 0], 'dynamic': [0, 1],
        'raw_class_id': [40, 10], 'intensity': [0.5, 0.6],
    }
    norm = normalize_points(d)
    assert norm['drivable'][0] == 1
    assert norm['drivable'][1] == 0
    assert norm['dynamic'][1] == 1
    assert norm['intensity'][0] == 0.5
    assert norm['raw_class_id'][0] == 40


def test_invalid_input():
    with pytest.raises(ValueError):
        normalize_points(np.array([[1, 2, 3]]))