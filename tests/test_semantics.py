import numpy as np
from core.semantics import SemanticStats
from core.cell import CellData

def test_semantic_dominant():
    classes = np.array([1, 1, 2, 2, 2])
    conf = np.array([0.9, 0.9, 0.8, 0.8, 0.8])
    sem = SemanticStats.from_values(classes, conf)
    assert sem.dominant_class == 2
    # weights: class1: 1.8, class2: 2.4 -> confidence of dominant = 2.4/4.2
    assert abs(sem.dominant_confidence - 2.4/4.2) < 1e-6

def test_semantic_unweighted():
    classes = np.array([1, 1, 2])
    conf = np.array([-1, -1, -1])  # unknown confidence
    sem = SemanticStats.from_values(classes, conf)
    assert sem.dominant_class == 1
    assert sem.dominant_confidence == 2/3

def test_cell_semantics():
    cell = CellData(band_id=0, cell_size=0.1, ix=0, iy=0)
    cell.add_point(0.0, 1, 0.9)
    cell.add_point(0.0, 1, 0.8)
    cell.add_point(0.0, 2, 0.7)
    sem = cell.semantics
    assert sem.dominant_class == 1
    assert sem.dominant_confidence > 0.5