from core.cell import CellData


def test_traversability_unknown_empty():
    c = CellData(0, 0.1, 0, 0)
    assert c.traversability() == 'unknown'


def test_traversability_drivable():
    c = CellData(0, 0.1, 0, 0)
    c.add_point(0.0, 1, 0.9, drivable=1, dynamic=0)
    c.add_point(0.01, 1, 0.9, drivable=1, dynamic=0)
    assert c.traversability() == 'drivable'


def test_traversability_non_drivable_by_ratio():
    c = CellData(0, 0.1, 0, 0)
    c.add_point(0.0, 1, 0.9, drivable=0)
    c.add_point(0.01, 1, 0.9, drivable=0)
    assert c.traversability() == 'non_drivable'


def test_traversability_obstacle_override():
    c = CellData(0, 0.1, 0, 0)
    c.add_point(0.0, 1, 0.9, drivable=1)
    c.add_point(0.5, 1, 0.9, drivable=1)  # 0.5 m height > 0.3 m threshold
    assert c.traversability() == 'non_drivable'


def test_traversability_unknown_when_no_drivable_flag():
    c = CellData(0, 0.1, 0, 0)
    c.add_point(0.0, 1, 0.9)  # drivable defaults to 255
    assert c.traversability() == 'unknown'


def test_dynamic_ratio():
    c = CellData(0, 0.1, 0, 0)
    c.add_point(0.0, 1, 0.9, dynamic=1)
    c.add_point(0.0, 1, 0.9, dynamic=1)
    c.add_point(0.0, 1, 0.9, dynamic=0)
    assert abs(c.dynamic_ratio - 2 / 3) < 1e-6
    assert c.is_dynamic() is True