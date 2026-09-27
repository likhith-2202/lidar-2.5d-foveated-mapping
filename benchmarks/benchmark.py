"""
Benchmarking utilities for comparing uniform vs foveated mappers.
"""

import numpy as np
import time
from typing import Dict, Any
from core.uniform_mapper import UniformMapper
from core.foveated_mapper import FoveatedMapper
from config.config import UniformConfig, FoveatedConfig
import tracemalloc

def generate_synthetic_points(n_points: int = 100000, max_range: float = 100.0) -> np.ndarray:
    """
    Generate random points within a circular area up to max_range.
    Returns structured array with fields: x, y, z, class_id, confidence, frame_id, timestamp.
    """
    # Random angles and radii (uniform in area)
    r = max_range * np.sqrt(np.random.random(n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.normal(0, 0.1, n_points)  # mostly ground
    class_id = np.random.randint(0, 20, n_points).astype(np.int32)
    confidence = np.random.random(n_points).astype(np.float32)
    frame_id = np.zeros(n_points, dtype=np.int64)
    timestamp = np.zeros(n_points, dtype=np.float64)
    
    from core.points import CANONICAL_DTYPE
    points = np.empty(n_points, dtype=CANONICAL_DTYPE)
    points['x'] = x.astype(np.float32)
    points['y'] = y.astype(np.float32)
    points['z'] = z.astype(np.float32)
    points['class_id'] = class_id
    points['confidence'] = confidence
    points['frame_id'] = frame_id
    points['timestamp'] = timestamp
    return points

def measure_memory_usage(obj) -> int:
    """Return memory usage in bytes using sys.getsizeof recursively."""
    import sys
    def get_size(obj, seen=None):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum([get_size(k, seen) + get_size(v, seen) for k, v in obj.items()])
        elif isinstance(obj, (list, tuple, set)):
            size += sum([get_size(i, seen) for i in obj])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        return size
    return get_size(obj)

def benchmark_mappers(points: np.ndarray, uniform_config: UniformConfig = None,
                      foveated_config: FoveatedConfig = None) -> Dict[str, Any]:
    """
    Run both mappers on the same points and return statistics.
    """
    if uniform_config is None:
        uniform_config = UniformConfig()
    if foveated_config is None:
        foveated_config = FoveatedConfig()
    
    results = {}
    
    # Uniform
    tracemalloc.start()
    um = UniformMapper(uniform_config)
    t0 = time.perf_counter()
    um_cells = um.build(points)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['uniform'] = {
        'num_cells': len(um_cells),
        'time_ms': (t1 - t0) * 1000,
        'memory_bytes': current,
        'peak_memory_bytes': peak,
        'statistics': um.statistics,
    }
    
    # Foveated
    tracemalloc.start()
    fm = FoveatedMapper(foveated_config)
    t0 = time.perf_counter()
    fm_cells = fm.build(points)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['foveated'] = {
        'num_cells': len(fm_cells),
        'time_ms': (t1 - t0) * 1000,
        'memory_bytes': current,
        'peak_memory_bytes': peak,
        'statistics': fm.statistics,
    }
    
    return results