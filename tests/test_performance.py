import numpy as np
from core.uniform_mapper import UniformMapper
from core.foveated_mapper import FoveatedMapper
from config.config import UniformConfig, FoveatedConfig
from benchmarks.benchmark import generate_synthetic_points

def test_foveated_uses_fewer_cells():
    # Generate points biased to far distances to show foveated advantage
    n = 20000
    r = np.random.uniform(0, 100, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(n)
    class_id = np.zeros(n, dtype=np.int32)
    conf = np.ones(n, dtype=np.float32)
    from core.points import CANONICAL_DTYPE
    pts = np.empty(n, dtype=CANONICAL_DTYPE)
    pts['x'] = x.astype(np.float32)
    pts['y'] = y.astype(np.float32)
    pts['z'] = z.astype(np.float32)
    pts['class_id'] = class_id
    pts['confidence'] = conf
    pts['frame_id'] = np.zeros(n, dtype=np.int64)
    pts['timestamp'] = np.zeros(n, dtype=np.float64)
    
    uniform_config = UniformConfig(resolution=0.05, max_range=100.0)
    foveated_config = FoveatedConfig()
    
    um = UniformMapper(uniform_config)
    fm = FoveatedMapper(foveated_config)
    
    um_cells = um.build(pts)
    fm_cells = fm.build(pts)
    
    # Foveated should have fewer cells (because coarse far away)
    assert len(fm_cells) < len(um_cells)