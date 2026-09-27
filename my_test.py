import numpy as np
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig

# 6 fake classified points: x, y, z, class_id, confidence
points = np.array([
    [ 2.0,  0.0, 0.1, 1, 0.9],   # near  -> fine cell  (5 cm)
    [ 8.0,  0.0, 0.2, 1, 0.9],   # near  -> fine cell
    [15.0,  0.0, 0.3, 2, 0.8],   # mid   -> 10 cm cell
    [30.0,  0.0, 0.5, 3, 0.7],   # far   -> 25 cm cell
    [75.0,  0.0, 1.0, 4, 0.6],   # far   -> 50 cm cell
    [200.0, 0.0, 1.5, 4, 0.6],   # OUT OF RANGE -> dropped
])

mapper = FoveatedMapper(FoveatedConfig())
cells = mapper.build(points)

print(f"\nTotal cells: {len(cells)}  (expected 5, out-of-range point dropped)\n")
for key, cell in sorted(cells.items()):
    band_id, ix, iy = key
    print(f"band={band_id}  size={cell.cell_size}m  "
          f"pos=({cell.center_x:.2f},{cell.center_y:.2f})  "
          f"pts={cell.point_count}  "
          f"z=[{cell.elevation.min:.2f},{cell.elevation.max:.2f}]  "
          f"class={cell.semantics.dominant_class}")
    