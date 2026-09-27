"""
Example pipeline demonstrating integration with Person 1 (semantic perception)
and Person 3 (output consumption).
"""

import numpy as np
from core.foveated_mapper import FoveatedMapper
from core.uniform_mapper import UniformMapper
from config.config import FoveatedConfig, UniformConfig
from core.points import normalize_points
import time

def simulate_person1_output(n_points=100000):
    """
    Simulate output from Person 1: classified points as a NumPy array.
    Format: N x 6 (x, y, z, class_id, confidence, frame_id)
    """
    r = np.random.uniform(0, 80, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.normal(0, 0.1, n_points)
    class_id = np.random.randint(0, 20, n_points)
    confidence = np.random.random(n_points)
    frame_id = np.full(n_points, 0)  # single frame
    return np.column_stack([x, y, z, class_id, confidence, frame_id])

def main():
    # Example configuration
    fov_config = FoveatedConfig()
    uniform_config = UniformConfig(resolution=0.05, max_range=100.0)
    
    # Simulate Person 1 output
    print("Simulating Person 1 output...")
    points = simulate_person1_output()
    print(f"Received {len(points)} points")
    
    # Build foveated map
    print("\nBuilding foveated map...")
    fov_mapper = FoveatedMapper(fov_config)
    t0 = time.perf_counter()
    fov_cells = fov_mapper.build(points)
    t1 = time.perf_counter()
    print(f"Foveated map: {len(fov_cells)} cells in {(t1-t0)*1000:.2f} ms")
    print(f"Statistics: {fov_mapper.statistics}")
    
    # Build uniform baseline
    print("\nBuilding uniform baseline...")
    uni_mapper = UniformMapper(uniform_config)
    t0 = time.perf_counter()
    uni_cells = uni_mapper.build(points)
    t1 = time.perf_counter()
    print(f"Uniform map: {len(uni_cells)} cells in {(t1-t0)*1000:.2f} ms")
    print(f"Statistics: {uni_mapper.statistics}")
    
    # Output for Person 3: list of cell dictionaries
    output_list = fov_mapper.to_list()
    print(f"\nExported {len(output_list)} cells for Person 3")
    # Example: print first cell
    if output_list:
        print("First cell:", output_list[0])
    
    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        from visualization.plot import plot_cells_2d
        plot_cells_2d(output_list, color_by='resolution', show=False)
        plt.savefig('foveated_map.png')
        print("Saved visualization to foveated_map.png")
    except ImportError:
        print("Matplotlib not installed, skipping visualization")

if __name__ == "__main__":
    main()