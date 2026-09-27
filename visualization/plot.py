"""
Simple visualization helpers for debugging.
Uses matplotlib for 2D and optional Open3D for 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def plot_cells_2d(cells: List[Dict], color_by: str = 'elevation', ax=None, show=True):
    """
    Plot cells as colored rectangles on XY plane.
    color_by: 'elevation', 'semantic', 'resolution', 'occupancy'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    for cell in cells:
        cx = cell['center_x']
        cy = cell['center_y']
        size = cell['cell_size']
        if color_by == 'elevation':
            color = cell['mean_z']
            cmap = 'viridis'
        elif color_by == 'semantic':
            color = cell['dominant_class']
            cmap = 'tab20'
        elif color_by == 'resolution':
            color = cell['cell_size']
            cmap = 'plasma'
        elif color_by == 'occupancy':
            color = 0 if cell['occupancy'] == 'occupied' else 1
            cmap = 'gray'
        else:
            color = 0
            cmap = 'gray'
        rect = plt.Rectangle((cx - size/2, cy - size/2), size, size,
                             linewidth=0.5, edgecolor='none', facecolor=plt.cm.get_cmap(cmap)(color))
        ax.add_patch(rect)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'2.5D Map (colored by {color_by})')
    if show:
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
        plt.show()
    return ax

def plot_points_3d(points, color_by='z', max_points=10000):
    """
    Plot raw points in 3D using Open3D (if installed).
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Skipping 3D plot.")
        return
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([points['x'], points['y'], points['z']], axis=1))
    if color_by == 'z':
        colors = plt.cm.viridis((points['z'] - points['z'].min()) / (points['z'].max() - points['z'].min()))[:, :3]
    elif color_by == 'class':
        colors = plt.cm.tab20(points['class_id'] % 20)[:, :3]
    else:
        colors = np.zeros((len(points), 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])