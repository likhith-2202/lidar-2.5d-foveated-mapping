"""
Adapter for SemanticKITTI dataset.
SemanticKITTI scans are stored as .bin files (x,y,z,remission) and labels as .label files.
"""

import numpy as np
import os
from typing import Tuple, Optional

def load_semantickitti_frame(bin_path: str, label_path: Optional[str] = None) -> np.ndarray:
    """
    Load a single SemanticKITTI frame and return a structured array with
    fields: x, y, z, class_id, confidence (set to 1.0), frame_id (0), timestamp (0).
    
    Args:
        bin_path: Path to .bin file containing x,y,z,remission (float32 each)
        label_path: Optional path to .label file containing uint32 semantic labels.
                    If not provided, class_id will be set to 0 (unknown).
    Returns:
        Structured NumPy array with canonical fields.
    """
    # Load .bin file
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Bin file not found: {bin_path}")
    
    # SemanticKITTI .bin contains Nx4 float32: x,y,z,remission
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    n = scan.shape[0]
    
    # Load labels if available
    if label_path and os.path.exists(label_path):
        labels = np.fromfile(label_path, dtype=np.uint32)
        if len(labels) != n:
            raise ValueError(f"Label count {len(labels)} does not match point count {n}")
        # SemanticKITTI uses lower 16 bits for semantic class
        class_ids = (labels & 0xFFFF).astype(np.int32)
    else:
        class_ids = np.zeros(n, dtype=np.int32)
    
    # Create structured array
    from core.points import CANONICAL_DTYPE
    points = np.empty(n, dtype=CANONICAL_DTYPE)
    points['x'] = scan[:, 0].astype(np.float32)
    points['y'] = scan[:, 1].astype(np.float32)
    points['z'] = scan[:, 2].astype(np.float32)
    points['class_id'] = class_ids
    points['confidence'] = np.ones(n, dtype=np.float32)  # assume full confidence from ground truth
    points['frame_id'] = np.zeros(n, dtype=np.int64)
    points['timestamp'] = np.zeros(n, dtype=np.float64)
    return points