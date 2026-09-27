"""
Point representation and normalisation.
Sentinel values are passed in (defaults match SemanticKITTI conventions).
"""

import numpy as np
from typing import Union, List, Dict, Any

CANONICAL_DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('class_id', np.int32),
    ('confidence', np.float32),
    ('drivable', np.uint8),
    ('dynamic', np.uint8),
    ('raw_class_id', np.int32),
    ('intensity', np.float32),
    ('frame_id', np.int64),
    ('timestamp', np.float64),
])


def _extract_field(points, field, n, default_val, dtype):
    if isinstance(points, dict):
        v = points.get(field, None)
    elif isinstance(points, np.ndarray) and points.dtype.names:
        v = points[field] if field in points.dtype.names else None
    else:
        v = None
    if v is None:
        return np.full(n, default_val, dtype=dtype)
    return np.asarray(v, dtype=dtype)


def normalize_points(
    points: Union[np.ndarray, List, Dict],
    unknown_sentinel: int = 255,
    unknown_confidence_sentinel: float = -1.0,
    unknown_raw_class_sentinel: int = -1,
) -> np.ndarray:
    if isinstance(points, dict):
        for k in ('x', 'y', 'z', 'class_id'):
            if k not in points:
                raise ValueError(f"Missing required key '{k}' in input dict")
        x = np.asarray(points['x'], dtype=np.float32)
        y = np.asarray(points['y'], dtype=np.float32)
        z = np.asarray(points['z'], dtype=np.float32)
        class_id = np.asarray(points['class_id'], dtype=np.int32)
        n = len(x)
        confidence = _extract_field(points, 'confidence', n,
                                    unknown_confidence_sentinel, np.float32)

    elif isinstance(points, (list, tuple)):
        return normalize_points(
            np.asarray(points),
            unknown_sentinel=unknown_sentinel,
            unknown_confidence_sentinel=unknown_confidence_sentinel,
            unknown_raw_class_sentinel=unknown_raw_class_sentinel,
        )

    elif isinstance(points, np.ndarray):
        if points.dtype.names:
            for f in ('x', 'y', 'z', 'class_id'):
                if f not in points.dtype.names:
                    raise ValueError(f"Structured array missing field '{f}'")
            x = points['x'].astype(np.float32)
            y = points['y'].astype(np.float32)
            z = points['z'].astype(np.float32)
            class_id = points['class_id'].astype(np.int32)
            n = len(points)
            confidence = _extract_field(points, 'confidence', n,
                                        unknown_confidence_sentinel, np.float32)
        else:
            if points.ndim != 2:
                raise ValueError("Point array must be 2D")
            ncols = points.shape[1]
            if ncols < 4:
                raise ValueError("At least 4 columns required: x, y, z, class_id")
            n = len(points)
            x = points[:, 0].astype(np.float32)
            y = points[:, 1].astype(np.float32)
            z = points[:, 2].astype(np.float32)
            class_id = points[:, 3].astype(np.int32)
            confidence = (points[:, 4].astype(np.float32) if ncols >= 5
                          else np.full(n, unknown_confidence_sentinel, dtype=np.float32))
    else:
        raise TypeError("Unsupported points input type")

    drivable = _extract_field(points, 'drivable', n, unknown_sentinel, np.uint8)
    dynamic = _extract_field(points, 'dynamic', n, unknown_sentinel, np.uint8)
    raw_class_id = _extract_field(points, 'raw_class_id', n,
                                  unknown_raw_class_sentinel, np.int32)
    intensity = _extract_field(points, 'intensity', n, 0.0, np.float32)
    frame_id = _extract_field(points, 'frame_id', n, 0, np.int64)
    timestamp = _extract_field(points, 'timestamp', n, 0.0, np.float64)

    result = np.empty(n, dtype=CANONICAL_DTYPE)
    result['x'] = x
    result['y'] = y
    result['z'] = z
    result['class_id'] = class_id
    result['confidence'] = confidence
    result['drivable'] = drivable
    result['dynamic'] = dynamic
    result['raw_class_id'] = raw_class_id
    result['intensity'] = intensity
    result['frame_id'] = frame_id
    result['timestamp'] = timestamp
    return result