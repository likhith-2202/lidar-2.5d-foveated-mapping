"""
Semantic aggregation for a cell.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class SemanticStats:
    """
    Aggregated semantic information for a cell.
    """
    dominant_class: int
    dominant_confidence: float
    histogram: Dict[int, float] = field(default_factory=dict)
    total_weight: float = 0.0

    @classmethod
    def from_values(cls, class_ids: np.ndarray, confidences: np.ndarray, use_confidence_weighting: bool = True) -> 'SemanticStats':
        """
        Compute semantic statistics from arrays of class ids and confidences.
        If use_confidence_weighting is True, confidences are used as weights;
        otherwise each point counts equally. Confidence values of -1 are ignored
        (treated as weight 1).
        """
        if len(class_ids) == 0:
            return cls(dominant_class=0, dominant_confidence=0.0, histogram={})
        
        class_ids = np.asarray(class_ids)
        confidences = np.asarray(confidences)
        
        # Determine weights
        if use_confidence_weighting:
            # Use confidence as weight, but if any confidence is -1, treat all as weight 1
            if np.any(confidences < 0):
                weights = np.ones(len(class_ids))
            else:
                weights = confidences.astype(np.float64)
        else:
            weights = np.ones(len(class_ids))
        
        # Build histogram
        hist = {}
        for cid, w in zip(class_ids, weights):
            cid = int(cid)
            hist[cid] = hist.get(cid, 0.0) + w
        
        total = sum(hist.values())
        if total == 0:
            return cls(dominant_class=0, dominant_confidence=0.0, histogram={})
        
        # Dominant class
        dominant_class = max(hist, key=hist.get)
        dominant_confidence = hist[dominant_class] / total
        
        return cls(
            dominant_class=dominant_class,
            dominant_confidence=float(dominant_confidence),
            histogram=hist,
            total_weight=float(total)
        )

    def to_dict(self):
        return {
            'dominant_class': self.dominant_class,
            'dominant_confidence': self.dominant_confidence,
            'class_histogram': self.histogram,
        }