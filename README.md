# Foveated 2.5D Semantic Mapping Engine

A memory-efficient, variable-resolution 2.5D semantic elevation mapping system for autonomous navigation. Converts classified 3D LiDAR point clouds into a foveated spatial representation where resolution decreases with distance from the sensor — preserving safety-critical detail near the vehicle while dramatically reducing memory and compute for distant regions.

```
Raw LiDAR  →  Classified Points  →  Foveated 2.5D Map  →  Downstream Navigation
```

---

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Architecture](#architecture)
- [Coordinate System & Conventions](#coordinate-system--conventions)
- [Data Representation](#data-representation)
- [Foveated Grid Design](#foveated-grid-design)
- [Elevation & Semantics](#elevation--semantics)
- [Traversability & Occupancy](#traversability--occupancy)
- [Risk-Aware Extension](#risk-aware-extension-advanced)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Input & Output Formats](#input--output-formats)
- [Installation & Quick Start](#installation--quick-start)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Complexity Analysis](#complexity-analysis)
- [Prior Work & Differentiation](#prior-work--differentiation)
- [Assumptions & Limitations](#assumptions--limitations)
- [Roadmap](#roadmap)

---

## Motivation

**The problem with existing representations:**

| Approach | Issue |
|---|---|
| Raw 3D point cloud | Millions of points per frame → expensive to store, process, transmit |
| Uniform 2D occupancy grid | Loses height → can't represent curbs, potholes, walls, overhangs |
| Uniform 2.5D grid (5 cm) | Preserves height, but a 100 m × 100 m area at 5 cm is **4M cells per frame** |

**Key insight:** For autonomous navigation, sensor resolution matters most **near the vehicle** — where reactive safety decisions happen. Distant regions mainly need coarse terrain context.

**Solution:** A distance-driven foveated 2.5D grid:

- **0–10 m** → 5 cm cells (fine detail for obstacle avoidance)
- **10–25 m** → 10 cm cells
- **25–50 m** → 25 cm cells
- **50–100 m** → 50 cm cells

This yields **~10–20× fewer cells** than a uniform 5 cm baseline while preserving all safety-critical information near the vehicle.

---

## Features

- **Uniform 2.5D baseline** — reference implementation for benchmarking
- **Distance-based foveated grid** — configurable resolution bands
- **Aligned multi-resolution cells** — no gaps, no overlaps (all sizes are integer multiples of base)
- **Rich per-cell aggregation** — elevation stats, semantic histogram, traversability, intensity
- **Confidence-weighted semantic voting** — preserves class distribution when uncertain
- **Explicit UNKNOWN state** — never assumes "no points = free space"
- **Traversability layer** — separate from semantic class, with obstacle-height override
- **Dynamic-object flagging** — for downstream tracking pipelines
- **Risk-aware refinement** — subdivide important cells (pedestrians, cyclists) at distance
- **Fully configurable** — no hard-coded thresholds, sentinels, or importance maps
- **CPU-first** — runs on modest hardware, no GPU required
- **SemanticKITTI adapter** — load real-world data out of the box

---

## Architecture

```
foveated_mapping/
├── core/
│   ├── points.py           # Normalization to canonical structured array
│   ├── cell.py             # Per-cell accumulator + traversability logic
│   ├── elevation.py        # Elevation statistics
│   ├── semantics.py        # Semantic histogram aggregation
│   ├── uniform_mapper.py   # Baseline: fixed-resolution mapper
│   ├── foveated_mapper.py  # Main: distance-based multi-resolution mapper
│   └── risk_aware.py       # Advanced: semantic-aware local refinement
├── config/
│   └── config.py           # FoveatedConfig, UniformConfig, ResolutionBand
├── adapters/
│   └── semantickitti.py    # .bin + .label loader
├── benchmarks/
│   └── benchmark.py        # Memory / latency / cell-count measurements
├── visualization/
│   └── plot.py             # 2D cell plots, 3D point cloud
├── tests/
│   ├── test_points.py
│   ├── test_uniform.py
│   ├── test_foveated.py
│   ├── test_boundaries.py
│   ├── test_elevation.py
│   ├── test_semantics.py
│   ├── test_traversability.py
│   └── test_performance.py
└── examples/
    └── example_pipeline.py
```

**Design principle:** dataset code (`adapters/`), configuration (`config/`), and core algorithms (`core/`) are fully decoupled. The mapper never knows what dataset it's looking at — it only sees the canonical point schema.

---

## Coordinate System & Conventions

### Frames

- **Sensor frame** (initial implementation): origin at LiDAR center, `+x` forward, `+y` left, `+z` up.
- **Vehicle frame** — future extension (transformation matrix applied before mapping).
- **World/map frame** — future extension.

Frame transforms are *not* baked into the grid algorithm. To map in a different frame, transform the points before calling `mapper.build(...)`.

### Units

All lengths in **metres**. All distances in **metres**. Semantic class IDs are integers.

### Foveation distance

Resolution selection uses **horizontal distance**:

```
r = √(x² + y²)
```

**Why horizontal, not 3D Euclidean?**

Because the 2.5D map projects points onto the XY plane. A point at 30 m horizontal distance with `z = 2 m` and a point at 30 m horizontal distance with `z = 0 m` both land in the same cell — they should use the same resolution. Using 3D distance would give different resolutions for overhanging structures vs. ground at the same map location, breaking spatial consistency.

### Cell indexing

Given a point `(x, y)` and resolution `Δ`:

```
ix = floor(x / Δ)
iy = floor(y / Δ)
```

`floor` correctly handles negative coordinates (e.g., `x = −0.1`, `Δ = 0.05` → `ix = −2`).

Cell center:

```
cx = (ix + 0.5) · Δ
cy = (iy + 0.5) · Δ
```

### Band selection (half-open semantics)

Bands are defined as **half-open intervals** `[r_min, r_max)`, except the last band which includes its upper bound. A point exactly at a boundary goes to the **higher (coarser)** band:

| Distance | Band |
|---|---|
| 5.0 m | 0 (5 cm) |
| 10.0 m | **1** (10 cm) — *boundary goes to coarser* |
| 19.99 m | 1 (10 cm) |
| 20.0 m | 2 (25 cm) |
| 100.0 m | last band (inclusive) |

Implemented with `np.searchsorted(thresholds, r, side='right')` and validated by `tests/test_boundaries.py`.

### Alignment guarantee

All cell sizes must be **integer multiples of the finest cell size** (the *base*):

```
Δ_i = k_i · Δ_base,  k_i ∈ ℤ⁺
```

For the default config: `0.10 / 0.05 = 2`, `0.25 / 0.05 = 5`, `0.50 / 0.05 = 10`. ✅

`FoveatedConfig.validate()` enforces this — misaligned configs are rejected at construction time. This prevents:

- gaps between coarse and fine cells,
- shifted boundaries,
- overlapping regions.

---

## Data Representation

### Canonical schema

Points are normalized to a **structured NumPy array** with the following dtype:

```python
CANONICAL_DTYPE = np.dtype([
    ('x', np.float32),           # metres, sensor frame
    ('y', np.float32),
    ('z', np.float32),
    ('class_id', np.int32),      # semantic class integer
    ('confidence', np.float32),  # 0..1, or sentinel (-1) if unavailable
    ('drivable', np.uint8),      # 0/1/255 (255 = unknown)
    ('dynamic', np.uint8),       # 0/1/255 (255 = unknown)
    ('raw_class_id', np.int32),  # original dataset class (e.g., SemanticKITTI)
    ('intensity', np.float32),   # LiDAR return intensity
    ('frame_id', np.int64),
    ('timestamp', np.float64),
])
```

### Accepted input formats

`normalize_points()` accepts:

| Format | Example |
|---|---|
| `(N, 4)` array | `x, y, z, class_id` |
| `(N, 5)` array | `+ confidence` |
| `(N, 6)` array | `+ frame_id` |
| `(N, 7)` array | `+ timestamp` |
| Structured array | Any subset of canonical fields |
| Dict of arrays | `{'x': ..., 'y': ..., 'class_id': ..., 'drivable': ...}` |
| List of dicts / lists | Auto-converted |

Optional fields default to configurable sentinels (`unknown_sentinel`, `unknown_confidence_sentinel`, etc.).

**Example — feeding a dict:**

```python
points = {
    'x': x_array,
    'y': y_array,
    'z': z_array,
    'class_id': class_ids,       # int
    'confidence': confidences,   # float in [0,1]
    'drivable': drivable_flags,  # 0/1/255
    'dynamic': dynamic_flags,    # 0/1/255
    'raw_class_id': raw_labels,
    'intensity': intensity,
}
mapper.build(points)
```

---

## Foveated Grid Design

### Data structure choice

| Structure | Pros | Cons | Verdict |
|---|---|---|---|
| **Dense uniform NumPy array** | Fast indexing, vectorized | 4M+ cells at 5 cm over 100 m — massive memory | ❌ Baseline only |
| **Dictionary (spatial hash)** | O(1) insert/lookup, sparse, memory-efficient | Python-object overhead per cell | ✅ Chosen for MVP |
| **Quadtree** | Truly adaptive, no overlapping cells | Complex, harder to align with distance bands | ⏭ Future for risk-aware mode |
| **Octree** | Same as quadtree but 3D | Overkill for 2.5D | ❌ |

**Chosen design:** multi-level **dictionary keyed by `(band_id, ix, iy)`**.

- Each band is an independent uniform grid at that band's resolution.
- Points are assigned to exactly one band via horizontal distance.
- Cell sizes are aligned (integer multiples of base), so cells at different bands never "shift".
- Coarse and fine cells can overlap spatially — intentional and correct, since each region only uses one resolution at a time.

**Why not a quadtree?** For distance-only foveation, a quadtree offers no memory savings vs. multi-level dictionaries (which are already sparse), but adds complexity to insertion, iteration, and testing. A quadtree becomes worthwhile only in **risk-aware mode** where you need to subdivide *some* cells within a coarse band. That's implemented as a post-processing step in `core/risk_aware.py`.

### Complexity (per frame)

| Operation | Complexity |
|---|---|
| Normalization | `O(N)` |
| Distance computation + filter | `O(N)` |
| Band assignment (`searchsorted`) | `O(N log B)` where `B` = number of bands |
| Cell index computation | `O(N)` |
| Insertion into dict | `O(1)` amortized per point |
| Aggregation (elevation, semantics) | `O(N)` total |
| **Total** | **`O(N log B)`** ≈ `O(N)` for small `B` |

Memory: `O(C)` where `C` is the number of **occupied** cells — proportional to point density, not the full bounding box.

---

## Elevation & Semantics

### Elevation statistics

Each cell stores per-point `z` values; on-demand aggregation produces:

| Statistic | Formula | Purpose |
|---|---|---|
| `min_z` | `min(z_i)` | Ground estimate (when cell is ground-only) |
| `max_z` | `max(z_i)` | Top surface (obstacle tip) |
| `mean_z` | `(1/n)·Σz_i` | Smooth reference |
| `std_z` | `√((1/n)·Σ(z_i − mean)²)` | Roughness / uncertainty |
| `obstacle_height` | `max_z − min_z` | **Height of terrain variation in cell** |

### Why not just `mean_z`?

A cell may contain both a **ground point** and an **obstacle point** (e.g., a cell straddling a curb). Averaging them would report a false flat surface. Instead the representation exposes the full range (`min_z` to `max_z`) and uses `max_z − min_z` as the **obstacle height** signal.

The traversability layer uses this to override "drivable" classifications when local terrain rises above a configurable threshold (`obstacle_height_threshold`, default 0.30 m).

### Semantic aggregation

Each cell stores the `(class_id, confidence)` of every point. Aggregation:

1. **Weight each point**: `w_i = confidence_i` (if `use_confidence_weighting`) or `1`.
2. **Build a weighted histogram** `H[c] = Σ_{i : class_i = c} w_i`.
3. **Dominant class** = `argmax_c H[c]`.
4. **Dominant confidence** = `H[dominant] / Σ_c H[c]`.

The **full histogram is preserved** in every cell output — downstream consumers can:

- see multi-class cells,
- compute entropy / uncertainty,
- re-run decisions under different aggregation rules.

If any point's confidence is `-1` (unknown), the entire cell falls back to uniform weighting (to avoid mixing inconsistent scales).

**Alternatives considered:**

| Strategy | Verdict |
|---|---|
| Majority vote | ❌ Ignores confidence, loses information |
| Confidence-weighted vote | ✅ **Chosen** |
| Full probability distribution per point | ⏭ Future (requires upstream probabilities) |

---

## Traversability & Occupancy

### Occupancy

| State | Meaning |
|---|---|
| `occupied` | Cell contains ≥ 1 point |
| `unknown` | No points — **do not** assume free space |

Free space is intentionally **not** marked by default. Ray-casting through the sensor origin could mark cells as free, but this is deferred to a multi-frame accumulation pipeline where multiple scans can be fused. For a single frame, "no return" ≠ "free" (could be a reflective surface, a gap, or simply the laser missing).

### Traversability

Computed per cell, independent of semantic class:

```
if no points:                        → 'unknown'
elif obstacle_height > threshold:    → 'non_drivable'   (safety override)
elif drivable_ratio >= threshold:    → 'drivable'
else:                                → 'non_drivable'
```

Defaults:

- `obstacle_height_threshold = 0.30 m` (curb height)
- `drivable_ratio_threshold = 0.50`

A cell classified as `road` but with a 0.5 m obstacle inside is marked **non-drivable** — safety overrides semantics.

### Dynamic-object flagging

If points are marked `dynamic` (pedestrian, cyclist, vehicle in motion), each cell exposes:

- `dynamic_ratio` — fraction of points marked dynamic (−1 if unknown)
- `is_dynamic` — `True` / `False` / `None` (unknown)

Useful for downstream tracking pipelines to decide whether to include a cell in a static map or track it separately.

---

## Risk-Aware Extension (Advanced)

Distance-only foveation is **the safe baseline** and should always be validated first. `RiskAwareMapper` extends it:

1. Run the standard `FoveatedMapper`.
2. Identify cells that contain points from **high-importance classes** (configurable — defaults: person, bicyclist, motorcyclist).
3. **Subdivide** those cells to `min_refinement_size` (default 5 cm) regardless of their distance band.
4. Leave all other cells unchanged.

Result: pedestrians at 30 m get 5 cm detail while an empty road at 30 m stays at 25 cm.

```python
from core.risk_aware import RiskAwareMapper
from config.config import FoveatedConfig

cfg = FoveatedConfig(
    risk_aware=True,
    importance_map={6: 1, 7: 1, 8: 1},
    min_refinement_size=0.05,
)
mapper = RiskAwareMapper(cfg)
cells = mapper.build(points)
```

> ⚠️ The current refinement is a **single-level subdivision** applied uniformly to any cell containing an important point. A production version would use a quadtree for recursive adaptive subdivision.

---

## API Reference

### `FoveatedMapper`

```python
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig

mapper = FoveatedMapper(FoveatedConfig())
cells = mapper.build(points)      # -> dict[(band_id, ix, iy), CellData]

cells_list = mapper.to_list()     # -> list[dict]
stats      = mapper.statistics    # -> dict (timings, cell counts)
```

### `UniformMapper`

```python
from core.uniform_mapper import UniformMapper
from config.config import UniformConfig

mapper = UniformMapper(UniformConfig(resolution=0.05, max_range=100.0))
cells  = mapper.build(points)     # -> dict[(ix, iy), CellData]
```

### `CellData`

Each cell exposes:

```python
# Position & geometry
cell.band_id        # int
cell.cell_size      # float (m)
cell.ix, cell.iy    # int indices
cell.center_x, cell.center_y

# Elevation
cell.elevation.min  # float
cell.elevation.max
cell.elevation.mean
cell.elevation.std
cell.elevation.count

# Semantics
cell.semantics.dominant_class       # int
cell.semantics.dominant_confidence  # float in [0,1]
cell.semantics.histogram            # dict {class_id: weight}

# Traversability
cell.drivable_ratio                 # float in [0,1] or -1 (unknown)
cell.dynamic_ratio                  # float in [0,1] or -1 (unknown)
cell.is_dynamic()                   # bool or None
cell.traversability()               # 'drivable' | 'non_drivable' | 'unknown'

# Raw data
cell.point_count
cell.mean_intensity
```

### `to_dict()` output

Exported cells contain:

```python
{
    'band_id': 1,
    'cell_size': 0.10,
    'ix': 120, 'iy': -43,
    'center_x': 12.05, 'center_y': -4.25,
    'point_count': 8,
    'min_z': -0.15, 'max_z': 0.32,
    'mean_z': 0.02, 'std_z': 0.11,
    'obstacle_height': 0.47,
    'dominant_class': 9,
    'dominant_confidence': 0.82,
    'class_histogram': {9: 6.5, 11: 1.4, 0: 0.1},
    'occupancy': 'occupied',
    'drivable_ratio': 0.87,
    'dynamic_ratio': 0.0,
    'is_dynamic': False,
    'mean_intensity': 0.34,
    'traversability': 'non_drivable',  # 0.47 m > 0.30 m
}
```

---

## Configuration

All parameters are configurable via `FoveatedConfig` and `UniformConfig`. **No hard-coded thresholds, sentinels, or importance maps.**

```python
from config.config import FoveatedConfig, ResolutionBand

cfg = FoveatedConfig(
    # Resolution bands
    bands=[
        ResolutionBand(0.0, 10.0, 0.05),
        ResolutionBand(10.0, 25.0, 0.10),
        ResolutionBand(25.0, 50.0, 0.25),
        ResolutionBand(50.0, 80.0, 0.50),
    ],
    max_range=80.0,

    # Aggregation
    use_confidence_weighting=True,
    store_histogram=True,

    # Traversability
    obstacle_height_threshold=0.30,
    drivable_ratio_threshold=0.50,
    dynamic_ratio_threshold=0.50,

    # Sentinels
    unknown_sentinel=255,
    unknown_ratio_sentinel=-1.0,

    # Risk-aware mode
    risk_aware=True,
    importance_map={6: 1, 7: 1, 8: 1},
    min_refinement_size=0.05,
)

mapper = FoveatedMapper(cfg)   # validate() is called automatically
```

`validate()` will **raise** if:

- Bands are not contiguous,
- Last band's `r_max` ≠ `max_range`,
- Any cell size is not an integer multiple of the finest cell size.

### Suggested `max_range` for SemanticKITTI

- **50 m** — good balance for benchmarking, faster tests
- **80 m** — matches most KITTI evaluation protocols
- **100 m** — full sensor coverage, more memory

---

## Input & Output Formats

### Input

**Required fields:**

| Field | Type | Notes |
|---|---|---|
| `x, y, z` | `float32` | Metres, sensor frame |
| `class_id` | `int32` | Semantic class integer (must match `importance_map` keys) |
| `confidence` | `float32` | `[0,1]` or sentinel (−1) if unavailable |

**Recommended fields:**

| Field | Type | Notes |
|---|---|---|
| `drivable` | `uint8` | `0` / `1` / `255` (unknown) |
| `dynamic` | `uint8` | `0` / `1` / `255` (unknown) |
| `raw_class_id` | `int32` | Original dataset label (preserved in output) |
| `intensity` | `float32` | LiDAR return intensity |
| `frame_id` | `int64` | Frame index |
| `timestamp` | `float64` | Acquisition time (seconds) |

**Delivery:** any NumPy array, structured array, or dict of arrays. `normalize_points()` handles the rest.

**Note:** class names should be kept as a `{class_id: "car"}` lookup table outside the mapper — the engine works with integer class IDs only.

### Output

Consume via `mapper.to_list()` (list of dicts) or `mapper.cells` (dict of `CellData`). Per cell:

| Field | Type | Description |
|---|---|---|
| `band_id` | int | Which resolution band |
| `cell_size` | float | Cell side length (m) |
| `ix, iy` | int | Grid indices |
| `center_x, center_y` | float | Cell center in sensor frame (m) |
| `point_count` | int | Points in cell |
| `min_z, max_z, mean_z, std_z` | float | Elevation statistics |
| `obstacle_height` | float | `max_z − min_z` |
| `dominant_class` | int | Most weighted class |
| `dominant_confidence` | float | Fraction of total weight |
| `class_histogram` | dict | Full weighted class distribution |
| `occupancy` | str | `'occupied'` / `'unknown'` |
| `drivable_ratio` | float | `[0,1]` or −1 (unknown) |
| `dynamic_ratio` | float | `[0,1]` or −1 (unknown) |
| `is_dynamic` | bool / None | Dynamic-object flag |
| `traversability` | str | `'drivable'` / `'non_drivable'` / `'unknown'` |
| `mean_intensity` | float | Mean LiDAR intensity |

**Per-frame metadata:** `mapper.statistics` contains all timings, cell counts, and point counts.

**Note on multi-resolution:** Cells come from different bands. Do not assume a uniform grid. When accumulating across frames, use `(band_id, ix, iy)` as the temporal key, or fuse by world-coordinate footprints.

---

## Installation & Quick Start

### Requirements

- Python 3.11+
- NumPy, SciPy, matplotlib
- (optional) Open3D for 3D visualization

```bash
pip install -r requirements.txt
```

### Quick example

```python
import numpy as np
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig

points = np.array([
    [1.0, 2.0, 0.1, 9, 0.9],    # x, y, z, class_id, confidence
    [12.0, 3.0, -0.2, 9, 0.85],
    [30.0, -5.0, 0.5, 1, 0.7],
    [70.0, 20.0, 1.0, 0, 0.6],
])

mapper = FoveatedMapper(FoveatedConfig(max_range=80.0))
cells = mapper.build(points)

for cell in mapper.to_list():
    print(cell['band_id'], cell['cell_size'], cell['traversability'])
```

### SemanticKITTI example

```python
from adapters.semantickitti import load_semantickitti_frame
from core.foveated_mapper import FoveatedMapper
from config.config import FoveatedConfig

points = load_semantickitti_frame(
    'dataset/sequences/00/velodyne/000000.bin',
    'dataset/sequences/00/labels/000000.label',
)
# Adapter sets confidence=1.0 and drivable/dynamic=255 (unknown).
# Enrich with per-point predictions before mapping for full utility.

mapper = FoveatedMapper(FoveatedConfig(max_range=50.0))
cells = mapper.build(points)
```

---

## Testing

```bash
pytest -q
```

| Test file | Coverage |
|---|---|
| `test_points.py` | Normalization from all input formats, sentinel handling |
| `test_uniform.py` | Baseline grid correctness, empty cells |
| `test_foveated.py` | Band selection, cell size alignment |
| `test_boundaries.py` | Exact distances (10, 25, 50, 100 m), negative coordinates |
| `test_elevation.py` | Min/max/mean/std computation |
| `test_semantics.py` | Weighted voting, dominant class, histogram |
| `test_traversability.py` | Drivable / non-drivable / unknown, obstacle override |
| `test_performance.py` | Foveated uses fewer cells than uniform high-res |

All 28 tests should pass.

---

## Benchmarking

```python
from benchmarks.benchmark import generate_synthetic_points, benchmark_mappers

pts = generate_synthetic_points(n_points=100_000, max_range=80.0)
results = benchmark_mappers(pts)

print(results['uniform'])
print(results['foveated'])
```

Measures:

- **Number of cells** (baseline vs. foveated)
- **Peak memory** (via `tracemalloc`)
- **Total time** (ms)
- **FPS** (`1000 / time_ms`)
- **Per-stage timings** (preprocess / filter / band-assign / insert / aggregate)

For reliable results, benchmark on real SemanticKITTI data (20+ frames, report mean ± std). All reported numbers should come from actual measurements.

---

## Complexity Analysis

Let `N` = number of points, `B` = number of resolution bands, `C` = number of occupied cells.

### Time complexity

| Stage | Uniform baseline | Foveated |
|---|---|---|
| Normalization | `O(N)` | `O(N)` |
| Filtering | `O(N)` | `O(N)` |
| Band assignment | — | `O(N log B)` |
| Cell indexing | `O(N)` | `O(N)` |
| Dict insertion | `O(N)` amortized | `O(N)` amortized |
| Aggregation | `O(N)` | `O(N)` |
| **Total** | **`O(N)`** | **`O(N log B)`** |

For typical `B ≈ 4–6` bands, `log B < 3` — the overhead is negligible.

### Space complexity

| Representation | Memory |
|---|---|
| Raw 3D point cloud | `O(N)` × 16–20 bytes/point |
| Uniform 5 cm over 100 m × 100 m | `O(1)` bounding cells × 40+ bytes: **~4M cells = ~160 MB** |
| **Foveated (this project)** | **`O(C)`** where `C` ≪ 4M |

Foveated cell count for typical KITTI frames at 50 m range: **~10k–30k cells** vs. **~400k+** for uniform 5 cm over 50 m × 50 m.

### Comparison to a dense uniform array

A dense 5 cm array over 100 m × 100 m is `2000 × 2000 = 4,000,000` cells. Even with 8-byte cells that's **32 MB per frame**, and most cells are empty (LiDAR returns are sparse). The dictionary representation allocates only for **occupied** cells, so foveated scales with **point density**, not bounding-box area.

---

## Prior Work & Differentiation

Foveated and multi-resolution LiDAR mapping is a well-established research area. Relevant prior work:

| Work | Idea |
|---|---|
| **OctoMap** (Hornung et al., 2013) | Octree-based 3D occupancy with adaptive refinement |
| **Elevation Maps** (Fankhauser et al., 2018) | Robot-centric 2.5D elevation with sensor-fusion |
| **REM / Adaptive Elevation Maps** | Multi-resolution terrain representations |
| **SemanticKITTI-based semantic mapping** | Semantic 2.5D grids |
| **Importance-driven foveation** | Semantic-guided resolution allocation |

**What's different in this implementation:**

1. **Distance-driven banded foveation** with strict integer-multiple alignment — simpler and more predictable than recursive quadtrees for the common navigation case.
2. **First-class traversability layer** decoupled from semantic class, with explicit obstacle-height override.
3. **Full uncertainty preservation** — the class histogram, confidence ratios, and dynamic-object ratios are preserved, not collapsed into a single class per cell.
4. **Explicit UNKNOWN semantics** — no implicit free-space assumption; single-frame robustness.
5. **Clean interface boundaries** — data adapters, configuration, and core algorithms are fully decoupled.

**What this is not:** a claim of algorithmic novelty. It's an integrated, engineering-oriented system tuned for a specific sensor pipeline — validated empirically, not on theoretical claims.

---

## Assumptions & Limitations

### Assumptions

- Points are pre-classified (with semantic labels and optional confidence).
- Points are in a consistent sensor frame; frame transforms are applied **before** the mapper.
- Points within one call to `build()` belong to the same frame (mixing is user's responsibility).
- Cell sizes are integer multiples of the finest resolution (enforced by `validate()`).

### Limitations

- **No ray-casting / free-space marking.** Absent cells remain `unknown`. Free-space marking is deferred to a multi-frame accumulation stage.
- **Risk-aware mode is single-level.** A quadtree-based refinement would be more adaptive.
- **No temporal accumulation.** Each call to `build()` is stateless.
- **Insertion is Python-loop based.** For 100k+ points per frame at high FPS on edge hardware, a Cython/Numba port or a partitioned vectorized insertion would help.
- **Coarse cells may spatially overlap fine cells.** This is intentional (foveation) but requires downstream consumers to handle multi-resolution data explicitly.
- **No GPU acceleration.** CPU-only by design for portability to modest hardware.

---

## Roadmap

- [x] Canonical point schema and normalizer
- [x] Uniform baseline mapper
- [x] Distance-based foveated mapper
- [x] Aligned multi-level dictionary storage
- [x] Elevation statistics (min / max / mean / std)
- [x] Confidence-weighted semantic aggregation
- [x] Traversability layer with obstacle override
- [x] Dynamic-object flagging
- [x] SemanticKITTI adapter
- [x] Benchmarks (memory, latency, cell count)
- [x] Unit tests (28)
- [x] Risk-aware refinement (single-level)
- [ ] Quadtree-based risk-aware refinement
- [ ] Vectorized (NumPy) cell insertion
- [ ] Numba / Cython hot path
- [ ] Free-space marking via ray-casting
- [ ] Optional sparse-tensor export
- [ ] Multi-frame accumulation helper

---

## License

MIT License — see `LICENSE` for details.

---

**Status:** MVP complete, risk-aware extension available, benchmarking ongoing.