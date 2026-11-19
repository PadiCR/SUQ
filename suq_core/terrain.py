# -*- coding: utf-8 -*-
"""
Terrain derivatives (Horn's 3x3 slope/aspect, 3D surface normals, auto-plains).
"""

from typing import Optional, Tuple
import numpy as np
from scipy import ndimage


def compute_slope_aspect(
    dem: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Horn (1981) 3x3 slope, aspect, sin/cos aspect, and 3D unit normal vectors.

    Aspect is measured in radians clockwise from North [0, 2*pi).
    Slope is in radians [0, pi/2].
    """
    z = dem.astype(np.float64, copy=False)
    if valid_mask is not None and not np.all(valid_mask):
        # Fill nodata cells using nearest valid value to avoid edge derivative spikes
        _, nearest_idx = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
        z = z[nearest_idx[0], nearest_idx[1]]

    padded = np.pad(z, 1, mode="edge")
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_size_x)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_size_y)

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))

    # Geographic aspect: 0 = North, pi/2 = East, pi = South, 3*pi/2 = West
    # Note: dz_dy is positive southward (increasing row index), so North-facing has dz_dy > 0.
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_rad = np.mod(aspect_rad, 2.0 * np.pi)
    # Flat cells have no aspect; GRASS r.slope.aspect codes them 0 (= East), keep the same
    aspect_rad = np.where((dz_dx == 0.0) & (dz_dy == 0.0), 0.5 * np.pi, aspect_rad)

    sin_a = np.sin(aspect_rad)
    cos_a = np.cos(aspect_rad)

    sin_s = np.sin(slope_rad)
    nx = sin_s * sin_a
    ny = sin_s * cos_a
    nz = np.cos(slope_rad)

    if valid_mask is not None:
        sin_a = np.where(valid_mask, sin_a, 0.0)
        cos_a = np.where(valid_mask, cos_a, 0.0)
        nx = np.where(valid_mask, nx, 0.0)
        ny = np.where(valid_mask, ny, 0.0)
        nz = np.where(valid_mask, nz, 0.0)

    return slope_rad, aspect_rad, sin_a, cos_a, nx, ny, nz


def detect_alluvial_plains(
    slope_rad: np.ndarray,
    cell_area: float,
    slope_thresh_deg: float = 5.0,
    min_plain_area_m2: float = 100000.0,
    valid_mask: Optional[np.ndarray] = None,
    edge_slope_deg: Optional[float] = None,
    dem: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    max_rel_position: Optional[float] = None,
    position_radius_m: float = 1000.0,
) -> np.ndarray:
    """
    Automatically detect contiguous flat/alluvial plain regions to mask out.

    Plains are connected areas with a slope of at most ``slope_thresh_deg`` and at least
    ``min_plain_area_m2``. With ``edge_slope_deg`` (> slope_thresh_deg) they are then grown
    into the connected cells with a slope of at most that value (the gentle sides of the
    valley floor). With ``dem`` and ``max_rel_position`` (< 1), patches whose relative
    position within ``position_radius_m`` is higher are dropped (see
    :func:`filter_plains_by_position`).

    Returns a boolean mask where True indicates an alluvial plain cell.
    """
    thresh_rad = np.deg2rad(slope_thresh_deg)
    flat = slope_rad <= thresh_rad
    if valid_mask is not None:
        flat &= valid_mask

    # Morphological closing removes isolated 1-2 px noise holes inside plains
    flat = ndimage.binary_closing(flat, structure=np.ones((3, 3), dtype=bool))
    if valid_mask is not None:
        flat &= valid_mask

    labeled, num_features = ndimage.label(flat)
    if num_features == 0:
        return np.zeros_like(flat, dtype=bool)

    counts = np.bincount(labeled.ravel())
    min_cells = max(1, int(np.ceil(min_plain_area_m2 / cell_area)))
    keep = counts >= min_cells
    keep[0] = False
    plains = keep[labeled]
    if edge_slope_deg is not None and edge_slope_deg > slope_thresh_deg and np.any(plains):
        gentle = slope_rad <= np.deg2rad(edge_slope_deg)
        if valid_mask is not None:
            gentle &= valid_mask
        gentle |= plains
        lab2, _ = ndimage.label(gentle)
        touched = np.unique(lab2[plains])
        grow = np.zeros(int(lab2.max()) + 1, dtype=bool)
        grow[touched] = True
        grow[0] = False
        plains = grow[lab2]
    if dem is not None and max_rel_position is not None and max_rel_position < 1.0 and np.any(plains):
        plains = filter_plains_by_position(
            plains, dem, cell_size, max_rel_position, position_radius_m, valid_mask
        )
    return plains


def relative_position(
    patch: np.ndarray, dem: np.ndarray, radius_cells: int, valid: Optional[np.ndarray] = None
) -> float:
    """
    Position of a patch in its surroundings: median elevation of the patch scaled between
    the lowest (0) and highest (1) elevation within ``radius_cells`` of it.
    """
    rows, cols = np.nonzero(patch)
    r0, r1 = max(rows.min() - radius_cells, 0), min(rows.max() + radius_cells + 1, dem.shape[0])
    c0, c1 = max(cols.min() - radius_cells, 0), min(cols.max() + radius_cells + 1, dem.shape[1])
    m = patch[r0:r1, c0:c1]
    z = dem[r0:r1, c0:c1]
    win = ndimage.distance_transform_edt(~m) <= radius_cells
    win &= np.isfinite(z)
    if valid is not None:
        win &= valid[r0:r1, c0:c1]
    zmin, zmax = float(z[win].min()), float(z[win].max())
    if zmax <= zmin:
        return 0.0
    return (float(np.median(z[m])) - zmin) / (zmax - zmin)


def filter_plains_by_position(
    plains: np.ndarray,
    dem: np.ndarray,
    cell_size: float,
    max_rel_position: float = 0.3,
    radius_m: float = 1000.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Drop the plain patches that are not at the bottom of their surroundings (relative
    position above ``max_rel_position``): flat hilltops, saddles and terraces high on a
    slope, which are not alluvial plains.
    """
    radius = max(1, int(round(radius_m / cell_size)))
    comp, n = ndimage.label(plains)
    out = plains.copy()
    for i, sl in enumerate(ndimage.find_objects(comp), 1):
        if sl is None:
            continue
        r0, r1 = max(sl[0].start - radius, 0), min(sl[0].stop + radius, dem.shape[0])
        c0, c1 = max(sl[1].start - radius, 0), min(sl[1].stop + radius, dem.shape[1])
        patch = comp[r0:r1, c0:c1] == i
        v = valid_mask[r0:r1, c0:c1] if valid_mask is not None else None
        if relative_position(patch, dem[r0:r1, c0:c1], radius, v) > max_rel_position:
            out[comp == i] = False
    return out


def aspect_cover_mask(valid_mask: np.ndarray) -> np.ndarray:
    """
    Cells for which GRASS r.slope.aspect returns an aspect: the full 3x3 window must be
    inside the raster and free of nodata. r.slopeunits ignores the other cells when it
    counts cells and sums aspect vectors.
    """
    rows, cols = valid_mask.shape
    padded = np.pad(valid_mask, 1, mode="constant", constant_values=False)
    cover = np.ones((rows, cols), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            cover &= padded[dr:dr + rows, dc:dc + cols]
    return cover


def detect_water_bodies(
    dem: np.ndarray,
    cell_area: float,
    min_area_m2: float = 5000.0,
    tolerance_m: float = 0.05,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect water bodies as connected, perfectly flat areas of the DEM.

    Lakes and reservoirs are stored flat in most DEMs (one elevation for the whole water
    surface). A cell is flat when all its valid 4-neighbours lie within ``tolerance_m`` of
    its elevation; connected flat areas of at least ``min_area_m2`` are kept and grown by
    one cell onto the shore cells at the same level.

    Returns a boolean mask where True indicates a water body cell.
    """
    valid = np.isfinite(dem) if valid_mask is None else (valid_mask & np.isfinite(dem))
    z = np.where(valid, dem, np.nan)
    pad = np.pad(z, 1, constant_values=np.nan)
    core = pad[1:-1, 1:-1]
    flat = valid.copy()
    for nb in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
        with np.errstate(invalid="ignore"):
            flat &= ~(np.abs(nb - core) > tolerance_m)       # NaN neighbours are ignored
    labeled, n = ndimage.label(flat)
    if n == 0:
        return np.zeros(dem.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    min_cells = max(2, int(np.ceil(float(min_area_m2) / cell_area)))
    keep = counts >= min_cells
    keep[0] = False
    if not np.any(keep):
        return np.zeros(dem.shape, dtype=bool)
    water = keep[labeled]
    # level of each water body, then one cell of shore at the same level
    level = ndimage.mean(np.where(valid, dem, 0.0), labels=labeled, index=np.arange(n + 1))
    level = np.asarray(level, dtype=np.float64)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    near = ndimage.grey_dilation(np.where(water, labeled, 0), footprint=cross)
    shore = valid & ~water & (near > 0)
    with np.errstate(invalid="ignore"):
        shore &= np.abs(np.where(valid, dem, np.nan) - level[near]) <= tolerance_m
    return water | shore
