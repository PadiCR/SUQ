# -*- coding: utf-8 -*-
"""
Core slope unit delineation, hierarchical basin tree, and nested multi-scale pruning.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .hydrology import HydroGrid, extract_half_basins


@dataclass
class SlopeUnitResult:
    """Result of slope unit delineation."""

    labels: np.ndarray        # (rows, cols) int32, 0 = nodata/masked
    cv_map: np.ndarray        # (rows, cols) float32, circular/normal variance per unit
    area_map: np.ndarray      # (rows, cols) float32, unit area in m^2
    n_units: int
    iterations_used: int


@dataclass
class BasinNode:
    """A node in the top-down hierarchical half-basin subdivision tree."""

    node_id: int
    parent_id: int            # 0 for root nodes
    iteration: int
    thresh_m2: float
    pixel_indices: np.ndarray # Flat indices belonging to this node (if leaf or intermediate)
    area_m2: float
    cv: float
    children: List[int] = field(default_factory=list)


@dataclass
class BasinHierarchy:
    """Complete precomputed top-down half-basin tree for instant nested & optimize."""

    shape: Tuple[int, int]
    nodes: Dict[int, BasinNode]
    roots: List[int]


def compute_unit_stats(
    labels: np.ndarray,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    cell_area: float,
    use_3d_normal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-label pixel count, area (m^2), and circular/normal variance in O(N).

    Returns (counts, areas_m2, cvs) indexed by label ID (index 0 is background).
    """
    lbl = labels.ravel()
    max_id = int(lbl.max())
    if max_id == 0:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    counts = np.bincount(lbl, minlength=max_id + 1)
    areas = counts.astype(np.float64) * cell_area
    cvs = np.zeros(max_id + 1, dtype=np.float64)

    valid_ids = counts > 0
    valid_ids[0] = False

    if use_3d_normal:
        sx = np.bincount(lbl, weights=nx.ravel(), minlength=max_id + 1)
        sy = np.bincount(lbl, weights=ny.ravel(), minlength=max_id + 1)
        sz = np.bincount(lbl, weights=nz.ravel(), minlength=max_id + 1)
        r_len = np.sqrt(sx * sx + sy * sy + sz * sz)
    else:
        ss = np.bincount(lbl, weights=sin_a.ravel(), minlength=max_id + 1)
        sc = np.bincount(lbl, weights=cos_a.ravel(), minlength=max_id + 1)
        r_len = np.hypot(ss, sc)

    with np.errstate(divide="ignore", invalid="ignore"):
        cvs[valid_ids] = 1.0 - (r_len[valid_ids] / counts[valid_ids])
    cvs = np.clip(cvs, 0.0, 1.0)
    return counts, areas, cvs


def threshold_schedule(
    thresh: float, cell_area: float, rf: int = 2, maxiteration: int = 20
) -> List[int]:
    """
    Accumulation thresholds (in cells) visited by r.slopeunits.create:
    thc = int(thresh / cell_area), then thc = int(thc - thc / rf) while thc >= rf.
    """
    rf = max(1, int(rf))
    thc = int(float(thresh) / float(cell_area))
    schedule: List[int] = []
    while thc >= rf and thc >= 1 and len(schedule) < max(1, int(maxiteration)) + 1:
        schedule.append(thc)
        nxt = int(thc - thc / rf)
        if nxt == thc:
            break
        thc = nxt
    return schedule


def clump_labels(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Equivalent of GRASS r.clump: give every 4-connected patch of equal value inside
    ``mask`` its own ID, numbered 1..K in raster scan order. Returns (labels, K).
    """
    rows, cols = values.shape
    n = rows * cols
    idx = np.arange(n, dtype=np.int64).reshape(rows, cols)
    h = mask[:, :-1] & mask[:, 1:] & (values[:, :-1] == values[:, 1:])
    v = mask[:-1, :] & mask[1:, :] & (values[:-1, :] == values[1:, :])
    src = np.concatenate([idx[:, :-1][h], idx[:-1, :][v]])
    dst = np.concatenate([idx[:, 1:][h], idx[1:, :][v]])
    graph = sparse.coo_matrix(
        (np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n)
    ).tocsr()
    _, comp = csgraph.connected_components(graph, directed=False)

    out = np.zeros(n, dtype=np.int32)
    m = mask.ravel()
    if not np.any(m):
        return out.reshape(rows, cols), 0
    uniq, first, inv = np.unique(comp[m], return_index=True, return_inverse=True)
    rank = np.empty(uniq.size, dtype=np.int32)
    rank[np.argsort(first, kind="stable")] = np.arange(1, uniq.size + 1, dtype=np.int32)
    out[m] = rank[inv]
    return out.reshape(rows, cols), int(uniq.size)


def create_slope_units(
    hydro: Optional[HydroGrid],
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    cell_area: float,
    thresh: float,
    areamin: float,
    cvmin: float,
    rf: int = 2,
    maxiteration: int = 20,
    areamax: Optional[float] = None,
    plains_mask: Optional[np.ndarray] = None,
    use_3d_normal: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
    cover_mask: Optional[np.ndarray] = None,
    half_basin_provider: Optional[Callable[[int], np.ndarray]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> SlopeUnitResult:
    """
    Line-by-line in-memory port of the r.slopeunits.create (v2.0) iteration.

    Given the same half-basins it returns the same partition as the GRASS module.

    Parameters beyond the GRASS ones:
      cover_mask           cells whose aspect counts in the statistics (GRASS has no aspect
                           on the one-cell DEM border). Default: every valid cell.
      half_basin_provider  callable(threshold_in_cells) -> int array of half-basin IDs
                           (0 = none). Default: the native SUQ engine on ``hydro``.
      valid_mask           DEM mask; only needed when ``hydro`` is None.
    """
    if hydro is not None:
        valid = hydro.valid_mask
    elif valid_mask is not None:
        valid = valid_mask
    else:
        raise ValueError("create_slope_units needs either hydro or valid_mask")
    if half_basin_provider is None:
        if hydro is None:
            raise ValueError("create_slope_units needs hydro or a half_basin_provider")

        def half_basin_provider(thc: int) -> np.ndarray:
            return extract_half_basins(hydro, thresh_cells=thc)

    shape = valid.shape
    dem_mask = valid.copy()
    if plains_mask is not None:
        dem_mask &= ~plains_mask

    cover = dem_mask if cover_mask is None else (cover_mask & dem_mask)
    cov = cover.astype(np.float64)
    if use_3d_normal:
        comps = [nx * cov, ny * cov, nz * cov]
    else:
        comps = [sin_a * cov, cos_a * cov]

    rf = max(1, int(rf))
    maxiteration = max(1, int(maxiteration))
    aminc = int(float(areamin) / cell_area)
    amaxc = int(float(areamax) / cell_area) if areamax is not None and areamax > 0 else None
    thc = int(float(thresh) / cell_area)
    n_planned = max(1, len(threshold_schedule(thresh, cell_area, rf, maxiteration)))

    def zonal_cv(zones: np.ndarray, sel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-zone count of cover cells and circular / normal-vector variance."""
        z = zones[sel]
        size = int(zones.max()) + 1
        cnt = np.bincount(z, weights=cov[sel], minlength=size)
        sq = np.zeros(size, dtype=np.float64)
        for comp in comps:
            sq += np.bincount(z, weights=comp[sel], minlength=size) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = 1.0 - np.sqrt(sq) / cnt
        return cnt, cv

    counter = 0
    control = 1
    slu: Optional[np.ndarray] = None                    # half-basins of the current iteration
    slu_done = np.zeros(shape, dtype=np.int64)          # slu_r_<counter>: finalised units
    todo = dem_mask.astype(np.int64)                    # slu_r_todo: still to be subdivided
    prev_todo = todo                                    # slu_r_todo_<last_counter>

    while control > 0 and counter < maxiteration and thc >= rf and thc >= 1:
        slu = np.asarray(half_basin_provider(thc)).astype(np.int64, copy=True)
        if slu.shape != shape:
            raise ValueError(
                f"Half-basin raster shape {slu.shape} does not match the DEM {shape}"
            )
        slu[~dem_mask] = 0

        # Statistics of each half-basin restricted to the cells still to be subdivided
        sel = (todo > 0) & (slu > 0)
        if not np.any(sel):
            break
        cnt, cv = zonal_cv(slu, sel)

        # Half-basins that are large and heterogeneous (or larger than areamax) stay "todo"
        with np.errstate(invalid="ignore"):
            keep = (cnt > aminc) & (cv > cvmin)
            if amaxc is not None:
                keep |= cnt > amaxc
        todo_new = np.where(sel & keep[slu], slu, 0)
        control = int(np.count_nonzero(todo_new))
        if control == 0:
            break

        counter += 1
        # Finalise everything that is not "todo"; earlier results take precedence
        cur = np.where(todo_new == 0, np.where(slu_done > 0, slu_done, slu), 0)

        # Reject over-fragmentation: where the mean size of the half-basins just finalised
        # inside a parent (previous "todo" half-basin) is below areamin, they get the ID of
        # the parent back and so form a single unit.
        par = (cur > 0) & (prev_todo > 0)
        if np.any(par):
            base = prev_todo[par]
            m = int(slu.max()) + 1
            pairs = np.unique(base * m + slu[par])
            diversity = np.bincount(pairs // m)
            cells = np.bincount(base, weights=cov[par], minlength=diversity.size)
            with np.errstate(divide="ignore", invalid="ignore"):
                revert = (cells / diversity) < aminc
            hit = np.zeros(shape, dtype=bool)
            hit[par] = revert[base]
            cur = np.where(hit, prev_todo, cur)

        slu_done = cur
        prev_todo = todo_new
        todo = todo_new
        thc = int(thc - thc / rf)

        if progress_cb is not None:
            progress_cb(min(90, 45 + int(45 * counter / n_planned)))

    # Cells still "todo" keep the last half-basins; DEM cells without any unit get value 1
    raw = slu_done if slu is None else np.where(slu_done > 0, slu_done, slu)
    raw = np.where((raw == 0) & dem_mask, 1, raw)
    labels, n_units = clump_labels(raw, dem_mask)

    # Area and variance of the final units
    cv_map = np.zeros(shape, dtype=np.float32)
    area_map = np.zeros(shape, dtype=np.float32)
    if n_units > 0:
        lab64 = labels.astype(np.int64)
        cnt, cv = zonal_cv(lab64, dem_mask)
        cells = np.bincount(lab64[dem_mask], minlength=n_units + 1)
        cv = np.clip(np.nan_to_num(cv, nan=0.0), 0.0, 1.0)
        cv_map[dem_mask] = cv[lab64[dem_mask]].astype(np.float32)
        area_map[dem_mask] = (cells[lab64[dem_mask]] * cell_area).astype(np.float32)

    return SlopeUnitResult(
        labels=labels,
        cv_map=cv_map,
        area_map=area_map,
        n_units=n_units,
        iterations_used=counter,
    )


def build_basin_hierarchy(
    hydro: HydroGrid,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    cell_area: float,
    thresh: float,
    min_stop_area: float,
    rf: int = 2,
    maxiteration: int = 20,
    plains_mask: Optional[np.ndarray] = None,
    use_3d_normal: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> BasinHierarchy:
    """
    Build a top-down hierarchical tree of half-basins once across the threshold schedule.

    Every child basin is strictly nested inside its parent basin, enabling
    instantaneous multi-scale nested layers and ultra-fast parameter optimization.
    """
    rows, cols = hydro.rows, hydro.cols
    active = hydro.valid_mask.copy()
    if plains_mask is not None:
        active &= ~plains_mask

    nodes: Dict[int, BasinNode] = {}
    roots: List[int] = []
    next_node_id = 1

    # Track which node_id each active pixel currently belongs to
    current_owner = np.zeros(rows * cols, dtype=np.int32)
    current_thresh = float(thresh)

    for it in range(max(1, int(maxiteration))):
        if not np.any(active):
            break

        hb = extract_half_basins(
            hydro, thresh_cells=current_thresh / cell_area, active_mask=active
        )
        if hb.max() == 0:
            break
        # Active cells that reach no channel (they leave the DEM directly) form one group
        hb = np.where(active & (hb == 0), hb.max() + 1, hb)

        hb_flat = hb.ravel()
        active_idx = np.flatnonzero(hb_flat > 0)
        if active_idx.size == 0:
            break

        # Group active pixels by (parent_owner, local_hb_id) so nesting is strictly preserved
        parents_at_idx = current_owner[active_idx]
        local_at_idx = hb_flat[active_idx]

        # Encode pair (parent, local) into 64-bit key for O(N log N) grouping
        keys = (parents_at_idx.astype(np.int64) << 32) | local_at_idx.astype(np.int64)
        order = np.argsort(keys)
        sorted_keys = keys[order]
        sorted_idx = active_idx[order]

        split_points = np.flatnonzero(np.diff(sorted_keys) != 0) + 1
        groups = np.split(sorted_idx, split_points)

        for grp_idx in groups:
            parent_id = int(current_owner[grp_idx[0]])
            g_count = grp_idx.size
            g_area = float(g_count) * cell_area

            # Compute exact variance for this (parent, local_hb) group
            if use_3d_normal:
                sx = float(nx.ravel()[grp_idx].sum())
                sy = float(ny.ravel()[grp_idx].sum())
                sz = float(nz.ravel()[grp_idx].sum())
                r_len = np.sqrt(sx * sx + sy * sy + sz * sz)
            else:
                ss = float(sin_a.ravel()[grp_idx].sum())
                sc = float(cos_a.ravel()[grp_idx].sum())
                r_len = float(np.hypot(ss, sc))
            g_cv = float(np.clip(1.0 - r_len / g_count, 0.0, 1.0))

            nid = next_node_id
            next_node_id += 1
            node = BasinNode(
                node_id=nid,
                parent_id=parent_id,
                iteration=it,
                thresh_m2=current_thresh,
                pixel_indices=grp_idx,
                area_m2=g_area,
                cv=g_cv,
            )
            nodes[nid] = node
            if parent_id == 0:
                roots.append(nid)
            else:
                nodes[parent_id].children.append(nid)

            current_owner[grp_idx] = nid
            if g_area < min_stop_area:
                active.ravel()[grp_idx] = False

        if progress_cb is not None:
            pct = 45 + int(40 * (it + 1) / max(1, int(maxiteration)))
            progress_cb(min(85, pct))

        if rf > 1:
            current_thresh = current_thresh - (current_thresh / float(rf))
        else:
            current_thresh = current_thresh * 0.5
        if current_thresh < cell_area:
            current_thresh = cell_area

    return BasinHierarchy(shape=(rows, cols), nodes=nodes, roots=roots)


def prune_basin_hierarchy(
    hierarchy: BasinHierarchy,
    cvmin: float,
    areamin: float,
    areamax: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Prune a precomputed BasinHierarchy at (cvmin, areamin, areamax) in milliseconds.

    Returns:
      labels: (rows, cols) int32 array of slope unit IDs (1..K)
      node_to_su: mapping from BasinNode.node_id -> final SU ID
    """
    rows, cols = hierarchy.shape
    labels_flat = np.zeros(rows * cols, dtype=np.int32)
    node_to_su: Dict[int, int] = {}
    su_id = 1

    stack = list(hierarchy.roots)
    while stack:
        nid = stack.pop()
        node = hierarchy.nodes[nid]
        is_leaf = len(node.children) == 0
        stop_area = node.area_m2 < areamin
        stop_cv = node.cv <= cvmin and (areamax is None or areamax <= 0 or node.area_m2 <= areamax)

        if is_leaf or stop_area or stop_cv:
            labels_flat[node.pixel_indices] = su_id
            node_to_su[nid] = su_id
            su_id += 1
        else:
            stack.extend(node.children)

    return labels_flat.reshape((rows, cols)), node_to_su


def create_nested_slope_units(
    hierarchy: BasinHierarchy,
    cvmin_levels: Sequence[float],
    areamin: float,
    areamax: Optional[float] = None,
) -> List[Tuple[float, np.ndarray, Dict[int, int]]]:
    """
    Generate multiple strictly-nested slope unit layers from a single BasinHierarchy.

    Returns a list of (cvmin, labels_2d, parent_su_map) ordered from finest to coarsest.
    """
    sorted_cvs = sorted(float(c) for c in cvmin_levels)
    layers: List[Tuple[float, np.ndarray, Dict[int, int]]] = []
    node_maps: List[Dict[int, int]] = []

    for cv_val in sorted_cvs:
        lbl, n2su = prune_basin_hierarchy(hierarchy, cvmin=cv_val, areamin=areamin, areamax=areamax)
        layers.append((cv_val, lbl, n2su))
        node_maps.append(n2su)

    # Build parent_su_id mapping from layer L to layer L+1
    results_with_parents: List[Tuple[float, np.ndarray, Dict[int, int]]] = []
    for idx, (cv_val, lbl, n2su) in enumerate(layers):
        parent_map: Dict[int, int] = {}
        if idx + 1 < len(layers):
            coarser_n2su = node_maps[idx + 1]
            for nid, fine_su in n2su.items():
                curr = nid
                while curr != 0 and curr not in coarser_n2su:
                    curr = hierarchy.nodes[curr].parent_id
                if curr in coarser_n2su:
                    parent_map[fine_su] = coarser_n2su[curr]
        results_with_parents.append((cv_val, lbl, parent_map))

    return results_with_parents


class CachedHalfBasins:
    """
    Memoises a half-basin provider (callable(threshold_in_cells) -> array) so that several
    delineations with different (cvmin, areamin) reuse the half-basins of each threshold.
    Least recently used rasters are dropped once ``max_bytes`` is exceeded.
    """

    def __init__(self, provider: Callable[[int], np.ndarray], max_bytes: float = 1.5e9):
        self.provider = provider
        self.max_bytes = float(max_bytes)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __call__(self, thc: int) -> np.ndarray:
        thc = int(thc)
        if thc in self._cache:
            self._cache.move_to_end(thc)
            return self._cache[thc]
        hb = np.asarray(self.provider(thc)).astype(np.int32, copy=False)
        self._cache[thc] = hb
        while len(self._cache) > 1 and sum(a.nbytes for a in self._cache.values()) > self.max_bytes:
            self._cache.popitem(last=False)
        return hb


def enforce_nesting(
    fine: np.ndarray, coarse: np.ndarray
) -> Tuple[np.ndarray, Dict[int, int], float]:
    """
    Make ``coarse`` a strict aggregation of the units of ``fine``.

    Every fine unit is given to the coarse unit that holds most of its cells; the coarse
    layer is then rebuilt as the union of the fine units it received (split into
    contiguous patches). Where the two layers already nest, nothing changes.

    Returns (nested_coarse_labels, {fine_id: coarse_id}, fraction_of_cells_reassigned).
    """
    mask = fine > 0
    if not np.any(mask):
        return np.zeros_like(fine, dtype=np.int32), {}, 0.0
    f = fine[mask].astype(np.int64)
    c = coarse[mask].astype(np.int64)
    k = int(c.max()) + 1
    pairs, counts = np.unique(f * k + c, return_counts=True)
    pf = pairs // k
    pc = pairs % k
    # prefer a real coarse unit (> 0), then the largest overlap
    score = counts.astype(np.float64) + np.where(pc > 0, 0.0, -1e15)
    order = np.lexsort((score, pf))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = pf[order][1:] != pf[order][:-1]
    parent_of = np.zeros(int(f.max()) + 1, dtype=np.int64)
    parent_of[pf[order][last]] = pc[order][last]

    grouped = np.zeros(fine.shape, dtype=np.int64)
    grouped[mask] = parent_of[f]
    moved = float(np.count_nonzero(grouped[mask] != c)) / float(f.size)
    # units of the coarse layer that received no fine unit vanish; give leftovers (0) an ID
    grouped[mask & (grouped == 0)] = k
    nested, _ = clump_labels(grouped, mask)

    ids, first = np.unique(f, return_index=True)
    parents = nested[mask][first]
    parent_map = {int(i): int(p) for i, p in zip(ids, parents)}
    return nested, parent_map, moved


def nest_layers(
    layers: Sequence[np.ndarray],
) -> List[Tuple[np.ndarray, Dict[int, int], float]]:
    """
    Turn independent delineations, ordered from finest to coarsest, into strictly nested
    layers. The finest layer is kept as it is; each following layer is rebuilt from the
    units of the (nested) layer below it with ``enforce_nesting``.

    Returns one (labels, parent_map, fraction_reassigned) per layer; ``parent_map`` links
    the units of a layer to the units of the next coarser layer (empty for the last one)
    and ``fraction_reassigned`` is the share of cells whose unit changed to obtain nesting.
    """
    out: List[Tuple[np.ndarray, Dict[int, int], float]] = []
    if not layers:
        return out
    current = np.asarray(layers[0]).astype(np.int32)
    moved_current = 0.0
    for nxt in layers[1:]:
        nested, parent_map, moved = enforce_nesting(current, np.asarray(nxt))
        out.append((current, parent_map, moved_current))
        current, moved_current = nested, moved
    out.append((current, {}, moved_current))
    return out
