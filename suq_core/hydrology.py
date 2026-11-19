# -*- coding: utf-8 -*-
"""
Pure NumPy/SciPy hydrological routing and half-basin delineation.

Implements:
1. Priority-Flood depression filling with flat resolution (Barnes et al., 2014).
2. Acyclic D8 flow direction and vectorized wavefront topological ordering.
3. Vectorized flow accumulation.
4. Left/right half-basin extraction from stream segments, built like GRASS r.watershed
   (inflow/outflow bank rule and thalweg extension above channel heads).
"""

from dataclasses import dataclass
import heapq
from typing import Callable, List, Optional, Tuple
import numpy as np

# D8 neighbor offsets: E, SE, S, SW, W, NW, N, NE
_DR = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int32)
_DC = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)
_DIST = np.array([1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)], dtype=np.float64)


@dataclass
class HydroGrid:
    """Precomputed acyclic D8 hydrological structure for a DEM."""

    rows: int
    cols: int
    valid_mask: np.ndarray      # (rows, cols) bool
    receiver: np.ndarray        # (rows * cols,) int32, downstream flat index or -1
    flow_dr: np.ndarray         # (rows * cols,) int8, row step to receiver
    flow_dc: np.ndarray         # (rows * cols,) int8, col step to receiver
    waves: List[np.ndarray]     # Topological wavefronts from ridge -> outlet
    acc: np.ndarray             # (rows, cols) float64, upslope contributing cells
    main_donor: np.ndarray      # (rows * cols,) int32, donor with the largest accumulation or -1


def _priority_flood_fill(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Fill depressions using Priority-Flood so every valid cell drains to an outlet.
    Uses a tiny epsilon step on filled/flat cells to guarantee a strictly acyclic D8 graph.
    """
    rows, cols = dem.shape
    filled = dem.astype(np.float64, copy=True)
    closed = ~valid_mask

    heap: List[Tuple[float, int, int]] = []
    seq = 0

    # Seed with all valid cells on the raster border or adjacent to nodata
    border_mask = np.zeros((rows, cols), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    # Also seed valid cells adjacent to invalid/nodata cells
    if not np.all(valid_mask):
        invalid = ~valid_mask
        adj_invalid = np.zeros_like(valid_mask)
        adj_invalid[1:, :] |= invalid[:-1, :]
        adj_invalid[:-1, :] |= invalid[1:, :]
        adj_invalid[:, 1:] |= invalid[:, :-1]
        adj_invalid[:, :-1] |= invalid[:, 1:]
        seed_mask = valid_mask & (border_mask | adj_invalid)
    else:
        seed_mask = valid_mask & border_mask

    seed_rc = np.argwhere(seed_mask)
    for r, c in seed_rc:
        r_i, c_i = int(r), int(c)
        closed[r_i, c_i] = True
        heapq.heappush(heap, (float(filled[r_i, c_i]), seq, r_i * cols + c_i))
        seq += 1

    total_valid = int(np.count_nonzero(valid_mask))
    processed = len(seed_rc)
    last_pct = -1

    dr_list = _DR.tolist()
    dc_list = _DC.tolist()

    while heap:
        elev, _, flat_idx = heapq.heappop(heap)
        r = flat_idx // cols
        c = flat_idx % cols

        for k in range(8):
            nr = r + dr_list[k]
            nc = c + dc_list[k]
            if 0 <= nr < rows and 0 <= nc < cols and not closed[nr, nc]:
                closed[nr, nc] = True
                n_elev = filled[nr, nc]
                if n_elev <= elev:
                    # Enforce tiny positive gradient out of pits/flats
                    n_elev = np.nextafter(elev, np.inf)
                    filled[nr, nc] = n_elev
                heapq.heappush(heap, (float(n_elev), seq, nr * cols + nc))
                seq += 1
                processed += 1

        if progress_cb is not None and total_valid > 0:
            pct = int(30 * processed / total_valid)
            if pct > last_pct:
                last_pct = pct
                progress_cb(pct)

    return filled


def prepare_hydro_grid(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> HydroGrid:
    """
    Build the acyclic D8 flow network, topological wavefronts, and flow accumulation.
    """
    rows, cols = dem.shape
    n_cells = rows * cols
    filled = _priority_flood_fill(dem, valid_mask, progress_cb=progress_cb)

    # Compute physical neighbor distances
    phys_dist = np.hypot(_DR * cell_size_y, _DC * cell_size_x)

    # Vectorized D8 steepest descent over all 8 directions
    best_slope = np.zeros((rows, cols), dtype=np.float64)
    best_dir = np.full((rows, cols), -1, dtype=np.int8)

    for k in range(8):
        dr = int(_DR[k])
        dc = int(_DC[k])
        r_src_start = max(0, -dr)
        r_src_end = rows - max(0, dr)
        c_src_start = max(0, -dc)
        c_src_end = cols - max(0, dc)

        r_dst_start = max(0, dr)
        r_dst_end = rows - max(0, -dr)
        c_dst_start = max(0, dc)
        c_dst_end = cols - max(0, -dc)

        src_z = filled[r_src_start:r_src_end, c_src_start:c_src_end]
        dst_z = filled[r_dst_start:r_dst_end, c_dst_start:c_dst_end]
        dst_valid = valid_mask[r_dst_start:r_dst_end, c_dst_start:c_dst_end]

        slope_k = (src_z - dst_z) / phys_dist[k]
        better = dst_valid & (slope_k > best_slope[r_src_start:r_src_end, c_src_start:c_src_end])
        best_slope[r_src_start:r_src_end, c_src_start:c_src_end] = np.where(
            better, slope_k, best_slope[r_src_start:r_src_end, c_src_start:c_src_end]
        )
        best_dir[r_src_start:r_src_end, c_src_start:c_src_end] = np.where(
            better, k, best_dir[r_src_start:r_src_end, c_src_start:c_src_end]
        )

    best_dir[~valid_mask] = -1

    flat_dir = best_dir.ravel()
    receiver = np.full(n_cells, -1, dtype=np.int32)
    flow_dr = np.zeros(n_cells, dtype=np.int8)
    flow_dc = np.zeros(n_cells, dtype=np.int8)

    has_rec = flat_dir >= 0
    src_idx = np.flatnonzero(has_rec).astype(np.int32)
    dirs = flat_dir[src_idx]
    dr_vec = _DR[dirs]
    dc_vec = _DC[dirs]
    receiver[src_idx] = src_idx + dr_vec * cols + dc_vec
    flow_dr[src_idx] = dr_vec.astype(np.int8)
    flow_dc[src_idx] = dc_vec.astype(np.int8)

    # Vectorized wavefront topological sort & flow accumulation
    valid_flat = valid_mask.ravel()
    in_degree = np.zeros(n_cells, dtype=np.int32)
    valid_targets = receiver[has_rec]
    np.add.at(in_degree, valid_targets, 1)

    acc_flat = np.where(valid_flat, 1.0, 0.0)
    frontier = np.flatnonzero(valid_flat & (in_degree == 0)).astype(np.int32)
    waves: List[np.ndarray] = []

    while frontier.size > 0:
        waves.append(frontier)
        rec = receiver[frontier]
        mask_rec = rec >= 0
        if not np.any(mask_rec):
            break
        f_with_rec = frontier[mask_rec]
        targets = rec[mask_rec]

        np.add.at(acc_flat, targets, acc_flat[f_with_rec])
        np.add.at(in_degree, targets, -1)

        uniq_targets = np.unique(targets)
        frontier = uniq_targets[in_degree[uniq_targets] == 0]

    # Donor with the largest accumulation for every cell (threshold independent).
    # It defines the upstream continuation of a channel and the thalweg above channel heads.
    main_donor = np.full(n_cells, -1, dtype=np.int32)
    if src_idx.size > 0:
        order = np.lexsort((acc_flat[src_idx], receiver[src_idx]))
        s_sorted = src_idx[order]
        r_sorted = receiver[s_sorted]
        last = np.ones(r_sorted.size, dtype=bool)
        last[:-1] = r_sorted[1:] != r_sorted[:-1]
        main_donor[r_sorted[last]] = s_sorted[last]

    if progress_cb is not None:
        progress_cb(45)

    return HydroGrid(
        rows=rows,
        cols=cols,
        valid_mask=valid_mask,
        receiver=receiver,
        flow_dr=flow_dr,
        flow_dc=flow_dc,
        waves=waves,
        acc=acc_flat.reshape((rows, cols)),
        main_donor=main_donor,
    )


# Lookup (dr + 1) * 3 + (dc + 1) -> clockwise direction index (E, SE, S, SW, W, NW, N, NE)
_K_OF = np.full(9, -1, dtype=np.int8)
for _k in range(8):
    _K_OF[(int(_DR[_k]) + 1) * 3 + (int(_DC[_k]) + 1)] = _k


def extract_half_basins(
    hydro: HydroGrid,
    thresh_cells: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract left/right half-basins for an accumulation threshold (in cells), following
    the construction used by GRASS r.watershed:

    * stream segments are split at confluences, one basin (two half-basins) per segment;
    * above each channel head the thalweg is extended up to the divide along the donor
      with the largest accumulation, so headwater areas are split into left and right;
    * the side of a hillslope cell is decided from BOTH the inflow and the outflow
      direction of the channel cell it enters (the channel path splits the 8 neighbours
      into two arcs), which keeps each bank on one side through bends of a D8 channel;
    * cells that leave the DEM without reaching a channel belong to no half-basin (0).

    Returns an int32 array (rows, cols): 0 = no half-basin / outside, 1..K = half-basin IDs.
    If ``active_mask`` is given the result is clipped to it (routing is never re-done).
    """
    rows, cols = hydro.rows, hydro.cols
    n_cells = rows * cols
    valid = hydro.valid_mask.ravel()
    rec = hydro.receiver
    acc = hydro.acc.ravel()
    main_donor = hydro.main_donor

    is_stream = valid & (acc >= max(1.0, float(thresh_cells)))
    if not np.any(is_stream):
        return np.zeros((rows, cols), dtype=np.int32)

    has_rec = rec >= 0
    recs = np.maximum(rec, 0)

    # --- stream segments: a new segment starts below every confluence -----------------
    stream_cells = np.flatnonzero(is_stream)
    s_rec = rec[stream_cells]
    s_ok = (s_rec >= 0) & is_stream[np.maximum(s_rec, 0)]
    stream_in_deg = np.bincount(s_rec[s_ok], minlength=n_cells)

    seg = np.zeros(n_cells, dtype=np.int64)
    is_bottom = is_stream & ((~has_rec) | (~is_stream[recs]) | (stream_in_deg[recs] > 1))
    bottoms = np.flatnonzero(is_bottom)
    seg[bottoms] = np.arange(1, bottoms.size + 1, dtype=np.int64)

    # --- channel = streams + thalweg extension above the heads ------------------------
    chan = is_stream.copy()
    is_main = has_rec & (main_donor[recs] == np.arange(n_cells))
    for wave in reversed(hydro.waves):
        w = wave[has_rec[wave]]
        if w.size == 0:
            continue
        s = is_stream[w]
        sw = w[s & (seg[w] == 0)]
        if sw.size:
            seg[sw] = seg[rec[sw]]
        vw = w[(~s) & is_main[w] & chan[rec[w]]]
        if vw.size:
            chan[vw] = True
            seg[vw] = seg[rec[vw]]

    # --- direction indices of the channel path through each channel cell --------------
    k_flow = _K_OF[(hydro.flow_dr.astype(np.int64) + 1) * 3 + (hydro.flow_dc.astype(np.int64) + 1)]
    k_flow = np.where(has_rec, k_flow, -1).astype(np.int64)

    up = main_donor.astype(np.int64)
    has_up = (up >= 0) & chan[np.maximum(up, 0)]
    k_up = np.where(has_up, (k_flow[np.maximum(up, 0)] + 4) % 8, -1)
    k_down = k_flow.copy()
    k_down = np.where((k_down < 0) & (k_up >= 0), (k_up + 4) % 8, k_down)
    k_up = np.where((k_up < 0) & (k_down >= 0), (k_down + 4) % 8, k_up)
    k_down = np.where(k_down < 0, 0, k_down)
    k_up = np.where(k_up < 0, 0, k_up)
    span = (k_up - k_down) % 8          # clockwise arc from outflow to inflow = "right" side

    # --- hillslope cells entering a channel cell --------------------------------------
    hb = np.zeros(n_cells, dtype=np.int64)
    bank = np.flatnonzero(valid & (~chan) & has_rec & chan[recs])
    n_right = np.zeros(n_cells, dtype=np.int64)
    n_left = np.zeros(n_cells, dtype=np.int64)
    if bank.size:
        c = rec[bank]
        k_in = (k_flow[bank] + 4) % 8    # direction from the channel cell towards the bank cell
        right = ((k_in - k_down[c]) % 8) < span[c]
        hb[bank] = 2 * seg[c] + right
        n_right = np.bincount(c[right], minlength=n_cells)
        n_left = np.bincount(c[~right], minlength=n_cells)

    # Channel cells join the side that sends them more hillslope cells (as r.watershed does)
    ch = np.flatnonzero(chan)
    hb[ch] = 2 * seg[ch] + (n_right[ch] > n_left[ch])

    # --- propagate upslope through the remaining hillslope cells ----------------------
    todo = valid & (~chan) & has_rec
    for wave in reversed(hydro.waves):
        w = wave[todo[wave]]
        w = w[hb[w] == 0]
        if w.size:
            hb[w] = hb[rec[w]]

    if active_mask is not None:
        hb[~active_mask.ravel()] = 0

    # Relabel to compact 1..K, keeping 0 as "no half-basin"
    out = np.zeros(n_cells, dtype=np.int32)
    pos = hb > 0
    if np.any(pos):
        _, inv = np.unique(hb[pos], return_inverse=True)
        out[pos] = (inv + 1).astype(np.int32)
    return out.reshape((rows, cols))
