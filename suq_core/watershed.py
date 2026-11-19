# -*- coding: utf-8 -*-
"""
Native port of GRASS GIS r.watershed (in-memory "ram" version) for SUQ.

This follows the C code of r.watershed step by step, so that SUQ can produce the same
half-basins as GRASS without GRASS being installed:

  init_vars.c   elevation scaling to integers, start cells along the DEM edge
  do_astar.c    A* least-cost search (no depression filling), drainage directions
  do_flatarea.c flat area "beautification" after Garbrecht & Martz (1997), flag -b
  do_cum.c      flow accumulation, single (SFD) or multiple flow direction (MFD,
                Holmgren 1994 with a convergence exponent), main drainage direction
                and stream ("swale") initiation
  find_pour.c, def_basin.c, split_str.c, no_stream.c, haf_side.c, over_cells.c
                basins and left/right half-basins

r.watershed is (C) the GRASS Development Team, GPL >= 2 (C. Ehlschlaeger, M. Metz et al.).

The port reproduces r.watershed as called by r.slopeunits (flags -a -b). Routing runs in
pure Python: expect roughly 15-20 s per million cells.

Direction codes are those of r.watershed: 1 = NE, 2 = N, 3 = NW, 4 = W, 5 = SW, 6 = S,
7 = SE, 8 = E; negative = the cell drains out of the DEM.
"""

from collections import deque
from dataclasses import dataclass
import heapq
from typing import Callable, List, Optional

import numpy as np

# Neighbour order used by r.watershed: S, N, W, E, NE, SW, SE, NW
_NEXTDR = (1, -1, 0, 0, -1, 1, 1, -1)
_NEXTDC = (0, 0, -1, 1, 1, -1, 1, -1)
# For a diagonal neighbour: index of the cardinal neighbour one E-W / N-S step away from it
_NBR_EW = (0, 1, 2, 3, 1, 0, 0, 1)
_NBR_NS = (0, 1, 2, 3, 3, 2, 3, 2)
# drain[rr][cc]: direction a neighbour at offset (rr-1, cc-1) needs to flow into the centre
_DRAIN = ((7, 6, 5), (8, 0, 4), (1, 2, 3))
# updrain[rr][cc]: direction from the centre towards the neighbour at offset (rr-1, cc-1)
_UPDRAIN = ((3, 2, 1), (4, 0, 8), (5, 6, 7))
_ASP_R = (0, -1, -1, -1, 0, 1, 1, 1, 0)
_ASP_C = (0, 1, 0, -1, -1, -1, 0, 1, 1)
_ELE_SCALE = 10000


@dataclass
class WatershedGrid:
    """Threshold-independent result of the r.watershed routing (padded by one cell)."""

    rows: int
    cols: int
    valid_mask: np.ndarray        # (rows, cols) bool
    mfd: bool
    pcols: int                    # padded number of columns
    asp: List[int]                # drainage direction codes (padded, flat)
    wat: List[float]              # flow accumulation in cells (padded, flat)
    order: List[int]              # cells from highest to lowest in A* order
    pos: np.ndarray               # position of each cell in ``order``
    edge: List[bool]              # cell touches nodata or the raster border
    edge_code: List[int]          # for edge cells that drain inwards: direction to nodata
    rmax: List[int]               # main downstream cell (continues a stream)
    flat: np.ndarray              # bool, no elevation difference to downstream cells
    max_up: np.ndarray            # largest accumulation among higher neighbours
    active: np.ndarray            # bool, cells that may start or continue a stream
    waves: List[np.ndarray]       # topological wavefronts of the overland flow graph
    receiver: np.ndarray          # overland receiver (padded flat index) or -1

    @property
    def accumulation(self) -> np.ndarray:
        """Flow accumulation (number of cells) as a (rows, cols) array."""
        a = np.asarray(self.wat, dtype=np.float64).reshape(self.rows + 2, self.pcols)
        return a[1:-1, 1:-1].copy()

    @property
    def drainage(self) -> np.ndarray:
        """Drainage direction codes as a (rows, cols) array (0 outside the DEM)."""
        a = np.asarray(self.asp, dtype=np.int32).reshape(self.rows + 2, self.pcols)
        return a[1:-1, 1:-1].copy()


def prepare_watershed_grid(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    mfd: bool = True,
    convergence: int = 5,
    progress_cb: Optional[Callable[[int], None]] = None,
    beautify_flats: bool = True,
) -> WatershedGrid:
    """
    A* search and flow accumulation of r.watershed (SFD or MFD).
    ``beautify_flats`` corresponds to the -b flag, which r.slopeunits always sets.
    """
    rows, cols = dem.shape
    prow, pcol = rows + 2, cols + 2
    n = prow * pcol
    ew_res = float(cell_size_x)
    ns_res = float(cell_size_y)
    diag = float(np.hypot(ew_res, ns_res))
    convergence = min(10, max(1, int(convergence)))

    valid_p = np.zeros((prow, pcol), dtype=bool)
    valid_p[1:-1, 1:-1] = valid_mask
    in_region_p = np.zeros((prow, pcol), dtype=bool)
    in_region_p[1:-1, 1:-1] = True

    # init_vars.c: elevation as scaled, rounded integers
    scaled = np.where(valid_mask, dem, 0.0).astype(np.float64) * _ELE_SCALE
    alt_np = np.where(scaled >= 0.0, np.floor(scaled + 0.5), -np.floor(-scaled + 0.5))
    alt_p = np.zeros((prow, pcol), dtype=np.int64)
    alt_p[1:-1, 1:-1] = alt_np.astype(np.int64)

    # A cell is an "edge" if one of its 8 neighbours is nodata or outside the raster
    edge_p = np.zeros((prow, pcol), dtype=bool)
    for k in range(8):
        dr, dc = _NEXTDR[k], _NEXTDC[k]
        edge_p[1:-1, 1:-1] |= ~valid_p[1 + dr:prow - 1 + dr, 1 + dc:pcol - 1 + dc]
    edge_p &= valid_p

    V = valid_p.ravel().tolist()
    IN = in_region_p.ravel().tolist()
    alt = alt_p.ravel().tolist()
    edge = edge_p.ravel().tolist()
    off = [_NEXTDR[k] * pcol + _NEXTDC[k] for k in range(8)]
    drain_to = [_DRAIN[_NEXTDR[k] + 1][_NEXTDC[k] + 1] for k in range(8)]   # nbr k -> centre
    up_code = [_UPDRAIN[_NEXTDR[k] + 1][_NEXTDC[k] + 1] for k in range(8)]  # centre -> nbr k
    asp_off = [_ASP_R[a] * pcol + _ASP_C[a] for a in range(9)]

    def report(pct: int) -> None:
        if progress_cb is not None:
            progress_cb(pct)

    # ---------------------------------------------------------------- start cells
    asp = [0] * n
    in_list = [not v for v in V]
    worked = [not v for v in V]
    heap = []
    seq = 0
    for i in np.flatnonzero(edge_p.ravel()).tolist():
        r = i // pcol - 1
        c = i % pcol - 1
        if r == 0:
            a = -2
        elif c == 0:
            a = -4
        elif r == rows - 1:
            a = -6
        elif c == cols - 1:
            a = -8
        elif not V[i - pcol]:
            a = -2
        elif not V[i + pcol]:
            a = -6
        elif not V[i - 1]:
            a = -4
        elif not V[i + 1]:
            a = -8
        elif not V[i - pcol - 1]:
            a = -3
        elif not V[i - pcol + 1]:
            a = -1
        elif not V[i + pcol - 1]:
            a = -5
        else:
            a = -7
        asp[i] = a
        in_list[i] = True
        heap.append((alt[i], seq, i))
        seq += 1
    heapq.heapify(heap)

    # ---------------------------------------------------------------- do_flatarea.c
    # ``alt`` keeps the original elevations, ``alt_mod`` the ones modified inside flat areas
    # (only used to order the A* search; accumulation uses the original elevations).
    alt_mod = list(alt)
    flat_done = [False] * n

    def do_flatarea(index: int, ele: int) -> None:
        orders = {index: [0, -1, 0]}          # cell -> [uphill order, downhill order, flag]
        down_q = deque([index])
        up_q = deque([index])
        n_flat = 1
        # collect the flat area; cells already in the A* list are the uphill start points
        while down_q:
            d = down_q.popleft()
            flat_done[d] = True
            for k in range(8):
                u = d + off[k]
                if IN[u] and alt[u] == ele and not worked[u] and u not in orders:
                    n_flat += 1
                    down_q.append(u)
                    if in_list[u]:
                        up_q.append(u)
                        orders[u] = [0, -1, 0]
                    else:
                        orders[u] = [-1, -1, 0]
        if n_flat < 5:
            return                            # too small, not worth the effort

        # gradient away from the outlet side (uphill order)
        max_uphill = max_downhill = 0
        while up_q:
            d = up_q.popleft()
            this_in_list = in_list[d]
            o = orders[d]
            uphill = o[0]
            if max_uphill < uphill:
                max_uphill = uphill
            uphill += 1
            in_down_queue = False
            for k in range(8):
                u = d + off[k]
                if not IN[u]:
                    continue
                ele_nbr = alt[u]
                nbr_in_list = in_list[u]
                nbr_worked = worked[u]
                if ele_nbr == ele and not nbr_worked:
                    no = orders[u]
                    if no[0] == -1:
                        up_q.append(u)
                        no[0] = uphill
                # flat cells next to other, not yet visited terrain start the downhill order
                if (not this_in_list and not in_down_queue and ele_nbr != ele
                        and not nbr_in_list and not nbr_worked):
                    down_q.append(d)
                    o[1] = 0
                    in_down_queue = True

        # gradient away from higher terrain (downhill order)
        up_q = deque()
        while down_q:
            d = down_q.popleft()
            downhill = orders[d][1]
            if max_downhill < downhill:
                max_downhill = downhill
            downhill += 1
            for k in range(8):
                u = d + off[k]
                if IN[u] and alt[u] == ele and not worked[u]:
                    no = orders[u]
                    if no[1] == -1:
                        down_q.append(u)
                        no[1] = downhill
                        if in_list[u]:
                            up_q.append(u)
                            no[2] = 1

        # adjust the elevations
        while up_q:
            d = up_q.popleft()
            o = orders[d]
            uphill, downhill = o[0], o[1]
            if in_list[d]:
                downhill = max_downhill
                uphill = 0
            alt_mod[d] = int(
                alt_mod[d] + ((uphill + (max_downhill - downhill) / 2.0 + 0.5) / 2.0 + 0.5)
            )
            for k in range(8):
                u = d + off[k]
                if IN[u] and alt[u] == ele and not worked[u]:
                    no = orders[u]
                    if no[2] == 0:
                        up_q.append(u)
                        no[2] = 1

    # ---------------------------------------------------------------- do_astar.c
    dist = (ns_res, ns_res, ew_res, ew_res, diag, diag, diag, diag)
    total = max(1, int(np.count_nonzero(valid_mask)))
    step = max(1, total // 25)
    pop_order = []
    heappop, heappush = heapq.heappop, heapq.heappush
    slope = [-1.0] * 8
    alt_nbr = [0] * 8
    while heap:
        _, _, i = heappop(heap)
        pop_order.append(i)
        a_val = alt_mod[i]
        flat_is_done = flat_done[i] or not beautify_flats
        for k in range(8):
            j = i + off[k]
            slope[k] = -1.0
            alt_nbr[k] = 0
            if not IN[j]:
                continue
            is_in_list = in_list[j]
            is_worked = worked[j]
            aj = alt_mod[j]
            if beautify_flats and not is_in_list and not is_worked:
                a_val = alt[i]
                if not flat_is_done and alt[j] == a_val:
                    do_flatarea(i, a_val)
                    flat_is_done = True
                if flat_done[j]:
                    a_val = alt_mod[i]
                    aj = alt_mod[j]
                else:
                    aj = alt[j]
            alt_nbr[k] = aj
            if not is_worked:
                slope[k] = 0.0 if a_val >= aj else (aj - a_val) / dist[k]
            skip_diag = False
            # avoid diagonal flow direction bias
            if k > 3 and slope[k] > 0 and ((not is_in_list) or ((not is_worked) and asp[j] < 0)):
                e = _NBR_EW[k]
                if slope[e] >= 0:
                    ae = alt_nbr[e]
                    if ae < aj and slope[k] < (aj - ae) / ew_res:
                        skip_diag = True
                if not skip_diag:
                    m = _NBR_NS[k]
                    if slope[m] >= 0:
                        am = alt_nbr[m]
                        if am < aj and slope[k] < (aj - am) / ns_res:
                            skip_diag = True
            if not skip_diag:
                if not is_in_list:
                    in_list[j] = True
                    asp[j] = drain_to[k]
                    heappush(heap, (aj, seq, j))
                    seq += 1
                elif not is_worked:
                    if asp[j] < 0 and slope[k] > 0:
                        asp[j] = drain_to[k]      # edge cell that drains inwards
                    elif asp[j] == 0:
                        asp[j] = drain_to[k]
        worked[i] = True
        if len(pop_order) % step == 0:
            report(int(18 * len(pop_order) / total))
    del heap, in_list, worked, alt_mod, flat_done

    order = pop_order[::-1]
    pos_np = np.full(n, n, dtype=np.int64)
    pos_np[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)

    # ---------------------------------------------------------------- do_cum.c
    wat = [1.0 if v else 0.0 for v in V]
    edge_code = [0] * n
    rmax = [-1] * n
    flat = [False] * n
    max_up = [-1.0] * n
    active = [False] * n

    if not mfd:
        for idx, i in enumerate(order):
            a = asp[i]
            if a == 0:
                continue
            d = i + asp_off[a if a > 0 else -a]
            if not IN[d]:
                continue
            if edge[i]:
                # flow is not passed on along edges
                for k in range(8):
                    if not V[i + off[k]]:
                        edge_code[i] = up_code[k]
                        break
                continue
            wat[d] += wat[i]
            rmax[i] = d
            active[i] = True
            if idx % step == 0:
                report(18 + int(27 * idx / total))
    else:
        dist_m = tuple(x * _ELE_SCALE for x in dist)
        half_w = []
        for k in range(8):
            w = b = 0.5 / dist_m[k]
            for _ in range(2, convergence + 1):
                w *= b
            half_w.append(w)
        unworked = list(V)
        for idx, i in enumerate(order):
            unworked[i] = False
            a = asp[i]
            if a == 0:
                continue
            d = i + asp_off[a if a > 0 else -a]
            if not IN[d] or edge[i]:
                continue
            value = wat[i]
            ele = alt[i]
            targets = []
            weights = []
            sum_weight = 0.0
            max_weight = 0.0
            astar_set = False
            for k in range(8):
                j = i + off[k]
                if unworked[j]:
                    aj = alt[j]
                    if aj <= ele:
                        if aj < ele:
                            w = b = (ele - aj) / dist_m[k]
                            for _ in range(2, convergence + 1):
                                w *= b
                        else:
                            w = half_w[k]
                        targets.append(j)
                        weights.append(w)
                        sum_weight += w
                        if w > max_weight:
                            max_weight = w
                        if j == d:
                            astar_set = True
            # honour the A* path: it always receives flow
            if targets and not astar_set:
                targets.append(d)
                weights.append(max_weight)
                sum_weight += max_weight
            if len(targets) > 1:
                for j, w in zip(targets, weights):
                    if unworked[j]:
                        wat[j] += value * (w / sum_weight)
            else:
                wat[d] += value
            if idx % step == 0:
                report(18 + int(17 * idx / total))

        # Second pass: main drainage direction = downstream neighbour with most flow
        pending = list(V)
        for idx, i in enumerate(order):
            pending[i] = False
            a = asp[i]
            if a == 0:
                continue
            d = i + asp_off[a if a > 0 else -a]
            if not IN[d]:
                continue
            if edge[i]:
                for k in range(8):
                    if not V[i + off[k]]:
                        edge_code[i] = up_code[k]
                        break
                continue
            ele = alt[i]
            j_max = d
            k_max = -1
            max_val = -1.0
            is_flat = True
            mu = -1.0
            for k in range(8):
                j = i + off[k]
                wj = wat[j]
                aj = alt[j]
                if aj > ele and wj > mu:
                    mu = wj
                if pending[j]:
                    if aj != ele:
                        is_flat = False
                    if wj > max_val:
                        max_val = wj
                        j_max = j
                        k_max = k
            if j_max != d:
                asp[i] = up_code[k_max]
            rmax[i] = j_max
            flat[i] = is_flat
            max_up[i] = mu
            active[i] = True
            if idx % step == 0:
                report(35 + int(10 * idx / total))

    # ---------------------------------------------------------------- overland flow graph
    asp_np = np.asarray(asp, dtype=np.int64)
    has_rec = asp_np > 0
    receiver = np.full(n, -1, dtype=np.int64)
    src = np.flatnonzero(has_rec)
    receiver[src] = src + np.asarray(asp_off, dtype=np.int64)[asp_np[src]]
    in_deg = np.bincount(receiver[src], minlength=n)
    frontier = np.flatnonzero(valid_p.ravel() & (in_deg == 0))
    waves: List[np.ndarray] = []
    while frontier.size > 0:
        waves.append(frontier)
        t = receiver[frontier]
        t = t[t >= 0]
        if t.size == 0:
            break
        np.subtract.at(in_deg, t, 1)
        u = np.unique(t)
        frontier = u[in_deg[u] == 0]

    report(45)
    return WatershedGrid(
        rows=rows,
        cols=cols,
        valid_mask=valid_mask,
        mfd=bool(mfd),
        pcols=pcol,
        asp=asp,
        wat=wat,
        order=order,
        pos=pos_np,
        edge=edge,
        edge_code=edge_code,
        rmax=rmax,
        flat=np.asarray(flat, dtype=bool),
        max_up=np.asarray(max_up, dtype=np.float64),
        active=np.asarray(active, dtype=bool),
        waves=waves,
        receiver=receiver,
    )


def _haf_side(updir: int, downdir: int, thisdir: int) -> int:
    """haf_side.c: 2 = LEFT, 1 = RITE, 0 = NEITHER."""
    newup = (updir - downdir) % 8
    newthis = (thisdir - downdir) % 8
    if newthis < newup:
        return 2
    if newthis > newup:
        return 1
    return 0


def watershed_half_basins(grid: WatershedGrid, thresh_cells: float) -> np.ndarray:
    """
    Half-basins of r.watershed for a threshold in cells.
    Returns int32 (rows, cols); 0 = no half-basin (r.watershed writes NULL there).
    IDs are numbered like r.watershed (basin b: right side = b, left side = b - 1).
    """
    pcol = grid.pcols
    n = (grid.rows + 2) * pcol
    threshold = int(thresh_cells)
    wat = grid.wat
    wat_np = np.asarray(wat, dtype=np.float64)
    valid_p = np.zeros((grid.rows + 2, pcol), dtype=bool)
    valid_p[1:-1, 1:-1] = grid.valid_mask
    valid_flat = valid_p.ravel()

    asp = list(grid.asp)
    edge = grid.edge
    edge_code = grid.edge_code
    tweaked: List[int] = []

    # ------------------------------------------------------------ streams (swales)
    if not grid.mfd:
        swale_np = valid_flat & (wat_np >= threshold)
        swale = swale_np.tolist()
        for i in np.flatnonzero(swale_np).tolist():
            if edge[i] and asp[i] > 0 and edge_code[i]:
                asp[i] = -edge_code[i]
                tweaked.append(i)
    else:
        swale = [False] * n
        off = [_NEXTDR[k] * pcol + _NEXTDC[k] for k in range(8)]
        rmax = grid.rmax
        pos = grid.pos
        big = np.floor(wat_np + 0.5) >= threshold
        cand = np.flatnonzero(
            grid.active & big & (~grid.flat) & ((grid.max_up + 0.5) < threshold)
        )
        cand = cand[np.argsort(pos[cand], kind="stable")]
        events = [(int(pos[i]), int(i)) for i in cand.tolist()]   # sorted = valid heap
        is_active = grid.active
        while events:
            _, i = heapq.heappop(events)
            if not is_active[i]:
                # flagged edge cell: it becomes an outlet, the stream ends here
                if edge[i] and asp[i] > 0 and edge_code[i]:
                    asp[i] = -edge_code[i]
                    tweaked.append(i)
                continue
            if not swale[i]:
                # start a new stream only away from existing streams
                near = False
                for k in range(8):
                    if swale[i + off[k]]:
                        near = True
                        break
                if near:
                    continue
                swale[i] = True
            t = rmax[i]
            if not swale[t]:
                swale[t] = True
                heapq.heappush(events, (int(pos[t]), t))
        swale_np = np.asarray(swale, dtype=bool)

    # ------------------------------------------------------------ basins
    scan = []
    for rr in range(3):
        for cc in range(3):
            if rr == 1 and cc == 1:
                continue
            scan.append(((rr - 1) * pcol + (cc - 1), _DRAIN[rr][cc], _UPDRAIN[rr][cc]))

    haf = [0] * n
    asp_np = np.asarray(asp, dtype=np.int64)
    pour = np.flatnonzero(swale_np & valid_flat & (asp_np <= 0)).tolist()
    basin_num = 0
    for start in pour:
        stack = [start]
        while stack:
            cell = stack.pop()
            basin_num += 2
            b = basin_num
            while True:
                # upstream stream cells
                ups = []
                for o, dcode, ucode in scan:
                    j = cell + o
                    v = asp[j]
                    if (v if v >= 0 else -v) == dcode and swale[j]:
                        ups.append((j, ucode))
                a = asp[cell]
                downdir = a if a >= 0 else -a

                if not ups:
                    # no_stream.c: follow the thalweg up to the divide
                    while True:
                        best = -1.0
                        up = -1
                        updir = 0
                        for o, dcode, ucode in scan:
                            j = cell + o
                            if asp[j] == dcode and wat[j] > best:
                                best = wat[j]
                                up = j
                                updir = ucode
                        if up < 0:
                            haf[cell] = b
                            break
                        a = asp[cell]
                        downdir = a if a >= 0 else -a
                        left = rite = 0
                        for o, dcode, ucode in scan:
                            j = cell + o
                            if asp[j] == dcode:
                                side = _haf_side(updir, downdir, ucode)
                                if side == 2:
                                    haf[j] = b - 1
                                    left += 1
                                elif side == 1:
                                    haf[j] = b
                                    rite += 1
                        haf[cell] = b - 1 if left > rite else b
                        cell = up
                    break

                if len(ups) >= 2:
                    # split_str.c: confluence, every tributary starts a new basin
                    splitdirs = [u for _, u in ups]
                    updir = splitdirs[0]
                    left = rite = 0
                    for o, dcode, ucode in scan:
                        j = cell + o
                        if asp[j] == dcode and ucode not in splitdirs:
                            side = _haf_side(updir, downdir, ucode)
                            if side == 2:
                                haf[j] = b - 1
                                left += 1
                            elif side == 1:
                                haf[j] = b
                                rite += 1
                    haf[cell] = b - 1 if left >= rite else b
                    for j, _ in reversed(ups):
                        stack.append(j)
                    break

                # def_basin.c: one upstream stream cell, same basin continues
                up, updir = ups[0]
                left = rite = 0
                for o, dcode, ucode in scan:
                    j = cell + o
                    if asp[j] == dcode:
                        side = _haf_side(updir, downdir, ucode)
                        if side == 2:
                            haf[j] = b - 1
                            left += 1
                        elif side == 1:
                            haf[j] = b
                            rite += 1
                haf[cell] = b - 1 if left > rite else b
                cell = up

    # ------------------------------------------------------------ overland cells
    hb = np.asarray(haf, dtype=np.int64)
    receiver = grid.receiver
    if tweaked:
        receiver = receiver.copy()
        receiver[np.asarray(tweaked, dtype=np.int64)] = -1
    for wave in reversed(grid.waves):
        w = wave[(hb[wave] == 0) & (receiver[wave] >= 0)]
        if w.size:
            hb[w] = hb[receiver[w]]

    out = hb.reshape(grid.rows + 2, pcol)[1:-1, 1:-1].astype(np.int32)
    out[~grid.valid_mask] = 0
    return out
