# -*- coding: utf-8 -*-
"""
Cleaning of small slope units for SUQ.

Small units are merged one at a time (smallest first) into an adjacent unit. Original
borders are kept; only the borders between the merged units disappear.

Methods
  - 'hydrological' : merges must be hydrologically coherent. Every border between two
                     original units is classified once, from the drainage directions on
                     both sides of it:
                       ridge  : both sides drain away from the border
                       valley : both sides drain towards the border (a channel runs along it)
                       slope  : anything else (side by side on the same slope, or one unit
                                draining into the other)
                     Slope borders may always be removed. A ridge or valley border may be
                     removed only when the merged unit makes up at least
                     ``CATCHMENT_SHARE`` of the smallest catchment that contains it (the
                     area upstream of the point where all its water has joined): the merged
                     unit is then essentially one head catchment, so a main divide or a
                     river with water from further upstream is never crossed. Among the
                     allowed neighbours, a slope neighbour with similar mean aspect is
                     preferred, then a head-catchment one (largest share first); the most
                     similar aspect, then the longest shared border, decides. A small unit
                     without an allowed neighbour is kept.
  - 'detailed'     : neighbour with the most similar mean aspect (no hydrological check).
  - 'quick'        : neighbour sharing the longest border.

Raster specks (units of fewer than ``SPECK_CELLS`` cells, too small for their borders to be
classified) are always merged into the neighbour sharing the longest border, as
r.slopeunits.clean does with v.clean rmarea (10 cells) in its detailed method.

The original r.slopeunits.clean has no hydrological rules: its default removes small units
and grows the remaining ones into the gap (r.grow), which reshapes the borders.
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# drainage direction codes of r.watershed: 1=NE 2=N 3=NW 4=W 5=SW 6=S 7=SE 8=E
_ASP_R = np.array([0, -1, -1, -1, 0, 1, 1, 1, 0], dtype=np.int64)
_ASP_C = np.array([0, 1, 0, -1, -1, -1, 0, 1, 1], dtype=np.int64)

BORDER_SLOPE = 0
BORDER_VALLEY = 1
BORDER_RIDGE = 2
BORDER_NAMES = {BORDER_SLOPE: "slope", BORDER_VALLEY: "valley", BORDER_RIDGE: "ridge"}

# a side drains "towards" / "away from" a border when the mean component of its drainage
# directions across the border exceeds this value (1 = every cell drains straight across)
TOWARD_TOL = 0.2
# a ridge or valley border may be removed only if the merged unit covers at least this
# fraction of the smallest catchment containing it
CATCHMENT_SHARE = 0.5
# borders shorter than this (cell sides) are too short to be classified and are ignored
MIN_CLASS_CELLS = 3
# a neighbour must share at least this fraction of the small unit's perimeter, unless it is
# the neighbour with the longest shared border
MIN_BORDER_FRACTION = 0.05
# units smaller than this (cells) are raster specks
SPECK_CELLS = 10


@dataclass
class CleaningReport:
    n_input: int = 0
    n_small: int = 0
    n_merges: int = 0
    n_kept_small: int = 0
    kept_small_ids: list = field(default_factory=list)   # ids in the returned labels
    blocked: Dict[str, int] = field(default_factory=dict)  # rejected neighbours, by reason

    def _block(self, reason):
        self.blocked[reason] = self.blocked.get(reason, 0) + 1


def clean_slope_units(
    labels: np.ndarray,
    cell_area: float,
    cleansize: float,
    sin_a: Optional[np.ndarray] = None,
    cos_a: Optional[np.ndarray] = None,
    method: str = "detailed",
    plains_mask: Optional[np.ndarray] = None,
    drainage: Optional[np.ndarray] = None,
    return_report: bool = False,
):
    """
    Merge slope units smaller than ``cleansize`` (m^2) into neighbours.

    ``drainage`` (r.watershed direction codes, e.g. ``WatershedGrid.drainage``) is required
    by method 'hydrological'. Returns the relabelled array (1..K, 0 = nodata), plus a
    ``CleaningReport`` when ``return_report`` is True.
    """
    if method not in ("hydrological", "detailed", "quick"):
        raise ValueError(f"unknown cleaning method {method!r}")
    lab = np.where(labels > 0, labels, 0).astype(np.int64)
    if plains_mask is not None:
        lab[plains_mask] = 0
    lab = _relabel_sequential(lab).astype(np.int64)
    n_units = int(lab.max())
    report = CleaningReport(n_input=n_units)

    min_cells = max(1, int(np.ceil(float(cleansize) / cell_area)))
    if n_units == 0 or min_cells <= 1:
        out = _relabel_sequential(lab)
        return (out, report) if return_report else out

    hydro = method == "hydrological"
    if hydro and drainage is None:
        raise ValueError("method 'hydrological' needs the drainage directions")
    use_aspect = method != "quick" and sin_a is not None and cos_a is not None
    graph = _UnitGraph(lab, n_units, sin_a if use_aspect else None, cos_a if use_aspect else None,
                       drainage if hydro else None)
    report.n_small = int(np.count_nonzero((graph.size[1:] > 0) & (graph.size[1:] < min_cells)))

    pending = [u for u in range(1, n_units + 1) if 0 < graph.size[u] < min_cells]
    # A unit left alone may find a valid neighbour once its neighbours have merged, so the
    # kept units get further passes while merges still happen.
    for _ in range(50):
        heap = [(int(graph.size[u]), u) for u in pending if graph.alive[u] and graph.size[u] < min_cells]
        heapq.heapify(heap)
        kept = []
        merges_before = report.n_merges
        while heap:
            sz, u = heapq.heappop(heap)
            if not graph.alive[u] or graph.size[u] != sz:
                continue
            target = graph.choose_neighbour(u, method, use_aspect, min(SPECK_CELLS, min_cells), report)
            if target is None:
                kept.append(u)
                continue
            graph.merge(target, u)
            report.n_merges += 1
            if graph.size[target] < min_cells:
                heapq.heappush(heap, (int(graph.size[target]), target))
        pending = kept
        if not kept or report.n_merges == merges_before:
            break

    final = graph.final_labels()
    out = _relabel_sequential(final)
    roots = np.unique(final[final > 0])
    small_roots = [u for u in roots.tolist() if graph.size[u] < min_cells]
    report.kept_small_ids = [int(i) + 1 for i in np.searchsorted(roots, small_roots)]
    report.n_kept_small = len(report.kept_small_ids)
    return (out, report) if return_report else out


def classify_borders(labels: np.ndarray, drainage: np.ndarray):
    """
    Border class between every pair of adjacent units of ``labels`` (> 0).
    Returns {(u, v): (class, shared cell sides, mean drainage of u towards v, of v towards u)}
    with u < v.
    """
    lab = np.where(labels > 0, labels, 0).astype(np.int64)
    g = _UnitGraph(lab, int(lab.max()), None, None, drainage)
    out = {}
    for e in range(len(g.e_u)):
        u, v = g.e_u[e], g.e_v[e]
        k = g.e_k[e]
        out[(u, v)] = (g.e_cls[e], k, g.e_tu[e] / k, g.e_tv[e] / k)
    return out


class _UnitGraph:
    """Adjacency of (merged) slope units; each original border piece keeps its own class."""

    def __init__(self, lab, n_units, sin_a, cos_a, drainage):
        self.lab = lab
        n = n_units + 1
        flat = lab.ravel()
        self.size = np.bincount(flat, minlength=n).astype(np.int64)
        self.size[0] = 0
        self.alive = [True] * n
        self.alive[0] = False
        self.parent = list(range(n))
        self.has_aspect = sin_a is not None
        if self.has_aspect:
            s = np.where(np.isfinite(sin_a), sin_a, 0.0).ravel()
            c = np.where(np.isfinite(cos_a), cos_a, 0.0).ravel()
            self.ssum = np.bincount(flat, weights=s, minlength=n)
            self.csum = np.bincount(flat, weights=c, minlength=n)
        rows, cols = lab.shape

        # perimeter in cell sides (incl. the outer border of the data)
        pad = np.zeros((rows + 2, cols + 2), dtype=np.int64)
        pad[1:-1, 1:-1] = lab
        core = pad[1:-1, 1:-1]
        perim = np.zeros(n, dtype=np.int64)
        for nb in (pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]):
            m = (core > 0) & (nb != core)
            perim += np.bincount(core[m], minlength=n)
        self.perim = perim

        self.hydro = drainage is not None
        if self.hydro:
            code = np.abs(np.asarray(drainage, dtype=np.int64))
            code[(code > 8) | (lab == 0)] = 0
            dr = _ASP_R[code].astype(np.float64)
            dc = _ASP_C[code].astype(np.float64)
            nrm = np.hypot(dr, dc)
            nrm[nrm == 0] = 1.0
            dr /= nrm
            dc /= nrm

        # Original border pieces (u < v): shared cell sides, and the mean component of the
        # drainage towards the border on each side, taken over the two cell rows next to it
        # (the crest or channel line itself often drains along the border).
        P = 2
        lp = np.zeros((rows + 2 * P, cols + 2 * P), dtype=np.int64)
        lp[P:-P, P:-P] = lab
        if self.hydro:
            drp = np.zeros(lp.shape)
            dcp = np.zeros(lp.shape)
            drp[P:-P, P:-P] = dr
            dcp[P:-P, P:-P] = dc

        def win(arr, oy, ox):
            return arr[P + oy : P + oy + rows, P + ox : P + ox + cols]

        acc_k: Dict[int, list] = {}
        for ddr, ddc in ((0, 1), (1, 0)):
            a = win(lp, 0, 0)
            b = win(lp, ddr, ddc)
            m = (a > 0) & (b > 0) & (a != b)
            ua, ub = a[m], b[m]
            if self.hydro:
                # side a: the cell and the one behind it (if in the same unit), towards +d
                a2 = win(lp, -ddr, -ddc)[m] == ua
                ta = (win(drp, 0, 0) * ddr + win(dcp, 0, 0) * ddc)[m] + np.where(
                    a2, (win(drp, -ddr, -ddc) * ddr + win(dcp, -ddr, -ddc) * ddc)[m], 0.0)
                na = 1.0 + a2
                b2 = win(lp, 2 * ddr, 2 * ddc)[m] == ub
                tb = -(win(drp, ddr, ddc) * ddr + win(dcp, ddr, ddc) * ddc)[m] - np.where(
                    b2, (win(drp, 2 * ddr, 2 * ddc) * ddr + win(dcp, 2 * ddr, 2 * ddc) * ddc)[m], 0.0)
                nb = 1.0 + b2
            else:
                ta = tb = np.zeros(ua.size)
                na = nb = np.ones(ua.size)
            swap = ua > ub
            lo = np.where(swap, ub, ua)
            hi = np.where(swap, ua, ub)
            key = lo * n + hi
            uk, kinv = np.unique(key, return_inverse=True)
            cols_ = (
                np.bincount(kinv),
                np.bincount(kinv, weights=np.where(swap, tb, ta)),
                np.bincount(kinv, weights=np.where(swap, nb, na)),
                np.bincount(kinv, weights=np.where(swap, ta, tb)),
                np.bincount(kinv, weights=np.where(swap, na, nb)),
            )
            for i, kk in enumerate(uk.tolist()):
                e = acc_k.setdefault(kk, [0, 0.0, 0.0, 0.0, 0.0])
                for j in range(5):
                    e[j] += cols_[j][i]
        self.e_u, self.e_v, self.e_k, self.e_tu, self.e_tv, self.e_cls = [], [], [], [], [], []
        # adj[g][h] = list of border pieces between groups g and h
        self.adj = [dict() for _ in range(n)]
        for kk in sorted(acc_k):
            u, v = divmod(kk, n)
            k, tu, nu, tv, nv = acc_k[kk]
            k = int(k)
            # stored as sums scaled to the border length, so that tu / k is the mean
            tu = tu / nu * k if nu > 0 else 0.0
            tv = tv / nv * k if nv > 0 else 0.0
            eid = len(self.e_u)
            self.e_u.append(u)
            self.e_v.append(v)
            self.e_k.append(k)
            self.e_tu.append(tu)
            self.e_tv.append(tv)
            self.e_cls.append(self._classify(k, tu, tv) if self.hydro else BORDER_SLOPE)
            self.adj[u].setdefault(v, []).append(eid)
            self.adj[v].setdefault(u, []).append(eid)

        # Single-flow tree: preorder index (tin) and upstream area (acc, cells) of every
        # cell; the upstream area of a cell is the tin interval [tin, tin + acc). Each group
        # keeps the cells with its lowest and highest tin: the smallest catchment containing
        # the group is the first common downstream cell of these two.
        if self.hydro:
            self.rec, self.acc = _sfd_receiver_and_accumulation(code, lab > 0)
            self.tin = _preorder(self.rec, flat > 0)
            idx = np.flatnonzero(flat > 0)
            t = self.tin[idx]
            o = np.lexsort((t, flat[idx]))
            ids = flat[idx][o]
            first = np.searchsorted(ids, np.arange(n))
            last = np.searchsorted(ids, np.arange(n), side="right") - 1
            self.tmin_cell = [-1] * n
            self.tmax_cell = [-1] * n
            for u in range(1, n):
                if first[u] <= last[u]:
                    self.tmin_cell[u] = int(idx[o[first[u]]])
                    self.tmax_cell[u] = int(idx[o[last[u]]])
            self._path_cache = {}

    @staticmethod
    def _classify(k, tu, tv):
        if k < MIN_CLASS_CELLS:
            return BORDER_SLOPE
        mu, mv = tu / k, tv / k
        if max(mu, mv) < -TOWARD_TOL:
            return BORDER_RIDGE
        if mu + mv > TOWARD_TOL and min(mu, mv) > -TOWARD_TOL:
            return BORDER_VALLEY
        return BORDER_SLOPE

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def shared(self, g, h):
        return sum(self.e_k[e] for e in self.adj[g][h])

    def catchment_share(self, g, h):
        """Share of the smallest catchment containing groups g and h covered by g + h."""
        tin = self.tin
        a = min(self.tmin_cell[g], self.tmin_cell[h], key=lambda c: tin[c])
        b = max(self.tmax_cell[g], self.tmax_cell[h], key=lambda c: tin[c])
        t_hi = tin[b]
        path = self._path_cache.get(a)
        if path is None:
            cells = [a]
            r = self.rec[a]
            while r >= 0:
                cells.append(int(r))
                r = self.rec[r]
            path = np.asarray(cells, dtype=np.int64)
            self._path_cache[a] = path
        cover = tin[path] + self.acc[path] > t_hi
        if not np.any(cover):
            return 0.0                 # the two groups drain to different outlets
        root = path[int(np.argmax(cover))]
        return float(self.size[g] + self.size[h]) / float(self.acc[root])

    def aspect_cos(self, g, h):
        if not self.has_aspect:
            return 0.0
        a = np.hypot(self.ssum[g], self.csum[g])
        b = np.hypot(self.ssum[h], self.csum[h])
        if a <= 0 or b <= 0:
            return 0.0
        return float((self.ssum[g] * self.ssum[h] + self.csum[g] * self.csum[h]) / (a * b))

    def choose_neighbour(self, g, method, use_aspect, speck_cells, report):
        nbrs = self.adj[g]
        if not nbrs:
            return None
        shared = {h: self.shared(g, h) for h in nbrs}
        longest = max(shared.values())
        if self.size[g] < speck_cells:
            return max(shared, key=lambda h: (shared[h], -h))
        min_len = min(longest, max(MIN_CLASS_CELLS, MIN_BORDER_FRACTION * self.perim[g]))
        best, best_key = None, None
        for h, k in shared.items():
            if k < min_len:
                report._block("short border")
                continue
            cos_gh = self.aspect_cos(g, h) if use_aspect else 0.0
            if method == "hydrological":
                classes = {self.e_cls[e] for e in nbrs[h]}
                if BORDER_RIDGE in classes or BORDER_VALLEY in classes:
                    share = self.catchment_share(g, h)
                    if share < CATCHMENT_SHARE:
                        report._block("ridge" if BORDER_RIDGE in classes else "river")
                        continue
                    # head catchment: the merged unit is most of one small catchment
                    key = (-1, share, k)
                else:
                    key = (0 if cos_gh > 0 else -2, cos_gh, k)
            elif method == "detailed" and use_aspect:
                key = (0, cos_gh, k)
            else:
                key = (0, 0.0, k)
            if best_key is None or key > best_key:
                best_key, best = key, h
        return best

    def merge(self, a, b):
        """Merge group b into group a."""
        self.parent[b] = a
        self.alive[b] = False
        if self.hydro:
            tin = self.tin
            self.tmin_cell[a] = min(self.tmin_cell[a], self.tmin_cell[b], key=lambda c: tin[c])
            self.tmax_cell[a] = max(self.tmax_cell[a], self.tmax_cell[b], key=lambda c: tin[c])
        self.size[a] += self.size[b]
        self.size[b] = 0
        self.perim[a] = self.perim[a] + self.perim[b] - 2 * self.shared(a, b)
        if self.has_aspect:
            self.ssum[a] += self.ssum[b]
            self.csum[a] += self.csum[b]
        del self.adj[a][b]
        del self.adj[b][a]
        for v, pieces in self.adj[b].items():
            del self.adj[v][b]
            self.adj[a].setdefault(v, []).extend(pieces)
            self.adj[v].setdefault(a, []).extend(pieces)
        self.adj[b] = {}

    def final_labels(self):
        n = len(self.parent)
        root = np.array([self.find(u) for u in range(n)], dtype=np.int64)
        root[0] = 0
        return root[self.lab]


def _preorder(rec: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Preorder (DFS entry) index of every cell in the single-flow forest (-1 if invalid)."""
    n = rec.size
    has = rec >= 0
    donors = np.flatnonzero(has)
    order = donors[np.argsort(rec[donors], kind="stable")]
    rs = rec[order]
    start = np.searchsorted(rs, np.arange(n)).tolist()
    end = np.searchsorted(rs, np.arange(n), side="right").tolist()
    order = order.tolist()
    tin = np.full(n, -1, dtype=np.int64)
    out = [0] * n
    t = 0
    for r in np.flatnonzero(valid & ~has).tolist():
        stack = [r]
        while stack:
            x = stack.pop()
            out[x] = t
            t += 1
            stack.extend(order[start[x]:end[x]])
    tin_list = np.asarray(out, dtype=np.int64)
    tin[valid] = tin_list[valid]
    return tin


def _sfd_receiver_and_accumulation(code: np.ndarray, valid: np.ndarray):
    """Single-flow receivers (flat index, -1 where the flow leaves) and accumulation (cells)."""
    rows, cols = code.shape
    rr = np.arange(rows)[:, None] + _ASP_R[code]
    cc = np.arange(cols)[None, :] + _ASP_C[code]
    ok = valid & (code > 0) & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    rec = np.where(ok, np.clip(rr, 0, rows - 1) * cols + np.clip(cc, 0, cols - 1), -1).ravel()
    vflat = valid.ravel()
    good = rec >= 0
    good[good] = vflat[rec[good]]
    rec = np.where(good, rec, -1)
    acc = vflat.astype(np.float64)
    src = np.flatnonzero(rec >= 0)
    in_deg = np.bincount(rec[src], minlength=rows * cols)
    frontier = np.flatnonzero(vflat & (in_deg == 0))
    while frontier.size > 0:
        t = rec[frontier]
        m = t >= 0
        if not np.any(m):
            break
        np.add.at(acc, t[m], acc[frontier[m]])
        np.subtract.at(in_deg, t[m], 1)
        u = np.unique(t[m])
        frontier = u[in_deg[u] == 0]
    return rec, acc


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Relabel positive IDs sequentially 1..K while keeping 0 as 0."""
    out = np.zeros(labels.shape, dtype=np.int32)
    mask = labels > 0
    if not np.any(mask):
        return out
    _, inv = np.unique(labels[mask], return_inverse=True)
    out[mask] = (inv + 1).astype(np.int32)
    return out


# Classes written for the areas left out of the slope units
CLASS_PLAIN = -1
CLASS_WATER = -2
CLASS_NAMES = {CLASS_PLAIN: "Alluvial plain", CLASS_WATER: "Water body"}
SLOPE_UNIT_NAME = "Slope unit"


def absorb_enclosed_units(
    labels: np.ndarray, barrier: np.ndarray, min_cells: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Units (> 0) smaller than ``min_cells`` whose whole border touches ``barrier`` cells or
    the outside of the data (and at least one barrier cell) are removed: islands of slope
    units inside plains or water bodies. Returns (labels with those units set to 0, mask of
    the removed cells).
    """
    lab = np.where(labels > 0, labels, 0).astype(np.int64)
    n = int(lab.max()) + 1
    absorbed = np.zeros(lab.shape, dtype=bool)
    if n <= 1 or not np.any(barrier):
        return lab, absorbed
    pad = np.pad(lab, 1)
    pb = np.pad(barrier, 1)
    core = pad[1:-1, 1:-1]
    to_unit = np.zeros(n)
    to_barrier = np.zeros(n)
    for sl in ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
               (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None))):
        nb = pad[sl]
        m = (core > 0) & (nb != core)
        to_unit += np.bincount(core[m & (nb > 0)], minlength=n)
        to_barrier += np.bincount(core[m & pb[sl]], minlength=n)
    size = np.bincount(lab.ravel(), minlength=n)
    island = (size < min_cells) & (to_unit == 0) & (to_barrier > 0)
    island[0] = False
    if np.any(island):
        absorbed = island[lab]
        lab[absorbed] = 0
    return lab, absorbed


def assign_classes(
    labels: np.ndarray,
    valid: np.ndarray,
    plains: Optional[np.ndarray] = None,
    water: Optional[np.ndarray] = None,
    absorbed: Optional[np.ndarray] = None,
    relabel: bool = True,
) -> np.ndarray:
    """
    Final labels: slope units > 0 (renumbered 1..K when ``relabel``), plains -1, water
    bodies -2, 0 outside the data. ``absorbed`` cells take the class of the nearest plain
    or water cell.
    """
    out = np.where(labels > 0, labels, 0).astype(np.int64)
    if relabel:
        out = _relabel_sequential(out).astype(np.int64)
    cls = np.zeros(out.shape, dtype=np.int64)
    if plains is not None:
        cls[plains & valid] = CLASS_PLAIN
    if water is not None:
        cls[water & valid] = CLASS_WATER
    free = (out == 0) & valid
    out[free & (cls < 0)] = cls[free & (cls < 0)]
    if absorbed is not None and np.any(absorbed) and np.any(cls < 0):
        from scipy import ndimage

        _, idx = ndimage.distance_transform_edt(cls == 0, return_indices=True)
        near = cls[idx[0], idx[1]]
        out[absorbed & valid] = near[absorbed & valid]
    if np.any(out == CLASS_PLAIN) and np.any(out == CLASS_WATER):
        out = absorb_enclosed_plains(out)
    return out.astype(np.int32)


def absorb_enclosed_plains(out: np.ndarray) -> np.ndarray:
    """
    Plain patches (-1) whose whole border touches water (-2) or the outside of the data,
    with no slope unit next to them, become water: islands of plain inside a water body.
    """
    from scipy import ndimage

    comp, n = ndimage.label(out == CLASS_PLAIN)
    if n == 0:
        return out
    pad_out = np.pad(out, 1)
    core = np.pad(comp, 1)[1:-1, 1:-1]
    to_unit = np.zeros(n + 1, dtype=bool)
    to_water = np.zeros(n + 1, dtype=bool)
    for sl in ((slice(None, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
               (slice(1, -1), slice(None, -2)), (slice(1, -1), slice(2, None))):
        nb = pad_out[sl]
        to_unit[core[(core > 0) & (nb > 0)]] = True
        to_water[core[(core > 0) & (nb == CLASS_WATER)]] = True
    island = to_water & ~to_unit
    island[0] = False
    if np.any(island):
        out = out.copy()
        out[island[comp]] = CLASS_WATER
    return out
