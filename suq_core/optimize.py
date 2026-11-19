# -*- coding: utf-8 -*-
"""
Parameter optimization for SUQ (equivalent to r.slopeunits.optimize).

As in GRASS, every candidate (cvmin, areamin) is evaluated by running the slope unit
delineation (the exact r.slopeunits.create iteration), cleaning the result and computing
the segmentation metrics. Unlike GRASS, flow routing is done only once and the half-basins
of each threshold are cached, so a candidate costs one pass of the iteration instead of
a full series of r.watershed runs.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .cleaning import clean_slope_units
from .hydrology import HydroGrid, extract_half_basins
from .metrics import compute_f_scores, compute_segmentation_metrics
from .slope_units import CachedHalfBasins, create_slope_units


@dataclass
class OptimizationCandidate:
    """One evaluated (cvmin, areamin) parameter point."""

    cvmin: float
    areamin: float
    v_fin: float
    i_fin: float
    f_score: float
    n_units: int


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_cvmin: float
    best_areamin: float
    best_f: float
    best_v: float
    best_i: float
    best_labels: np.ndarray
    best_cleaned_labels: np.ndarray
    candidates: List[OptimizationCandidate]


def optimize_slope_units(
    hydro: Optional[HydroGrid],
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    nz: np.ndarray,
    cell_area: float,
    thresh: float,
    cvmin_range: Tuple[float, float] = (0.05, 0.35),
    areamin_range: Tuple[float, float] = (50000.0, 200000.0),
    epsilon_cv: float = 0.02,
    epsilon_area: float = 25000.0,
    rf: int = 2,
    maxiteration: int = 20,
    cleansize: float = 25000.0,
    areamax: Optional[float] = None,
    plains_mask: Optional[np.ndarray] = None,
    use_3d_normal: bool = False,
    clean_method: str = "quick",
    clean_drainage: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    cover_mask: Optional[np.ndarray] = None,
    half_basin_provider: Optional[Callable[[int], np.ndarray]] = None,
    valid_mask: Optional[np.ndarray] = None,
    log: Optional[Callable[[str], None]] = None,
) -> OptimizationResult:
    """
    Find optimal (cvmin, areamin) maximizing the Espindola/Alvioli objective function F.

    Half-basins come from ``half_basin_provider`` (any engine) or, if it is None, from the
    fast D8 engine on ``hydro``. ``valid_mask`` is only needed when ``hydro`` is None.
    """
    cv_lo, cv_hi = sorted((float(cvmin_range[0]), float(cvmin_range[1])))
    area_lo, area_hi = sorted((float(areamin_range[0]), float(areamin_range[1])))

    if half_basin_provider is None:
        if hydro is None:
            raise ValueError("optimize_slope_units needs hydro or a half_basin_provider")

        def half_basin_provider(thc: int) -> np.ndarray:
            return extract_half_basins(hydro, thresh_cells=thc)

    if valid_mask is None:
        if hydro is None:
            raise ValueError("optimize_slope_units needs hydro or valid_mask")
        valid_mask = hydro.valid_mask
    provider = (
        half_basin_provider
        if isinstance(half_basin_provider, CachedHalfBasins)
        else CachedHalfBasins(half_basin_provider)
    )

    def delineate(cv_val: float, area_val: float) -> Tuple[np.ndarray, np.ndarray]:
        raw = create_slope_units(
            hydro=None,
            sin_a=sin_a,
            cos_a=cos_a,
            nx=nx,
            ny=ny,
            nz=nz,
            cell_area=cell_area,
            thresh=thresh,
            areamin=area_val,
            cvmin=cv_val,
            rf=rf,
            maxiteration=maxiteration,
            areamax=areamax,
            plains_mask=plains_mask,
            use_3d_normal=use_3d_normal,
            cover_mask=cover_mask,
            half_basin_provider=provider,
            valid_mask=valid_mask,
        ).labels
        if cleansize > 0:
            cleaned = clean_slope_units(
                raw,
                cell_area=cell_area,
                cleansize=cleansize,
                sin_a=sin_a,
                cos_a=cos_a,
                method=clean_method,
                plains_mask=plains_mask,
                drainage=clean_drainage,
            )
        else:
            cleaned = raw
        return raw, cleaned

    evaluated: Dict[Tuple[float, float], Tuple[float, float, int]] = {}

    def eval_point(cv_val: float, area_val: float) -> Tuple[float, float, int]:
        key = (round(cv_val, 5), round(area_val, 2))
        if key in evaluated:
            return evaluated[key]
        _, cleaned = delineate(key[0], key[1])
        m = compute_segmentation_metrics(cleaned, sin_a, cos_a, cell_area)
        evaluated[key] = (m.v_fin, m.i_fin, m.n_units)
        if log is not None:
            log(
                f"  cvmin={key[0]:.4f} areamin={key[1]:.0f}: "
                f"{m.n_units} units, V={m.v_fin:.4f}, I={m.i_fin:.4f}"
            )
        return evaluated[key]

    # Iterative 3x3 zoom-in search over (cvmin, areamin) space
    max_zoom = 8
    for zoom_step in range(max_zoom):
        cv_pts = np.linspace(cv_lo, cv_hi, 3)
        area_pts = np.linspace(area_lo, area_hi, 3)
        for n_done, (cv_v, ar_v) in enumerate((c, a) for c in cv_pts for a in area_pts):
            eval_point(float(cv_v), float(ar_v))
            if progress_cb is not None:
                progress_cb(min(97, 45 + int(52 * (zoom_step + (n_done + 1) / 9.0) / max_zoom)))

        # Compute F across all evaluated points so far
        all_keys = list(evaluated.keys())
        v_arr = [evaluated[k][0] for k in all_keys]
        i_arr = [evaluated[k][1] for k in all_keys]
        f_arr = compute_f_scores(v_arr, i_arr)
        best_idx = int(np.argmax(f_arr))
        best_cv, best_area = all_keys[best_idx]

        if (cv_hi - cv_lo) <= epsilon_cv and (area_hi - area_lo) <= epsilon_area:
            break

        # Shrink search window around best point
        half_cv = max(epsilon_cv * 0.5, (cv_hi - cv_lo) * 0.35)
        half_area = max(epsilon_area * 0.5, (area_hi - area_lo) * 0.35)
        cv_lo = max(float(cvmin_range[0]), best_cv - half_cv)
        cv_hi = min(float(cvmin_range[1]), best_cv + half_cv)
        area_lo = max(float(areamin_range[0]), best_area - half_area)
        area_hi = min(float(areamin_range[1]), best_area + half_area)

    all_keys = list(evaluated.keys())
    v_arr = [evaluated[k][0] for k in all_keys]
    i_arr = [evaluated[k][1] for k in all_keys]
    f_arr = compute_f_scores(v_arr, i_arr)

    candidates: List[OptimizationCandidate] = []
    for idx, (cv_k, area_k) in enumerate(all_keys):
        v_v, i_v, n_u = evaluated[(cv_k, area_k)]
        candidates.append(
            OptimizationCandidate(
                cvmin=cv_k,
                areamin=area_k,
                v_fin=v_v,
                i_fin=i_v,
                f_score=float(f_arr[idx]),
                n_units=n_u,
            )
        )

    best_cand = max(candidates, key=lambda c: c.f_score)
    best_labels, best_cleaned = delineate(best_cand.cvmin, best_cand.areamin)

    return OptimizationResult(
        best_cvmin=best_cand.cvmin,
        best_areamin=best_cand.areamin,
        best_f=best_cand.f_score,
        best_v=best_cand.v_fin,
        best_i=best_cand.i_fin,
        best_labels=best_labels,
        best_cleaned_labels=best_cleaned,
        candidates=sorted(candidates, key=lambda c: c.f_score, reverse=True),
    )
