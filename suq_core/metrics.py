# -*- coding: utf-8 -*-
"""
Slope unit segmentation metrics (Espindola et al., 2006; Alvioli et al., 2016, 2020).
Implements V (internal variance), I (external autocorrelation), and F (objective function).
"""

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np


@dataclass
class SlopeUnitMetrics:
    """Segmentation quality metrics for a slope unit partition."""

    v_fin: float          # Area-weighted internal circular variance (lower = more homogeneous)
    i_fin: float          # External aspect autocorrelation across adjacent units (lower = better contrast)
    n_units: int          # Number of slope units
    n_edges: int          # Number of adjacent unit pairs
    total_area_m2: float  # Total segmented area in m^2


def compute_segmentation_metrics(
    labels: np.ndarray,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    cell_area: float,
) -> SlopeUnitMetrics:
    """
    Compute V (internal variance) and I (external autocorrelation) matching r.slopeunits.metrics.
    """
    labels = np.where(labels > 0, labels, 0)      # plains / water bodies (< 0) are ignored
    lbl = labels.ravel()
    valid = lbl > 0
    if not np.any(valid):
        return SlopeUnitMetrics(v_fin=1.0, i_fin=1.0, n_units=0, n_edges=0, total_area_m2=0.0)

    max_id = int(lbl.max())
    counts = np.bincount(lbl, minlength=max_id + 1)
    unit_mask = counts > 0
    unit_mask[0] = False
    n_units = int(np.count_nonzero(unit_mask))

    s_sin = np.bincount(lbl, weights=sin_a.ravel(), minlength=max_id + 1)
    s_cos = np.bincount(lbl, weights=cos_a.ravel(), minlength=max_id + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_len = np.hypot(s_sin, s_cos)
        v_i = np.where(unit_mask, 1.0 - (r_len / counts), 0.0)

    total_cells = int(counts[unit_mask].sum())
    total_area_m2 = float(total_cells) * cell_area
    # V_fin = sum(area_i * v_i) / sum(area_i)
    v_fin = float(np.sum(counts[unit_mask] * v_i[unit_mask]) / total_cells)

    # Mean aspect angle per unit (a_i) and global mean aspect angle (a_all)
    a_i = np.arctan2(s_sin, s_cos)
    a_all = float(np.arctan2(s_sin[unit_mask].sum(), s_cos[unit_mask].sum()))

    # Find all unique adjacent slope unit pairs (i < j) via horizontal and vertical pixel borders
    h_diff = (labels[:, :-1] != labels[:, 1:]) & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    v_diff = (labels[:-1, :] != labels[1:, :]) & (labels[:-1, :] > 0) & (labels[1:, :] > 0)

    p_left = np.concatenate([labels[:, :-1][h_diff], labels[:-1, :][v_diff]])
    p_right = np.concatenate([labels[:, 1:][h_diff], labels[1:, :][v_diff]])

    if p_left.size == 0:
        return SlopeUnitMetrics(
            v_fin=v_fin, i_fin=0.0, n_units=n_units, n_edges=0, total_area_m2=total_area_m2
        )

    u_min = np.minimum(p_left, p_right).astype(np.int64)
    u_max = np.maximum(p_left, p_right).astype(np.int64)
    edge_keys = np.unique((u_min << 32) | u_max)
    ei = (edge_keys >> 32).astype(np.int32)
    ej = (edge_keys & 0xFFFFFFFF).astype(np.int32)

    # Exact ALT2 formula from r.slopeunits.metrics.py (lines 408-454)
    sin_all = np.sin(a_all)
    cos_all = np.cos(a_all)

    ai = a_i[ei]
    aj = a_i[ej]

    ang_i = np.arctan2(np.sin(ai) + sin_all, np.cos(ai) + cos_all)
    ang_j = np.arctan2(np.sin(aj) + sin_all, np.cos(aj) + cos_all)

    ci = np.cos(ang_i)
    si = np.sin(ang_i)
    cj = np.cos(ang_j)
    sj = np.sin(ang_j)

    i_fin = float(np.mean(ci * cj + si * sj))

    return SlopeUnitMetrics(
        v_fin=v_fin,
        i_fin=i_fin,
        n_units=n_units,
        n_edges=int(edge_keys.size),
        total_area_m2=total_area_m2,
    )


def compute_f_scores(v_values: Sequence[float], i_values: Sequence[float]) -> np.ndarray:
    """
    Compute the normalized segmentation objective function F = F_V + F_I.
    Higher F indicates a better balance between internal homogeneity (low V)
    and external contrast (low I).
    """
    v = np.asarray(v_values, dtype=np.float64)
    i = np.asarray(i_values, dtype=np.float64)
    if v.size == 0:
        return np.array([], dtype=np.float64)

    v_min, v_max = float(v.min()), float(v.max())
    i_min, i_max = float(i.min()), float(i.max())

    fv = (v_max - v) / (v_max - v_min) if v_max > v_min else np.ones_like(v)
    fi = (i_max - i) / (i_max - i_min) if i_max > i_min else np.ones_like(i)
    return fv + fi
