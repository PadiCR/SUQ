# -*- coding: utf-8 -*-
"""
suq_core: Pure Python/NumPy/SciPy/GDAL Slope Units engine for SUQ.
Zero external GRASS GIS or SAGA dependencies.
"""

from .dem_io import RasterData, read_raster, write_raster, polygonize_labels, rasterize_mask
from .terrain import compute_slope_aspect, detect_alluvial_plains, aspect_cover_mask
from .hydrology import HydroGrid, prepare_hydro_grid, extract_half_basins
from .slope_units import (
    SlopeUnitResult,
    create_slope_units,
    threshold_schedule,
    clump_labels,
    CachedHalfBasins,
    enforce_nesting,
    nest_layers,
    create_nested_slope_units,
    build_basin_hierarchy,
    prune_basin_hierarchy,
)
from .watershed import WatershedGrid, prepare_watershed_grid, watershed_half_basins
from .cleaning import clean_slope_units
from .metrics import SlopeUnitMetrics, compute_segmentation_metrics, compute_f_scores
from .optimize import OptimizationResult, optimize_slope_units

__all__ = [
    "RasterData",
    "read_raster",
    "write_raster",
    "polygonize_labels",
    "rasterize_mask",
    "compute_slope_aspect",
    "detect_alluvial_plains",
    "aspect_cover_mask",
    "HydroGrid",
    "prepare_hydro_grid",
    "extract_half_basins",
    "WatershedGrid",
    "prepare_watershed_grid",
    "watershed_half_basins",
    "SlopeUnitResult",
    "create_slope_units",
    "threshold_schedule",
    "clump_labels",
    "CachedHalfBasins",
    "enforce_nesting",
    "nest_layers",
    "create_nested_slope_units",
    "build_basin_hierarchy",
    "prune_basin_hierarchy",
    "clean_slope_units",
    "SlopeUnitMetrics",
    "compute_segmentation_metrics",
    "compute_f_scores",
    "OptimizationResult",
    "optimize_slope_units",
]
