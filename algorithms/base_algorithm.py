# -*- coding: utf-8 -*-
"""
Shared base class and helpers for SUQ Processing algorithms.
"""

import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtGui import QIcon

from ..suq_core.dem_io import RasterData, read_raster
from ..suq_core.grass_backend import GrassBackendError, GrassHalfBasinProvider
from ..suq_core.hydrology import extract_half_basins, prepare_hydro_grid
from ..suq_core.slope_units import CachedHalfBasins, threshold_schedule
from ..suq_core.terrain import (
    compute_slope_aspect,
    detect_alluvial_plains,
    detect_water_bodies,
)
from ..suq_core.cleaning import absorb_enclosed_units, assign_classes
from ..suq_core.watershed import prepare_watershed_grid, watershed_half_basins


class SUQBaseAlgorithm(QgsProcessingAlgorithm):
    """Base class providing shared icon, terrain prep, and morphometric stats."""

    ENGINE = "ENGINE"
    CONVERGENCE = "CONVERGENCE"

    ENGINE_NATIVE_MFD = 0
    ENGINE_NATIVE_SFD = 1
    ENGINE_FAST_D8 = 2
    ENGINE_GRASS_MFD = 3
    ENGINE_GRASS_SFD = 4

    # Tooltip texts (parameter help). A tool's PARAM_HELP overrides COMMON_HELP.
    PARAM_HELP: Dict[str, str] = {}
    COMMON_HELP: Dict[str, str] = {
        "DEM": "Elevation raster in a projected coordinate system with metres as units. "
        "No-data cells are left out.",
        "INPUT_SU": "Raster of slope unit IDs, for example the output of Create Slope "
        "Units. 0 or no-data = no unit; alluvial plains (-1) and water bodies (-2) are kept "
        "as they are.",
        "PLAINS": "Optional raster of flat areas to leave out (valley floors, alluvial "
        "plains): every cell with a value gets no slope unit.",
        "THRESH": "Flow accumulation area (m²) that defines the first, coarsest stream "
        "network and its half-basins. It is lowered at every iteration. Larger values "
        "start from fewer, larger half-basins.",
        "AREAMIN": "Half-basins smaller than this area (m²) are not split further, and a "
        "split is undone when its new units are on average smaller than this. It is not "
        "a minimum size of the result: use Clean Slope Units for that.",
        "CVMIN": "Circular variance of aspect: 0 = all cells face the same way, 1 = "
        "directions completely mixed. Half-basins above this value keep being split. "
        "Lower values give more, smaller and more uniform units.",
        "RF": "At each iteration the flow accumulation threshold becomes threshold - "
        "threshold / rf: 2 halves it each time; larger values lower it more slowly, in "
        "more and finer steps.",
        "MAXITERATION": "Maximum number of times the threshold is lowered and the "
        "half-basins are split again.",
        "AREAMAX": "Units larger than this area (m²) keep being split even if their aspect "
        "is already uniform. 0 = no limit.",
        "USE_3D_NORMAL": "Measure the variability with the 3D surface normal (slope and "
        "aspect together) instead of aspect only; nearly flat cells, whose aspect is "
        "unreliable, then count less.",
        "AUTO_PLAINS": "Detect flat valley floors automatically and leave them out, as with a "
        "plains mask: connected areas below the maximum slope and larger than the minimum "
        "area, optionally widened up to the edge slope. They are written as -1 (raster) and "
        "class 'Alluvial plain' (vector); small units enclosed by them are merged into them.",
        "AUTO_PLAINS_EDGE": "Widen the detected plains into the connected cells with a slope "
        "up to this angle (the gentle sides of the valley floor). Must be larger than the "
        "maximum slope to have an effect; 0 = no widening. Try 8-10°.",
        "AUTO_PLAINS_POSITION": "Keep only the plains that lie at the bottom of their "
        "surroundings. Each detected plain gets a relative position: its median elevation "
        "scaled between the lowest (0) and the highest (1) elevation within the search radius "
        "around it. Plains above this value (flat hilltops, saddles, high terraces) are not "
        "left out and become slope units. Valley floors are usually below 0.2; 1 = no check.",
        "AUTO_PLAINS_RADIUS": "Distance (m) around each plain used for its relative position. "
        "It should reach the valley sides; increase it for very wide valley floors.",
        "WATER_BODIES": "Detect lakes and reservoirs automatically and leave them out: "
        "connected areas where the DEM is flat (neighbouring cells at the same elevation). "
        "They are written as -2 (raster) and class 'Water body' (vector); small units "
        "enclosed by them are merged into them.",
        "WATER_MIN_AREA": "Only flat areas at least this large (m²) are taken as water bodies.",
        "WATER_TOLERANCE": "Largest elevation difference (m) between neighbouring cells of a "
        "water surface. Use 0 for DEMs with perfectly flat lakes; raise it slightly for noisy "
        "water surfaces.",
        "AUTO_PLAINS_SLOPE": "Cells with a slope at or below this angle count as flat for "
        "the automatic plains detection.",
        "AUTO_PLAINS_AREA": "Only connected flat areas at least this large (m²) are left "
        "out as plains.",
        "ENGINE": "How flow directions and half-basins are computed:<br>"
        "• Native, multiple flow direction (default): the flow of each cell is shared "
        "among its lower neighbours.<br>"
        "• Native, single flow direction: each cell drains to one neighbour.<br>"
        "• Native fast D8: much faster on very large DEMs; results close to single flow."
        "<br>"
        "• GRASS r.watershed, multiple or single flow: needs GRASS GIS installed.",
        "CONVERGENCE": "Multiple flow engines only: how strongly the flow concentrates "
        "towards the steepest neighbours. Higher values give narrower flow paths, closer "
        "to single flow.",
        "ADD_MORPHOMETRY": "Add the circular variance, mean elevation, mean slope and mean "
        "aspect of each unit to the vector (the area is always added).",
        "CLEANSIZE": "Units smaller than this area (m²) are merged into a neighbour.",
        "METHOD": "Which neighbour a small unit joins:<br>"
        "• Quick: the one sharing the longest border.<br>"
        "• Aspect: the one with the most similar mean aspect.<br>"
        "• Hydrological: like Aspect, but never across a ridge or a river.",
        "OUTPUT_RASTER": "Raster of slope unit IDs (1, 2, ...); alluvial plains = -1, water "
        "bodies = -2, 0 = no data.",
        "OUTPUT_VECTOR": "Polygons of the slope units, alluvial plains and water bodies, with "
        "their class, area (m²) and, for slope units, the morphometric attributes.",
        "OUTPUT_CV_MAP": "Raster with the circular variance of the unit each cell "
        "belongs to.",
        "OUTPUT_AREA_MAP": "Raster with the area (m²) of the unit each cell belongs to.",
    }
    COMMON_HELP["CLEAN_METHOD"] = COMMON_HELP["METHOD"]

    def addParameter(self, parameterDefinition, createOutput=True):
        """Adds the tooltip text (help) of the parameter before registering it."""
        text = self.PARAM_HELP.get(parameterDefinition.name()) or self.COMMON_HELP.get(
            parameterDefinition.name()
        )
        if text:
            parameterDefinition.setHelp(text)
        return super().addParameter(parameterDefinition, createOutput)

    # cleaning method shared by Clean, Optimize and Nested (keys of clean_slope_units)
    CLEAN_METHODS = [
        "Hydrological (never across ridges or rivers)",
        "Aspect (most similar mean aspect, no hydrological check)",
        "Quick (longest shared border)",
    ]
    CLEAN_METHOD_KEYS = ["hydrological", "detailed", "quick"]
    CLEAN_METHOD_DEFAULT = 2  # Quick

    def add_clean_method_parameter(self, name, label="Cleaning method"):
        self.addParameter(
            QgsProcessingParameterEnum(
                name, label, options=self.CLEAN_METHODS, defaultValue=self.CLEAN_METHOD_DEFAULT
            )
        )

    def clean_method_and_drainage(self, parameters, name, context, feedback, dem_data, valid_mask,
                                  needed=True):
        """Selected cleaning method; for 'hydrological' also the drainage directions."""
        method = self.CLEAN_METHOD_KEYS[self.parameterAsEnum(parameters, name, context)]
        drainage = None
        if method == "hydrological" and needed:
            feedback.pushInfo("Computing drainage directions for the hydrological cleaning...")
            grid = prepare_watershed_grid(
                dem_data.array, valid_mask, dem_data.cell_size_x, dem_data.cell_size_y, mfd=True
            )
            drainage = grid.drainage
        return method, drainage

    def add_engine_parameters(self):
        """Half-basin engine and MFD convergence, shared by Create, Optimize and Nested."""
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ENGINE,
                "Half-basin engine",
                options=[
                    "Native - multiple flow direction",
                    "Native - single flow direction",
                    "Native fast D8 (for very large DEMs)",
                    "GRASS r.watershed - multiple flow direction (needs GRASS)",
                    "GRASS r.watershed - single flow direction (needs GRASS)",
                ],
                defaultValue=self.ENGINE_NATIVE_MFD,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONVERGENCE,
                "MFD convergence factor, 1-10 (multiple flow engines only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=10,
                optional=True,
            )
        )

    def prepareAlgorithm(self, parameters, context, feedback):
        # Give the dialog time to switch to the Log tab before the computation takes the
        # Python interpreter (the native engines are pure Python and would block redraws).
        time.sleep(0.3)
        return True

    def make_half_basin_provider(
        self,
        parameters,
        context,
        feedback,
        dem_layer,
        dem_data,
        valid_mask,
        thresh,
        rf,
        maxiter,
        cached=False,
    ):
        """
        Run the flow routing of the selected engine and return ``(provider, close)``:
        ``provider(threshold_in_cells)`` gives the half-basins, ``close()`` frees resources.
        With ``cached=True`` half-basins are kept in memory for repeated delineations.
        """
        engine = self.parameterAsEnum(parameters, self.ENGINE, context)
        convergence = self.parameterAsInt(parameters, self.CONVERGENCE, context) or 5
        pcb = self.make_progress_cb(feedback)
        close = lambda: None  # noqa: E731

        if engine in (self.ENGINE_NATIVE_MFD, self.ENGINE_NATIVE_SFD):
            use_mfd = engine == self.ENGINE_NATIVE_MFD
            feedback.pushInfo(
                "Running A* search and {} flow accumulation (r.watershed port)...".format(
                    "multiple" if use_mfd else "single"
                )
            )
            grid = prepare_watershed_grid(
                dem_data.array,
                valid_mask,
                dem_data.cell_size_x,
                dem_data.cell_size_y,
                mfd=use_mfd,
                convergence=convergence,
                progress_cb=pcb,
            )

            def provider(thc):
                return watershed_half_basins(grid, thc)

        elif engine == self.ENGINE_FAST_D8:
            feedback.pushInfo("Running Priority-Flood depression filling & D8 flow routing...")
            hydro = prepare_hydro_grid(
                dem_data.array,
                valid_mask,
                dem_data.cell_size_x,
                dem_data.cell_size_y,
                progress_cb=pcb,
            )

            def provider(thc):
                return extract_half_basins(hydro, thresh_cells=thc)

        else:
            feedback.pushInfo("Half-basins will be computed by GRASS r.watershed.")
            try:
                grass = GrassHalfBasinProvider(
                    dem_path=dem_layer.source(),
                    shape=dem_data.shape,
                    schedule=threshold_schedule(thresh, dem_data.cell_area, rf, maxiter),
                    sfd=(engine == self.ENGINE_GRASS_SFD),
                    convergence=convergence,
                    log=feedback.pushInfo,
                    is_canceled=feedback.isCanceled,
                )
            except GrassBackendError as err:
                raise QgsProcessingException(str(err))
            close = grass.close

            def provider(thc):
                try:
                    return grass(thc)
                except GrassBackendError as err:
                    raise QgsProcessingException(str(err))

        if cached:
            provider = CachedHalfBasins(provider)
        return provider, close

    def icon(self):
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.svg")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()

    def createInstance(self):
        return type(self)()

    @staticmethod
    def make_progress_cb(feedback: QgsProcessingFeedback):
        def _cb(pct: int):
            if feedback.isCanceled():
                raise InterruptedError("Execution canceled by user.")
            feedback.setProgress(int(pct))
            # let the QGIS interface redraw (the native engines are pure Python)
            time.sleep(0.002)

        return _cb

    # ------------------------------------------------------------ plains and water bodies
    AUTO_PLAINS_SLOPE = "AUTO_PLAINS_SLOPE"
    AUTO_PLAINS_AREA = "AUTO_PLAINS_AREA"
    AUTO_PLAINS_EDGE = "AUTO_PLAINS_EDGE"
    AUTO_PLAINS_POSITION = "AUTO_PLAINS_POSITION"
    AUTO_PLAINS_RADIUS = "AUTO_PLAINS_RADIUS"
    WATER_BODIES = "WATER_BODIES"
    WATER_MIN_AREA = "WATER_MIN_AREA"
    WATER_TOLERANCE = "WATER_TOLERANCE"

    def add_plains_water_parameters(self, slope_and_area: bool = True):
        """Automatic plains settings and water body detection (Create, Optimize, Nested)."""
        from qgis.core import QgsProcessingParameterBoolean

        if slope_and_area:
            self.addParameter(QgsProcessingParameterNumber(
                self.AUTO_PLAINS_SLOPE, "Auto-plains max slope threshold (degrees)",
                type=QgsProcessingParameterNumber.Double, defaultValue=5.0, minValue=0.1,
                maxValue=45.0, optional=True))
            self.addParameter(QgsProcessingParameterNumber(
                self.AUTO_PLAINS_AREA, "Auto-plains minimum contiguous area (m²)",
                type=QgsProcessingParameterNumber.Double, defaultValue=100000.0, minValue=1.0,
                optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.AUTO_PLAINS_EDGE, "Auto-plains edge slope (degrees, 0 = no growth)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.0, minValue=0.0,
            maxValue=45.0, optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.AUTO_PLAINS_POSITION, "Auto-plains maximum relative position (0-1, 1 = no check)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.3, minValue=0.0,
            maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber(
            self.AUTO_PLAINS_RADIUS, "Auto-plains relative position search radius (m)",
            type=QgsProcessingParameterNumber.Double, defaultValue=1000.0, minValue=1.0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.WATER_BODIES, "Detect and leave out water bodies automatically",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterNumber(
            self.WATER_MIN_AREA, "Water bodies minimum area (m²)",
            type=QgsProcessingParameterNumber.Double, defaultValue=5000.0, minValue=1.0,
            optional=True))
        self.addParameter(QgsProcessingParameterNumber(
            self.WATER_TOLERANCE, "Water bodies elevation tolerance (m)",
            type=QgsProcessingParameterNumber.Double, defaultValue=0.05, minValue=0.0,
            optional=True))

    def exclusion_kwargs(self, parameters, context) -> dict:
        """Plains / water settings of the tool, for load_dem_and_terrain."""
        kw = {}

        def has(name):
            return self.parameterDefinition(name) is not None

        if has(self.AUTO_PLAINS_SLOPE):
            kw["auto_plains_slope"] = self.parameterAsDouble(parameters, self.AUTO_PLAINS_SLOPE, context) or 5.0
        if has(self.AUTO_PLAINS_AREA):
            kw["auto_plains_area"] = self.parameterAsDouble(parameters, self.AUTO_PLAINS_AREA, context) or 100000.0
        if has(self.AUTO_PLAINS_EDGE):
            kw["auto_plains_edge"] = self.parameterAsDouble(parameters, self.AUTO_PLAINS_EDGE, context)
        if has(self.AUTO_PLAINS_POSITION):
            kw["auto_plains_position"] = self.parameterAsDouble(parameters, self.AUTO_PLAINS_POSITION, context)
            kw["auto_plains_radius"] = self.parameterAsDouble(parameters, self.AUTO_PLAINS_RADIUS, context) or 1000.0
        if has(self.WATER_BODIES):
            kw["water"] = self.parameterAsBool(parameters, self.WATER_BODIES, context)
            kw["water_min_area"] = self.parameterAsDouble(parameters, self.WATER_MIN_AREA, context) or 5000.0
            kw["water_tolerance"] = self.parameterAsDouble(parameters, self.WATER_TOLERANCE, context)
        return kw

    def finalize_labels(self, labels, valid_mask, min_cells, relabel=True, feedback=None):
        """
        Merge the slope units enclosed by plains / water bodies (smaller than ``min_cells``)
        into them and write the classes: units > 0, plains -1, water bodies -2.
        """
        plains = getattr(self, "plains_class_mask", None)
        water = getattr(self, "water_class_mask", None)
        if plains is None and water is None:
            return labels
        barrier = np.zeros(labels.shape, dtype=bool)
        for m in (plains, water):
            if m is not None:
                barrier |= m
        lab, absorbed = absorb_enclosed_units(labels, barrier & valid_mask, int(max(1, min_cells)))
        if feedback is not None and np.any(absorbed):
            n = len(np.unique(np.asarray(labels)[absorbed]))
            feedback.pushInfo(f"{n} small units enclosed by plains or water bodies merged into them.")
        return assign_classes(lab, valid_mask, plains, water, absorbed, relabel=relabel)

    @staticmethod
    def report_classes(labels, cell_area, feedback):
        for value, name in ((-1, "Alluvial plains"), (-2, "Water bodies")):
            n = int(np.count_nonzero(labels == value))
            if n:
                feedback.pushInfo(f"{name}: {n * cell_area / 1e6:.3f} km² (value {value}).")

    def load_dem_and_terrain(
        self,
        dem_path: str,
        plains_path: Optional[str] = None,
        auto_plains: bool = False,
        auto_plains_slope: float = 5.0,
        auto_plains_area: float = 100000.0,
        auto_plains_edge: float = 0.0,
        auto_plains_position: float = 1.0,
        auto_plains_radius: float = 1000.0,
        water: bool = False,
        water_min_area: float = 5000.0,
        water_tolerance: float = 0.05,
    ) -> Tuple[
        RasterData,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """Read DEM, build valid mask, compute Horn slope/aspect/normals and plains mask."""
        dem_data = read_raster(dem_path, dtype=np.float64)
        dem = dem_data.array
        valid_mask = np.isfinite(dem)
        if dem_data.nodata is not None:
            valid_mask &= dem != dem_data.nodata

        slope_rad, aspect_rad, sin_a, cos_a, nx, ny, nz = compute_slope_aspect(
            dem,
            dem_data.cell_size_x,
            dem_data.cell_size_y,
            valid_mask=valid_mask,
        )

        plains_mask = None
        if plains_path:
            pl_data = read_raster(plains_path, dtype=np.float64)
            pl_arr = pl_data.array
            if pl_data.nodata is not None:
                plains_mask = (pl_arr != pl_data.nodata) & (pl_arr > 0) & np.isfinite(pl_arr)
            else:
                plains_mask = (pl_arr > 0) & np.isfinite(pl_arr)
        elif auto_plains:
            plains_mask = detect_alluvial_plains(
                slope_rad=slope_rad,
                cell_area=dem_data.cell_area,
                slope_thresh_deg=auto_plains_slope,
                min_plain_area_m2=auto_plains_area,
                valid_mask=valid_mask,
                edge_slope_deg=auto_plains_edge if auto_plains_edge and auto_plains_edge > 0 else None,
                dem=np.where(valid_mask, dem, np.nan),
                cell_size=float(np.sqrt(dem_data.cell_area)),
                max_rel_position=auto_plains_position,
                position_radius_m=auto_plains_radius,
            )
        water_mask = None
        if water:
            water_mask = detect_water_bodies(
                dem, dem_data.cell_area, water_min_area, water_tolerance, valid_mask
            )
        # classes written in the outputs; the returned mask is everything left out
        self.plains_class_mask = plains_mask
        self.water_class_mask = water_mask
        if water_mask is not None:
            plains_mask = water_mask if plains_mask is None else (plains_mask | water_mask)

        return dem_data, valid_mask, slope_rad, aspect_rad, sin_a, cos_a, nx, ny, nz, plains_mask

    @staticmethod
    def build_morphometric_attributes(
        labels: np.ndarray,
        dem: np.ndarray,
        slope_rad: np.ndarray,
        sin_a: np.ndarray,
        cos_a: np.ndarray,
        cell_area: float,
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-slope-unit morphometric summary attributes for vector export."""
        lbl = np.where(labels > 0, labels, 0).ravel()
        max_id = int(lbl.max())
        if max_id == 0:
            return {}

        counts = np.bincount(lbl, minlength=max_id + 1)
        slope_deg = np.rad2deg(slope_rad).ravel()
        dem_flat = dem.ravel()
        safe_dem = np.where(lbl > 0, dem_flat, 0.0)
        safe_slope = np.where(lbl > 0, slope_deg, 0.0)

        sum_elev = np.bincount(lbl, weights=safe_dem, minlength=max_id + 1)
        sum_slope = np.bincount(lbl, weights=safe_slope, minlength=max_id + 1)
        s_sin = np.bincount(lbl, weights=sin_a.ravel(), minlength=max_id + 1)
        s_cos = np.bincount(lbl, weights=cos_a.ravel(), minlength=max_id + 1)

        attrs: Dict[int, Dict[str, float]] = {}
        for uid in range(1, max_id + 1):
            cnt = int(counts[uid])
            if cnt == 0:
                continue
            r_len = float(np.hypot(s_sin[uid], s_cos[uid]))
            cv = float(np.clip(1.0 - r_len / cnt, 0.0, 1.0))
            mean_asp = float(np.mod(np.rad2deg(np.arctan2(s_sin[uid], s_cos[uid])), 360.0))
            attrs[uid] = {
                "area_m2": float(cnt * cell_area),
                "cv": round(cv, 5),
                "mean_elev": round(float(sum_elev[uid] / cnt), 2),
                "mean_slope": round(float(sum_slope[uid] / cnt), 2),
                "mean_asp": round(mean_asp, 2),
            }
        return attrs
