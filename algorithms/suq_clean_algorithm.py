# -*- coding: utf-8 -*-
"""
SUQ Clean Slope Units algorithm (equivalent to r.slopeunits.clean).
"""

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorDestination,
)

from .base_algorithm import SUQBaseAlgorithm
from ..suq_core.cleaning import absorb_enclosed_units, assign_classes, clean_slope_units
from ..suq_core.watershed import prepare_watershed_grid
from ..suq_core.dem_io import polygonize_labels, read_raster, write_raster


class SUQCleanAlgorithm(SUQBaseAlgorithm):
    """Clean small slope units and slivers below a size threshold."""

    INPUT_SU = "INPUT_SU"
    DEM = "DEM"
    PLAINS = "PLAINS"
    CLEANSIZE = "CLEANSIZE"
    METHOD = "METHOD"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    OUTPUT_VECTOR = "OUTPUT_VECTOR"


    PARAM_HELP = {
        "DEM": "The DEM used to create the slope units, on the same grid as the slope "
        "unit raster. It provides the aspect and, for the hydrological method, the "
        "drainage directions.",
        "OUTPUT_VECTOR": "Polygons of the cleaned units with their area, circular "
        "variance, mean elevation, slope and aspect, and small_kept (1 = still smaller "
        "than the minimum size because no valid neighbour was found).",
    }

    def name(self):
        return "clean"

    def displayName(self):
        return "2. Clean Slope Units (r.slopeunits.clean)"

    def group(self):
        return "Slope Units"

    def groupId(self):
        return "slope_units"

    def shortHelpString(self):
        return (
            'Removes slope units smaller than a minimum size by merging each of them into a neighbouring unit. Units are merged one at a time, smallest first, and a merged unit that is still too small is merged again. Existing borders are kept: only the border between the two merged units disappears.\n\n'
            'The cleaning method decides which neighbour a small unit joins:\n'
            '• Quick (default): the neighbour sharing the longest border.\n'
            '• Aspect: the neighbour with the most similar mean aspect.\n'
            '• Hydrological: like Aspect, but never across a ridge or a river; ridges and valleys are only crossed inside a small head catchment. A unit without a valid neighbour stays as it is and is marked small_kept = 1 in the vector.\n\n'
            'Areas of fewer than 10 cells always join the neighbour sharing the longest border.'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT_SU, "Input Slope Units Raster")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "Input Digital Elevation Model (DEM)")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PLAINS, "Alluvial Plains Mask Raster", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLEANSIZE,
                "Minimum slope unit size to keep (cleansize, m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=25000.0,
                minValue=1.0,
            )
        )
        self.add_clean_method_parameter(self.METHOD)
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER, "Output Cleaned Slope Units Raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                "Output Cleaned Slope Units Vector",
                optional=True,
                createByDefault=True,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        su_layer = self.parameterAsRasterLayer(parameters, self.INPUT_SU, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        plains_layer = self.parameterAsRasterLayer(parameters, self.PLAINS, context)
        cleansize = self.parameterAsDouble(parameters, self.CLEANSIZE, context)
        method = self.CLEAN_METHOD_KEYS[self.parameterAsEnum(parameters, self.METHOD, context)]

        out_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        out_vector = self.parameterAsOutputLayer(parameters, self.OUTPUT_VECTOR, context)

        su_data = read_raster(su_layer.source(), dtype=np.int32)
        dem_data, valid_mask, slope_rad, _, sin_a, cos_a, _, _, _, plains_mask = (
            self.load_dem_and_terrain(
                dem_layer.source(),
                plains_path=plains_layer.source() if plains_layer else None,
            )
        )

        if su_data.array.shape != dem_data.array.shape:
            raise QgsProcessingException(
                "The slope-unit raster and the DEM must have the same grid "
                f"(got {su_data.array.shape} and {dem_data.array.shape} cells)."
            )
        drainage = None
        if method == "hydrological":
            feedback.pushInfo("Computing drainage directions (r.watershed, multiple flow)...")
            grid = prepare_watershed_grid(
                dem_data.array,
                valid_mask,
                dem_data.cell_size_x,
                dem_data.cell_size_y,
                mfd=True,
                progress_cb=lambda p: feedback.setProgress(int(p * 0.8)),
            )
            if feedback.isCanceled():
                return {}
            drainage = grid.drainage
            del grid

        feedback.pushInfo(f"Cleaning slope units < {cleansize} m² using '{method}' mode...")
        # plains (-1) and water bodies (-2) of the input stay as they are
        src = su_data.array
        plains_cls = src == -1
        if plains_mask is not None:
            plains_cls = plains_cls | (plains_mask & (src >= 0))
        water_cls = src == -2
        left_out = plains_cls | water_cls
        cleaned, report = clean_slope_units(
            labels=src,
            cell_area=su_data.cell_area,
            cleansize=cleansize,
            sin_a=sin_a,
            cos_a=cos_a,
            method=method,
            plains_mask=left_out if np.any(left_out) else None,
            drainage=drainage,
            return_report=True,
        )
        min_cells = max(1, int(np.ceil(cleansize / su_data.cell_area)))
        if np.any(left_out):
            lab, absorbed = absorb_enclosed_units(cleaned, left_out & valid_mask, min_cells)
            if np.any(absorbed):
                feedback.pushInfo(
                    f"{len(np.unique(cleaned[absorbed]))} small units enclosed by plains or "
                    "water bodies merged into them."
                )
            cleaned = assign_classes(
                lab, valid_mask | left_out, plains_cls, water_cls, absorbed, relabel=True
            )
            self.report_classes(cleaned, su_data.cell_area, feedback)

        n_units = int(cleaned.max())
        feedback.pushInfo(
            f"Cleaning complete: {report.n_input} -> {n_units} slope units; "
            f"{report.n_small} were smaller than {cleansize:g} m², {report.n_merges} merges."
        )
        if report.blocked:
            feedback.pushInfo(
                "Neighbours rejected: "
                + ", ".join(f"{k} {v}" for k, v in sorted(report.blocked.items()))
            )
        if report.n_kept_small:
            feedback.pushInfo(
                f"{report.n_kept_small} units are still smaller than {cleansize:g} m² because "
                "no neighbour could be merged without crossing a ridge or a river "
                "(small_kept = 1 in the vector)."
            )
        write_raster(out_raster, cleaned, su_data, nodata=0, gdal_dtype=gdal.GDT_Int32)

        results = {self.OUTPUT_RASTER: out_raster}
        if out_vector:
            attrs = self.build_morphometric_attributes(
                cleaned, dem_data.array, slope_rad, sin_a, cos_a, su_data.cell_area
            )
            sizes = np.bincount(np.where(cleaned > 0, cleaned, 0).ravel())
            for su_id, attr_dict in attrs.items():
                attr_dict["small_kept"] = int(sizes[su_id] < min_cells)
            polygonize_labels(cleaned, su_data, out_vector, extra_attributes=attrs)
            results[self.OUTPUT_VECTOR] = out_vector

        feedback.setProgress(100)
        return results
