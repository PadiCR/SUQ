# -*- coding: utf-8 -*-
"""
SUQ Slope Unit Metrics algorithm (equivalent to r.slopeunits.metrics).
"""

import csv
import numpy as np
from qgis.core import (
    QgsProcessing,
    QgsProcessingOutputNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
)

from .base_algorithm import SUQBaseAlgorithm
from ..suq_core.dem_io import rasterize_mask, read_raster
from ..suq_core.metrics import compute_segmentation_metrics


class SUQMetricsAlgorithm(SUQBaseAlgorithm):
    """Compute Espindola/Alvioli V (internal variance) and I (external autocorrelation)."""

    INPUT_SU = "INPUT_SU"
    DEM = "DEM"
    BASIN_MASK = "BASIN_MASK"
    OUTPUT_CSV = "OUTPUT_CSV"

    V_FIN = "V_FIN"
    I_FIN = "I_FIN"
    N_UNITS = "N_UNITS"

    PARAM_HELP = {
        "DEM": "The DEM used to create the slope units, on the same grid; it provides "
        "the aspect.",
        "BASIN_MASK": "Optional polygon layer: only the cells inside it are measured.",
        "OUTPUT_CSV": "Table with V, I, the number of units, the number of neighbouring "
        "pairs and the total area.",
    }

    def name(self):
        return "metrics"

    def displayName(self):
        return "3. Slope Unit Metrics (r.slopeunits.metrics)"

    def group(self):
        return "Slope Units"

    def groupId(self):
        return "slope_units"

    def shortHelpString(self):
        return (
            'Measures the quality of a slope unit map with two indices based on aspect:\n'
            '• V, internal variance: the circular variance of aspect inside each unit, averaged by area. Low V = units facing one direction.\n'
            '• I, external autocorrelation: how similar the mean aspect of neighbouring units is. Low I = each unit differs from its neighbours.\n\n'
            'A good map has both low. Small units lower V but raise I, large units do the opposite; Optimize Slope Unit Parameters looks for the best balance.\n\n'
            'Reference: Espindola et al. (2006); Alvioli et al. (2016, 2020).'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.INPUT_SU, "Input Slope Units Raster")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "Input Digital Elevation Model (DEM)")
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.BASIN_MASK,
                "Basin Polygon Mask",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output Metrics CSV",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )

        self.addOutput(QgsProcessingOutputNumber(self.V_FIN, "Internal Variance (V_fin)"))
        self.addOutput(QgsProcessingOutputNumber(self.I_FIN, "External Autocorrelation (I_fin)"))
        self.addOutput(QgsProcessingOutputNumber(self.N_UNITS, "Number of Slope Units"))

    def processAlgorithm(self, parameters, context, feedback):
        su_layer = self.parameterAsRasterLayer(parameters, self.INPUT_SU, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        basin_layer = self.parameterAsVectorLayer(parameters, self.BASIN_MASK, context)
        out_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        su_data = read_raster(su_layer.source(), dtype=np.int32)
        labels = su_data.array.copy()
        if basin_layer:
            mask = rasterize_mask(basin_layer.source(), su_data)
            labels[~mask] = 0

        _, _, _, _, sin_a, cos_a, _, _, _, _ = self.load_dem_and_terrain(dem_layer.source())
        m = compute_segmentation_metrics(labels, sin_a, cos_a, su_data.cell_area)

        feedback.pushInfo(
            f"Metrics: V_fin={m.v_fin:.6f}, I_fin={m.i_fin:.6f}, "
            f"n_units={m.n_units}, n_edges={m.n_edges}, area={m.total_area_m2:.1f} m²"
        )

        if out_csv:
            with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["v_fin", "i_fin", "n_units", "n_edges", "total_area_m2"])
                writer.writerow([m.v_fin, m.i_fin, m.n_units, m.n_edges, m.total_area_m2])

        feedback.setProgress(100)
        return {
            self.V_FIN: m.v_fin,
            self.I_FIN: m.i_fin,
            self.N_UNITS: m.n_units,
            self.OUTPUT_CSV: out_csv,
        }
