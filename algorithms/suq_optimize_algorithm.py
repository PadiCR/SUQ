# -*- coding: utf-8 -*-
"""
SUQ Optimize Parameters algorithm (equivalent to r.slopeunits.optimize).
"""

import csv
import datetime
import os
from osgeo import gdal
from qgis.core import (
    QgsProcessingOutputNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorDestination,
)

from .base_algorithm import SUQBaseAlgorithm
from ..suq_core.dem_io import polygonize_labels, write_raster
from ..suq_core.optimize import optimize_slope_units
from ..suq_core.terrain import aspect_cover_mask


class SUQOptimizeAlgorithm(SUQBaseAlgorithm):
    """Automatically optimize (cvmin, areamin) maximizing the F objective function."""

    DEM = "DEM"
    PLAINS = "PLAINS"
    THRESH = "THRESH"
    RF = "RF"
    MAXITERATION = "MAXITERATION"
    CLEANSIZE = "CLEANSIZE"
    CLEAN_METHOD = "CLEAN_METHOD"
    CVMIN_MIN = "CVMIN_MIN"
    CVMIN_MAX = "CVMIN_MAX"
    AREAMIN_MIN = "AREAMIN_MIN"
    AREAMIN_MAX = "AREAMIN_MAX"
    EPSILON_CV = "EPSILON_CV"
    EPSILON_AREA = "EPSILON_AREA"
    AREAMAX = "AREAMAX"
    USE_3D_NORMAL = "USE_3D_NORMAL"
    AUTO_PLAINS = "AUTO_PLAINS"

    OUTPUT_RASTER = "OUTPUT_RASTER"
    OUTPUT_VECTOR = "OUTPUT_VECTOR"
    OUTPUT_CSV = "OUTPUT_CSV"

    BEST_CVMIN = "BEST_CVMIN"
    BEST_AREAMIN = "BEST_AREAMIN"
    BEST_F = "BEST_F"

    PARAM_HELP = {
        "CLEANSIZE": "Each candidate map is cleaned of units smaller than this area (m²) "
        "before it is scored. 0 = no cleaning.",
        "CVMIN_MIN": "Lowest cvmin value searched.",
        "CVMIN_MAX": "Highest cvmin value searched.",
        "AREAMIN_MIN": "Lowest minimum area (m²) searched.",
        "AREAMIN_MAX": "Highest minimum area (m²) searched.",
        "EPSILON_CV": "The search stops once the cvmin range is narrower than this "
        "(and the area range narrower than its tolerance).",
        "EPSILON_AREA": "The search stops once the area range (m²) is narrower than this "
        "(and the cvmin range narrower than its tolerance).",
        "OUTPUT_RASTER": "Cleaned slope units for the best (cvmin, areamin) pair.",
        "OUTPUT_VECTOR": "Polygons of the best slope units with their area, circular "
        "variance, mean elevation, slope and aspect.",
        "OUTPUT_CSV": "Every pair tried, with its V, I, F and number of units.",
    }

    def name(self):
        return "optimize"

    def displayName(self):
        return "4. Optimize Slope Unit Parameters (r.slopeunits.optimize)"

    def group(self):
        return "Slope Units"

    def groupId(self):
        return "slope_units"

    def shortHelpString(self):
        return (
            'Finds the cvmin and minimum area that give the best slope unit map for a DEM.\n\n'
            'For each (cvmin, areamin) pair tried, slope units are created as in Create Slope Units, cleaned with the chosen size and method, and scored with V and I (see Slope Unit Metrics). V and I are rescaled from 0 to 1 over all the pairs tried and added into F = F_V + F_I; the pair with the highest F is the best.\n\n'
            'The search tries a 3 × 3 grid of pairs over the given ranges, then repeatedly narrows the ranges around the best pair, until they are smaller than the tolerances.\n\n'
            'Outputs: the best cvmin and areamin, the cleaned slope units for that pair and, optionally, a table of every pair tried.'
        )

    def initAlgorithm(self, config=None):
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
                self.THRESH,
                "Initial flow accumulation threshold (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000000.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RF,
                "Threshold reduction factor (rf)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXITERATION,
                "Maximum iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=20,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLEANSIZE,
                "Minimum unit size for cleaning (cleansize, m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=25000.0,
                minValue=0.0,
            )
        )
        self.add_clean_method_parameter(self.CLEAN_METHOD)
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CVMIN_MIN,
                "Minimum cvmin search bound",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.05,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CVMIN_MAX,
                "Maximum cvmin search bound",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.35,
                minValue=0.0,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AREAMIN_MIN,
                "Minimum areamin search bound (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=50000.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AREAMIN_MAX,
                "Maximum areamin search bound (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=200000.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EPSILON_CV,
                "Convergence tolerance for cvmin",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.02,
                minValue=0.001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EPSILON_AREA,
                "Convergence tolerance for areamin (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=25000.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AREAMAX,
                "Maximum slope unit area (m², 0 = no limit)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_3D_NORMAL,
                "Use 3D surface normal variance (slope + aspect)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.AUTO_PLAINS,
                "Detect and leave out alluvial plains automatically",
                defaultValue=False,
            )
        )

        self.add_plains_water_parameters(slope_and_area=True)
        self.add_engine_parameters()
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER, "Output Optimal Cleaned Slope Units Raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                "Output Optimal Slope Units Vector",
                optional=True,
                createByDefault=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output Optimization Candidates Table (CSV)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )

        self.addOutput(QgsProcessingOutputNumber(self.BEST_CVMIN, "Optimal cvmin"))
        self.addOutput(QgsProcessingOutputNumber(self.BEST_AREAMIN, "Optimal areamin (m²)"))
        self.addOutput(QgsProcessingOutputNumber(self.BEST_F, "Best Objective Score (F)"))

    @staticmethod
    def _best_parameters_path(out_raster):
        """Text file next to the output raster, or in the project folder (else the user's
        home folder) when the raster is a temporary output."""
        from qgis.core import QgsProcessingUtils, QgsProject

        raster_dir = os.path.dirname(os.path.abspath(out_raster))
        temp_dir = os.path.abspath(QgsProcessingUtils.tempFolder())
        if os.path.normcase(raster_dir).startswith(os.path.normcase(temp_dir)):
            folder = QgsProject.instance().homePath() or os.path.expanduser("~")
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(folder, f"SUQ_optimize_best_parameters_{stamp}.txt")
        base = os.path.splitext(os.path.basename(out_raster))[0]
        return os.path.join(raster_dir, f"{base}_best_parameters.txt")

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        plains_layer = self.parameterAsRasterLayer(parameters, self.PLAINS, context)

        thresh = self.parameterAsDouble(parameters, self.THRESH, context)
        rf = self.parameterAsInt(parameters, self.RF, context)
        maxiter = self.parameterAsInt(parameters, self.MAXITERATION, context)
        cleansize = self.parameterAsDouble(parameters, self.CLEANSIZE, context)
        cv_lo = self.parameterAsDouble(parameters, self.CVMIN_MIN, context)
        cv_hi = self.parameterAsDouble(parameters, self.CVMIN_MAX, context)
        ar_lo = self.parameterAsDouble(parameters, self.AREAMIN_MIN, context)
        ar_hi = self.parameterAsDouble(parameters, self.AREAMIN_MAX, context)
        eps_cv = self.parameterAsDouble(parameters, self.EPSILON_CV, context)
        eps_ar = self.parameterAsDouble(parameters, self.EPSILON_AREA, context)
        areamax = self.parameterAsDouble(parameters, self.AREAMAX, context)
        use_3d = self.parameterAsBool(parameters, self.USE_3D_NORMAL, context)
        auto_plains = self.parameterAsBool(parameters, self.AUTO_PLAINS, context)

        out_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        out_vector = self.parameterAsOutputLayer(parameters, self.OUTPUT_VECTOR, context)
        out_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        pcb = self.make_progress_cb(feedback)
        (
            dem_data,
            valid_mask,
            slope_rad,
            _,
            sin_a,
            cos_a,
            nx,
            ny,
            nz,
            plains_mask,
        ) = self.load_dem_and_terrain(
            dem_layer.source(),
            plains_path=plains_layer.source() if plains_layer else None,
            auto_plains=auto_plains,
            **self.exclusion_kwargs(parameters, context),
        )

        clean_method, clean_drainage = self.clean_method_and_drainage(
            parameters, self.CLEAN_METHOD, context, feedback, dem_data, valid_mask,
            needed=cleansize > 0,
        )
        provider, close_engine = self.make_half_basin_provider(
            parameters, context, feedback, dem_layer, dem_data, valid_mask, thresh, rf, maxiter,
            cached=True,
        )

        feedback.pushInfo("Evaluating (cvmin, areamin) candidates: create + clean + metrics...")
        try:
            opt = optimize_slope_units(
                hydro=None,
                sin_a=sin_a,
                cos_a=cos_a,
                nx=nx,
                ny=ny,
                nz=nz,
                cell_area=dem_data.cell_area,
                thresh=thresh,
                cvmin_range=(cv_lo, cv_hi),
                areamin_range=(ar_lo, ar_hi),
                epsilon_cv=eps_cv,
                epsilon_area=eps_ar,
                rf=rf,
                maxiteration=maxiter,
                cleansize=cleansize,
                clean_method=clean_method,
                clean_drainage=clean_drainage,
                areamax=areamax if areamax > 0 else None,
                plains_mask=plains_mask,
                use_3d_normal=use_3d,
                progress_cb=pcb,
                cover_mask=aspect_cover_mask(valid_mask),
                half_basin_provider=provider,
                valid_mask=valid_mask,
                log=feedback.pushInfo,
            )
        finally:
            close_engine()

        feedback.pushInfo(
            f"Optimization complete across {len(opt.candidates)} candidates: "
            f"best cvmin={opt.best_cvmin}, areamin={opt.best_areamin} m², "
            f"F={opt.best_f:.4f} (V={opt.best_v:.4f}, I={opt.best_i:.4f})"
        )

        best_labels = self.finalize_labels(
            opt.best_cleaned_labels, valid_mask,
            max(opt.best_areamin, cleansize) / dem_data.cell_area, feedback=feedback,
        )
        self.report_classes(best_labels, dem_data.cell_area, feedback)
        write_raster(out_raster, best_labels, dem_data, nodata=0, gdal_dtype=gdal.GDT_Int32)

        results = {
            self.OUTPUT_RASTER: out_raster,
            self.BEST_CVMIN: opt.best_cvmin,
            self.BEST_AREAMIN: opt.best_areamin,
            self.BEST_F: opt.best_f,
        }

        if out_vector:
            attrs = self.build_morphometric_attributes(
                best_labels,
                dem_data.array,
                slope_rad,
                sin_a,
                cos_a,
                dem_data.cell_area,
            )
            polygonize_labels(best_labels, dem_data, out_vector, extra_attributes=attrs)
            results[self.OUTPUT_VECTOR] = out_vector

        if out_csv:
            # best pair first (best = 1), then the other candidates by decreasing F
            ranked = sorted(opt.candidates, key=lambda c: -c.f_score)
            with open(out_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    ["best", "cvmin", "areamin", "v_fin", "i_fin", "f_score", "n_units"]
                )
                for c in ranked:
                    is_best = int(c.cvmin == opt.best_cvmin and c.areamin == opt.best_areamin)
                    writer.writerow(
                        [is_best, c.cvmin, c.areamin, c.v_fin, c.i_fin, c.f_score, c.n_units]
                    )
            results[self.OUTPUT_CSV] = out_csv
        else:
            txt = self._best_parameters_path(out_raster)
            best = next(
                (c for c in opt.candidates
                 if c.cvmin == opt.best_cvmin and c.areamin == opt.best_areamin),
                None,
            )
            engine = self.parameterDefinition(self.ENGINE).options()[
                self.parameterAsEnum(parameters, self.ENGINE, context)
            ]
            lines = [
                "SUQ - Optimize Slope Unit Parameters",
                f"Date: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                "",
                "Best parameters",
                f"  cvmin   = {opt.best_cvmin}",
                f"  areamin = {opt.best_areamin} m2",
                f"  F = {opt.best_f:.4f} (V = {opt.best_v:.4f}, I = {opt.best_i:.4f})"
                + (f", {best.n_units} units" if best else ""),
                "",
                "Settings of the run",
                f"  DEM: {dem_layer.source()}",
                f"  Initial flow accumulation threshold: {thresh} m2",
                f"  Threshold reduction factor: {rf}",
                f"  Maximum iterations: {maxiter}",
                f"  Half-basin engine: {engine}",
                f"  Convergence factor: "
                f"{self.parameterAsInt(parameters, self.CONVERGENCE, context) or 5}",
                f"  Plains mask: {plains_layer.source() if plains_layer else 'none'}",
                f"  Automatic plains detection: {'yes' if auto_plains else 'no'}",
                f"  Maximum slope unit area: {areamax if areamax > 0 else 'no limit'}",
                f"  3D surface normal variance: {'yes' if use_3d else 'no'}",
                f"  Cleaning: {cleansize} m2, method {clean_method}",
                f"  cvmin searched: {cv_lo} - {cv_hi} (tolerance {eps_cv})",
                f"  areamin searched: {ar_lo} - {ar_hi} m2 (tolerance {eps_ar} m2)",
                f"  Candidates evaluated: {len(opt.candidates)}",
                f"  Output raster: {out_raster}",
            ]
            with open(txt, "w", encoding="utf-8") as fh:
                fh.write(chr(10).join(lines) + chr(10))
            feedback.pushInfo(f"Best parameters written to {txt}")

        feedback.setProgress(100)
        return results
