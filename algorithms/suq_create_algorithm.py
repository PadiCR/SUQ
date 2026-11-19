# -*- coding: utf-8 -*-
"""
SUQ Create Slope Units algorithm (r.slopeunits.create + optional SUQ enhancements).
"""

from osgeo import gdal
from qgis.core import (
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorDestination,
)

from .base_algorithm import SUQBaseAlgorithm
from ..suq_core.dem_io import polygonize_labels, write_raster
from ..suq_core.slope_units import create_slope_units
from ..suq_core.terrain import aspect_cover_mask


class SUQCreateAlgorithm(SUQBaseAlgorithm):
    """Create slope units from a DEM using iterative half-basin segmentation."""

    DEM = "DEM"
    PLAINS = "PLAINS"
    THRESH = "THRESH"
    AREAMIN = "AREAMIN"
    AREAMAX = "AREAMAX"
    CVMIN = "CVMIN"
    RF = "RF"
    MAXITERATION = "MAXITERATION"
    USE_3D_NORMAL = "USE_3D_NORMAL"
    AUTO_PLAINS = "AUTO_PLAINS"
    AUTO_PLAINS_SLOPE = "AUTO_PLAINS_SLOPE"
    AUTO_PLAINS_AREA = "AUTO_PLAINS_AREA"
    ADD_MORPHOMETRY = "ADD_MORPHOMETRY"
    OUTPUT_RASTER = "OUTPUT_RASTER"
    OUTPUT_VECTOR = "OUTPUT_VECTOR"
    OUTPUT_CV_MAP = "OUTPUT_CV_MAP"
    OUTPUT_AREA_MAP = "OUTPUT_AREA_MAP"

    def name(self):
        return "create"

    def displayName(self):
        return "1. Create Slope Units (r.slopeunits.create)"

    def group(self):
        return "Slope Units"

    def groupId(self):
        return "slope_units"

    def shortHelpString(self):
        return (
            'Delineates slope units from a DEM in a projected coordinate system (metres).\n\n'
            'The DEM is first divided into half-basins: the areas draining to the left and to the right side of each stream segment, with streams defined by the initial flow accumulation threshold. A half-basin is split again, with a lower threshold, while it is larger than the minimum area and its aspect is still too variable (circular variance above cvmin). This repeats until no half-basin needs splitting or the maximum number of iterations is reached.\n\n'
            'The result usually contains some very small units and slivers; remove them with Clean Slope Units.\n\n'
            'Options:\n'
            '• Alluvial plains (given as a mask, or detected automatically and optionally widened up to an edge slope) and water bodies (flat areas of the DEM): left out of the slope units and written as -1 and -2 in the raster and as classes in the vector. Small units enclosed by them are merged into them.\n'
            '• Maximum area: larger units keep being split even if their aspect is uniform.\n'
            '• 3D surface normal variance: slope and aspect together instead of aspect only.\n'
            '• Morphometric attributes in the vector output.'
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
                self.AREAMIN,
                "Minimum slope unit area (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=50000.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CVMIN,
                "Minimum circular variance (0.0 - 1.0)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.15,
                minValue=0.0,
                maxValue=1.0,
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AUTO_PLAINS_SLOPE,
                "Auto-plains max slope threshold (degrees)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.1,
                maxValue=45.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AUTO_PLAINS_AREA,
                "Auto-plains minimum contiguous area (m²)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100000.0,
                minValue=1.0,
                optional=True,
            )
        )
        self.add_plains_water_parameters(slope_and_area=False)
        self.add_engine_parameters()
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ADD_MORPHOMETRY,
                "Compute morphometric summary attributes on vector output",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER, "Output Slope Units Raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_VECTOR,
                "Output Slope Units Vector",
                optional=True,
                createByDefault=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_CV_MAP,
                "Output Circular Variance Raster",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_AREA_MAP,
                "Output Area (m²) Raster",
                optional=True,
                createByDefault=False,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        plains_layer = self.parameterAsRasterLayer(parameters, self.PLAINS, context)

        thresh = self.parameterAsDouble(parameters, self.THRESH, context)
        areamin = self.parameterAsDouble(parameters, self.AREAMIN, context)
        cvmin = self.parameterAsDouble(parameters, self.CVMIN, context)
        rf = self.parameterAsInt(parameters, self.RF, context)
        maxiter = self.parameterAsInt(parameters, self.MAXITERATION, context)
        areamax = self.parameterAsDouble(parameters, self.AREAMAX, context)
        use_3d = self.parameterAsBool(parameters, self.USE_3D_NORMAL, context)
        auto_plains = self.parameterAsBool(parameters, self.AUTO_PLAINS, context)
        add_morph = self.parameterAsBool(parameters, self.ADD_MORPHOMETRY, context)

        out_raster = self.parameterAsOutputLayer(parameters, self.OUTPUT_RASTER, context)
        out_vector = self.parameterAsOutputLayer(parameters, self.OUTPUT_VECTOR, context)
        out_cv = self.parameterAsOutputLayer(parameters, self.OUTPUT_CV_MAP, context)
        out_area = self.parameterAsOutputLayer(parameters, self.OUTPUT_AREA_MAP, context)

        pcb = self.make_progress_cb(feedback)
        feedback.pushInfo("Loading DEM and computing Horn 3x3 terrain derivatives...")
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

        provider, close_engine = self.make_half_basin_provider(
            parameters, context, feedback, dem_layer, dem_data, valid_mask, thresh, rf, maxiter
        )

        feedback.pushInfo("Delineating slope units...")
        try:
            res = create_slope_units(
                hydro=None,
                sin_a=sin_a,
                cos_a=cos_a,
                nx=nx,
                ny=ny,
                nz=nz,
                cell_area=dem_data.cell_area,
                thresh=thresh,
                areamin=areamin,
                cvmin=cvmin,
                rf=rf,
                maxiteration=maxiter,
                areamax=areamax if areamax > 0 else None,
                plains_mask=plains_mask,
                use_3d_normal=use_3d,
                progress_cb=pcb,
                cover_mask=aspect_cover_mask(valid_mask),
                half_basin_provider=provider,
                valid_mask=valid_mask,
            )
        finally:
            close_engine()

        labels = self.finalize_labels(
            res.labels, valid_mask, areamin / dem_data.cell_area, feedback=feedback
        )
        res.cv_map[labels <= 0] = 0.0
        res.area_map[labels <= 0] = 0.0
        self.report_classes(labels, dem_data.cell_area, feedback)
        feedback.pushInfo(
            f"Created {int(labels.max())} slope units in {res.iterations_used} iterations."
        )
        write_raster(out_raster, labels, dem_data, nodata=0, gdal_dtype=gdal.GDT_Int32)

        results = {self.OUTPUT_RASTER: out_raster}
        if out_cv:
            write_raster(out_cv, res.cv_map, dem_data, nodata=0.0, gdal_dtype=gdal.GDT_Float32)
            results[self.OUTPUT_CV_MAP] = out_cv
        if out_area:
            write_raster(out_area, res.area_map, dem_data, nodata=0.0, gdal_dtype=gdal.GDT_Float32)
            results[self.OUTPUT_AREA_MAP] = out_area

        if out_vector:
            feedback.pushInfo("Polygonizing slope units to vector layer...")
            attrs = (
                self.build_morphometric_attributes(
                    labels, dem_data.array, slope_rad, sin_a, cos_a, dem_data.cell_area
                )
                if add_morph
                else None
            )
            polygonize_labels(
                labels, dem_data, out_vector, extra_attributes=attrs, progress_cb=pcb
            )
            results[self.OUTPUT_VECTOR] = out_vector

        feedback.setProgress(100)
        return results
