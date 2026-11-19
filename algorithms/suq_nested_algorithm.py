# -*- coding: utf-8 -*-
"""
SUQ Nested Multi-Scale Slope Units algorithm (equivalent to r.slopeunits.nested).
"""

import os
from osgeo import gdal
from qgis.core import (
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
)

from .base_algorithm import SUQBaseAlgorithm
from ..suq_core.dem_io import polygonize_labels, write_raster
import numpy as np

from ..suq_core.cleaning import absorb_enclosed_units, assign_classes, clean_slope_units
from ..suq_core.slope_units import create_slope_units, nest_layers
from ..suq_core.terrain import aspect_cover_mask


class SUQNestedAlgorithm(SUQBaseAlgorithm):
    """Generate hierarchically nested multi-scale slope unit layers in a single pass."""

    DEM = "DEM"
    PLAINS = "PLAINS"
    THRESH = "THRESH"
    AREAMIN = "AREAMIN"
    CVMIN_LEVELS = "CVMIN_LEVELS"
    RF = "RF"
    MAXITERATION = "MAXITERATION"
    AREAMAX = "AREAMAX"
    USE_3D_NORMAL = "USE_3D_NORMAL"
    AUTO_PLAINS = "AUTO_PLAINS"
    CLEANSIZE = "CLEANSIZE"
    CLEAN_METHOD = "CLEAN_METHOD"

    OUTPUT_DIR = "OUTPUT_DIR"
    OUTPUT_BASE_VECTOR = "OUTPUT_BASE_VECTOR"

    PARAM_HELP = {
        "CVMIN_LEVELS": "Comma-separated cvmin values, one layer each (they are sorted "
        "from smallest to largest). Values above which the units no longer change give "
        "identical layers.",
        "CLEANSIZE": "If greater than 0, units smaller than this area (m²) are merged "
        "into a neighbour in every layer before nesting. 0 = no cleaning.",
        "OUTPUT_DIR": "Folder where every layer is written as a raster (.tif) and a "
        "vector (.gpkg), named su_layer_<n>_cv<value>.",
        "OUTPUT_BASE_VECTOR": "Optional extra copy of the finest layer (layer 0) as a "
        "vector.",
    }

    def name(self):
        return "nested"

    def displayName(self):
        return "5. Nested Multi-Scale Slope Units (r.slopeunits.nested)"

    def group(self):
        return "Slope Units"

    def groupId(self):
        return "slope_units"

    def shortHelpString(self):
        return (
            'Creates several slope unit maps, from fine to coarse, that fit exactly inside each other: one map (layer) for each cvmin value of the list.\n\n'
            'The smallest cvmin gives the finest layer (layer 0), identical to Create Slope Units with the same settings. Each following layer is built only from whole units of the layer below, so every border of a coarse layer is also a border of the finer layers. A fine unit that would cross a border of the coarse layer is assigned to the coarse unit holding most of its area; the log reports the share of cells moved.\n\n'
            'The field parent_su gives, for each unit, the unit that contains it in the next coarser layer.\n\n'
            'Optionally, small units can be removed from every layer before nesting (removal size and cleaning method, as in Clean Slope Units).'
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
            QgsProcessingParameterString(
                self.CVMIN_LEVELS,
                "Comma-separated cvmin thresholds (finest to coarsest)",
                defaultValue="0.2, 0.3, 0.4, 0.5, 0.6, 0.7",
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
                self.CLEANSIZE,
                "Remove units smaller than this area before nesting (m², 0 = keep all)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
            )
        )
        self.add_clean_method_parameter(self.CLEAN_METHOD, "Cleaning method (used when the size above is > 0)")
        self.add_plains_water_parameters(slope_and_area=True)
        self.add_engine_parameters()
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR, "Output Directory for All Nested Layers"
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_BASE_VECTOR,
                "Output Base Layer (Layer 0) Vector",
                optional=True,
                createByDefault=True,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        plains_layer = self.parameterAsRasterLayer(parameters, self.PLAINS, context)

        thresh = self.parameterAsDouble(parameters, self.THRESH, context)
        areamin = self.parameterAsDouble(parameters, self.AREAMIN, context)
        cv_str = self.parameterAsString(parameters, self.CVMIN_LEVELS, context)
        cv_levels = [float(x.strip()) for x in cv_str.split(",") if x.strip()]
        rf = self.parameterAsInt(parameters, self.RF, context)
        maxiter = self.parameterAsInt(parameters, self.MAXITERATION, context)
        areamax = self.parameterAsDouble(parameters, self.AREAMAX, context)
        use_3d = self.parameterAsBool(parameters, self.USE_3D_NORMAL, context)
        auto_plains = self.parameterAsBool(parameters, self.AUTO_PLAINS, context)
        cleansize = self.parameterAsDouble(parameters, self.CLEANSIZE, context)

        out_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        out_base_vec = self.parameterAsOutputLayer(parameters, self.OUTPUT_BASE_VECTOR, context)
        os.makedirs(out_dir, exist_ok=True)

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

        cv_levels = sorted(set(cv_levels))
        cover = aspect_cover_mask(valid_mask)
        raw_layers = []
        try:
            for n_done, cv_val in enumerate(cv_levels):
                feedback.pushInfo(f"Delineating slope units for cvmin={cv_val}...")
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
                    cvmin=cv_val,
                    rf=rf,
                    maxiteration=maxiter,
                    areamax=areamax if areamax > 0 else None,
                    plains_mask=plains_mask,
                    use_3d_normal=use_3d,
                    cover_mask=cover,
                    half_basin_provider=provider,
                    valid_mask=valid_mask,
                )
                labels = res.labels
                if cleansize > 0:
                    labels = clean_slope_units(
                        labels,
                        cell_area=dem_data.cell_area,
                        cleansize=cleansize,
                        sin_a=sin_a,
                        cos_a=cos_a,
                        method=clean_method,
                        plains_mask=plains_mask,
                        drainage=clean_drainage,
                    )
                raw_layers.append(labels)
                pcb(45 + int(40 * (n_done + 1) / len(cv_levels)))
        finally:
            close_engine()

        # small units enclosed by plains / water bodies are merged into them in the finest
        # layer; the coarser layers, built from it, follow
        plains_cls = getattr(self, "plains_class_mask", None)
        water_cls = getattr(self, "water_class_mask", None)
        absorbed = None
        if raw_layers and (plains_cls is not None or water_cls is not None):
            barrier = np.zeros(valid_mask.shape, dtype=bool)
            for m in (plains_cls, water_cls):
                if m is not None:
                    barrier |= m
            min_cells = max(areamin, cleansize) / dem_data.cell_area
            lab0, absorbed = absorb_enclosed_units(raw_layers[0], barrier & valid_mask, min_cells)
            if np.any(absorbed):
                feedback.pushInfo(
                    f"{len(np.unique(raw_layers[0][absorbed]))} small units enclosed by plains "
                    "or water bodies merged into them."
                )
            raw_layers[0] = assign_classes(lab0, valid_mask, relabel=True)

        feedback.pushInfo("Enforcing strict nesting between layers...")
        nested_layers = []
        for cv_val, (lbl, parent_map, moved) in zip(cv_levels, nest_layers(raw_layers)):
            if plains_cls is not None or water_cls is not None:
                lbl = assign_classes(lbl, valid_mask, plains_cls, water_cls, absorbed, relabel=False)
            nested_layers.append((cv_val, lbl, parent_map))
            if moved > 0:
                feedback.pushInfo(
                    f"  cvmin={cv_val}: {100.0 * moved:.2f}% of cells reassigned to obtain nesting"
                )

        for idx, (cv_val, lbl, parent_map) in enumerate(nested_layers):
            tag = f"layer_{idx}_cv{cv_val:.2f}".replace(".", "_")
            tif_path = os.path.join(out_dir, f"su_{tag}.tif")
            gpkg_path = os.path.join(out_dir, f"su_{tag}.gpkg")

            write_raster(tif_path, lbl, dem_data, nodata=0, gdal_dtype=gdal.GDT_Int32)
            attrs = self.build_morphometric_attributes(
                lbl, dem_data.array, slope_rad, sin_a, cos_a, dem_data.cell_area
            )
            for su_id, attr_dict in attrs.items():
                attr_dict["parent_su"] = int(parent_map.get(su_id, 0))
            polygonize_labels(lbl, dem_data, gpkg_path, extra_attributes=attrs)

            if idx == 0 and out_base_vec:
                polygonize_labels(lbl, dem_data, out_base_vec, extra_attributes=attrs)

            feedback.pushInfo(
                f"Layer {idx} (cvmin={cv_val}): {int(lbl.max())} polygons -> {gpkg_path}"
            )

        feedback.setProgress(100)
        return {
            self.OUTPUT_DIR: out_dir,
            self.OUTPUT_BASE_VECTOR: out_base_vec,
        }
