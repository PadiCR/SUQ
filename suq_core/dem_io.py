# -*- coding: utf-8 -*-
"""
GDAL-based raster and vector I/O helpers for SUQ.
"""

from dataclasses import dataclass
import os
from typing import Dict, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr, osr

gdal.AllRegister()
ogr.RegisterAll()
gdal.UseExceptions()
ogr.UseExceptions()


@dataclass
class RasterData:
    """In-memory raster container with spatial metadata."""

    array: np.ndarray
    geotransform: Tuple[float, float, float, float, float, float]
    projection: str
    nodata: Optional[float]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.array.shape

    @property
    def cell_size_x(self) -> float:
        return abs(float(self.geotransform[1]))

    @property
    def cell_size_y(self) -> float:
        return abs(float(self.geotransform[5]))

    @property
    def cell_area(self) -> float:
        """Pixel area in map units^2 (projected CRS assumed in meters)."""
        return self.cell_size_x * self.cell_size_y


def read_raster(path: str, band_index: int = 1, dtype=np.float64) -> RasterData:
    """Read a single-band raster into a RasterData object."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise IOError(f"Could not open raster file: {path}")

    band = ds.GetRasterBand(band_index)
    arr = band.ReadAsArray().astype(dtype, copy=False)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection() or ""
    ds = None
    return RasterData(array=arr, geotransform=gt, projection=proj, nodata=nodata)


def write_raster(
    path: str,
    array: np.ndarray,
    reference: RasterData,
    nodata: float = 0,
    gdal_dtype: int = gdal.GDT_Int32,
) -> str:
    """Write a 2D NumPy array to GeoTIFF using reference geotransform/projection."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    out_ds = driver.Create(
        path,
        cols,
        rows,
        1,
        gdal_dtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(reference.geotransform)
    if reference.projection:
        out_ds.SetProjection(reference.projection)
    band = out_ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(array)
    band.FlushCache()
    out_ds = None
    return path


def _ogr_driver_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".shp":
        return "ESRI Shapefile"
    if ext in (".geojson", ".json"):
        return "GeoJSON"
    return "GPKG"


AREA_FIELD = "area_m2"
CLASS_FIELD = "class"


def _class_name(value: int) -> str:
    """Name of the class of a label: slope unit (> 0), alluvial plain (-1), water body (-2)."""
    if value > 0:
        return "Slope unit"
    return {-1: "Alluvial plain", -2: "Water body"}.get(value, "Other")


def polygonize_labels(
    labels: np.ndarray,
    reference: RasterData,
    output_path: str,
    field_name: str = "su_id",
    extra_attributes: Optional[Dict[int, Dict[str, float]]] = None,
    progress_cb=None,
) -> str:
    """Polygonize an integer slope-unit label array (0 = nodata) to vector.

    Every feature gets an ``area_m2`` field with the area of its polygon (map units^2,
    i.e. m^2 for a projected CRS in metres).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows, cols = labels.shape
    mem_driver = gdal.GetDriverByName("MEM")
    mem_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Int32)
    mem_ds.SetGeoTransform(reference.geotransform)
    if reference.projection:
        mem_ds.SetProjection(reference.projection)
    band = mem_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(labels.astype(np.int32, copy=False))

    mask_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(reference.geotransform)
    mask_ds.GetRasterBand(1).WriteArray((labels != 0).astype(np.uint8))

    drv_name = _ogr_driver_for_path(output_path)
    vec_driver = ogr.GetDriverByName(drv_name)
    if os.path.exists(output_path):
        vec_driver.DeleteDataSource(output_path)

    srs = None
    if reference.projection:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(reference.projection)

    vec_ds = vec_driver.CreateDataSource(output_path)
    layer = vec_ds.CreateLayer("slope_units", srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn(AREA_FIELD, ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn(CLASS_FIELD, ogr.OFTString))

    extra_fields = []
    if extra_attributes:
        sample = next(iter(extra_attributes.values()))
        for key, val in sample.items():
            if key in (field_name, AREA_FIELD, CLASS_FIELD):
                continue
            ftype = ogr.OFTInteger if isinstance(val, (int, np.integer)) else ogr.OFTReal
            layer.CreateField(ogr.FieldDefn(key, ftype))
            extra_fields.append(key)

    def _gdal_progress(complete, message, unknown):
        if progress_cb is not None:
            try:
                progress_cb(90 + int(6 * complete))
            except InterruptedError:
                return 0
        return 1

    # Wrap Polygonize and attribute updates in a single OGR transaction so SQLite/GPKG
    # does not fsync on every single feature (critical for 10,000+ slope units).
    in_tx = False
    if vec_ds.TestCapability(ogr.ODsCTransactions):
        vec_ds.StartTransaction()
        in_tx = True

    try:
        gdal.Polygonize(
            band,
            mask_ds.GetRasterBand(1),
            layer,
            0,
            callback=_gdal_progress if progress_cb is not None else None,
        )

        layer.ResetReading()
        total_feats = layer.GetFeatureCount() or 1
        for idx, feat in enumerate(layer):
            geom = feat.GetGeometryRef()
            feat.SetField(AREA_FIELD, float(geom.GetArea()) if geom is not None else 0.0)
            feat.SetField(CLASS_FIELD, _class_name(int(feat.GetField(field_name))))
            attrs = extra_attributes.get(int(feat.GetField(field_name))) if extra_attributes else None
            if attrs:
                for key in extra_fields:
                    if key in attrs:
                        val = attrs[key]
                        feat.SetField(
                            key,
                            int(val) if isinstance(val, (int, np.integer)) else float(val),
                        )
            layer.SetFeature(feat)
            if progress_cb is not None and idx % 2000 == 0:
                progress_cb(96 + int(3 * idx / total_feats))

        if in_tx:
            vec_ds.CommitTransaction()
    except Exception:
        if in_tx:
            vec_ds.RollbackTransaction()
        raise
    finally:
        vec_ds = None
        mem_ds = None
        mask_ds = None

    return output_path


def rasterize_mask(vector_path: str, reference: RasterData) -> np.ndarray:
    """Rasterize a polygon vector layer onto the reference raster grid as a boolean mask."""
    rows, cols = reference.shape
    mem_driver = gdal.GetDriverByName("MEM")
    mem_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    mem_ds.SetGeoTransform(reference.geotransform)
    if reference.projection:
        mem_ds.SetProjection(reference.projection)
    mem_ds.GetRasterBand(1).Fill(0)

    vec_ds = ogr.Open(vector_path)
    if vec_ds is None:
        raise IOError(f"Could not open vector mask: {vector_path}")
    layer = vec_ds.GetLayer(0)
    gdal.RasterizeLayer(mem_ds, [1], layer, burn_values=[1])
    mask = mem_ds.GetRasterBand(1).ReadAsArray() > 0

    vec_ds = None
    mem_ds = None
    return mask
