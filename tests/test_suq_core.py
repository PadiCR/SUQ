# -*- coding: utf-8 -*-
r"""
Unit and integration tests for SUQ core engine and QGIS Processing provider.
Run with:
  "C:\OSGeo4W\bin\python-qgis-ltr.bat" -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from osgeo import gdal, ogr, osr

# Allow importing SUQ as a package when run from SUQ/
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(REPO_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from SUQ.suq_core.dem_io import RasterData, polygonize_labels, read_raster, write_raster
from SUQ.suq_core.terrain import compute_slope_aspect, detect_alluvial_plains, aspect_cover_mask
from SUQ.suq_core.hydrology import prepare_hydro_grid, extract_half_basins
from SUQ.suq_core.slope_units import (
    create_slope_units,
    clump_labels,
    CachedHalfBasins,
    enforce_nesting,
    nest_layers,
    threshold_schedule,
    build_basin_hierarchy,
    create_nested_slope_units,
)
from SUQ.suq_core.watershed import prepare_watershed_grid, watershed_half_basins
from SUQ.suq_core.grass_backend import GrassHalfBasinProvider, find_grass_command
from SUQ.suq_core.cleaning import clean_slope_units
from SUQ.suq_core.metrics import compute_segmentation_metrics, compute_f_scores
from SUQ.suq_core.optimize import optimize_slope_units


def make_synthetic_dem(rows: int = 80, cols: int = 80, cell_size: float = 25.0):
    """
    Create a synthetic mountain valley DEM (2000m x 2000m at 25m resolution)
    with a central N-S river valley, E-W side tributaries, and a flat southern plain.
    """
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # V-shaped main valley centered at col=40 sloping down toward row=rows-1
    dist_center = np.abs(x - cols * 0.5) * cell_size
    valley_slope = (rows - y) * 3.0
    ridge_height = 0.25 * dist_center + 40.0 * np.sin(y / 8.0) * (dist_center / 1000.0)
    dem = 100.0 + valley_slope + ridge_height

    # Add a flat alluvial plain at the bottom 12 rows
    dem[-12:, :] = 100.0 + np.linspace(2.0, 0.0, 12)[:, None]

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)  # WGS84 / UTM zone 32N (meters)
    gt = (500000.0, cell_size, 0.0, 4800000.0, 0.0, -cell_size)
    return RasterData(array=dem, geotransform=gt, projection=srs.ExportToWkt(), nodata=-9999.0)


class TestSUQCore(unittest.TestCase):
    def setUp(self):
        self.dem_data = make_synthetic_dem(80, 80, 25.0)
        self.valid = np.ones(self.dem_data.shape, dtype=bool)
        (
            self.slope_rad,
            self.aspect_rad,
            self.sin_a,
            self.cos_a,
            self.nx,
            self.ny,
            self.nz,
        ) = compute_slope_aspect(
            self.dem_data.array,
            self.dem_data.cell_size_x,
            self.dem_data.cell_size_y,
            valid_mask=self.valid,
        )
        self.hydro = prepare_hydro_grid(
            self.dem_data.array,
            self.valid,
            self.dem_data.cell_size_x,
            self.dem_data.cell_size_y,
        )

    def test_hydrology_and_half_basins(self):
        # Flow accumulation should be >= 1 everywhere and max at valley outlet
        self.assertGreaterEqual(self.hydro.acc.min(), 1.0)
        self.assertGreater(self.hydro.acc.max(), 500.0)

        hb = extract_half_basins(self.hydro, thresh_cells=50.0)
        self.assertEqual(hb.shape, self.dem_data.shape)
        self.assertGreaterEqual(hb.max(), 2)
        # Cells that leave the DEM without reaching a channel have no half-basin (0),
        # as in r.watershed (here: the flat plain along the southern edge).
        self.assertGreater(np.count_nonzero(hb > 0), 0.6 * hb.size)
        self.assertTrue(np.all(hb[:60, :] > 0))

        # Every half-basin must be one contiguous patch (8-connected): the left/right
        # bank rule must not flip sides along a bending channel.
        from scipy import ndimage

        fragments = sum(
            ndimage.label(hb == v, structure=np.ones((3, 3), dtype=bool))[1]
            for v in np.unique(hb[hb > 0])
        )
        self.assertLessEqual(fragments, int(1.2 * hb.max()) + 1)

    def test_two_sided_valley_is_split_along_the_thalweg(self):
        # Straight N-S valley whose channel head lies well inside the DEM: the area above
        # the head must still be split into a west and an east side (thalweg extension).
        rows, cols = 60, 61
        y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
        dem = (rows - y) * 2.0 + np.abs(x - 30) * 5.0
        valid = np.ones(dem.shape, dtype=bool)
        hydro = prepare_hydro_grid(dem, valid, 10.0, 10.0)
        hb = extract_half_basins(hydro, thresh_cells=600.0)
        west = hb[5:50, 5:25]
        east = hb[5:50, 36:56]
        self.assertEqual(np.unique(west).size, 1)
        self.assertEqual(np.unique(east).size, 1)
        self.assertGreater(west[0, 0], 0)
        self.assertGreater(east[0, 0], 0)
        self.assertNotEqual(west[0, 0], east[0, 0])

    def test_schedule_clump_and_cover_mask(self):
        # Same sequence as r.slopeunits.create: thc = int(thc - thc / rf) while thc >= rf
        self.assertEqual(threshold_schedule(1000.0, 10.0, rf=2, maxiteration=20), [100, 50, 25, 12, 6, 3])

        values = np.array([[5, 5, 7], [7, 5, 7], [5, 0, 7]])
        labels, n = clump_labels(values, values > 0)
        self.assertEqual(n, 4)
        self.assertEqual(labels[2, 1], 0)
        self.assertEqual(labels[0, 0], labels[1, 1])
        self.assertNotEqual(labels[1, 0], labels[0, 2])   # same value, not 4-connected
        self.assertNotEqual(labels[2, 0], labels[0, 0])

        valid = np.ones((6, 6), dtype=bool)
        valid[0, 0] = False
        cover = aspect_cover_mask(valid)
        self.assertFalse(cover[0, :].any())
        self.assertFalse(cover[1, 1])
        self.assertTrue(cover[3, 3])

    def test_create_with_external_half_basins(self):
        # The loop accepts half-basins from any engine (e.g. GRASS r.watershed)
        calls = []

        def provider(thc):
            calls.append(thc)
            hb = np.ones(self.dem_data.shape, dtype=np.int32)
            hb[:, 40:] = 2
            return hb

        res = create_slope_units(
            hydro=None,
            valid_mask=self.valid,
            half_basin_provider=provider,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=25000.0,
            cvmin=0.99,
            rf=2,
            maxiteration=15,
        )
        self.assertEqual(calls, [640])
        self.assertEqual(res.n_units, 2)
        self.assertTrue(np.all(res.labels > 0))

    def test_auto_plains_detection(self):
        plains = detect_alluvial_plains(
            self.slope_rad,
            self.dem_data.cell_area,
            slope_thresh_deg=2.0,
            min_plain_area_m2=50000.0,
            valid_mask=self.valid,
        )
        # Southern flat rows should be detected as plains
        self.assertTrue(np.any(plains[-10:, :]))
        # Steep upper ridges should NOT be plains
        self.assertFalse(np.any(plains[:20, :10]))

    def test_create_slope_units_and_areamax(self):
        res_classic = create_slope_units(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=25000.0,
            cvmin=0.25,
            rf=2,
            maxiteration=15,
        )
        self.assertGreater(res_classic.n_units, 1)
        self.assertTrue(np.all(res_classic.labels > 0))
        # Like r.clump: IDs are 1..K and every unit is a single 4-connected patch
        self.assertEqual(res_classic.n_units, int(res_classic.labels.max()))
        _, n_patches = clump_labels(res_classic.labels, res_classic.labels > 0)
        self.assertEqual(n_patches, res_classic.n_units)

        # Test areamax cap forces finer subdivision
        res_capped = create_slope_units(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=25000.0,
            cvmin=0.5,
            rf=2,
            maxiteration=15,
            areamax=150000.0,
        )
        self.assertGreaterEqual(res_capped.n_units, res_classic.n_units)

        # Test 3D normal variance
        res_3d = create_slope_units(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=25000.0,
            cvmin=0.25,
            rf=2,
            maxiteration=15,
            use_3d_normal=True,
        )
        self.assertGreater(res_3d.n_units, 1)

    def test_clean_and_metrics(self):
        res = create_slope_units(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=10000.0,
            cvmin=0.15,
            rf=2,
            maxiteration=15,
        )
        cleaned = clean_slope_units(
            res.labels,
            cell_area=self.dem_data.cell_area,
            cleansize=25000.0,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            method="detailed",
        )
        self.assertGreater(cleaned.max(), 0)
        self.assertLessEqual(cleaned.max(), res.n_units)

        m = compute_segmentation_metrics(cleaned, self.sin_a, self.cos_a, self.dem_data.cell_area)
        self.assertGreaterEqual(m.v_fin, 0.0)
        self.assertLessEqual(m.v_fin, 1.0)
        self.assertGreaterEqual(m.i_fin, -1.0)
        self.assertLessEqual(m.i_fin, 1.0)
        self.assertGreater(m.n_edges, 0)

        f = compute_f_scores([0.1, 0.3, 0.5], [0.2, 0.4, 0.6])
        self.assertAlmostEqual(float(f[0]), 2.0)
        self.assertAlmostEqual(float(f[2]), 0.0)

    def test_cached_half_basins(self):
        calls = []

        def provider(thc):
            calls.append(thc)
            return np.full((4, 4), thc, dtype=np.int32)

        cache = CachedHalfBasins(provider)
        self.assertEqual(int(cache(7)[0, 0]), 7)
        cache(9)
        cache(7)
        self.assertEqual(calls, [7, 9])
        # a tiny budget keeps only the most recent raster
        small = CachedHalfBasins(provider, max_bytes=1)
        small(1), small(2), small(1)
        self.assertEqual(calls, [7, 9, 1, 2, 1])

    def test_enforce_nesting(self):
        fine = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 0]])
        # coarse unit boundary cuts through fine unit 2 (one cell belongs to coarse 20)
        coarse = np.array([[10, 10, 10, 10], [10, 10, 10, 20], [20, 20, 20, 20], [20, 20, 20, 0]])
        nested, parent_map, moved = enforce_nesting(fine, coarse)
        self.assertEqual(nested[3, 3], 0)
        self.assertAlmostEqual(moved, 1.0 / 15.0)
        self.assertEqual(set(parent_map), {1, 2, 3, 4})
        for fid, pid in parent_map.items():
            self.assertTrue(np.all(nested[fine == fid] == pid))
        self.assertEqual(parent_map[1], parent_map[2])
        self.assertEqual(parent_map[3], parent_map[4])
        self.assertNotEqual(parent_map[1], parent_map[3])

        # already nested layers are not changed
        same, _, moved = enforce_nesting(fine, np.where(fine > 0, (fine + 1) // 2, 0))
        self.assertEqual(moved, 0.0)
        self.assertEqual(np.unique(same[same > 0]).size, 2)

    def test_nested_layers_from_exact_loop(self):
        grid = prepare_watershed_grid(
            self.dem_data.array, self.valid, self.dem_data.cell_size_x, self.dem_data.cell_size_y
        )
        provider = CachedHalfBasins(lambda thc: watershed_half_basins(grid, thc))
        raw = [
            create_slope_units(
                hydro=None,
                valid_mask=self.valid,
                half_basin_provider=provider,
                sin_a=self.sin_a,
                cos_a=self.cos_a,
                nx=self.nx,
                ny=self.ny,
                nz=self.nz,
                cell_area=self.dem_data.cell_area,
                thresh=400000.0,
                areamin=25000.0,
                cvmin=cv,
                rf=2,
                maxiteration=15,
            ).labels
            for cv in (0.1, 0.3, 0.6)
        ]
        layers = nest_layers(raw)
        self.assertEqual(len(layers), 3)
        self.assertTrue(np.array_equal(layers[0][0], raw[0]))      # base layer untouched
        self.assertEqual(layers[-1][1], {})
        for (fine, parent_map, _), (coarse, _, _) in zip(layers[:-1], layers[1:]):
            self.assertGreaterEqual(fine.max(), coarse.max())
            self.assertTrue(np.all(coarse[self.valid] > 0))
            for fid, pid in parent_map.items():
                self.assertTrue(np.all(coarse[fine == fid] == pid))

    def test_optimize_with_r_watershed_port(self):
        grid = prepare_watershed_grid(
            self.dem_data.array, self.valid, self.dem_data.cell_size_x, self.dem_data.cell_size_y
        )
        opt = optimize_slope_units(
            hydro=None,
            valid_mask=self.valid,
            half_basin_provider=lambda thc: watershed_half_basins(grid, thc),
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            cvmin_range=(0.1, 0.4),
            areamin_range=(20000.0, 80000.0),
            epsilon_cv=0.1,
            epsilon_area=30000.0,
            rf=2,
            maxiteration=12,
            cleansize=10000.0,
        )
        self.assertGreaterEqual(len(opt.candidates), 9)
        self.assertEqual(opt.candidates[0].f_score, opt.best_f)
        self.assertTrue(np.all(opt.best_labels[self.valid] > 0))
        # the reported optimum is reproducible with a plain delineation
        again = create_slope_units(
            hydro=None,
            valid_mask=self.valid,
            half_basin_provider=lambda thc: watershed_half_basins(grid, thc),
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            areamin=opt.best_areamin,
            cvmin=opt.best_cvmin,
            rf=2,
            maxiteration=12,
        )
        self.assertTrue(np.array_equal(again.labels, opt.best_labels))

    def test_strict_nested_hierarchy(self):
        hierarchy = build_basin_hierarchy(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            min_stop_area=25000.0,
            rf=2,
            maxiteration=15,
        )
        layers = create_nested_slope_units(
            hierarchy, cvmin_levels=[0.1, 0.3, 0.6], areamin=25000.0
        )
        self.assertEqual(len(layers), 3)
        cv0, lbl0, parent0 = layers[0]
        cv1, lbl1, _ = layers[1]
        # Number of units must decrease (or stay equal) as cvmin increases
        self.assertGreaterEqual(lbl0.max(), lbl1.max())

        # Verify every pixel in Layer 0 maps to its exact parent_su in Layer 1!
        for fine_id, parent_id in parent0.items():
            mask = lbl0 == fine_id
            coarser_ids = np.unique(lbl1[mask])
            self.assertEqual(len(coarser_ids), 1)
            self.assertEqual(int(coarser_ids[0]), parent_id)

    def test_optimize_and_gdal_io(self):
        opt = optimize_slope_units(
            hydro=self.hydro,
            sin_a=self.sin_a,
            cos_a=self.cos_a,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            cell_area=self.dem_data.cell_area,
            thresh=400000.0,
            cvmin_range=(0.1, 0.4),
            areamin_range=(20000.0, 80000.0),
            epsilon_cv=0.05,
            epsilon_area=15000.0,
            rf=2,
            maxiteration=12,
            cleansize=10000.0,
        )
        self.assertGreater(len(opt.candidates), 0)
        self.assertGreater(opt.best_cleaned_labels.max(), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = os.path.join(tmpdir, "su.tif")
            gpkg_path = os.path.join(tmpdir, "su.gpkg")
            write_raster(tif_path, opt.best_cleaned_labels, self.dem_data)
            read_back = read_raster(tif_path, dtype=np.int32)
            self.assertTrue(np.array_equal(read_back.array, opt.best_cleaned_labels))
            polygonize_labels(opt.best_cleaned_labels, self.dem_data, gpkg_path)
            self.assertTrue(os.path.exists(gpkg_path))

            # every polygon carries its own area, and the areas add up to the labelled area
            from osgeo import ogr

            ds = ogr.Open(gpkg_path)
            areas = [f.GetField("area_m2") for f in ds.GetLayer(0)]
            ds = None
            self.assertTrue(all(a is not None and a > 0 for a in areas))
            self.assertAlmostEqual(
                sum(areas),
                float(np.count_nonzero(opt.best_cleaned_labels)) * self.dem_data.cell_area,
                places=3,
            )


def make_rough_dem(rows: int = 70, cols: int = 90, cell_size: float = 10.0):
    """Synthetic terrain without flat areas or equal neighbours (two valleys plus relief)."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    dem = (
        500.0
        + (rows - y) * 1.7
        + 60.0 * np.abs(np.sin(x / 14.0))
        + 9.0 * np.sin(x / 3.1 + y / 4.3)
        + 7.0 * np.cos(y / 2.7 - x / 6.1)
        + 0.013 * x + 0.007 * y
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    gt = (500000.0, cell_size, 0.0, 4800000.0, 0.0, -cell_size)
    return RasterData(array=dem, geotransform=gt, projection=srs.ExportToWkt(), nodata=-9999.0)


class TestWatershedPort(unittest.TestCase):
    """Native port of GRASS r.watershed (A* search, SFD/MFD accumulation, half-basins)."""

    def setUp(self):
        self.dem_data = make_rough_dem()
        self.valid = np.ones(self.dem_data.shape, dtype=bool)
        self.valid[:6, :9] = False          # a nodata corner
        self.dem = np.where(self.valid, self.dem_data.array, np.nan)

    def _grid(self, mfd):
        return prepare_watershed_grid(
            self.dem, self.valid, self.dem_data.cell_size_x, self.dem_data.cell_size_y, mfd=mfd
        )

    def test_accumulation_and_half_basins(self):
        for mfd in (False, True):
            grid = self._grid(mfd)
            acc = grid.accumulation
            self.assertTrue(np.all(acc[self.valid] >= 1.0 - 1e-9))
            self.assertTrue(np.all(acc[~self.valid] == 0.0))
            self.assertGreater(acc.max(), 200.0)
            # every cell drains somewhere: direction code 1..8, negative on outlets
            drn = grid.drainage
            self.assertTrue(np.all(np.abs(drn[self.valid]) >= 1))
            self.assertTrue(np.all(np.abs(drn[self.valid]) <= 8))

            hb = watershed_half_basins(grid, 150)
            self.assertEqual(hb.shape, self.dem.shape)
            self.assertTrue(np.all(hb[~self.valid] == 0))
            self.assertGreater(np.count_nonzero(hb), 0.7 * self.valid.sum())
            ids = np.unique(hb[hb > 0])
            self.assertGreaterEqual(ids.size, 4)
            # a lower threshold gives more basins
            self.assertGreater(np.unique(watershed_half_basins(grid, 40)).size, ids.size)

    def test_mfd_spreads_flow(self):
        # With multiple flow directions fewer cells keep the minimum accumulation of 1
        acc_sfd = self._grid(False).accumulation
        acc_mfd = self._grid(True).accumulation
        ones_sfd = np.count_nonzero(np.isclose(acc_sfd[self.valid], 1.0))
        ones_mfd = np.count_nonzero(np.isclose(acc_mfd[self.valid], 1.0))
        self.assertLessEqual(ones_mfd, ones_sfd)
        self.assertFalse(np.allclose(acc_sfd, acc_mfd))

    def test_flat_areas_are_drained(self):
        # Terraced DEM with a lake: every flat cell must still get a drainage direction
        dem = np.round(self.dem_data.array / 10.0) * 10.0
        dem[30:50, 20:60] = np.minimum(dem[30:50, 20:60], 560.0)
        dem = np.where(self.valid, dem, np.nan)
        for flats in (True, False):
            grid = prepare_watershed_grid(
                dem, self.valid, 10.0, 10.0, mfd=True, beautify_flats=flats
            )
            drn = grid.drainage
            self.assertTrue(np.all(np.abs(drn[self.valid]) >= 1))
            self.assertAlmostEqual(
                float(grid.accumulation[self.valid].min()), 1.0, places=6
            )
            hb = watershed_half_basins(grid, 150)
            self.assertGreater(np.count_nonzero(hb), 0)
            self.assertTrue(np.all(hb[~self.valid] == 0))

    @unittest.skipUnless(find_grass_command(), "GRASS GIS is not installed")
    def test_identical_to_grass_r_watershed(self):
        self._compare_with_grass(self.dem_data.array)

    @unittest.skipUnless(find_grass_command(), "GRASS GIS is not installed")
    def test_identical_to_grass_on_flat_areas(self):
        # elevation in 5 m steps plus a lake: exercises the flat-area beautification (-b)
        dem = np.round(self.dem_data.array / 5.0) * 5.0
        dem[30:50, 20:60] = np.minimum(dem[30:50, 20:60], 560.0)
        self._compare_with_grass(dem)

    def _compare_with_grass(self, dem_array):
        thresholds = [400, 150, 60]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            dem_path = os.path.join(tmpdir, "dem.tif")
            write_raster(
                dem_path,
                np.where(self.valid, dem_array, -9999.0),
                self.dem_data,
                nodata=-9999.0,
                gdal_dtype=gdal.GDT_Float32,
            )
            dem32 = read_raster(dem_path).array      # what GRASS reads (float32)
            for mfd in (True, False):
                grid = prepare_watershed_grid(
                    dem32, self.valid, self.dem_data.cell_size_x, self.dem_data.cell_size_y, mfd=mfd
                )
                grass = GrassHalfBasinProvider(
                    dem_path, self.dem.shape, thresholds, sfd=not mfd, batch=len(thresholds)
                )
                try:
                    for t in thresholds:
                        np.testing.assert_array_equal(watershed_half_basins(grid, t), grass(t))
                finally:
                    grass.close()


def make_valley_dem(rows: int = 60, cols: int = 41, ridge: bool = False):
    """V-shaped valley along the central column draining south (or an inverted V: a ridge)."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    side = np.abs(x - cols // 2) * 2.0
    dem = 500.0 + (rows - y) * 1.0 + (-side if ridge else side)
    dem += 0.01 * np.sin(x * 1.7 + y * 0.9)  # break ties
    return dem


class TestHydrologicalCleaning(unittest.TestCase):
    """Cleaning that never merges across ridges or rivers."""

    def _drainage(self, dem):
        valid = np.ones(dem.shape, dtype=bool)
        return prepare_watershed_grid(dem, valid, 10.0, 10.0, mfd=True).drainage

    @staticmethod
    def _halves(shape, top=0):
        rows, cols = shape
        lab = np.zeros(shape, dtype=np.int32)
        lab[top:, : cols // 2] = 1
        lab[top:, cols // 2 :] = 2
        if top:
            lab[:top, :] = 3
        return lab

    def test_border_classes(self):
        from SUQ.suq_core.cleaning import classify_borders, BORDER_RIDGE, BORDER_VALLEY, BORDER_SLOPE

        lab = self._halves((60, 41), top=20)
        valley = classify_borders(lab, self._drainage(make_valley_dem()))
        self.assertEqual(valley[(1, 2)][0], BORDER_VALLEY)
        self.assertEqual(valley[(1, 3)][0], BORDER_SLOPE)   # upper unit drains into lower one
        ridge = classify_borders(lab, self._drainage(make_valley_dem(ridge=True)))
        self.assertEqual(ridge[(1, 2)][0], BORDER_RIDGE)

    def test_head_of_catchment_halves_are_merged(self):
        dem = make_valley_dem()
        lab = self._halves(dem.shape)
        half = 1260 * 100.0  # each half is 60 x 20-21 cells of 100 m2
        out, rep = clean_slope_units(
            lab, 100.0, 1.5 * half, method="hydrological",
            drainage=self._drainage(dem), return_report=True,
        )
        self.assertEqual(int(out.max()), 1)
        self.assertEqual(rep.n_kept_small, 0)

    def test_never_across_a_ridge(self):
        dem = make_valley_dem(ridge=True)
        lab = self._halves(dem.shape)
        out, rep = clean_slope_units(
            lab, 100.0, 1.5 * 1260 * 100.0, method="hydrological",
            drainage=self._drainage(dem), return_report=True,
        )
        self.assertEqual(int(out.max()), 2)
        self.assertEqual(rep.n_kept_small, 2)
        # the non-hydrological methods do merge them
        self.assertEqual(int(clean_slope_units(lab, 100.0, 1.5 * 1260 * 100.0, method="quick").max()), 1)

    def test_never_across_a_river(self):
        # the two banks of the lower valley receive the water of the upper unit 3
        dem = make_valley_dem()
        lab = self._halves(dem.shape, top=20)
        out = clean_slope_units(
            lab, 100.0, 500 * 100.0, method="hydrological", drainage=self._drainage(dem)
        )
        a, b, c = out[40, 5], out[40, 35], out[5, 20]
        # the banks may only end up together if the whole valley (a closed catchment) is one unit
        self.assertTrue(a != b or a == c)

    def test_small_units_merge_one_at_a_time(self):
        # a chain of small units must not be glued into one block
        lab = np.zeros((10, 40), dtype=np.int32)
        lab[:, :10] = 1          # big
        lab[:, 10:13] = 2        # small
        lab[:, 13:16] = 3        # small
        lab[:, 16:] = 4          # big
        out = clean_slope_units(lab, 1.0, 50.0, method="quick")
        self.assertEqual(int(out.max()), 2)
        self.assertEqual(out[0, 11], out[0, 0])
        self.assertEqual(out[0, 14], out[0, 39])


class TestPlainsAndWaterBodies(unittest.TestCase):
    """Plains (-1) and water bodies (-2) in the outputs, islands merged into them."""

    def test_water_bodies(self):
        from SUQ.suq_core.terrain import detect_water_bodies

        y, x = np.mgrid[0:60, 0:60].astype(np.float64)
        dem = 500.0 + 0.8 * x + 0.5 * y + 0.03 * np.sin(x * 1.3 + y)
        dem[20:40, 15:45] = 510.0                     # flat lake surface, 600 cells
        valid = np.ones(dem.shape, dtype=bool)
        water = detect_water_bodies(dem, 100.0, 5000.0, 0.01, valid)
        self.assertTrue(np.all(water[21:39, 16:44]))
        self.assertLess(water.sum(), 700)
        self.assertFalse(np.any(water[:10, :]))

    def test_plains_edge_slope(self):
        # valley: flat floor (~1°), gentle sides (~6°), steep slopes (~25°), 10 m cells
        x = np.abs(np.arange(80) - 40.0)
        prof = np.where(x <= 8, x * 0.17, np.where(x <= 16, 1.4 + (x - 8) * 1.05, 9.8 + (x - 16) * 4.7))
        dem = np.tile(prof, (60, 1)) + np.arange(60)[:, None] * 0.1
        valid = np.ones(dem.shape, dtype=bool)
        sl = compute_slope_aspect(dem, 10.0, 10.0, valid)[0]
        core = detect_alluvial_plains(sl, 100.0, 5.0, 10000.0, valid)
        wide = detect_alluvial_plains(sl, 100.0, 5.0, 10000.0, valid, edge_slope_deg=8.0)
        self.assertTrue(core[30, 40] and wide[30, 40])
        self.assertFalse(core[30, 40 + 12])
        self.assertTrue(wide[30, 40 + 12])
        self.assertFalse(wide[30, 40 + 30])

    def test_islands_merged_and_classes(self):
        from SUQ.suq_core.cleaning import absorb_enclosed_units, assign_classes

        lab = np.zeros((20, 20), dtype=np.int32)
        lab[:, :8] = 1
        lab[:, 12:] = 3
        plains = np.zeros(lab.shape, dtype=bool)
        plains[:, 8:12] = True
        plains[5:15, 4:8] = True          # plain reaching into unit 1 ...
        lab[plains] = 0
        lab[9:11, 9:11] = 2               # ... and a small unit inside the plain
        plains[9:11, 9:11] = False
        valid = np.ones(lab.shape, dtype=bool)
        out_lab, absorbed = absorb_enclosed_units(lab, plains, min_cells=10)
        self.assertTrue(np.all(absorbed[9:11, 9:11]))
        self.assertEqual(int(absorbed.sum()), 4)
        out = assign_classes(out_lab, valid, plains, None, absorbed)
        self.assertTrue(np.all(out[9:11, 9:11] == -1))
        self.assertTrue(np.all(out[plains] == -1))
        self.assertEqual(sorted(np.unique(out[out > 0]).tolist()), [1, 2])   # renumbered

    def test_hilltop_plains_dropped(self):
        from SUQ.suq_core.terrain import filter_plains_by_position

        y, x = np.mgrid[0:100, 0:100]
        dem = np.abs(x - 50) * 2.0 + 100.0          # V-shaped valley along x = 50
        dem = np.maximum(dem, 0)
        dem[10:20, 5:15] = dem[10:20, 5:15].max()     # flat terrace high on the side
        plains = np.zeros(dem.shape, dtype=bool)
        plains[:, 48:53] = True                     # valley floor
        plains[10:20, 5:15] = True                  # high flat patch
        out = filter_plains_by_position(plains, dem, 1.0, 0.3, 40.0)
        self.assertTrue(np.all(out[:, 48:53]))
        self.assertFalse(np.any(out[10:20, 5:15]))
        self.assertTrue(np.array_equal(filter_plains_by_position(plains, dem, 1.0, 1.0, 40.0), plains))

    def test_plains_inside_water_become_water(self):
        from SUQ.suq_core.cleaning import assign_classes

        lab = np.zeros((20, 20), dtype=np.int32)
        lab[:, :6] = 1
        water = np.zeros(lab.shape, dtype=bool)
        water[2:18, 8:18] = True
        plains = np.zeros(lab.shape, dtype=bool)
        plains[9:11, 12:14] = True        # plain island inside the lake -> water
        plains[:, 6:8] = True             # shore plain next to unit 1 -> stays plain
        water[plains] = False
        out = assign_classes(lab, np.ones(lab.shape, dtype=bool), plains, water)
        self.assertTrue(np.all(out[9:11, 12:14] == -2))
        self.assertTrue(np.all(out[:, 6:8] == -1))

    def test_classes_in_vector_clean_and_metrics(self):
        dem_data = make_synthetic_dem(40, 40, 25.0)
        lab = np.zeros((40, 40), dtype=np.int32)
        lab[:, :18] = 1
        lab[:, 22:] = 2
        lab[:20, :18] = 3
        lab[:, 18:22] = -1
        lab[0:2, 0:2] = 4                                  # tiny unit to clean
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            gpkg = os.path.join(tmpdir, "c.gpkg")
            polygonize_labels(lab, dem_data, gpkg)
            ds = ogr.Open(gpkg)
            classes = {f.GetField("su_id"): f.GetField("class") for f in ds.GetLayer(0)}
            ds = None
        self.assertEqual(classes[-1], "Alluvial plain")
        self.assertEqual(classes[1], "Slope unit")
        cleaned = clean_slope_units(lab, 625.0, 10000.0, method="quick")
        self.assertTrue(np.all(cleaned[lab == -1] == 0))     # left out of the cleaning
        _, _, _, _, s, c, *_ = compute_slope_aspect(dem_data.array, 25.0, 25.0)
        m1 = compute_segmentation_metrics(lab, s, c, 625.0)
        m0 = compute_segmentation_metrics(np.where(lab > 0, lab, 0), s, c, 625.0)
        self.assertAlmostEqual(m1.v_fin, m0.v_fin)
        self.assertAlmostEqual(m1.i_fin, m0.i_fin)


class TestQGISProviderIntegration(unittest.TestCase):
    """Verify SUQPlugin & all 5 Processing algorithms load and run inside QGIS."""

    def test_qgis_processing_provider(self):
        from qgis.core import QgsApplication, QgsProcessingFeedback
        from SUQ.suq_plugin import SUQPlugin

        qgs = QgsApplication([], False)
        qgs.initQgis()
        try:
            self._run_provider_checks()
        finally:
            qgs.exitQgis()
            # exitQgis() unregisters the GDAL drivers; restore them for the other tests
            gdal.AllRegister()
            ogr.RegisterAll()

    def _run_provider_checks(self):
        from qgis.core import QgsApplication, QgsProcessingFeedback
        from SUQ.suq_plugin import SUQPlugin

        if True:
            plugin = SUQPlugin()
            plugin.initProcessing()

            registry = QgsApplication.processingRegistry()
            provider = registry.providerById("suq")
            self.assertIsNotNone(provider)
            alg_ids = {alg.id() for alg in provider.algorithms()}
            self.assertEqual(
                alg_ids,
                {"suq:create", "suq:clean", "suq:metrics", "suq:optimize", "suq:nested"},
            )

            # Execute suq:create end-to-end via Processing
            dem_data = make_synthetic_dem(60, 60, 25.0)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
                dem_path = os.path.join(tmpdir, "dem.tif")
                out_tif = os.path.join(tmpdir, "out_su.tif")
                out_gpkg = os.path.join(tmpdir, "out_su.gpkg")
                write_raster(dem_path, dem_data.array, dem_data, nodata=-9999.0, gdal_dtype=gdal.GDT_Float32)

                alg = registry.algorithmById("suq:create")
                feedback = QgsProcessingFeedback()
                from qgis.core import QgsProcessingContext

                context = QgsProcessingContext()
                params = {
                    "DEM": dem_path,
                    "THRESH": 300000.0,
                    "AREAMIN": 20000.0,
                    "CVMIN": 0.25,
                    "RF": 2,
                    "MAXITERATION": 12,
                    "OUTPUT_RASTER": out_tif,
                    "OUTPUT_VECTOR": out_gpkg,
                }
                res, ok = alg.run(params, context, feedback)
                self.assertTrue(ok)
                self.assertTrue(os.path.exists(out_tif))
                self.assertTrue(os.path.exists(out_gpkg))

                # the three tools with a cleaning step default to Quick
                for alg_id, pname in (("suq:clean", "METHOD"), ("suq:optimize", "CLEAN_METHOD"),
                                      ("suq:nested", "CLEAN_METHOD")):
                    pdef = registry.algorithmById(alg_id).parameterDefinition(pname)
                    self.assertIsNotNone(pdef, (alg_id, pname))
                    self.assertEqual(pdef.options()[int(pdef.defaultValue())], "Quick (longest shared border)")

                # suq:create with automatic plains and water bodies: classes -1 / -2
                lake_dem = dem_data.array.copy()
                lake_dem[25:35, 25:35] = float(np.nanmin(lake_dem[25:35, 25:35]))
                lake_path = os.path.join(tmpdir, "lake.tif")
                write_raster(lake_path, lake_dem, dem_data, nodata=-9999.0, gdal_dtype=gdal.GDT_Float32)
                cls_tif = os.path.join(tmpdir, "cls.tif")
                cls_gpkg = os.path.join(tmpdir, "cls.gpkg")
                res, ok = registry.algorithmById("suq:create").run(
                    dict(params, DEM=lake_path, WATER_BODIES=True, WATER_TOLERANCE=0.0,
                         WATER_MIN_AREA=20000.0, OUTPUT_RASTER=cls_tif, OUTPUT_VECTOR=cls_gpkg),
                    QgsProcessingContext(), QgsProcessingFeedback(),
                )
                self.assertTrue(ok)
                cls = read_raster(cls_tif, dtype=np.int32).array
                self.assertGreater(int(np.count_nonzero(cls == -2)), 90)
                ds = ogr.Open(cls_gpkg)
                names = {f.GetField("class") for f in ds.GetLayer(0)}
                ds = None
                self.assertIn("Water body", names)

                # suq:clean keeps the water body as it is
                cls_clean = os.path.join(tmpdir, "cls_clean.tif")
                res, ok = registry.algorithmById("suq:clean").run(
                    {"INPUT_SU": cls_tif, "DEM": lake_path, "CLEANSIZE": 60000.0,
                     "OUTPUT_RASTER": cls_clean},
                    QgsProcessingContext(), QgsProcessingFeedback(),
                )
                self.assertTrue(ok)
                np.testing.assert_array_equal(
                    read_raster(cls_clean, dtype=np.int32).array == -2, cls == -2
                )

                # suq:clean, hydrological method, on the units just created
                clean_tif = os.path.join(tmpdir, "clean.tif")
                clean_gpkg = os.path.join(tmpdir, "clean.gpkg")
                alg = registry.algorithmById("suq:clean")
                res, ok = alg.run(
                    {
                        "INPUT_SU": out_tif,
                        "DEM": dem_path,
                        "CLEANSIZE": 60000.0,
                        "METHOD": 0,
                        "OUTPUT_RASTER": clean_tif,
                        "OUTPUT_VECTOR": clean_gpkg,
                    },
                    QgsProcessingContext(),
                    QgsProcessingFeedback(),
                )
                self.assertTrue(ok)

                # suq:nested with hydrological cleaning of every layer
                nest_dir = os.path.join(tmpdir, "nested")
                res, ok = registry.algorithmById("suq:nested").run(
                    {
                        "DEM": dem_path,
                        "THRESH": 300000.0,
                        "AREAMIN": 20000.0,
                        "CVMIN_LEVELS": "0.2, 0.4",
                        "CLEANSIZE": 40000.0,
                        "CLEAN_METHOD": 0,
                        "OUTPUT_DIR": nest_dir,
                    },
                    QgsProcessingContext(),
                    QgsProcessingFeedback(),
                )
                self.assertTrue(ok)
                self.assertTrue(any(f.endswith(".gpkg") for f in os.listdir(nest_dir)))

                # suq:optimize: best pair marked in the CSV, or a text file when no CSV
                opt = registry.algorithmById("suq:optimize")
                common = {
                    "DEM": dem_path, "THRESH": 300000.0, "CLEANSIZE": 20000.0,
                    "CVMIN_MIN": 0.1, "CVMIN_MAX": 0.3, "AREAMIN_MIN": 20000.0,
                    "AREAMIN_MAX": 40000.0, "EPSILON_CV": 0.2, "EPSILON_AREA": 20000.0,
                }
                csv_path = os.path.join(tmpdir, "cand.csv")
                res, ok = opt.run(
                    dict(common, OUTPUT_RASTER=os.path.join(tmpdir, "opt1.tif"), OUTPUT_CSV=csv_path),
                    QgsProcessingContext(), QgsProcessingFeedback(),
                )
                self.assertTrue(ok)
                import csv as _csv
                with open(csv_path, encoding="utf-8") as fh:
                    rows = list(_csv.DictReader(fh))
                self.assertEqual(rows[0]["best"], "1")
                self.assertEqual(sum(r["best"] == "1" for r in rows), 1)
                self.assertAlmostEqual(float(rows[0]["cvmin"]), res["BEST_CVMIN"])
                self.assertAlmostEqual(float(rows[0]["areamin"]), res["BEST_AREAMIN"])

                res, ok = opt.run(
                    dict(common, OUTPUT_RASTER=os.path.join(tmpdir, "opt2.tif")),
                    QgsProcessingContext(), QgsProcessingFeedback(),
                )
                self.assertTrue(ok)
                txt = os.path.join(tmpdir, "opt2_best_parameters.txt")
                self.assertTrue(os.path.exists(txt))
                with open(txt, encoding="utf-8") as fh:
                    content = fh.read()
                self.assertIn(f"cvmin   = {res['BEST_CVMIN']}", content)
                ds = ogr.Open(clean_gpkg)
                lyr = ds.GetLayer(0)
                names = [lyr.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(lyr.GetLayerDefn().GetFieldCount())]
                self.assertIn("small_kept", names)
                self.assertIn("area_m2", names)
                n_feat = lyr.GetFeatureCount()
                lyr = None
                ds = None
                self.assertGreater(n_feat, 0)

            plugin.unload()


if __name__ == "__main__":
    unittest.main()
