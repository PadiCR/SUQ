# -*- coding: utf-8 -*-
"""
Optional GRASS GIS backend for SUQ: half-basins from r.watershed.

SUQ never needs GRASS. This module is only used when the user selects one of the
"GRASS r.watershed" engines, to obtain half-basins identical to those used by
r.slopeunits.create (MFD accumulation is not implemented by the native engine yet).

GRASS runs in its own temporary project, created from the DEM file so that the region
matches the DEM grid cell by cell. Nothing is written to the user's GRASS database.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from osgeo import gdal


class GrassBackendError(RuntimeError):
    """GRASS is missing or a GRASS command failed."""


def find_grass_command() -> Optional[str]:
    """Locate the GRASS start script, first through the QGIS GRASS provider, then on PATH."""
    for module, cls in (
        ("grassprovider.grass_utils", "GrassUtils"),
        ("grassprovider.Grass7Utils", "Grass7Utils"),
        ("processing.algs.grass7.Grass7Utils", "Grass7Utils"),
    ):
        try:
            utils = getattr(__import__(module, fromlist=[cls]), cls)
            utils.grassBin()
            if utils.command:
                return utils.command
        except Exception:
            continue
    for name in ("grass", "grass85", "grass84", "grass83", "grass82", "grass80", "grass78"):
        found = shutil.which(name)
        if found:
            return found
    return None


class GrassHalfBasinProvider:
    """
    Callable ``provider(threshold_in_cells) -> half-basin array`` backed by r.watershed.

    schedule    thresholds the caller is expected to ask for, in order. On a cache miss
                the requested threshold and the next ``batch - 1`` ones are computed in a
                single GRASS start-up, which is much cheaper than one start-up each.
    sfd         True = r.watershed -s (single flow direction), False = MFD (GRASS default).
    """

    def __init__(
        self,
        dem_path: str,
        shape,
        schedule: Sequence[int],
        sfd: bool = False,
        convergence: int = 5,
        batch: int = 3,
        log: Optional[Callable[[str], None]] = None,
        is_canceled: Optional[Callable[[], bool]] = None,
    ):
        self.grass = find_grass_command()
        if self.grass is None:
            raise GrassBackendError(
                "GRASS GIS was not found. Install GRASS (it ships with the standard QGIS "
                "installers) or choose the native SUQ engine."
            )
        self.dem_path = dem_path
        self.shape = tuple(shape)
        self.schedule: List[int] = [int(t) for t in schedule]
        self.sfd = bool(sfd)
        self.convergence = int(convergence)
        self.batch = max(1, int(batch))
        self.log = log or (lambda msg: None)
        self.is_canceled = is_canceled or (lambda: False)
        self.workdir = tempfile.mkdtemp(prefix="suq_grass_")
        self.project = os.path.join(self.workdir, "project")
        self.cache: Dict[int, np.ndarray] = {}
        self._imported = False

    # ------------------------------------------------------------------ GRASS plumbing
    def _run(self, args: List[str]) -> None:
        env = os.environ.copy()
        env["GRASS_MESSAGE_FORMAT"] = "plain"
        env.pop("GISBASE", None)
        kw = {}
        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kw["startupinfo"] = si
        proc = subprocess.run(
            [self.grass] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            **kw,
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        errors = [ln.strip() for ln in out.splitlines() if "ERROR" in ln]
        if proc.returncode != 0 or errors:
            raise GrassBackendError(
                "GRASS command failed:\n" + ("\n".join(errors) if errors else out[-2000:])
            )

    def _run_batch(self, commands: List[str]) -> None:
        """Run GRASS module calls inside the temporary project through a batch job."""
        if sys.platform == "win32":
            job = os.path.join(self.workdir, "suq_job.cmd")
            lines = ["@echo off"] + commands + ["exit"]
        else:
            job = os.path.join(self.workdir, "suq_job.sh")
            lines = ["#!/bin/sh", "set -e"] + commands
        with open(job, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(job, 0o755)
        self._run([os.path.join(self.project, "PERMANENT"), "--exec", job])

    def _out_path(self, thc: int) -> str:
        return os.path.join(self.workdir, f"hb_{thc}.tif")

    # ------------------------------------------------------------------ public API
    def __call__(self, thc: int) -> np.ndarray:
        thc = int(thc)
        if thc in self.cache:
            return self.cache.pop(thc)
        if self.is_canceled():
            raise InterruptedError("Execution canceled by user.")

        todo = [thc]
        if thc in self.schedule:
            i = self.schedule.index(thc)
            todo = self.schedule[i:i + self.batch]

        commands: List[str] = []
        if not self._imported:
            # Project with the CRS of the DEM, then import and align the region to it
            self._run(["-c", self.dem_path, "-e", self.project])
            commands += [
                f'r.in.gdal -o input="{self.dem_path}" output=suq_dem --overwrite --quiet',
                "g.region raster=suq_dem",
            ]
        flags = "-abs" if self.sfd else "-ab"
        for t in todo:
            commands += [
                f"r.watershed {flags} elevation=suq_dem threshold={t} "
                f"convergence={self.convergence} hbasin=suq_hb --overwrite --quiet",
                f'r.out.gdal -c -f input=suq_hb output="{self._out_path(t)}" type=Int32 '
                f"nodata=0 createopt=COMPRESS=LZW --overwrite --quiet",
            ]
        self.log(
            "GRASS r.watershed ({}) for thresholds {} cells...".format(
                "SFD" if self.sfd else "MFD", ", ".join(str(t) for t in todo)
            )
        )
        self._run_batch(commands)
        self._imported = True

        for t in todo:
            path = self._out_path(t)
            if not os.path.exists(path):
                raise GrassBackendError(f"GRASS did not produce half-basins for threshold {t}")
            ds = gdal.Open(path, gdal.GA_ReadOnly)
            arr = ds.GetRasterBand(1).ReadAsArray().astype(np.int32)
            ds = None
            try:
                os.remove(path)
            except OSError:
                pass
            if arr.shape != self.shape:
                raise GrassBackendError(
                    f"GRASS half-basin raster is {arr.shape}, the DEM is {self.shape}"
                )
            self.cache[t] = arr
        return self.cache.pop(thc)

    def close(self) -> None:
        self.cache.clear()
        shutil.rmtree(self.workdir, ignore_errors=True)
