<p align="center"><img src="docs/suq_banner.png" alt="SUQ banner" width="300"></p>

# SUQ — Slope Units for QGIS

**SUQ** is a QGIS Processing plugin for delineating **slope units** from a digital elevation model (DEM).

Slope units are terrain units bounded by drainage and divide lines: each one is a hillslope, the area between a ridge and a stream. They are widely used as mapping units for landslide susceptibility, hazard and geomorphological studies, because they follow the shape of the landscape instead of an arbitrary grid.

SUQ creates slope units, cleans them, measures their quality, finds the best parameters for an area, and builds multi-scale nested maps. It is based on the GRASS GIS add-on [r.slopeunits](#based-on-rslopeunits), and everything runs inside QGIS, with no GRASS GIS installation needed.

## Features

- Slope unit delineation from a DEM, with the r.slopeunits parameters (flow accumulation threshold, minimum area, minimum circular variance, reduction factor).
- Removal of small units by merging them into a neighbour, keeping the original borders. Three methods are available: longest shared border, most similar aspect, or hydrologically consistent.
- Quality metrics and automatic search for the best parameters.
- Nested slope units at several scales, where every unit fits exactly inside a unit of the coarser layer.
- Optional detection of **alluvial plains** and **water bodies**, which are left out of the slope units and written as their own classes.
- Optional maximum unit area, 3D surface-normal variance (slope and aspect), and morphometric attributes.
- Raster and vector outputs. Every vector output includes the unit area (`area_m2`).

## Requirements

- QGIS 3.28 or later (QGIS 4 supported).
- Nothing else: SUQ uses NumPy, SciPy and GDAL, which come with QGIS.
- GRASS GIS is optional, as an alternative flow-routing engine.
- The DEM must be in a projected coordinate system with units in metres.

## Installation

1. Download `SUQ.zip` from the [Releases](https://github.com/PadiCR/SUQ/releases) page, or zip the `SUQ` folder of this repository.
2. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP** and select the zip file.
3. Open the **Processing Toolbox** (`Ctrl+Alt+T`); the tools are under **SUQ (Slope Units for QGIS)**.

## Typical workflow

1. **Optimize Slope Unit Parameters** on your DEM, to find the best minimum circular variance and minimum area.
2. **Create Slope Units** with those values, adding alluvial plains and water bodies if needed.
3. **Clean Slope Units** to merge the units that are too small.
4. **Slope Unit Metrics** to evaluate the result, or to compare alternatives.

Each tool explains its parameters in the help panel, and in the tooltip of each parameter.

## Tools

### 1. Create Slope Units

Divides the DEM into half-basins: the areas draining to each side of the stream segments. A half-basin is split again, with a lower flow accumulation threshold, while it is larger than the minimum area and its aspect is too variable (above the minimum circular variance).

- **Flow-routing engine:**
  - multiple flow direction (default);
  - single flow direction;
  - a fast single-direction engine for very large DEMs;
  - GRASS `r.watershed`, if GRASS is installed.
- **Optional:**
  - maximum unit area;
  - 3D surface-normal variance;
  - an alluvial plains mask, or automatic detection of plains and water bodies;
  - morphometric attributes (`cv`, `mean_elev`, `mean_slope`, `mean_asp`);
  - circular variance and area rasters.

The raw result contains small units and slivers; remove them with **Clean Slope Units**.

### 2. Clean Slope Units

Merges every unit smaller than the given size into a neighbour, one at a time, smallest first. The remaining borders stay exactly where they were. Choose the neighbour with one of three methods:

- **Quick** (default): the neighbour sharing the longest border.
- **Aspect:** the neighbour with the most similar mean aspect.
- **Hydrological:** units are merged only along the same hillslope, never across a ridge or a river. The exception is the head of a catchment, whose small units can be joined into one. A unit with no valid neighbour is kept and flagged `small_kept = 1` in the vector.

### 3. Slope Unit Metrics

Measures how well the slope units divide the terrain:

- **V:** internal aspect variance, low when each unit is homogeneous.
- **I:** autocorrelation between neighbouring units, low when neighbours differ.
- **F:** a combined objective of the two, used to compare alternatives.

### 4. Optimize Slope Unit Parameters

Creates, cleans and scores the slope units for a range of minimum circular variance and minimum area values. It keeps the combination with the best objective F and writes those slope units.

- The candidates table marks the best parameters in its `best` column.
- If you don't save the table, the best parameters are written to a text file next to the output raster.

### 5. Nested Multi-Scale Slope Units

Creates one slope unit layer for each minimum circular variance in a list, from fine to coarse. Each coarser layer is built from the units of the finer one, so its borders always coincide with borders of the layer below. A `parent_su` field links each unit to the unit containing it.

## Output classes

| Raster value | Vector `class`  | Meaning                                   |
|--------------|-----------------|-------------------------------------------|
| 1, 2, 3, …   | Slope unit      | Slope unit ID                             |
| -1           | Alluvial plain  | Plains left out of the slope units        |
| -2           | Water body      | Lakes and reservoirs detected in the DEM  |
| 0            | —               | No data                                   |

Small slope units enclosed by a plain or water body are merged into it. Clean keeps plains and water bodies unchanged, and Metrics ignores them.

## Based on r.slopeunits

SUQ is based on **r.slopeunits**, the GRASS GIS toolset for slope unit delineation and optimization developed at CNR-IRPI (Perugia, Italy):

- **Authors:** Ivan Marchesini and Massimiliano Alvioli (CNR-IRPI). The Python version is by Markus Metz and Carmen Tawalika (mundialis).
- **Manual:** [r.slopeunits (GRASS GIS add-on)](https://grass.osgeo.org/grass-stable/manuals/addons/r.slopeunits.html).
- **Source code:** [OSGeo/grass-addons](https://github.com/OSGeo/grass-addons), `src/raster/r.slopeunits`. It is released under the GNU GPL v2 or later.
- **Project page:** [geomorphology.irpi.cnr.it/tools/slope-units](https://geomorphology.irpi.cnr.it/tools/slope-units).

The delineation, metrics, optimization and nested slope units follow the method of r.slopeunits v1.0 (Alvioli et al., 2016) and v2.0 (Alvioli & Marchesini, 2025). Flow routing uses the method of GRASS `r.watershed`. If you use SUQ in your work, please cite these papers (see [References](#references)).

## References

- Alvioli, M., Marchesini, I., Reichenbach, P., Rossi, M., Ardizzone, F., Fiorucci, F., & Guzzetti, F. (2016). Automatic delineation of geomorphological slope units with r.slopeunits v1.0 and their optimization for landslide susceptibility modeling. *Geoscientific Model Development*, 9, 3975–3991. https://doi.org/10.5194/gmd-9-3975-2016
- Alvioli, M., Guzzetti, F., & Marchesini, I. (2020). Parameter-free delineation of slope units and terrain subdivision of Italy. *Geomorphology*, 358, 107124. https://doi.org/10.1016/j.geomorph.2020.107124
- Alvioli, M., & Marchesini, I. (2025). Automatic optimization and delineation of nested slope units with r.slopeunits v2.0. *Proceedings of Geomorphometry 2025*, Perugia, Italy.
- Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-Flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*, 62, 117–127. https://doi.org/10.1016/j.cageo.2013.04.024

## Author

Cristobal Padilla — cristobalpadilla@gmail.com

Bug reports and suggestions are welcome in the [issue tracker](https://github.com/PadiCR/SUQ/issues).

## License

Copyright (C) 2026 Cristobal Padilla.

SUQ is free software, released under the GNU General Public License v2 or later (see [LICENSE](LICENSE)), the same licence as r.slopeunits and GRASS GIS.
