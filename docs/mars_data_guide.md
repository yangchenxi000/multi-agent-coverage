# Mars DEM Data Guide

## 1. Required File

- Filename: `Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`
- Place at repository root:
  - `multi-agent-coverage/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`

Mars-related configs already point to this relative path.

## 2. Download

Official sources:

- USGS Astropedia page:
  - `https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m`
- Direct GeoTIFF mirror:
  - `https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`

PowerShell example:

```powershell
Invoke-WebRequest `
  -Uri "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif" `
  -OutFile "Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
```

## 3. How This Project Processes DEM

Implemented in `src/mars_scene.py` (`build_mars_scene_from_dem`):

1. Read global DEM and select a crop (`_pick_auto_crop`) or use fixed crop coordinates from config.
2. Resample crop to workspace resolution (`scipy.ndimage.zoom`) and smooth with Gaussian filter.
3. Compute normalized elevation, slope, and roughness.
4. Build obstacle score from slope/roughness and threshold by percentile (`slope_percentile`).
5. Morphology cleanup (optional opening, small-component removal, dilation).
6. Build terrain-derived resource density prior, blend hotspots, and suppress obstacle regions.
7. Return:
   - `dem_patch`
   - `obstacle_mask`
   - `base_density`
   - metadata (`crop`, `obstacle_ratio`, etc.)

## 4. Minimal Runnable Verification

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run Mars example:

```bash
python scripts/run_two_agents_mars_easy.py
```

Check output:

- `outputs/two_agents_mars_easy/comparison_metrics.csv`
- `outputs/two_agents_mars_easy/baseline_vs_ours_2agents_mars_easy.png`
- `outputs/two_agents_mars_easy/baseline/trajectory_evolution.gif`
- `outputs/two_agents_mars_easy/ours/trajectory_evolution.gif`
