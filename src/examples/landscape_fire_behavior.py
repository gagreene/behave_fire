"""
landscape_fire_behavior.py
==========================
Simulates surface and crown fire behavior across a synthetic 10 km × 10 km
rasterized landscape at 30 m resolution using the behave_fire array-mode facade.

Coordinate system : EPSG:32611 (UTM Zone 11N, WGS 84)
Center            : Easting = 587 000 m, Northing = 5 500 000 m
Cell size         : 30 m
Grid dimensions   : 334 rows × 334 cols  (~10.02 km each side)

All inputs and outputs use metric units:
  - Lengths / heights  → metres
  - Wind speed         → km/h  (10-m mast height, converted to 20-ft internally)
  - Slope              → percent
  - Temperature        → °C
  - Area               → hectares
  - Fire spread rate   → m/min
  - Flame length       → m
  - Fireline intensity → kW/m

Input  GeoTIFFs → src/examples/data/
Output GeoTIFFs → src/examples/results/
PNG summary map  → src/examples/results/fire_behavior_summary.png
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path setup — allow running as a plain script OR via pytest/import
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import matplotlib
matplotlib.use("Agg")                 # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from behave_fire.behave import BehaveRun
from behave_fire.components.fuel_models import FuelModels

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
DATA_DIR    = os.path.join(_HERE, "data")
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
CELL_SIZE   = 30          # metres
EXTENT_M    = 10_020      # 334 × 30 = 10 020 m (~10 km)
N_CELLS     = EXTENT_M // CELL_SIZE          # 334
CENTER_E    = 587_000.0   # easting  (UTM Zone 11N)
CENTER_N    = 5_500_000.0 # northing
ORIGIN_E    = CENTER_E - EXTENT_M / 2.0     # upper-left corner (west edge)
ORIGIN_N    = CENTER_N + EXTENT_M / 2.0     # upper-left corner (north edge)
EPSG        = 32611
RNG         = np.random.default_rng(42)      # reproducible random state

NROWS = NCOLS = N_CELLS  # 334 × 334

print(f"Grid: {NROWS} rows × {NCOLS} cols, {CELL_SIZE} m cells")
print(f"Extent: {EXTENT_M/1000:.2f} km × {EXTENT_M/1000:.2f} km")
print(f"Origin (UL): E={ORIGIN_E:.0f} m, N={ORIGIN_N:.0f} m  (EPSG:{EPSG})")

# ---------------------------------------------------------------------------
# Rasterio transform & CRS (shared by every layer)
# ---------------------------------------------------------------------------
TRANSFORM = from_origin(ORIGIN_E, ORIGIN_N, CELL_SIZE, CELL_SIZE)
CRS_OBJ   = CRS.from_epsg(EPSG)

def save_tif(array: np.ndarray, path: str, nodata: float = -9999.0,
             dtype=None, descriptions: list[str] | None = None):
    """Write a 2-D (rows×cols) or 3-D (bands×rows×cols) ndarray to a GeoTIFF."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]    # (rows,cols) → (1,rows,cols)
    bands, rows, cols = arr.shape
    dtype = dtype or arr.dtype
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=rows, width=cols,
        count=bands,
        dtype=dtype,
        crs=CRS_OBJ,
        transform=TRANSFORM,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        for b in range(bands):
            dst.write(arr[b].astype(dtype), b + 1)
            if descriptions and b < len(descriptions):
                dst.update_tags(b + 1, description=descriptions[b])


# ===========================================================================
# 1.  SYNTHETIC TERRAIN
# ===========================================================================
print("\n[1/5] Generating synthetic terrain …")

# Row/col index grids (normalised 0–1)
col_idx, row_idx = np.meshgrid(
    np.linspace(0, 1, NCOLS),
    np.linspace(0, 1, NROWS),
)

# Elevation: two overlapping Gaussian ridges + low-frequency noise
# Ridge 1 — SW–NE trending, sharp narrow crest (peak ~2 200 m)
elev  = 800.0 + 1400.0 * np.exp(
    -((col_idx - 0.35)**2 / 0.015 + (row_idx - 0.40)**2 / 0.025))
# Ridge 2 — secondary ridge, north sector (peak ~1 800 m)
elev += 800.0 * np.exp(
    -((col_idx - 0.65)**2 / 0.018 + (row_idx - 0.20)**2 / 0.012))
# Broad valley floor baseline (valley centre around col 0.55, row 0.75)
elev += 200.0 * np.exp(
    -((col_idx - 0.55)**2 / 0.12 + (row_idx - 0.75)**2 / 0.10))
# Medium-scale noise (smoothed)
noise_med = gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=12) * 120
# Fine-scale noise — sharper for realistic rugged terrain
noise_fine = gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=3) * 40
elev += noise_med + noise_fine
elev = np.clip(elev, 600.0, None).astype(np.float32)

# ---- Derive slope (%) and aspect (degrees) from elevation ----
# Central-difference gradient in metres; dy is positive downward (row axis),
# so we negate it for the northward component.
dz_dx = np.gradient(elev, CELL_SIZE, axis=1)    # east  component (rise / run)
dz_dy = np.gradient(elev, CELL_SIZE, axis=0)    # south component (rise / run)

slope_pct = np.sqrt(dz_dx**2 + dz_dy**2) * 100.0   # percent
slope_pct = np.clip(slope_pct, 0, 200).astype(np.float32)

# Aspect: degrees clockwise from north (0=N, 90=E, 180=S, 270=W)
aspect_deg = np.degrees(np.arctan2(dz_dx, -dz_dy)) % 360.0
aspect_deg = aspect_deg.astype(np.float32)

save_tif(elev,       os.path.join(DATA_DIR, "elevation_m.tif"),       dtype=np.float32)
save_tif(slope_pct,  os.path.join(DATA_DIR, "slope_pct.tif"),          dtype=np.float32)
save_tif(aspect_deg, os.path.join(DATA_DIR, "aspect_deg.tif"),         dtype=np.float32)
print(f"  elevation : {elev.min():.0f} – {elev.max():.0f} m")
print(f"  slope     : {slope_pct.min():.1f} – {slope_pct.max():.1f} %")


# ===========================================================================
# 2.  FUEL MODEL MAP  (Scott & Burgan 40 + classic 13 models)
# ===========================================================================
print("\n[2/5] Generating fuel model map …")

# Land-cover zones driven by elevation and a stochastic shrub patch layer
# Zone thresholds (m)
ELEV_LOW   = 900    # below → grass-dominated
ELEV_MID   = 1300   # low–mid → grass-shrub mix
ELEV_HIGH  = 1800   # mid–high → timber-litter mix
               #         above → sparse / rock

# Base assignment
fuel_grid = np.full((NROWS, NCOLS), 181, dtype=np.int32)  # TL1 default

# Low elevation / valley: GR4 (moderate load dry grass) and GS2 (grass-shrub)
fuel_grid[elev < ELEV_LOW]  = 104   # GR4
fuel_grid[(elev >= ELEV_LOW) & (elev < ELEV_MID)] = 122  # GS2

# Mid elevation: mix of TU1 (light dry timber-grass-shrub) and TL1
fuel_grid[(elev >= ELEV_MID) & (elev < ELEV_HIGH)] = 161  # TU1

# High elevation / ridge: TL3 (moderate conifer litter)
fuel_grid[elev >= ELEV_HIGH] = 183  # TL3

# Stochastic shrub patches (SH5 — high load dry shrub) scattered in mid-elevations
shrub_noise = gaussian_filter(
    RNG.standard_normal((NROWS, NCOLS)), sigma=15)
shrub_mask = (shrub_noise > 1.1) & (elev >= ELEV_LOW) & (elev < ELEV_HIGH)
fuel_grid[shrub_mask] = 145  # SH5

# Small riparian corridors (GR2 — low load dry grass) in valleys
riparian_noise = gaussian_filter(
    RNG.standard_normal((NROWS, NCOLS)), sigma=3)
riparian_mask = (riparian_noise > 1.8) & (elev < ELEV_LOW + 100)
fuel_grid[riparian_mask] = 102  # GR2

# Non-burnable rocky outcrops on steepest slopes
fuel_grid[slope_pct > 80] = 99   # NB9 bare ground

save_tif(fuel_grid, os.path.join(DATA_DIR, "fuel_models.tif"), dtype=np.int32, nodata=-1)

unique_fm, counts = np.unique(fuel_grid, return_counts=True)
print("  Fuel model distribution (model : cells):")
for fm, n in zip(unique_fm, counts):
    pct = n / fuel_grid.size * 100
    print(f"    {fm:4d} : {n:6d} cells  ({pct:.1f} %)")


# ===========================================================================
# 3.  FOREST STRUCTURE  (canopy cover, height, base height, bulk density)
# ===========================================================================
print("\n[3/5] Generating forest structure …")

# Canopy cover (fraction 0–1): highest in TU/TL zone, low in grass/shrub
canopy_cover = np.where(
    (elev >= ELEV_MID) & (elev < ELEV_HIGH), 0.65,
    np.where(elev >= ELEV_HIGH, 0.50,
    np.where(elev < ELEV_LOW, 0.10, 0.30))
).astype(np.float32)
# Add spatial heterogeneity
canopy_cover += gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=10).astype(np.float32) * 0.08
canopy_cover  = np.clip(canopy_cover, 0.0, 0.95).astype(np.float32)

# Canopy height (m): 5–35 m depending on zone
canopy_height_m = np.where(
    (elev >= ELEV_MID) & (elev < ELEV_HIGH), 22.0,
    np.where(elev >= ELEV_HIGH, 18.0,
    np.where(elev < ELEV_LOW, 3.0, 8.0))
).astype(np.float32)
canopy_height_m += gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=8).astype(np.float32) * 3.0
canopy_height_m  = np.clip(canopy_height_m, 1.0, 40.0).astype(np.float32)

# Crown ratio (fraction of canopy height that is live crown, 0–1)
crown_ratio = np.full((NROWS, NCOLS), 0.55, dtype=np.float32)
crown_ratio += gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=10).astype(np.float32) * 0.08
crown_ratio  = np.clip(crown_ratio, 0.20, 0.85).astype(np.float32)

# Canopy base height (m) = canopy_height × (1 – crown_ratio)
canopy_base_height_m = canopy_height_m * (1.0 - crown_ratio)
canopy_base_height_m = np.clip(canopy_base_height_m, 0.5, 20.0).astype(np.float32)

# Canopy bulk density (kg/m³): typical conifer range 0.05–0.35 kg/m³
canopy_bulk_density_kgm3 = np.where(
    (elev >= ELEV_MID) & (elev < ELEV_HIGH), 0.15,
    np.where(elev >= ELEV_HIGH, 0.10,
    np.where(elev < ELEV_LOW, 0.02, 0.06))
).astype(np.float32)
canopy_bulk_density_kgm3 += (
    gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=12).astype(np.float32) * 0.03)
canopy_bulk_density_kgm3 = np.clip(canopy_bulk_density_kgm3, 0.01, 0.40).astype(np.float32)

save_tif(canopy_cover,          os.path.join(DATA_DIR, "canopy_cover_frac.tif"),     dtype=np.float32)
save_tif(canopy_height_m,       os.path.join(DATA_DIR, "canopy_height_m.tif"),       dtype=np.float32)
save_tif(crown_ratio,           os.path.join(DATA_DIR, "crown_ratio_frac.tif"),      dtype=np.float32)
save_tif(canopy_base_height_m,  os.path.join(DATA_DIR, "canopy_base_height_m.tif"),  dtype=np.float32)
save_tif(canopy_bulk_density_kgm3,
         os.path.join(DATA_DIR, "canopy_bulk_density_kgm3.tif"), dtype=np.float32)
print(f"  canopy height : {canopy_height_m.min():.1f} – {canopy_height_m.max():.1f} m")
print(f"  canopy cover  : {canopy_cover.min():.2f} – {canopy_cover.max():.2f}")


# ===========================================================================
# 4.  WEATHER  (fuel moisture + wind, single weather scenario)
# ===========================================================================
print("\n[4/5] Generating weather inputs …")

# --- Fuel moisture (spatially varying, fractions) ---
# Mild elevation-moisture gradient + local random variation
elev_norm = (elev - elev.min()) / (elev.max() - elev.min())   # 0–1

def make_moisture(base, elev_gain, sigma=6, noise_amp=0.01, lo=0.02, hi=0.50):
    """Create a spatially smooth moisture grid.

    :param base: Base moisture fraction at lowest elevation.
    :param elev_gain: Additional moisture fraction added from low to high elevation.
    :param sigma: Gaussian smoothing radius (cells) for spatial noise.
    :param noise_amp: Amplitude of random spatial noise (fraction).
    :param lo: Lower clip bound (fraction).
    :param hi: Upper clip bound (fraction).
    """
    m = (base + elev_norm * elev_gain
         + gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=sigma) * noise_amp)
    return np.clip(m, lo, hi).astype(np.float32)


m1h   = make_moisture(0.06, 0.04, sigma=8,  noise_amp=0.015)           # 1-hr  dead  ~6–10 %
m10h  = make_moisture(0.08, 0.04, sigma=10, noise_amp=0.010)           # 10-hr dead  ~8–12 %
m100h = make_moisture(0.11, 0.04, sigma=12, noise_amp=0.008)           # 100-hr dead ~11–15 %
mlh   = make_moisture(0.60,  0.20, sigma=15, noise_amp=0.050, hi=1.50)  # live herb   ~60–80 %
mlw   = make_moisture(0.90,  0.15, sigma=15, noise_amp=0.050, hi=1.50)  # live woody  ~90–105 %
moisture_foliar = np.full((NROWS, NCOLS), 1.00, dtype=np.float32)  # 100 % (as fraction)

save_tif(m1h,   os.path.join(DATA_DIR, "m1h_frac.tif"),   dtype=np.float32)
save_tif(m10h,  os.path.join(DATA_DIR, "m10h_frac.tif"),  dtype=np.float32)
save_tif(m100h, os.path.join(DATA_DIR, "m100h_frac.tif"), dtype=np.float32)
save_tif(mlh,   os.path.join(DATA_DIR, "mlh_frac.tif"),   dtype=np.float32)
save_tif(mlw,   os.path.join(DATA_DIR, "mlw_frac.tif"),   dtype=np.float32)

# --- Wind (10-m mast, km/h, geographic north convention) ---
# Prevailing SW wind (225°) with orographic acceleration on ridges
wind_speed_base_kmh = 25.0    # prevailing open-land speed (km/h at 10 m)
wind_dir_deg        = 225.0   # compass bearing SW → NE

# Orographic enhancement: faster on exposed upper slopes, slower in valleys
wind_topo_factor = 1.0 + 0.5 * elev_norm
wind_speed_kmh = (wind_speed_base_kmh * wind_topo_factor
                  + gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=12) * 3.0)
wind_speed_kmh = np.clip(wind_speed_kmh, 3.0, 80.0).astype(np.float32)

wind_dir_grid = np.full((NROWS, NCOLS), wind_dir_deg, dtype=np.float32)
# Add ±15° directional variability
wind_dir_grid += gaussian_filter(
    RNG.standard_normal((NROWS, NCOLS)), sigma=10).astype(np.float32) * 15.0
wind_dir_grid = (wind_dir_grid % 360.0).astype(np.float32)

# Air temperature (°C) — declines with elevation (lapse rate ~6.5 °C/km)
air_temp_c = (33.0 - (elev - elev.min()) / 1000.0 * 6.5
              + gaussian_filter(RNG.standard_normal((NROWS, NCOLS)), sigma=8) * 1.5)
air_temp_c = np.clip(air_temp_c, 5.0, 40.0).astype(np.float32)

save_tif(wind_speed_kmh, os.path.join(DATA_DIR, "wind_speed_10m_kmh.tif"),  dtype=np.float32)
save_tif(wind_dir_grid,  os.path.join(DATA_DIR, "wind_dir_north_deg.tif"),  dtype=np.float32)
save_tif(air_temp_c,     os.path.join(DATA_DIR, "air_temp_c.tif"),           dtype=np.float32)
print(f"  wind speed (10 m): {wind_speed_kmh.min():.1f} – {wind_speed_kmh.max():.1f} km/h")
print(f"  air temperature  : {air_temp_c.min():.1f} – {air_temp_c.max():.1f} °C")


# ===========================================================================
# 5.  FIRE BEHAVIOR  (surface → crown → scorch → mortality → spotting)
# ===========================================================================
print("\n[5/5] Running behave_fire …")

fm     = FuelModels()
runner = BehaveRun(fm)

# --------------------------------------------------------------------------
# Unit enum integers (from behave_units.py):
#   SpeedUnitsEnum     : 6 = KilometersPerHour,  3 = MetersPerMinute
#   LengthUnitsEnum    : 4 = Meters,  7 = Kilometers
#   SlopeUnitsEnum     : 1 = Percent
#   TemperatureUnitsEnum: 1 = Celsius
#   FirelineIntensityUnitsEnum: 4 = KilowattsPerMeter
#   HeatPerUnitAreaUnitsEnum  : 1 = KilojoulesPerSquareMeter
#   AreaUnitsEnum      : 2 = Hectares
# --------------------------------------------------------------------------

# ---- 5a  Surface run ----
# Wind input is at 10-m mast height → wind_height_mode='TenMeter'
# Wind is compass-north bearing      → wind_orientation_mode='RelativeToNorth'
surface = runner.do_surface_run(
    fuel_model_grid      = fuel_grid,
    m1h=m1h, m10h=m10h, m100h=m100h, mlh=mlh, mlw=mlw,
    moisture_units       = 0,          # Fraction (data stored as 0.06, not 6%)
    wind_speed           = wind_speed_kmh,
    wind_speed_units     = 6,          # KilometersPerHour
    wind_direction       = wind_dir_grid,
    wind_orientation_mode= 'RelativeToNorth',
    slope                = slope_pct,
    slope_units          = 1,          # Percent
    aspect               = aspect_deg,
    canopy_cover         = canopy_cover,
    canopy_cover_units   = 0,          # Fraction (data stored as 0–1)
    canopy_height        = canopy_height_m,
    canopy_height_units  = 4,          # Meters
    crown_ratio          = crown_ratio,
    crown_ratio_units    = 0,          # Fraction (data stored as 0–1)
    wind_height_mode     = 'TenMeter',
    waf_method           = 'UseCrownRatio',
    out_units={
        'spread_rate'              : 3,   # MetersPerMinute
        'backing_spread_rate'      : 3,
        'flanking_spread_rate'     : 3,
        'no_wind_no_slope_spread_rate': 3,
        'midflame_wind_speed'      : 6,   # KilometersPerHour
        'effective_wind_speed'     : 6,
        'flame_length'             : 4,   # Meters
        'fireline_intensity'       : 4,   # KilowattsPerMeter
        'heat_per_unit_area'       : 1,   # KilojoulesPerSquareMeter
        'reaction_intensity'       : 4,   # KilowattsPerSquareMeter
        'residence_time'           : 0,   # Minutes (already default)
    },
)

# ---- 5b  Crown run ----
crown = runner.do_crown_run(
    surface_results      = surface,
    fuel_model_grid      = fuel_grid,
    m1h=m1h, m10h=m10h, m100h=m100h, mlh=mlh, mlw=mlw,
    moisture_units       = 0,          # Fraction (data stored as 0–1)
    wind_speed           = wind_speed_kmh,
    wind_speed_units     = 6,
    wind_direction       = wind_dir_grid,
    wind_orientation_mode= 'RelativeToNorth',
    slope                = slope_pct,
    slope_units          = 1,
    aspect               = aspect_deg,
    canopy_base_height        = canopy_base_height_m,
    canopy_base_height_units  = 4,                    # Meters
    canopy_height             = canopy_height_m,
    canopy_height_units       = 4,                    # Meters
    canopy_bulk_density       = canopy_bulk_density_kgm3,
    canopy_bulk_density_units = 1,                    # KilogramsPerCubicMeter
    moisture_foliar           = moisture_foliar * 100.0,  # fraction → percent (foliar still %)

    out_units={
        'crown_fire_spread_rate'             : 3,   # MetersPerMinute
        'crown_critical_fire_spread_rate'    : 3,
        'crown_flame_length'                 : 4,   # Meters
        'crown_fire_line_intensity'          : 4,   # KilowattsPerMeter
        'crown_critical_surface_fire_line_intensity': 4,
        'crown_fire_heat_per_unit_area'      : 1,   # KilojoulesPerSquareMeter
        'canopy_heat_per_unit_area'          : 1,
        'final_spread_rate'                  : 3,   # MetersPerMinute
        'final_fireline_intensity'           : 4,   # KilowattsPerMeter
        'final_flame_length'                 : 4,   # Meters
        'final_heat_per_unit_area'           : 1,   # KilojoulesPerSquareMeter
    },
)

# ---- 5c  Scorch height ----
# fireline_intensity from surface run is already in kW/m (unit 4);
# we need it in BTU/ft/s (unit 0) for calculate_scorch_height, which then
# converts internally — so pass kW/m with fireline_intensity_units=4.
scorch_height_m = runner.calculate_scorch_height(
    fireline_intensity       = surface['fireline_intensity'],
    fireline_intensity_units = 4,   # KilowattsPerMeter
    midflame_wind_speed      = surface['midflame_wind_speed'],
    wind_speed_units         = 6,   # KilometersPerHour
    air_temperature          = air_temp_c,
    temperature_units        = 1,   # Celsius
    out_units                = 4,   # Meters
)

# ---- 5d  Mortality  (tree height from canopy_height_m, DBH estimated) ----
# Estimate DBH (cm) from height via a simple allometric: DBH_cm ≈ 2.5 × H_m^0.8
dbh_cm = 2.5 * canopy_height_m ** 0.8

# Equation number: TU/TL zone → ponderosa pine eq. 15, elsewhere → Douglas-fir eq. 5
eq_grid = np.where(elev >= ELEV_MID, 15, 5).astype(np.int32)

mortality = runner.calculate_crown_scorch_mortality(
    scorch_height        = scorch_height_m,
    scorch_height_units  = 4,   # Meters
    tree_height          = canopy_height_m,
    tree_height_units    = 4,   # Meters
    crown_ratio          = crown_ratio,
    crown_ratio_units    = 0,   # Fraction (data stored as 0–1)
    dbh                  = dbh_cm,
    dbh_units            = 3,   # Centimeters
    equation_number_grid = eq_grid,
    out_units={
        'crown_length_scorch'  : 1,   # Percent
        'crown_volume_scorch'  : 1,
        'probability_mortality': 1,
    },
)

# ---- 5e  Spotting from surface fire ----
# Flame length in metres (unit 4), wind in km/h (unit 6), cover height in m (unit 4)
spot_distance_m = runner.calculate_spotting_from_surface_fire(
    flame_length       = surface['flame_length'],
    flame_length_units = 4,   # Meters
    wind_speed         = wind_speed_kmh,
    wind_speed_units   = 6,   # KilometersPerHour
    cover_height       = canopy_height_m,
    cover_height_units = 4,   # Meters
    out_units          = 4,   # Meters
)

# ---- 5f  Fire size at 60 min (surface) ----
fire_area_ha = runner.calculate_fire_area(
    forward_ros     = surface['spread_rate'],
    backing_ros     = surface['backing_spread_rate'],
    ros_units       = 3,   # MetersPerMinute
    lwr             = surface['fire_length_to_width_ratio'],
    elapsed_time    = 60.0,
    elapsed_time_units = 0,  # Minutes
    is_crown        = False,
    out_units       = 2,   # Hectares
)

print("  Surface run  : done")
print("  Crown run    : done")
print("  Scorch height: done")
print("  Mortality    : done")
print("  Spotting     : done")
print("  Fire size    : done")


# ===========================================================================
# 6.  SAVE OUTPUT GeoTIFFs
# ===========================================================================
print("\nSaving output rasters …")

def _save2d(arr, fname, desc=""):
    path = os.path.join(RESULTS_DIR, fname)
    a = np.asarray(arr).astype(np.float32)
    if a.ndim != 2:
        raise ValueError(f"{fname}: expected 2D array, got shape {a.shape}")
    save_tif(a, path, dtype=np.float32)
    print(f"  {fname:<55s}  min={np.nanmin(a):.3f}  max={np.nanmax(a):.3f}  {desc}")

_save2d(surface['spread_rate'],                "surface_spread_rate_mpm.tif",     "m/min")
_save2d(surface['backing_spread_rate'],        "surface_backing_ros_mpm.tif",     "m/min")
_save2d(surface['flanking_spread_rate'],       "surface_flanking_ros_mpm.tif",    "m/min")
_save2d(surface['flame_length'],               "surface_flame_length_m.tif",      "m")
_save2d(surface['fireline_intensity'],         "surface_fireline_intensity_kwm.tif", "kW/m")
_save2d(surface['heat_per_unit_area'],         "surface_hpua_kjm2.tif",           "kJ/m²")
_save2d(surface['effective_wind_speed'],       "surface_effective_wind_kmh.tif",  "km/h")
_save2d(surface['midflame_wind_speed'],        "surface_midflame_wind_kmh.tif",   "km/h")
_save2d(surface['fire_length_to_width_ratio'], "surface_lwr.tif",                 "dimensionless")
_save2d(surface['direction_of_max_spread'],    "surface_direction_max_spread_deg.tif", "deg")

_save2d(crown['crown_fire_spread_rate'],       "crown_spread_rate_mpm.tif",       "m/min")
_save2d(crown['crown_flame_length'],           "crown_flame_length_m.tif",        "m")
_save2d(crown['crown_fire_line_intensity'],    "crown_fireline_intensity_kwm.tif","kW/m")
_save2d(crown['fire_type'],                    "fire_type.tif",
        "0=surface 1=torching 2=conditional 3=active")
_save2d(crown['crown_fraction_burned'],        "crown_fraction_burned.tif",       "0-1")
_save2d(crown['final_spread_rate'],            "final_head_fire_ros_mpm.tif",     "m/min")
_save2d(crown['final_fireline_intensity'],     "final_head_fire_fli_kwm.tif",     "kW/m")
_save2d(crown['final_flame_length'],           "final_head_fire_flame_length_m.tif", "m")
_save2d(crown['final_heat_per_unit_area'],     "final_head_fire_hpua_kjm2.tif",   "kJ/m²")

_save2d(scorch_height_m,                       "scorch_height_m.tif",             "m")
_save2d(mortality['probability_mortality'],    "probability_mortality_pct.tif",   "%")
_save2d(mortality['crown_volume_scorch'],      "crown_volume_scorch_pct.tif",     "%")

_save2d(spot_distance_m,                       "spot_distance_m.tif",             "m")
_save2d(fire_area_ha,                          "fire_area_60min_ha.tif",          "ha")


# ===========================================================================
# 7.  SUMMARY MAP  (12-panel figure, 3 rows × 4 cols)
# ===========================================================================
print("\nGenerating summary map …")

def _r(arr):
    """Return the array as a 2D (NROWS,NCOLS) grid — outputs are already this shape."""
    return np.asarray(arr)

# Discrete fire-type colormap — 4 values matching C++ FireType enum
FIRE_TYPE_COLORS = {
    0: "#f7f7f7",   # Surface
    1: "#fdae6b",   # Torching (passive)
    2: "#a6cee3",   # ConditionalCrownFire
    3: "#d73027",   # Crowning (active)
}
fire_type_arr = _r(crown['fire_type'])
ft_rgba = np.zeros((NROWS, NCOLS, 4), dtype=np.float32)
for val, hexcol in FIRE_TYPE_COLORS.items():
    rgba = np.array(mcolors.to_rgba(hexcol))
    ft_rgba[fire_type_arr == val] = rgba

fig, axes = plt.subplots(3, 4, figsize=(22, 17),
                         constrained_layout=True)
fig.suptitle(
    "behave_fire — Synthetic Landscape Fire Behavior\n"
    f"10 km × 10 km, 30 m resolution  |  EPSG:{EPSG}  |  "
    f"Centre: E{CENTER_E:.0f} N{CENTER_N:.0f}",
    fontsize=13, fontweight="bold",
)

panels = [
    # (axis, data, title, cmap, unit_label, vmin, vmax)
    # Row 0 — terrain and surface fire
    (axes[0, 0], elev,
     "Elevation (m)",                  "terrain",   "m",       None, None),
    (axes[0, 1], _r(surface['spread_rate']),
     "Surface ROS (m/min)",            "YlOrRd",    "m/min",   0,    10),
    (axes[0, 2], _r(surface['flame_length']),
     "Surface Flame Length (m)",       "hot_r",     "m",       0,    8),
    (axes[0, 3], _r(surface['fireline_intensity']),
     "Surface FLI (kW/m)",             "inferno",   "kW/m",    0,    5000),
    # Row 1 — crown fire and classification
    (axes[1, 0], None,
     "Fire Type",                      None,        "",        None, None),
    (axes[1, 1], _r(crown['crown_fraction_burned']),
     "Crown Fraction Burned (0–1)",    "YlOrRd",    "CFB",     0,    1),
    (axes[1, 2], _r(crown['crown_flame_length']),
     "Active Crown Flame Length (m)",  "hot_r",     "m",       0,    30),
    (axes[1, 3], _r(crown['crown_fire_line_intensity']),
     "Active Crown FLI (kW/m)",        "inferno",   "kW/m",    0,    50000),
    # Row 2 — final head fire and effects
    (axes[2, 0], _r(crown['final_spread_rate']),
     "Final Head Fire ROS (m/min)",    "YlOrRd",    "m/min",   0,    30),
    (axes[2, 1], _r(crown['final_fireline_intensity']),
     "Final Head Fire FLI (kW/m)",     "inferno",   "kW/m",    0,    10000),
    (axes[2, 2], scorch_height_m,
     "Scorch Height (m)",              "RdPu",      "m",       0,    25),
    (axes[2, 3], _r(mortality['probability_mortality']),
     "Probability of Mortality (%)",   "RdYlGn_r",  "%",       0,    100),
]

for ax, data, title, cmap, unit, vmin, vmax in panels:
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Easting offset (km)", fontsize=7)
    ax.set_ylabel("Northing offset (km)", fontsize=7)
    tick_km = np.arange(0, NCOLS + 1, NCOLS // 5)
    ax.set_xticks(tick_km)
    ax.set_xticklabels([f"{t*CELL_SIZE/1000:.0f}" for t in tick_km], fontsize=6)
    ax.set_yticks(tick_km)
    ax.set_yticklabels([f"{t*CELL_SIZE/1000:.0f}" for t in tick_km], fontsize=6)

    if data is None:
        ax.imshow(ft_rgba, origin="upper", interpolation="nearest")
        legend_elements = [
            Patch(facecolor=FIRE_TYPE_COLORS[0], edgecolor="grey", label="0 — Surface"),
            Patch(facecolor=FIRE_TYPE_COLORS[1], edgecolor="grey", label="1 — Torching"),
            Patch(facecolor=FIRE_TYPE_COLORS[2], edgecolor="grey", label="2 — Conditional crown"),
            Patch(facecolor=FIRE_TYPE_COLORS[3], edgecolor="grey", label="3 — Active crown"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=6)
    else:
        im = ax.imshow(data, origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(unit, fontsize=7)

out_png = os.path.join(RESULTS_DIR, "fire_behavior_summary.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved -> {out_png}")

# ===========================================================================
# 8.  CONSOLE STATISTICS
# ===========================================================================
print("\n" + "=" * 60)
print("Landscape-wide fire behavior statistics")
print("=" * 60)

def stats(label, arr, unit):
    a = np.asarray(arr).ravel()
    print(f"  {label:<40s}  "
          f"mean={np.nanmean(a):8.2f}  "
          f"p50={np.nanpercentile(a,50):8.2f}  "
          f"p90={np.nanpercentile(a,90):8.2f}  "
          f"max={np.nanmax(a):8.2f}  [{unit}]")

stats("Surface ROS",               surface['spread_rate'],           "m/min")
stats("Backing ROS",               surface['backing_spread_rate'],   "m/min")
stats("Surface flame length",      surface['flame_length'],          "m")
stats("Surface FLI",               surface['fireline_intensity'],    "kW/m")
stats("Effective wind speed",      surface['effective_wind_speed'],  "km/h")
stats("Active crown ROS",          crown['crown_fire_spread_rate'],  "m/min")
stats("Active crown flame length", crown['crown_flame_length'],      "m")
stats("Crown fraction burned",     crown['crown_fraction_burned'],   "0-1")
stats("Final head fire ROS",       crown['final_spread_rate'],       "m/min")
stats("Final head fire FLI",       crown['final_fireline_intensity'],"kW/m")
stats("Final head fire FL",        crown['final_flame_length'],      "m")
stats("Scorch height",             scorch_height_m,                  "m")
stats("Probability of mortality",  mortality['probability_mortality'],"% ")
stats("Max spotting distance",     spot_distance_m,                  "m")
stats("60-min fire area",          fire_area_ha,                     "ha")

fire_type_flat = fire_type_arr.ravel()
n_total       = fire_type_flat.size
n_surface     = np.sum(fire_type_flat == 0)
n_torching    = np.sum(fire_type_flat == 1)
n_conditional = np.sum(fire_type_flat == 2)
n_active      = np.sum(fire_type_flat == 3)
print()
print(f"  Fire type breakdown (of {n_total:,} cells):")
print(f"    0 — Surface             : {n_surface:6,}  ({n_surface/n_total*100:5.1f} %)")
print(f"    1 — Torching (passive)  : {n_torching:6,}  ({n_torching/n_total*100:5.1f} %)")
print(f"    2 — Conditional crown   : {n_conditional:6,}  ({n_conditional/n_total*100:5.1f} %)")
print(f"    3 — Active crown        : {n_active:6,}  ({n_active/n_total*100:5.1f} %)")
print("=" * 60)
print("\nDone.")





