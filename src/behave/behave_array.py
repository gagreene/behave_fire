"""
behave_array.py — Array-Mode Facade (V9)

BehaveRunArray is the public entry point for array/raster mode.
It accepts NumPy arrays for all spatially-variable inputs and returns
dicts of output arrays of the same shape.

The scalar facade (BehaveRun in behave.py) is unchanged.

Usage
-----
    import numpy as np
    from behave.components.fuel_models import FuelModels
    from behave.behave_array import BehaveRunArray
    from behave.components.behave_units import SpeedUnits

    fm     = FuelModels()
    runner = BehaveRunArray(fm)

    results = runner.do_surface_run(
        fuel_model_grid=np.array([[124]]),
        m1h=np.array([[0.06]]), m10h=np.array([[0.07]]),
        m100h=np.array([[0.08]]), mlh=np.array([[0.60]]),
        mlw=np.array([[0.90]]),
        wind_speed=np.array([[5.0]]),
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        wind_direction=np.array([[0.0]]),
        wind_orientation_mode='RelativeToUpslope',
        slope=np.array([[np.degrees(np.arctan(0.30))]]),   # 30% → degrees
        aspect=np.array([[0.0]]),
        canopy_cover=np.array([[0.50]]),
        canopy_height=np.array([[30.0]]),
        crown_ratio=np.array([[0.50]]),
    )
    ros = results['spread_rate'][0, 0]   # ft/min

Notes
-----
* slope must be in DEGREES.  Convert percent before calling:
      slope_deg = np.degrees(np.arctan(slope_pct / 100.0))
* BehaveRunArray is NOT a scalar replacement for BehaveRun.  Single-value
  inputs return shape-(1,) or shape-(1,1) arrays, not Python scalars.
"""

import numpy as np

try:
    from .components.fuel_models import FuelModels
    from .components.fuel_models_array import build_fuel_lookup_arrays
    from .components.surface_array import (
        build_particle_arrays,
        calculate_fuelbed_intermediates,
        calculate_reaction_intensity,
        calculate_spread_rate,
        calculate_fire_area,
        calculate_fire_perimeter,
        calculate_fire_length,
        calculate_fire_width,
    )
    from .components.crown_array import calculate_crown_fire
    from .components.mortality_array import (
        calculate_scorch_height,
        build_mortality_lookup,
        calculate_crown_scorch_mortality,
    )
    from .components.spot_array import (
        calculate_spotting_from_surface_fire,
        calculate_spotting_from_burning_pile,
        calculate_spotting_from_torching_trees,
    )
    from .components.behave_units_array import (
        fireline_intensity_to_base,
        speed_to_base,
        speed_from_base,
        temp_to_base,
    )
except ImportError:
    from components.fuel_models import FuelModels
    from components.fuel_models_array import build_fuel_lookup_arrays
    from components.surface_array import (
        build_particle_arrays,
        calculate_fuelbed_intermediates,
        calculate_reaction_intensity,
        calculate_spread_rate,
        calculate_fire_area,
        calculate_fire_perimeter,
        calculate_fire_length,
        calculate_fire_width,
    )
    from components.crown_array import calculate_crown_fire
    from components.mortality_array import (
        calculate_scorch_height,
        build_mortality_lookup,
        calculate_crown_scorch_mortality,
    )
    from components.spot_array import (
        calculate_spotting_from_surface_fire,
        calculate_spotting_from_burning_pile,
        calculate_spotting_from_torching_trees,
    )
    from components.behave_units_array import (
        fireline_intensity_to_base,
        speed_to_base,
        speed_from_base,
        temp_to_base,
    )


class BehaveRunArray:
    """
    Array-mode facade for behave_py.

    Accepts NumPy arrays for all spatially-variable inputs.
    Scalar inputs (unit enums, mode strings) remain Python scalars.

    The fuel lookup table is built once at construction from the FuelModels
    instance and reused for all subsequent run calls.
    """

    def __init__(self, fuel_models: FuelModels):
        """
        Parameters
        ----------
        fuel_models : FuelModels
            Populated FuelModels instance (scalar, unchanged).
        """
        self._fuel_models = fuel_models
        self._lut = build_fuel_lookup_arrays(fuel_models)
        self._mortality_coeffs = build_mortality_lookup()

    # ------------------------------------------------------------------
    # Surface run
    # ------------------------------------------------------------------

    def do_surface_run(self,
                       fuel_model_grid,
                       m1h, m10h, m100h,
                       mlh, mlw,
                       wind_speed,
                       wind_speed_units,
                       wind_direction,
                       wind_orientation_mode,
                       slope,
                       aspect,
                       canopy_cover,
                       canopy_height,
                       crown_ratio,
                       wind_height_mode='TwentyFoot',
                       waf_method='UseCrownRatio',
                       user_waf=None) -> dict:
        """
        Run the vectorized surface fire pipeline.

        Parameters
        ----------
        fuel_model_grid      : int array (*S)
        m1h .. mlw           : float arrays (*S) — moisture fractions
        wind_speed           : (*S) or scalar — in wind_speed_units
        wind_speed_units     : scalar int — SpeedUnitsEnum
        wind_direction       : (*S) or scalar — degrees
        wind_orientation_mode: str — 'RelativeToUpslope' or 'RelativeToNorth'
        slope                : (*S) or scalar — DEGREES (not percent)
        aspect               : (*S) or scalar — degrees
        canopy_cover         : (*S) or scalar — fraction (0–1)
        canopy_height        : (*S) or scalar — feet
        crown_ratio          : (*S) or scalar — fraction (0–1)
        wind_height_mode     : str — 'TwentyFoot' or 'TenMeter'
        waf_method           : str — 'UseCrownRatio' or 'UserInput'
        user_waf             : (*S) or scalar or None

        Returns
        -------
        dict of (*S) ndarrays — same keys as calculate_spread_rate() output:
            spread_rate, backing_spread_rate, flanking_spread_rate,
            flame_length, fireline_intensity, heat_per_unit_area,
            effective_wind_speed, fire_length_to_width_ratio, eccentricity,
            direction_of_max_spread, residence_time, reaction_intensity,
            midflame_wind_speed, no_wind_no_slope_spread_rate

        IMPORTANT — slope must be in DEGREES.
        Convert percent slope before calling:
            slope_deg = np.degrees(np.arctan(slope_pct_grid / 100.0))
        """
        # Coerce all spatial inputs to at-least-1D ndarrays (G10 fix)
        fuel_model_grid = np.atleast_1d(np.asarray(fuel_model_grid, dtype=np.int32))
        m1h = np.atleast_1d(np.asarray(m1h, dtype=float))
        m10h = np.atleast_1d(np.asarray(m10h, dtype=float))
        m100h = np.atleast_1d(np.asarray(m100h, dtype=float))
        mlh = np.atleast_1d(np.asarray(mlh, dtype=float))
        mlw = np.atleast_1d(np.asarray(mlw, dtype=float))
        wind_speed = np.atleast_1d(np.asarray(wind_speed, dtype=float))
        wind_direction = np.atleast_1d(np.asarray(wind_direction, dtype=float))
        slope = np.atleast_1d(np.asarray(slope, dtype=float))
        aspect = np.atleast_1d(np.asarray(aspect, dtype=float))
        canopy_cover = np.atleast_1d(np.asarray(canopy_cover, dtype=float))
        canopy_height = np.atleast_1d(np.asarray(canopy_height, dtype=float))
        crown_ratio = np.atleast_1d(np.asarray(crown_ratio, dtype=float))

        p = build_particle_arrays(
            self._lut, fuel_model_grid,
            m1h, m10h, m100h, mlh, mlw
        )
        ib = calculate_fuelbed_intermediates(p)
        ri = calculate_reaction_intensity(ib)
        return calculate_spread_rate(
            ri, ib,
            wind_speed, wind_speed_units,
            wind_direction, wind_orientation_mode,
            slope, aspect,
            canopy_cover, canopy_height, crown_ratio,
            wind_height_mode=wind_height_mode,
            waf_method=waf_method,
            user_waf=user_waf,
        )

    # ------------------------------------------------------------------
    # Crown run
    # ------------------------------------------------------------------

    def do_crown_run(self,
                     surface_results,
                     fuel_model_grid,
                     m1h, m10h, m100h, mlh, mlw,
                     wind_speed, wind_speed_units,
                     wind_direction, wind_orientation_mode,
                     slope, aspect,
                     canopy_base_height, canopy_height,
                     canopy_bulk_density, moisture_foliar) -> dict:
        """
        Run the vectorized crown fire pipeline.

        Parameters
        ----------
        surface_results   : dict from do_surface_run()
        (other params same as do_surface_run / calculate_crown_fire)
        canopy_base_height: (*S) — feet
        canopy_bulk_density: (*S) — lb/ft³
        moisture_foliar   : (*S) or scalar — percent

        Returns
        -------
        dict — see calculate_crown_fire() for full key list
        """
        return calculate_crown_fire(
            surface_results, self._lut, fuel_model_grid,
            m1h, m10h, m100h, mlh, mlw,
            wind_speed, wind_speed_units,
            wind_direction, wind_orientation_mode,
            slope, aspect,
            canopy_base_height, canopy_height,
            canopy_bulk_density, moisture_foliar,
        )

    # ------------------------------------------------------------------
    # Scorch height
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_scorch_height(fireline_intensity,
                                fireline_intensity_units,
                                midflame_wind_speed,
                                wind_speed_units,
                                air_temperature,
                                temperature_units) -> np.ndarray:
        """
        Vectorized scorch height.

        Parameters
        ----------
        fireline_intensity       : (*S) or scalar
        fireline_intensity_units : scalar int — FirelineIntensityUnitsEnum
        midflame_wind_speed      : (*S) or scalar
        wind_speed_units         : scalar int — SpeedUnitsEnum
        air_temperature          : (*S) or scalar
        temperature_units        : scalar int — TemperatureUnitsEnum

        Returns
        -------
        (*S) ndarray — scorch height in feet
        """
        fi = fireline_intensity_to_base(fireline_intensity, fireline_intensity_units)
        ws_fpm = speed_to_base(midflame_wind_speed, wind_speed_units)
        ws_mph = speed_from_base(ws_fpm, 5)  # 5 = MilesPerHour
        t_f = temp_to_base(air_temperature, temperature_units)
        return calculate_scorch_height(fi, ws_mph, t_f)

    # ------------------------------------------------------------------
    # Mortality
    # ------------------------------------------------------------------

    def calculate_crown_scorch_mortality(self,
                                         scorch_height_ft,
                                         tree_height_ft,
                                         crown_ratio,
                                         dbh_inches,
                                         equation_number_grid) -> dict:
        """
        Vectorized crown scorch mortality.

        Parameters
        ----------
        scorch_height_ft     : (*S) — feet
        tree_height_ft       : (*S) — feet
        crown_ratio          : (*S) — fraction (0–1)
        dbh_inches           : (*S) — inches
        equation_number_grid : (*S) int — mortality equation number per cell

        Returns
        -------
        dict with keys:
            'crown_length_scorch', 'crown_volume_scorch', 'probability_mortality'
        """
        return calculate_crown_scorch_mortality(
            scorch_height_ft, tree_height_ft,
            crown_ratio, dbh_inches,
            equation_number_grid, self._mortality_coeffs,
        )

    # ------------------------------------------------------------------
    # Spotting
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_spotting_from_surface_fire(flame_length_ft,
                                             wind_mph,
                                             cover_height_ft) -> np.ndarray:
        """Vectorized surface fire spotting distance. Returns feet (*S)."""
        return calculate_spotting_from_surface_fire(
            flame_length_ft, wind_mph, cover_height_ft
        )

    @staticmethod
    def calculate_spotting_from_burning_pile(flame_height_ft,
                                             wind_mph,
                                             cover_height_ft) -> np.ndarray:
        """Vectorized burning pile spotting distance. Returns feet (*S)."""
        return calculate_spotting_from_burning_pile(
            flame_height_ft, wind_mph, cover_height_ft
        )

    @staticmethod
    def calculate_spotting_from_torching_trees(dbh_in, height_ft, count,
                                               wind_mph,
                                               cover_height_ft) -> np.ndarray:
        """Vectorized torching-tree spotting distance. Returns feet (*S)."""
        return calculate_spotting_from_torching_trees(
            dbh_in, height_ft, count, wind_mph, cover_height_ft
        )

    # ------------------------------------------------------------------
    # Fire size / shape
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_fire_area(forward_ros, backing_ros, lwr,
                            elapsed_min, is_crown=False) -> np.ndarray:
        """Fire area in square feet (*S)."""
        return calculate_fire_area(forward_ros, backing_ros, lwr,
                                   elapsed_min, is_crown)

    @staticmethod
    def calculate_fire_perimeter(forward_ros, backing_ros, lwr,
                                 elapsed_min, is_crown=False) -> np.ndarray:
        """Fire perimeter in feet (*S)."""
        return calculate_fire_perimeter(forward_ros, backing_ros, lwr,
                                        elapsed_min, is_crown)

    @staticmethod
    def calculate_fire_length(forward_ros, backing_ros,
                              elapsed_min) -> np.ndarray:
        """Fire length (major axis) in feet (*S)."""
        return calculate_fire_length(forward_ros, backing_ros, elapsed_min)

    @staticmethod
    def calculate_fire_width(forward_ros, backing_ros, lwr,
                             elapsed_min) -> np.ndarray:
        """Fire width (minor axis) in feet (*S)."""
        return calculate_fire_width(forward_ros, backing_ros, lwr, elapsed_min)
