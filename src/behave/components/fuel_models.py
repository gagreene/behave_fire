"""
FuelModels component for managing fuel model data.
Provides access to standard NFFL fuel models and custom fuel models.

All standard fuel model data is a direct translation of fuelModels.cpp.
  - fuel_bed_depth      : feet
  - moisture_of_extinction_dead : fraction (e.g. 0.25, not 25)
  - heat_of_combustion_* : BTU/lb
  - dead_1h/10h/100h, live_herb/woody : lbs/ft²  (using f = 2000/43560)
  - savr_1h, savr_live_herb, savr_live_woody : ft²/ft³
  - is_dynamic, is_reserved, is_defined : bool
"""

try:
    from .behave_units import (
        LengthUnits, FractionUnits, HeatOfCombustionUnits,
        LoadingUnits, SurfaceAreaToVolumeUnits,
    )
except ImportError:
    from behave_units import (
        LengthUnits, FractionUnits, HeatOfCombustionUnits,
        LoadingUnits, SurfaceAreaToVolumeUnits,
    )


class FuelModels:
    """
    FuelModels manages all fuel model data including standard NFFL models and custom models.
    Direct Python translation of C++ FuelModels class.
    """

    def __init__(self):
        self.fuel_models_ = {}
        self._initialize_standard_fuel_models()

    # ------------------------------------------------------------------
    def _initialize_standard_fuel_models(self):
        """
        Populate every standard fuel model record, mirroring
        FuelModels::populateFuelModels() in fuelModels.cpp exactly.
        """
        f = 2000.0 / 43560.0  # tons/acre -> lbs/ft²

        def rec(number, code, name,
                depth, moe, hoc_d, hoc_l,
                d1h, d10h, d100h, lherb, lwood,
                savr1, savr_lh, savr_lw,
                is_dynamic, is_reserved=True, is_defined=True):
            self.fuel_models_[number] = {
                'code': code,
                'name': name,
                'fuel_bed_depth': depth,
                'moisture_of_extinction_dead': moe,   # fraction
                'heat_of_combustion_dead': hoc_d,
                'heat_of_combustion_live': hoc_l,
                'dead_1h':    d1h,
                'dead_10h':   d10h,
                'dead_100h':  d100h,
                'live_herb':  lherb,
                'live_woody': lwood,
                'savr_1h':        savr1,
                'savr_live_herb': savr_lh,
                'savr_live_woody':savr_lw,
                'is_dynamic':   is_dynamic,
                'is_reserved':  is_reserved,
                'is_defined':   is_defined,
            }

        def reserved(number):
            self.fuel_models_[number] = {
                'code': 'RESERVED', 'name': 'Reserved',
                'fuel_bed_depth': 0, 'moisture_of_extinction_dead': 0,
                'heat_of_combustion_dead': 0, 'heat_of_combustion_live': 0,
                'dead_1h': 0, 'dead_10h': 0, 'dead_100h': 0,
                'live_herb': 0, 'live_woody': 0,
                'savr_1h': 0, 'savr_live_herb': 0, 'savr_live_woody': 0,
                'is_dynamic': False, 'is_reserved': True, 'is_defined': False,
            }

        # ---- Original 13 Fuel Models (FM1–FM13) -------------------------
        rec(1,  "FM1",  "Short grass [1]",
            1.0, 0.12, 8000, 8000,
            0.034, 0, 0, 0, 0,
            3500, 1500, 1500, False)
        rec(2,  "FM2",  "Timber grass and understory [2]",
            1.0, 0.15, 8000, 8000,
            0.092, 0.046, 0.023, 0.023, 0,
            3000, 1500, 1500, False)
        rec(3,  "FM3",  "Tall grass [3]",
            2.5, 0.25, 8000, 8000,
            0.138, 0, 0, 0, 0,
            1500, 1500, 1500, False)
        rec(4,  "FM4",  "Chaparral [4]",
            6.0, 0.20, 8000, 8000,
            0.230, 0.184, 0.092, 0, 0.230,
            2000, 1500, 1500, False)
        rec(5,  "FM5",  "Brush [5]",
            2.0, 0.20, 8000, 8000,
            0.046, 0.023, 0, 0, 0.092,
            2000, 1500, 1500, False)
        rec(6,  "FM6",  "Dormant brush, hardwood slash [6]",
            2.5, 0.25, 8000, 8000,
            0.069, 0.115, 0.092, 0, 0,
            1750, 1500, 1500, False)
        rec(7,  "FM7",  "Southern rough [7]",
            2.5, 0.40, 8000, 8000,
            0.052, 0.086, 0.069, 0, 0.017,
            1750, 1500, 1500, False)
        rec(8,  "FM8",  "Short needle litter [8]",
            0.2, 0.30, 8000, 8000,
            0.069, 0.046, 0.115, 0, 0,
            2000, 1500, 1500, False)
        rec(9,  "FM9",  "Long needle or hardwood litter [9]",
            0.2, 0.25, 8000, 8000,
            0.134, 0.019, 0.007, 0, 0,
            2500, 1500, 1500, False)
        rec(10, "FM10", "Timber litter & understory [10]",
            1.0, 0.25, 8000, 8000,
            0.138, 0.092, 0.230, 0, 0.092,
            2000, 1500, 1500, False)
        rec(11, "FM11", "Light logging slash [11]",
            1.0, 0.15, 8000, 8000,
            0.069, 0.207, 0.253, 0, 0,
            1500, 1500, 1500, False)
        rec(12, "FM12", "Medium logging slash [12]",
            2.3, 0.20, 8000, 8000,
            0.184, 0.644, 0.759, 0, 0,
            1500, 1500, 1500, False)
        rec(13, "FM13", "Heavy logging slash [13]",
            3.0, 0.25, 8000, 8000,
            0.322, 1.058, 1.288, 0, 0,
            1500, 1500, 1500, False)

        # 14–89 available for custom models (not populated)

        # ---- Non-burnable (NB) ------------------------------------------
        rec(91, "NB1", "Urban, developed [91]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        rec(92, "NB2", "Snow, ice [92]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        rec(93, "NB3", "Agricultural [93]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        # 94–95 reserved then given records
        reserved(94); reserved(95)
        rec(94, "NB4", "Future standard non-burnable [94]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        rec(95, "NB5", "Future standard non-burnable [95]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        # 96–97 available for custom NB
        rec(98, "NB8", "Open water [98]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)
        rec(99, "NB9", "Bare ground [99]",
            1.0, 0.10, 8000, 8000,
            0, 0, 0, 0, 0, 1500, 1500, 1500, False)

        # ---- Grass (GR) -------------------------------------------------
        # 100 available for custom GR model
        rec(101, "GR1", "Short, sparse, dry climate grass (D)",
            0.4, 0.15, 8000, 8000,
            0.10*f, 0, 0, 0.30*f, 0,
            2200, 2000, 1500, True)
        rec(102, "GR2", "Low load, dry climate grass (D)",
            1.0, 0.15, 8000, 8000,
            0.10*f, 0, 0, 1.0*f, 0,
            2000, 1800, 1500, True)
        rec(103, "GR3", "Low load, very coarse, humid climate grass (D)",
            2.0, 0.30, 8000, 8000,
            0.10*f, 0.40*f, 0, 1.50*f, 0,
            1500, 1300, 1500, True)
        rec(104, "GR4", "Moderate load, dry climate grass (D)",
            2.0, 0.15, 8000, 8000,
            0.25*f, 0, 0, 1.9*f, 0,
            2000, 1800, 1500, True)
        rec(105, "GR5", "Low load, humid climate grass (D)",
            1.5, 0.40, 8000, 8000,
            0.40*f, 0, 0, 2.50*f, 0,
            1800, 1600, 1500, True)
        rec(106, "GR6", "Moderate load, humid climate grass (D)",
            1.5, 0.40, 9000, 9000,
            0.10*f, 0, 0, 3.4*f, 0,
            2200, 2000, 1500, True)
        rec(107, "GR7", "High load, dry climate grass (D)",
            3.0, 0.15, 8000, 8000,
            1.0*f, 0, 0, 5.4*f, 0,
            2000, 1800, 1500, True)
        rec(108, "GR8", "High load, very coarse, humid climate grass (D)",
            4.0, 0.30, 8000, 8000,
            0.5*f, 1.0*f, 0, 7.3*f, 0,
            1500, 1300, 1500, True)
        rec(109, "GR9", "Very high load, humid climate grass (D)",
            5.0, 0.40, 8000, 8000,
            1.0*f, 1.0*f, 0, 9.0*f, 0,
            1800, 1600, 1500, True)
        rec(110, "V-Hb", "Short Gass, < 0.5 m (Dynamic)",
            0.35, 24, 19000, 19000,
            0.3*f, 0, 0, 1.2*f, 0,
            6000, 6000, 6000, True)
        rec(111, "V-Ha", "Tall Grass, > 0.5 m (Dynamic)",
            0.6, 24, 19000, 19000,
            0.5*f, 0.1*f, 0, 2.5*f, 0.3*f,
            4000, 6000, 4000, True)
        # 112 reserved
        reserved(112)
        # 113–119 available for custom grass models

        # ---- Grass-Shrub (GS) -------------------------------------------
        # 120 available for custom GS model
        rec(121, "GS1", "Low load, dry climate grass-shrub (D)",
            0.9, 0.15, 8000, 8000,
            0.2*f, 0, 0, 0.5*f, 0.65*f,
            2000, 1800, 1800, True)
        rec(122, "GS2", "Moderate load, dry climate grass-shrub (D)",
            1.5, 0.15, 8000, 8000,
            0.5*f, 0.5*f, 0, 0.6*f, 1.0*f,
            2000, 1800, 1800, True)
        rec(123, "GS3", "Moderate load, humid climate grass-shrub (D)",
            1.8, 0.40, 8000, 8000,
            0.3*f, 0.25*f, 0, 1.45*f, 1.25*f,
            1800, 1600, 1600, True)
        rec(124, "GS4", "High load, humid climate grass-shrub (D)",
            2.1, 0.40, 8000, 8000,
            1.9*f, 0.3*f, 0.1*f, 3.4*f, 7.1*f,
            1800, 1600, 1600, True)
        # 125–130 reserved
        for i in range(125, 131):
            reserved(i)
        # 131–139 available for custom GS models

        # ---- Shrub (SH) -------------------------------------------------
        # 140 available for custom SH model
        rec(141, "SH1", "Low load, dry climate shrub (D)",
            1.0, 0.15, 8000, 8000,
            0.25*f, 0.25*f, 0, 0.15*f, 1.3*f,
            2000, 1800, 1600, True)
        rec(142, "SH2", "Moderate load, dry climate shrub (S)",
            1.0, 0.15, 8000, 8000,
            1.35*f, 2.4*f, 0.75*f, 0, 3.85*f,
            2000, 1800, 1600, True)
        rec(143, "SH3", "Moderate load, humid climate shrub (S)",
            2.4, 0.40, 8000, 8000,
            0.45*f, 3.0*f, 0, 0, 6.2*f,
            1600, 1800, 1400, True)
        rec(144, "SH4", "Low load, humid climate timber-shrub (S)",
            3.0, 0.30, 8000, 8000,
            0.85*f, 1.15*f, 0.2*f, 0, 2.55*f,
            2000, 1800, 1600, True)
        rec(145, "SH5", "High load, dry climate shrub (S)",
            6.0, 0.15, 8000, 8000,
            3.6*f, 2.1*f, 0, 0, 2.9*f,
            750, 1800, 1600, True)
        rec(146, "SH6", "Low load, humid climate shrub (S)",
            2.0, 0.30, 8000, 8000,
            2.9*f, 1.45*f, 0, 0, 1.4*f,
            750, 1800, 1600, True)
        rec(147, "SH7", "Very high load, dry climate shrub (S)",
            6.0, 0.15, 8000, 8000,
            3.5*f, 5.3*f, 2.2*f, 0, 3.4*f,
            750, 1800, 1600, True)
        rec(148, "SH8", "High load, humid climate shrub (S)",
            3.0, 0.40, 8000, 8000,
            2.05*f, 3.4*f, 0.85*f, 0, 4.35*f,
            750, 1800, 1600, True)
        rec(149, "SH9", "Very high load, humid climate shrub (D)",
            4.4, 0.40, 8000, 8000,
            4.5*f, 2.45*f, 0, 1.55*f, 7.0*f,
            750, 1800, 1500, True)

        # ---- Standard Shrub / California models -------------------------
        rec(150, "SCAL17", "Chamise with Moderate Load Grass, 4 feet (Static)",
            4, 0.20, 8000, 8000,
            1.3*f, 1.0*f, 1.0*f, 2.0*f, 2.0*f,
            640, 2200, 640, False)
        rec(151, "SCAL15", "Chamise with Low Load Grass, 3 feet (Static)",
            3, 0.13, 10000, 10000,
            2.0*f, 3.0*f, 1.0*f, 0.5*f, 2.0*f,
            640, 2200, 640, False)
        rec(152, "SCAL16", "North Slope Ceanothus with Moderate Load Grass (Static)",
            6, 0.15, 8000, 8000,
            2.2*f, 4.8*f, 1.8*f, 3.0*f, 2.8*f,
            500, 1500, 500, False)
        rec(153, "SCAL14", "Manzanita/Scrub Oak with Low Load Grass (Static)",
            3, 0.15, 9211, 9211,
            3.0*f, 4.5*f, 1.1*f, 1.4*f, 5.0*f,
            350, 1500, 250, False)
        rec(154, "SCAL18", "Coastal Sage/Buckwheat Scrub with Low Load Grass (Static)",
            3, 0.25, 9200, 9200,
            5.5*f, 0.8*f, 0.1*f, 0.75*f, 2.5*f,
            640, 1500, 640, False)
        rec(155, "V-MH",
            "Short Green Shrub < 1 m With Grass, Discontinuous (< 1 m) often discontinuous and with grass (Dynamic)",
            0.55, 0.25, 19500, 19500,
            1.0*f, 1.0*f, 0, 1.5*f, 5.5*f,
            4500, 8500, 4000, True)
        rec(156, "V-MMb", "Short Shrub < 1 m, Low Dead Fraction and/or Thick Foliage (Static)",
            0.9, 0.20, 20500, 20500,
            4.0*f, 0.5*f, 0, 0, 7.0*f,
            3000, 3000, 3000, False)
        rec(157, "V-MAb", "Short Shrub < 1 m, High Dead Fraction and/or Thin Fuel (Static)",
            0.5, 0.35, 21000, 21000,
            6.0*f, 0.5*f, 0, 0, 7.5*f,
            4500, 4500, 4500, False)
        rec(158, "V-MMa", "Tall Shrub > 1 m, Low Dead Fraction and/or Thick Foliage (Static)",
            1.7, 0.24, 20500, 20500,
            6.0*f, 4.0*f, 0, 0, 13.0*f,
            2500, 3000, 3000, False)
        rec(159, "V-MAa", "Tall Shrub > 1 m, High Dead Fraction and/or Thin Fuel (Static)",
            1.05, 0.35, 21000, 21000,
            9.5*f, 2.5*f, 0, 0, 14.5*f,
            3500, 4000, 4000, False)

        # ---- Timber and understory (TU) ---------------------------------
        # 160 available for custom TU model
        rec(161, "TU1", "Light load, dry climate timber-grass-shrub (D)",
            0.6, 0.20, 8000, 8000,
            0.2*f, 0.9*f, 1.5*f, 0.2*f, 0.9*f,
            2000, 1800, 1600, True)
        rec(162, "TU2", "Moderate load, humid climate timber-shrub (S)",
            1.0, 0.30, 8000, 8000,
            0.95*f, 1.8*f, 1.25*f, 0, 0.2*f,
            2000, 1800, 1600, True)
        rec(163, "TU3", "Moderate load, humid climate timber-grass-shrub (D)",
            1.3, 0.30, 8000, 8000,
            1.1*f, 0.15*f, 0.25*f, 0.65*f, 1.1*f,
            1800, 1600, 1400, True)
        rec(164, "TU4", "Dwarf conifer understory (S)",
            0.5, 0.12, 8000, 8000,
            4.5*f, 0, 0, 0, 2.0*f,
            2300, 1800, 2000, True)
        rec(165, "TU5", "Very high load, dry climate timber-shrub (S)",
            1.0, 0.25, 8000, 8000,
            4.0*f, 4.0*f, 3.0*f, 0, 3.0*f,
            1500, 1800, 750, True)
        rec(166, "M-EUCd",
            "Discontinuous Litter Eucalyptus Plantation, With or Without Shrub Understory (Static)",
            0.4, 0.26, 21000, 20500,
            1.37*f, 2.89*f, 1.59*f, 0, 1.84*f,
            4500, 4200, 5000, False)
        rec(167, "M-H", "Deciduous or Conifer Litter, Shrub and Herb Understory",
            0.1, 0.30, 20500, 20500,
            2.71*f, 1.0*f, 0, 0.66*f, 0.1*f,
            5500, 8000, 4500, True)
        rec(168, "M-F", "Deciduous or Conifer Litter, Shrub and Fern Understory (Dynamic)",
            0.3, 0.35, 19500, 19500,
            4.5*f, 1.5*f, 0.5*f, 2.35*f, 0.48*f,
            6000, 8000, 4500, True)
        rec(169, "M-CAD", "Deciduous Litter, Shrub Understory (Static)",
            0.63, 0.30, 20000, 20000,
            4.54*f, 1.87*f, 0.61*f, 0, 9.08*f,
            6000, 4921, 5000, False)
        rec(170, "M-ESC", "Sclerophyll Broadleaf Litter, Shrub Understory (Static)",
            0.5, 0.27, 20500, 20500,
            5.65*f, 1.5*f, 0.48*f, 0, 7.89*f,
            5000, 4921, 5500, False)
        rec(171, "M-PIN", "Medium-Long Needle Pine Litter, Shrub Understory (Static)",
            0.5, 0.40, 20500, 21500,
            7.21*f, 3.0*f, 0, 0, 6.89*f,
            5500, 5500, 6000, False)
        rec(172, "M-EUC", "Eucalyptus Litter, Shrub Understory (Static)",
            0.64, 0.32, 21000, 21000,
            8.37*f, 3.81*f, 0, 0, 4.51*f,
            4700, 4200, 5000, False)
        # 173–179 available for custom TU models

        # ---- Timber and litter (TL) -------------------------------------
        # 180 available for custom TL model
        rec(181, "TL1", "Low load, compact conifer litter (S)",
            0.2, 0.30, 8000, 8000,
            1.0*f, 2.2*f, 3.6*f, 0, 0,
            2000, 1800, 1600, True)
        rec(182, "TL2", "Low load broadleaf litter (S)",
            0.2, 0.25, 8000, 8000,
            1.4*f, 2.3*f, 2.2*f, 0, 0,
            2000, 1800, 1600, True)
        rec(183, "TL3", "Moderate load conifer litter (S)",
            0.3, 0.20, 8000, 8000,
            0.5*f, 2.2*f, 2.8*f, 0, 0,
            2000, 1800, 1600, True)
        rec(184, "TL4", "Small downed logs (S)",
            0.4, 0.25, 8000, 8000,
            0.5*f, 1.5*f, 4.2*f, 0, 0,
            2000, 1800, 1600, True)
        rec(185, "TL5", "High load conifer litter (S)",
            0.6, 0.25, 8000, 8000,
            1.15*f, 2.5*f, 4.4*f, 0, 0,
            2000, 1800, 160, True)          # savr_lw = 160, matches C++ exactly
        rec(186, "TL6", "High load broadleaf litter (S)",
            0.3, 0.25, 8000, 8000,
            2.4*f, 1.2*f, 1.2*f, 0, 0,
            2000, 1800, 1600, True)
        rec(187, "TL7", "Large downed logs (S)",
            0.4, 0.25, 8000, 8000,
            0.3*f, 1.4*f, 8.1*f, 0, 0,
            2000, 1800, 1600, True)
        rec(188, "TL8", "Long-needle litter (S)",
            0.3, 0.35, 8000, 8000,
            5.8*f, 1.4*f, 1.1*f, 0, 0,
            1800, 1800, 1600, True)
        rec(189, "TL9", "Very high load broadleaf litter (S)",
            0.6, 0.35, 8000, 8000,
            6.65*f, 3.30*f, 4.15*f, 0, 0,
            1800, 1800, 1600, True)
        rec(190, "F-RAC", "Very Compact Litter, Short Needle Conifers (Static)",
            0.05, 0.28, 20500, 20500,
            3.75*f, 2.0*f, 1.0*f, 0, 1.18*f,
            6500, 4921, 4500, False)
        rec(191, "F-FOL", "Compact Litter, Deciduous or Evergreen Foliage (Static)",
            0.15, 0.25, 20500, 20500,
            2.67*f, 1.27*f, 0.69*f, 0, 1.16*f,
            4500, 5500, 5000, False)
        rec(192, "F-PIN", "Litter from Medium-Long Needle Pine Trees (Static)",
            0.1, 0.45, 20500, 21500,
            6.5*f, 1.5*f, 0, 0, 0,
            5500, 5500, 5500, False)
        rec(193, "F-EUC", "Pure Eucalyptus Litter, No Understory (Static)",
            0.32, 0.26, 21000, 20500,
            4.63*f, 2.96*f, 1.27*f, 0, 1.12*f,
            4200, 4200, 5000, False)
        # 194–199 available for custom TL models

        # ---- Slash and blowdown (SB) ------------------------------------
        # 200 available for custom SB model
        rec(201, "SB1", "Low load activity fuel (S)",
            1.0, 0.25, 8000, 8000,
            1.5*f, 3.0*f, 11.0*f, 0, 0,
            2000, 1800, 1600, True)
        rec(202, "SB2", "Moderate load activity or low load blowdown (S)",
            1.0, 0.25, 8000, 8000,
            4.5*f, 4.25*f, 4.0*f, 0, 0,
            2000, 1800, 1600, True)
        rec(203, "SB3", "High load activity fuel or moderate load blowdown (S)",
            1.2, 0.25, 8000, 8000,
            5.5*f, 2.75*f, 3.0*f, 0, 0,
            2000, 1800, 1600, True)
        rec(204, "SB4", "High load blowdown (S)",
            2.7, 0.25, 8000, 8000,
            5.25*f, 3.5*f, 5.25*f, 0, 0,
            2000, 1800, 1600, True)
        # 205–210 reserved
        for i in range(205, 211):
            reserved(i)
        # 211–219 available for custom SB models
        # 220–256 available for custom models

    # ------------------------------------------------------------------
    def get(self, fuel_model_number):
        """Dictionary-like get; returns None if not found."""
        return self.fuel_models_.get(fuel_model_number)

    def get_fuel_model(self, fuel_model_number):
        """Alias for get()."""
        return self.fuel_models_.get(fuel_model_number)

    def __getitem__(self, fuel_model_number):
        return self.fuel_models_[fuel_model_number]

    def __contains__(self, fuel_model_number):
        return fuel_model_number in self.fuel_models_

    # ------------------------------------------------------------------
    def set_custom_fuel_model(self, fuel_model_number, code, name,
                              fuel_bed_depth, length_units,
                              moisture_of_extinction_dead, moisture_units,
                              heat_of_combustion_dead, heat_of_combustion_live,
                              heat_of_combustion_units,
                              fuel_load_one_hour, fuel_load_ten_hour,
                              fuel_load_hundred_hour,
                              fuel_load_live_herbaceous, fuel_load_live_woody,
                              loading_units,
                              savr_one_hour, savr_live_herbaceous, savr_live_woody,
                              savr_units, is_dynamic):
        """
        Set a custom fuel model.  Mirrors C++ setCustomFuelModel():

        - Reserved slots cannot be overwritten (returns ``False``).
        - All numeric inputs are converted to internal base units before storage
          using the supplied unit enum arguments.
        - Base units are: depth=ft, moisture=fraction, heat=BTU/lb,
          load=lb/ft², SAVR=ft²/ft³.

        :param fuel_model_number: Fuel model slot number to populate.
        :param code: Three-character fuel model code (truncated to 3 chars).
        :param name: Descriptive fuel model name.
        :param fuel_bed_depth: Fuel bed depth, in ``length_units``.
        :param length_units: ``LengthUnitsEnum`` int for ``fuel_bed_depth``
            (e.g. ``LengthUnits.LengthUnitsEnum.Feet``).
        :param moisture_of_extinction_dead: Dead fuel moisture of extinction,
            in ``moisture_units``.
        :param moisture_units: ``FractionUnitsEnum`` int for moisture
            (0=Fraction, 1=Percent).
        :param heat_of_combustion_dead: Heat of combustion for dead fuel,
            in ``heat_of_combustion_units``.
        :param heat_of_combustion_live: Heat of combustion for live fuel,
            in ``heat_of_combustion_units``.
        :param heat_of_combustion_units: ``HeatOfCombustionUnitsEnum`` int
            (0=BtusPerPound, 1=KilojoulesPerKilogram).
        :param fuel_load_one_hour: 1-hr dead fuel load, in ``loading_units``.
        :param fuel_load_ten_hour: 10-hr dead fuel load, in ``loading_units``.
        :param fuel_load_hundred_hour: 100-hr dead fuel load, in ``loading_units``.
        :param fuel_load_live_herbaceous: Live herbaceous fuel load, in ``loading_units``.
        :param fuel_load_live_woody: Live woody fuel load, in ``loading_units``.
        :param loading_units: ``LoadingUnitsEnum`` int for all five fuel loads
            (e.g. ``LoadingUnits.LoadingUnitsEnum.TonsPerAcre``).
        :param savr_one_hour: 1-hr dead SAVR, in ``savr_units``.
        :param savr_live_herbaceous: Live herbaceous SAVR, in ``savr_units``.
        :param savr_live_woody: Live woody SAVR, in ``savr_units``.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for all three
            SAVRs (e.g. ``SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareFeetOverCubicFeet``).
        :param is_dynamic: ``True`` if the model uses dynamic live/dead
            load transfer.
        :return: ``True`` on success; ``False`` if the slot is reserved.
        """
        existing = self.fuel_models_.get(fuel_model_number)
        if existing and existing.get('is_reserved', False):
            return False

        # --- Convert all inputs to internal base units before storage ---
        fuel_bed_depth             = LengthUnits.toBaseUnits(fuel_bed_depth, length_units)
        moisture_of_extinction_dead = FractionUnits.toBaseUnits(moisture_of_extinction_dead, moisture_units)
        heat_of_combustion_dead    = HeatOfCombustionUnits.toBaseUnits(heat_of_combustion_dead, heat_of_combustion_units)
        heat_of_combustion_live    = HeatOfCombustionUnits.toBaseUnits(heat_of_combustion_live, heat_of_combustion_units)
        fuel_load_one_hour         = LoadingUnits.toBaseUnits(fuel_load_one_hour,         loading_units)
        fuel_load_ten_hour         = LoadingUnits.toBaseUnits(fuel_load_ten_hour,         loading_units)
        fuel_load_hundred_hour     = LoadingUnits.toBaseUnits(fuel_load_hundred_hour,     loading_units)
        fuel_load_live_herbaceous  = LoadingUnits.toBaseUnits(fuel_load_live_herbaceous,  loading_units)
        fuel_load_live_woody       = LoadingUnits.toBaseUnits(fuel_load_live_woody,       loading_units)
        savr_one_hour              = SurfaceAreaToVolumeUnits.toBaseUnits(savr_one_hour,          savr_units)
        savr_live_herbaceous       = SurfaceAreaToVolumeUnits.toBaseUnits(savr_live_herbaceous,   savr_units)
        savr_live_woody            = SurfaceAreaToVolumeUnits.toBaseUnits(savr_live_woody,        savr_units)

        self.fuel_models_[fuel_model_number] = {
            'code': code[:3],
            'name': name,
            'fuel_bed_depth':              fuel_bed_depth,
            'moisture_of_extinction_dead': moisture_of_extinction_dead,
            'heat_of_combustion_dead':     heat_of_combustion_dead,
            'heat_of_combustion_live':     heat_of_combustion_live,
            'dead_1h':    fuel_load_one_hour,
            'dead_10h':   fuel_load_ten_hour,
            'dead_100h':  fuel_load_hundred_hour,
            'live_herb':  fuel_load_live_herbaceous,
            'live_woody': fuel_load_live_woody,
            'savr_1h':        savr_one_hour,
            'savr_live_herb': savr_live_herbaceous,
            'savr_live_woody':savr_live_woody,
            'is_dynamic':  is_dynamic,
            'is_reserved': False,
            'is_defined':  True,
        }
        return True

    def clear_custom_fuel_model(self, fuel_model_number):
        """Clear a custom (non-reserved) fuel model.  Mirrors C++ clearCustomFuelModel()."""
        existing = self.fuel_models_.get(fuel_model_number)
        if existing and existing.get('is_reserved', False):
            return False
        if fuel_model_number in self.fuel_models_:
            del self.fuel_models_[fuel_model_number]
        return True

    # ------------------------------------------------------------------
    # Getter methods matching C++ API
    # ------------------------------------------------------------------
    def get_fuel_code(self, fuel_model_number):
        m = self.get(fuel_model_number)
        return m['code'] if m else ''

    def get_fuel_name(self, fuel_model_number):
        m = self.get(fuel_model_number)
        return m['name'] if m else ''

    def get_fuelbed_depth(self, fuel_model_number, length_units=None):
        """
        Return fuel bed depth.

        :param fuel_model_number: Fuel model number to look up.
        :param length_units: ``LengthUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in feet (base unit).
        :return: Fuel bed depth in ``length_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('fuel_bed_depth', 0.0)
        if length_units is not None:
            return LengthUnits.fromBaseUnits(val, length_units)
        return val

    def get_moisture_of_extinction_dead(self, fuel_model_number, moisture_units=None):
        """
        Return dead fuel moisture of extinction.

        :param fuel_model_number: Fuel model number to look up.
        :param moisture_units: ``FractionUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value as a fraction (base unit).
        :return: Moisture of extinction in ``moisture_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('moisture_of_extinction_dead', 0.0)
        if moisture_units is not None:
            return FractionUnits.fromBaseUnits(val, moisture_units)
        return val

    def get_heat_of_combustion_dead(self, fuel_model_number, heat_units=None):
        """
        Return heat of combustion for dead fuel.

        :param fuel_model_number: Fuel model number to look up.
        :param heat_units: ``HeatOfCombustionUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in BTU/lb (base unit).
        :return: Heat of combustion in ``heat_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('heat_of_combustion_dead', 0.0)
        if heat_units is not None:
            return HeatOfCombustionUnits.fromBaseUnits(val, heat_units)
        return val

    def get_heat_of_combustion_live(self, fuel_model_number, heat_units=None):
        """
        Return heat of combustion for live fuel.

        :param fuel_model_number: Fuel model number to look up.
        :param heat_units: ``HeatOfCombustionUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in BTU/lb (base unit).
        :return: Heat of combustion in ``heat_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('heat_of_combustion_live', 0.0)
        if heat_units is not None:
            return HeatOfCombustionUnits.fromBaseUnits(val, heat_units)
        return val

    def get_fuel_load_one_hour(self, fuel_model_number, loading_units=None):
        """
        Return 1-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ft² (base unit).
        :return: 1-hr fuel load in ``loading_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('dead_1h', 0.0)
        if loading_units is not None:
            return LoadingUnits.fromBaseUnits(val, loading_units)
        return val

    def get_fuel_load_ten_hour(self, fuel_model_number, loading_units=None):
        """
        Return 10-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ft² (base unit).
        :return: 10-hr fuel load in ``loading_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('dead_10h', 0.0)
        if loading_units is not None:
            return LoadingUnits.fromBaseUnits(val, loading_units)
        return val

    def get_fuel_load_hundred_hour(self, fuel_model_number, loading_units=None):
        """
        Return 100-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ft² (base unit).
        :return: 100-hr fuel load in ``loading_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('dead_100h', 0.0)
        if loading_units is not None:
            return LoadingUnits.fromBaseUnits(val, loading_units)
        return val

    def get_fuel_load_live_herbaceous(self, fuel_model_number, loading_units=None):
        """
        Return live herbaceous fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ft² (base unit).
        :return: Live herb fuel load in ``loading_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('live_herb', 0.0)
        if loading_units is not None:
            return LoadingUnits.fromBaseUnits(val, loading_units)
        return val

    def get_fuel_load_live_woody(self, fuel_model_number, loading_units=None):
        """
        Return live woody fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ft² (base unit).
        :return: Live woody fuel load in ``loading_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('live_woody', 0.0)
        if loading_units is not None:
            return LoadingUnits.fromBaseUnits(val, loading_units)
        return val

    def get_savr_one_hour(self, fuel_model_number, savr_units=None):
        """
        Return 1-hr dead fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ft²/ft³ (base unit).
        :return: 1-hr SAVR in ``savr_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('savr_1h', 0.0)
        if savr_units is not None:
            return SurfaceAreaToVolumeUnits.fromBaseUnits(val, savr_units)
        return val

    def get_savr_live_herbaceous(self, fuel_model_number, savr_units=None):
        """
        Return live herbaceous fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ft²/ft³ (base unit).
        :return: Live herb SAVR in ``savr_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('savr_live_herb', 0.0)
        if savr_units is not None:
            return SurfaceAreaToVolumeUnits.fromBaseUnits(val, savr_units)
        return val

    def get_savr_live_woody(self, fuel_model_number, savr_units=None):
        """
        Return live woody fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ft²/ft³ (base unit).
        :return: Live woody SAVR in ``savr_units``, or 0.0 if not found.
        """
        m = self.get(fuel_model_number)
        if not m:
            return 0.0
        val = m.get('savr_live_woody', 0.0)
        if savr_units is not None:
            return SurfaceAreaToVolumeUnits.fromBaseUnits(val, savr_units)
        return val

    def get_is_dynamic(self, fuel_model_number):
        m = self.get(fuel_model_number)
        return m.get('is_dynamic', False) if m else False

    def is_fuel_model_defined(self, fuel_model_number):
        m = self.get(fuel_model_number)
        return m.get('is_defined', False) if m else False

    def is_fuel_model_reserved(self, fuel_model_number):
        m = self.get(fuel_model_number)
        return m.get('is_reserved', False) if m else False

    def is_all_fuel_load_zero(self, fuel_model_number):
        m = self.get(fuel_model_number)
        if not m:
            return True
        return not any(m.get(k, 0.0) for k in
                       ('dead_1h', 'dead_10h', 'dead_100h', 'live_herb', 'live_woody'))
