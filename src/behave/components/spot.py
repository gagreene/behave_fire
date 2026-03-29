"""
Spot component for fire spotting distance calculations.
Complete Python implementation mirroring C++ spot.cpp.
Calculates spotting fire distances from burning piles, surface fires, torching trees, and active crown fires.
"""

import math

try:
    from .behave_units import LengthUnits, SpeedUnits, TimeUnits, FirelineIntensityUnits
except ImportError:
    from behave_units import LengthUnits, SpeedUnits, TimeUnits, FirelineIntensityUnits


class SpotTreeSpecies:
    ENGELMANN_SPRUCE = 0; DOUGLAS_FIR = 1; SUBALPINE_FIR = 2; WESTERN_HEMLOCK = 3
    PONDEROSA_PINE = 4; LODGEPOLE_PINE = 5; WESTERN_WHITE_PINE = 6; GRAND_FIR = 7
    BALSAM_FIR = 8; SLASH_PINE = 9; LONGLEAF_PINE = 10; POND_PINE = 11
    SHORTLEAF_PINE = 12; LOBLOLLY_PINE = 13


class SpotFireLocation:
    MIDSLOPE_WINDWARD = 0; VALLEY_BOTTOM = 1; MIDSLOPE_LEEWARD = 2; RIDGE_TOP = 3


class SpotDownWindCanopyMode:
    CLOSED = 0; OPEN = 1


def _parse_location(loc):
    if isinstance(loc, int):
        return loc
    s = str(loc).upper().replace(' ', '').replace('_', '')
    if 'RIDGE' in s:    return SpotFireLocation.RIDGE_TOP
    if 'VALLEY' in s:   return SpotFireLocation.VALLEY_BOTTOM
    if 'LEEWARD' in s:  return SpotFireLocation.MIDSLOPE_LEEWARD
    return SpotFireLocation.MIDSLOPE_WINDWARD


def _parse_canopy_mode(mode):
    if isinstance(mode, int):
        return mode
    return SpotDownWindCanopyMode.OPEN if 'OPEN' in str(mode).upper() else SpotDownWindCanopyMode.CLOSED


def _parse_species(sp):
    if isinstance(sp, int):
        return sp
    name = str(sp).upper().replace(' ', '_')
    _MAP = {
        'ENGELMANN_SPRUCE': 0, 'DOUGLAS_FIR': 1, 'SUBALPINE_FIR': 2, 'WESTERN_HEMLOCK': 3,
        'PONDEROSA_PINE': 4, 'LODGEPOLE_PINE': 5, 'WESTERN_WHITE_PINE': 6, 'GRAND_FIR': 7,
        'BALSAM_FIR': 8, 'SLASH_PINE': 9, 'LONGLEAF_PINE': 10, 'POND_PINE': 11,
        'SHORTLEAF_PINE': 12, 'LOBLOLLY_PINE': 13,
    }
    return _MAP.get(name, 0)


class CrownFirebrandProcessor:
    """Python port of C++ CrownFirebrandProcessor - Albini's active crown fire spotting."""

    def __init__(self, canopy_ht=0., fire_int=0., wind_speed=0., wind_ht=0., ember_diam=0.5, delta_step=0.2):
        self.m_canopy_ht = canopy_ht    # m
        self.m_fire_int = fire_int      # kW/m
        self.m_wind_speed = wind_speed  # km/h
        self.m_wind_ht = wind_ht        # m
        self.m_ember_diam = ember_diam  # mm
        self.m_delta_step = delta_step  # m
        self.m_is_dirty = True
        self._reset_outputs()

    def _reset_outputs(self):
        self.m_canopy_wind = 0.; self.m_flame_ht = 0.; self.m_loft_ht = 0.; self.m_spot_dist = 0.
        self.m_ang = 0.; self.m_bf = 0.; self.m_drift_x = 0.; self.m_eta = 1.
        self.m_fm = 0.; self.m_hf = 0.; self.m_layer = 0
        self.m_loft_x = 0.; self.m_loft_z = 0.; self.m_qfac = 0.; self.m_qreq = 0.
        self.m_rhof = 0.; self.m_uc = 0.; self.m_uf = 0.; self.m_vf = 0.; self.m_wf = 0.; self.m_wn = 0.

    def _update(self):
        if self.m_is_dirty:
            self._process()
            self.m_is_dirty = False

    def _process(self):
        self._reset_outputs()
        if self.m_canopy_ht > 0.1 and self.m_wind_speed > 0.1 and self.m_fire_int >= 1000.:
            self._canopy_wind()
            self._flame_height()
            self._flame_boundary()
            self._firebrand_loft()
            self._firebrand_drift()

    def _canopy_wind(self):
        factor = 1. + math.log(1. + 2.94 * (self.m_wind_ht / self.m_canopy_ht)) if self.m_canopy_ht > 0.1 else 1.
        self.m_canopy_wind = self.m_wind_speed / (3.6 * factor)

    def _flame_height(self):
        denom = self.m_canopy_ht * self.m_canopy_wind
        if denom > 0.:
            con = 7.791e-03 * self.m_fire_int / denom
            ylo, yhi = 1., math.exp(con)
            for _ in range(2000):
                y = 0.5 * (ylo + yhi)
                test = y * math.log(y)
                if abs(test - con) <= 1e-06:
                    self.m_flame_ht = self.m_canopy_ht * (y - 1.) / 2.94
                    return
                if test >= con: yhi = y
                else: ylo = y
            self.m_flame_ht = self.m_canopy_ht * (0.5 * (ylo + yhi) - 1.) / 2.94

    def _flame_boundary(self):
        self.m_hf = self.m_flame_ht / self.m_canopy_ht
        self.m_wn = math.sqrt(9.82 * self.m_canopy_ht)
        self.m_uc = self.m_canopy_wind / self.m_wn if self.m_wn > 1e-10 else 0.
        self.m_qfac = 0.00838 / (self.m_uc * self.m_uc) if self.m_uc > 1e-10 else 0.
        rfc = 1. + 2.94 * self.m_hf
        log_rfc = math.log(rfc)
        self.m_fm = 0.468 * rfc * log_rfc
        fmuf = 1.3765 * (self.m_hf + rfc * log_rfc * log_rfc)
        self.m_uf = fmuf / self.m_fm if self.m_fm > 1e-10 else 0.
        ctn2f = rfc - 1. + rfc * log_rfc * log_rfc
        tang = 1.40 * self.m_hf / (self.m_uc * math.sqrt(ctn2f)) if self.m_uc > 1e-10 and ctn2f > 1e-10 else 0.
        self.m_ang = math.atan(tang)
        self.m_wf = tang * self.m_uf
        self.m_vf = math.sqrt(self.m_uf * self.m_uf + self.m_wf * self.m_wf)
        self.m_rhof = 0.6
        self.m_bf = self.m_fm / (self.m_rhof * self.m_vf) if self.m_vf > 1e-10 else 0.

    def _firebrand_loft(self):
        dlos = 0.064 * self.m_canopy_ht
        dtop = self.m_ember_diam + dlos
        self.m_eta = dtop / dlos if dlos > 1e-10 else 1.
        zc2 = self.m_hf
        tan_ang = math.tan(self.m_ang)
        xc2 = self.m_hf / tan_ang if abs(tan_ang) > 1e-10 else 1e10
        fmadd = 0.2735 * self.m_fm
        hfarg = 1. + 2.94 * self.m_hf
        fmuadd = 0.3765 * (self.m_hf + hfarg * math.log(hfarg) * math.log(hfarg))
        dmwfac = 2. * self.m_fm / (3. * self.m_uc * self.m_uc) if self.m_uc > 1e-10 else 0.
        tratf = 2. * self.m_fm / 3.
        sing = math.sin(self.m_ang); cosg = math.cos(self.m_ang)
        delx = 0.5 * self.m_bf * sing; delz = 0.5 * self.m_bf * cosg
        x = xc2; z = self.m_hf; v = self.m_vf; w = self.m_wf; fmw = self.m_fm * self.m_wf
        xb_lo = delx; zb_lo = 0.; xb_hi = xc2 + delx; zb_hi = zc2 - delz
        q_lo = 0.5 * self.m_rhof * self.m_wf * self.m_wf; q_hi = q_lo
        self.m_layer = 1
        while True:
            self.m_qreq = self.m_qfac * (zb_hi + self.m_eta)
            if q_hi < self.m_qreq:
                self.m_loft_z = zb_lo; self.m_loft_x = xb_lo
                self.m_loft_ht = self.m_canopy_ht * self.m_loft_z; return
            self.m_layer += 1
            dx = self.m_delta_step * cosg; dz = self.m_delta_step * sing
            x += dx; z += dz
            zarg = 1. + 2.94 * z
            fm = 0.34 * zarg * math.log(zarg) + fmadd
            fmu = z + (0.34 * zarg * math.log(zarg) * math.log(zarg)) + fmuadd
            u = fmu / fm if fm > 1e-10 else 0.
            fmw += (dmwfac / v) * dz if v > 1e-10 else 0.
            w = fmw / fm if fm > 1e-10 else 0.
            v = math.sqrt(u * u + w * w)
            trat = 1. + tratf / fm if fm > 1e-10 else 1.
            b = fm * trat / v if v > 1e-10 else 0.
            sing = w / v if v > 1e-10 else 0.; cosg = u / v if v > 1e-10 else 1.
            delx = 0.5 * b * sing; delz = 0.5 * b * cosg
            q_lo = q_hi; q_hi = 0.5 * w * w / trat if trat > 1e-10 else 0.
            xb_lo = xb_hi; xb_hi = x + delx; zb_lo = zb_hi; zb_hi = z - delz
            if (self.m_layer * self.m_delta_step) > 10000.: return

    def _firebrand_drift(self):
        f0 = 1. + 2.94 * self.m_loft_z
        denom1 = self.m_eta + self.m_loft_z
        f1 = math.sqrt(self.m_eta / denom1) if denom1 > 1e-10 else 0.
        denom2 = self.m_eta - 0.34
        if denom2 <= 0.: denom2 = 1e-10
        f2 = math.sqrt(self.m_eta / denom2) if denom2 > 1e-10 else 0.
        f3 = f2 / f1 if f1 > 1e-10 else 0.
        f2log = math.log((f2 + 1.) / (f2 - 1.)) if f2 > 1. else 0.
        f3log = math.log((f3 + 1.) / (f3 - 1.)) if f3 > 1. else 0.
        f3d = f3 if f3 > 1e-10 else 1e-10
        f = 1. + math.log(f0) - f1 + (f3log - f2log) / f3d
        self.m_drift_x = 10.9 * f * self.m_uc * math.sqrt(self.m_loft_z + self.m_eta)
        self.m_spot_dist = self.m_canopy_ht * (self.m_loft_x + self.m_drift_x)

    def get_firebrand_distance(self):
        self._update(); return self.m_spot_dist  # meters


class SpotInputs:
    def __init__(self):
        self.active_crown_flame_length_ = 0.0; self.dbh_ = 0.0
        self.downwind_cover_height_ = 0.0; self.downwind_canopy_mode_ = SpotDownWindCanopyMode.CLOSED
        self.location_ = SpotFireLocation.RIDGE_TOP
        self.ridge_to_valley_distance_ = 0.0; self.ridge_to_valley_elevation_ = 0.0
        self.wind_speed_at_twenty_feet_ = 0.0; self.burning_pile_flame_height_ = 0.0
        self.surface_flame_length_ = 0.0; self.crown_fireline_intensity_ = 0.0
        self.torching_trees_ = 0; self.tree_height_ = 0.0
        self.tree_species_ = SpotTreeSpecies.ENGELMANN_SPRUCE

    def set_burning_pile_flame_height(self, v, u): self.burning_pile_flame_height_ = LengthUnits.toBaseUnits(v, u)
    def set_dbh(self, v, u): self.dbh_ = LengthUnits.toBaseUnits(v, u)
    def set_downwind_cover_height(self, v, u): self.downwind_cover_height_ = LengthUnits.toBaseUnits(v, u)
    def set_downwind_canopy_mode(self, m): self.downwind_canopy_mode_ = m
    def set_surface_flame_length(self, v, u): self.surface_flame_length_ = LengthUnits.toBaseUnits(v, u)
    def set_active_crown_flame_length(self, v, u): self.active_crown_flame_length_ = LengthUnits.toBaseUnits(v, u)
    def set_crown_fireline_intensity(self, v, u): self.crown_fireline_intensity_ = FirelineIntensityUnits.toBaseUnits(v, u)
    def set_location(self, loc): self.location_ = loc
    def set_ridge_to_valley_distance(self, v, u): self.ridge_to_valley_distance_ = LengthUnits.toBaseUnits(v, u)
    def set_ridge_to_valley_elevation(self, v, u): self.ridge_to_valley_elevation_ = LengthUnits.toBaseUnits(v, u)
    def set_torching_trees(self, n): self.torching_trees_ = n
    def set_tree_height(self, v, u): self.tree_height_ = LengthUnits.toBaseUnits(v, u)
    def set_tree_species(self, sp): self.tree_species_ = sp
    def set_wind_speed_at_twenty_feet(self, v, u): self.wind_speed_at_twenty_feet_ = SpeedUnits.toBaseUnits(v, u)

    def get_burning_pile_flame_height(self, u): return LengthUnits.fromBaseUnits(self.burning_pile_flame_height_, u)
    def get_dbh(self, u): return LengthUnits.fromBaseUnits(self.dbh_, u)
    def get_downwind_cover_height(self, u): return LengthUnits.fromBaseUnits(self.downwind_cover_height_, u)
    def get_down_wind_canopy_mode(self): return self.downwind_canopy_mode_
    def get_surface_flame_length(self, u): return LengthUnits.fromBaseUnits(self.surface_flame_length_, u)
    def get_active_crown_flame_length(self, u): return LengthUnits.fromBaseUnits(self.active_crown_flame_length_, u)
    def get_crown_fireline_intensity(self, u): return FirelineIntensityUnits.fromBaseUnits(self.crown_fireline_intensity_, u)
    def get_location(self): return self.location_
    def get_ridge_to_valley_distance(self, u): return LengthUnits.fromBaseUnits(self.ridge_to_valley_distance_, u)
    def get_ridge_to_valley_elevation(self, u): return LengthUnits.fromBaseUnits(self.ridge_to_valley_elevation_, u)
    def get_torching_trees(self): return self.torching_trees_
    def get_tree_height(self, u): return LengthUnits.fromBaseUnits(self.tree_height_, u)
    def get_tree_species(self): return self.tree_species_
    def get_wind_speed_at_twenty_feet(self, u): return SpeedUnits.fromBaseUnits(self.wind_speed_at_twenty_feet_, u)


class Spot:
    """Spot fire distance calculations. Mirrors C++ Spot class."""

    SPECIES_FLAME_HEIGHT_PARAMETERS = [
        [15.7,0.451],[15.7,0.451],[15.7,0.451],[15.7,0.451],[12.9,0.453],[12.9,0.453],
        [12.9,0.453],[16.5,0.515],[16.5,0.515],[2.71,1.000],[2.71,1.000],[2.71,1.000],
        [2.71,1.000],[2.71,1.000]
    ]
    SPECIES_FLAME_DURATION_PARAMETERS = [
        [12.6,-0.256],[10.7,-0.278],[10.7,-0.278],[6.30,-0.249],[12.6,-0.256],[12.6,-0.256],
        [10.7,-0.278],[10.7,-0.278],[10.7,-0.278],[11.9,-0.389],[11.9,-0.389],[7.91,-0.344],
        [7.91,-0.344],[13.5,-0.544]
    ]
    FIREBRAND_HEIGHT_FACTORS = [[4.24,0.332],[3.64,0.391],[2.78,0.418],[4.70,0.000]]

    def __init__(self):
        self.spot_inputs_ = SpotInputs()
        self.initialize_members()

    def initialize_members(self):
        self.cover_height_used_for_surface_fire_ = 0.0
        self.cover_height_used_for_burning_pile_ = 0.0
        self.cover_height_used_for_torching_trees_ = 0.0
        self.flame_height_for_torching_trees_ = 0.0
        self.flame_ratio_ = 0.0; self.firebrand_drift_ = 0.0; self.flame_duration_ = 0.0
        self.firebrand_height_from_burning_pile_ = 0.0
        self.firebrand_height_from_surface_fire_ = 0.0
        self.firebrand_height_from_torching_trees_ = 0.0
        self.flat_distance_from_burning_pile_ = 0.0
        self.flat_distance_from_surface_fire_ = 0.0
        self.flat_distance_from_torching_trees_ = 0.0
        self.flat_distance_from_active_crown_ = 0.0
        self.mountain_distance_from_burning_pile_ = 0.0
        self.mountain_distance_from_surface_fire_ = 0.0
        self.mountain_distance_from_torching_trees_ = 0.0
        self.mountain_distance_from_active_crown_ = 0.0

    # -- Helpers --

    @staticmethod
    def _critical_cover_ht(fb_ht, cover_ht):
        crit = 2.2 * (fb_ht ** 0.337) - 4.0 if fb_ht > 1e-7 else 0.0
        return cover_ht if cover_ht > crit else crit

    @staticmethod
    def _flat_dist(fb_ht, cover_ht, wind_mph):
        if cover_ht <= 1e-7: return 0.0
        return (0.000718 * wind_mph * math.sqrt(cover_ht)
                * (0.362 + math.sqrt(fb_ht / cover_ht) / 2.0 * math.log(fb_ht / cover_ht)))

    @staticmethod
    def _mtn_dist(flat_mi, location, rtv_mi, rtv_ft):
        mtn = flat_mi
        if rtv_ft > 1e-7 and rtv_mi > 1e-7:
            a1 = flat_mi / rtv_mi
            b1 = rtv_ft / (10.0 * math.pi) / 1000.0
            x = a1
            for _ in range(6):
                x = a1 - b1 * (math.cos(math.pi * x - location * math.pi / 2.0)
                                - math.cos(location * math.pi / 2.0))
            mtn = x * rtv_mi
        return mtn

    def _cover_ht_ft(self):
        ch = self.spot_inputs_.get_downwind_cover_height(LengthUnits.LengthUnitsEnum.Feet)
        return ch * 0.5 if self.spot_inputs_.get_down_wind_canopy_mode() == SpotDownWindCanopyMode.OPEN else ch

    def _tree_ht_ft(self):
        th = self.spot_inputs_.get_tree_height(LengthUnits.LengthUnitsEnum.Feet)
        return th * 0.5 if self.spot_inputs_.get_down_wind_canopy_mode() == SpotDownWindCanopyMode.OPEN else th

    def _tree_ht_m(self):
        return self._tree_ht_ft() * 0.3048

    # -- update_spot_inputs_for_* --

    def update_spot_inputs_for_burning_pile(self,
            location, rtv_dist, rtv_dist_u, rtv_elev, rtv_elev_u,
            cover_ht, cover_ht_u, canopy_mode,
            pile_flame, pile_flame_u, wind_20ft, wind_u):
        si = self.spot_inputs_
        si.set_location(_parse_location(location))
        si.set_ridge_to_valley_distance(rtv_dist, rtv_dist_u)
        si.set_ridge_to_valley_elevation(rtv_elev, rtv_elev_u)
        si.set_downwind_cover_height(cover_ht, cover_ht_u)
        si.set_downwind_canopy_mode(_parse_canopy_mode(canopy_mode))
        si.set_burning_pile_flame_height(pile_flame, pile_flame_u)
        si.set_wind_speed_at_twenty_feet(wind_20ft, wind_u)

    def update_spot_inputs_for_surface_fire(self,
            location, rtv_dist, rtv_dist_u, rtv_elev, rtv_elev_u,
            cover_ht, cover_ht_u, canopy_mode,
            wind_20ft, wind_u, flame_len, flame_len_u):
        si = self.spot_inputs_
        si.set_location(_parse_location(location))
        si.set_ridge_to_valley_distance(rtv_dist, rtv_dist_u)
        si.set_ridge_to_valley_elevation(rtv_elev, rtv_elev_u)
        si.set_downwind_cover_height(cover_ht, cover_ht_u)
        si.set_downwind_canopy_mode(_parse_canopy_mode(canopy_mode))
        si.set_wind_speed_at_twenty_feet(wind_20ft, wind_u)
        si.set_surface_flame_length(flame_len, flame_len_u)

    def update_spot_inputs_for_torching_trees(self,
            location, rtv_dist, rtv_dist_u, rtv_elev, rtv_elev_u,
            cover_ht, cover_ht_u, canopy_mode,
            num_trees, dbh, dbh_u, tree_ht, tree_ht_u,
            species, wind_20ft, wind_u):
        si = self.spot_inputs_
        si.set_location(_parse_location(location))
        si.set_ridge_to_valley_distance(rtv_dist, rtv_dist_u)
        si.set_ridge_to_valley_elevation(rtv_elev, rtv_elev_u)
        si.set_downwind_cover_height(cover_ht, cover_ht_u)
        si.set_downwind_canopy_mode(_parse_canopy_mode(canopy_mode))
        si.set_torching_trees(num_trees)
        si.set_dbh(dbh, dbh_u)
        si.set_tree_height(tree_ht, tree_ht_u)
        si.set_tree_species(_parse_species(species))
        si.set_wind_speed_at_twenty_feet(wind_20ft, wind_u)

    def update_spot_inputs_for_active_crown_fire(self,
            location, rtv_dist, rtv_dist_u, rtv_elev, rtv_elev_u,
            tree_ht, tree_ht_u, canopy_mode,
            wind_20ft, wind_u, crown_flame_len, crown_flame_len_u):
        si = self.spot_inputs_
        si.set_location(_parse_location(location))
        si.set_ridge_to_valley_distance(rtv_dist, rtv_dist_u)
        si.set_ridge_to_valley_elevation(rtv_elev, rtv_elev_u)
        si.set_tree_height(tree_ht, tree_ht_u)
        si.set_downwind_canopy_mode(_parse_canopy_mode(canopy_mode))
        si.set_wind_speed_at_twenty_feet(wind_20ft, wind_u)
        si.set_active_crown_flame_length(crown_flame_len, crown_flame_len_u)

    # -- Individual setters (backward compat) --

    def set_burning_pile_flame_height(self, v, u): self.spot_inputs_.set_burning_pile_flame_height(v, u)
    def set_dbh(self, v, u): self.spot_inputs_.set_dbh(v, u)
    def set_downwind_cover_height(self, v, u): self.spot_inputs_.set_downwind_cover_height(v, u)
    def set_downwind_canopy_mode(self, m): self.spot_inputs_.set_downwind_canopy_mode(_parse_canopy_mode(m))
    def set_flame_length(self, v, u): self.spot_inputs_.set_surface_flame_length(v, u)
    def set_active_crown_flame_length(self, v, u): self.spot_inputs_.set_active_crown_flame_length(v, u)
    def set_fireline_intensity(self, v, u): self.spot_inputs_.set_crown_fireline_intensity(v, u)
    def set_location(self, loc): self.spot_inputs_.set_location(_parse_location(loc))
    def set_ridge_to_valley_distance(self, v, u): self.spot_inputs_.set_ridge_to_valley_distance(v, u)
    def set_ridge_to_valley_elevation(self, v, u): self.spot_inputs_.set_ridge_to_valley_elevation(v, u)
    def set_torching_trees(self, n): self.spot_inputs_.set_torching_trees(n)
    def set_tree_height(self, v, u): self.spot_inputs_.set_tree_height(v, u)
    def set_tree_species(self, sp): self.spot_inputs_.set_tree_species(_parse_species(sp))
    def set_wind_speed_at_twenty_feet(self, v, u): self.spot_inputs_.set_wind_speed_at_twenty_feet(v, u)

    # -- Calculation methods --

    def calculate_spotting_distance_from_burning_pile(self):
        si = self.spot_inputs_
        loc = si.get_location()
        rtv_mi = si.get_ridge_to_valley_distance(LengthUnits.LengthUnitsEnum.Miles)
        rtv_ft = si.get_ridge_to_valley_elevation(LengthUnits.LengthUnitsEnum.Feet)
        ch_ft = self._cover_ht_ft()
        wind_mph = si.get_wind_speed_at_twenty_feet(SpeedUnits.SpeedUnitsEnum.MilesPerHour)
        pile_ft = si.get_burning_pile_flame_height(LengthUnits.LengthUnitsEnum.Feet)
        self.firebrand_height_from_burning_pile_ = 0.0
        self.flat_distance_from_burning_pile_ = 0.0
        self.mountain_distance_from_burning_pile_ = 0.0
        if wind_mph > 1e-7 and pile_ft > 1e-7:
            self.firebrand_height_from_burning_pile_ = 12.2 * pile_ft
            self.cover_height_used_for_burning_pile_ = self._critical_cover_ht(
                self.firebrand_height_from_burning_pile_, ch_ft)
            if self.cover_height_used_for_burning_pile_ > 1e-7:
                flat_mi = self._flat_dist(self.firebrand_height_from_burning_pile_,
                                          self.cover_height_used_for_burning_pile_, wind_mph)
                mtn_mi = self._mtn_dist(flat_mi, loc, rtv_mi, rtv_ft)
                self.flat_distance_from_burning_pile_ = LengthUnits.toBaseUnits(flat_mi, LengthUnits.LengthUnitsEnum.Miles)
                self.mountain_distance_from_burning_pile_ = LengthUnits.toBaseUnits(mtn_mi, LengthUnits.LengthUnitsEnum.Miles)

    def calculate_spotting_distance_from_surface_fire(self):
        si = self.spot_inputs_
        loc = si.get_location()
        rtv_mi = si.get_ridge_to_valley_distance(LengthUnits.LengthUnitsEnum.Miles)
        rtv_ft = si.get_ridge_to_valley_elevation(LengthUnits.LengthUnitsEnum.Feet)
        ch_ft = self._cover_ht_ft()
        wind_mph = si.get_wind_speed_at_twenty_feet(SpeedUnits.SpeedUnitsEnum.MilesPerHour)
        fl_ft = si.get_surface_flame_length(LengthUnits.LengthUnitsEnum.Feet)
        self.firebrand_height_from_surface_fire_ = 0.0
        self.flat_distance_from_surface_fire_ = 0.0
        self.mountain_distance_from_surface_fire_ = 0.0
        self.firebrand_drift_ = 0.0
        if wind_mph > 1e-7 and fl_ft > 1e-7:
            f = 322. * ((0.474 * wind_mph) ** -1.01)
            byrams = (fl_ft / 0.45) ** (1. / 0.46)
            val = f * byrams
            if val > 1e-7:
                self.firebrand_height_from_surface_fire_ = 1.055 * math.sqrt(val)
            self.cover_height_used_for_surface_fire_ = self._critical_cover_ht(
                self.firebrand_height_from_surface_fire_, ch_ft)
            if self.cover_height_used_for_surface_fire_ > 1e-7:
                self.firebrand_drift_ = (0.000278 * wind_mph
                                         * (self.firebrand_height_from_surface_fire_ ** 0.643))
                flat_mi = (self._flat_dist(self.firebrand_height_from_surface_fire_,
                                           self.cover_height_used_for_surface_fire_, wind_mph)
                           + self.firebrand_drift_)
                mtn_mi = self._mtn_dist(flat_mi, loc, rtv_mi, rtv_ft)
                self.flat_distance_from_surface_fire_ = LengthUnits.toBaseUnits(flat_mi, LengthUnits.LengthUnitsEnum.Miles)
                self.mountain_distance_from_surface_fire_ = LengthUnits.toBaseUnits(mtn_mi, LengthUnits.LengthUnitsEnum.Miles)

    def calculate_spotting_distance_from_torching_trees(self):
        si = self.spot_inputs_
        loc = si.get_location()
        rtv_mi = si.get_ridge_to_valley_distance(LengthUnits.LengthUnitsEnum.Miles)
        rtv_ft = si.get_ridge_to_valley_elevation(LengthUnits.LengthUnitsEnum.Feet)
        ch_ft = self._cover_ht_ft()
        wind_mph = si.get_wind_speed_at_twenty_feet(SpeedUnits.SpeedUnitsEnum.MilesPerHour)
        n = si.get_torching_trees()
        dbh_in = si.get_dbh(LengthUnits.LengthUnitsEnum.Inches)
        tree_ft = self._tree_ht_ft()
        sp = si.get_tree_species()
        self.flame_ratio_ = 0.0; self.flame_height_for_torching_trees_ = 0.0
        self.flame_duration_ = 0.0; self.firebrand_height_from_torching_trees_ = 0.0
        self.flat_distance_from_torching_trees_ = 0.0; self.mountain_distance_from_torching_trees_ = 0.0
        if wind_mph > 1e-7 and dbh_in > 1e-7 and n >= 1 and 0 <= sp < 14:
            a_ht, b_ht = self.SPECIES_FLAME_HEIGHT_PARAMETERS[sp]
            a_dur, b_dur = self.SPECIES_FLAME_DURATION_PARAMETERS[sp]
            self.flame_height_for_torching_trees_ = a_ht * (dbh_in ** b_ht) * (n ** 0.4)
            self.flame_ratio_ = (tree_ft / self.flame_height_for_torching_trees_
                                 if self.flame_height_for_torching_trees_ > 1e-7 else 0.0)
            self.flame_duration_ = a_dur * (dbh_in ** b_dur) * (n ** -0.2)
            if self.flame_ratio_ >= 1.0: i = 0
            elif self.flame_ratio_ >= 0.5: i = 1
            elif self.flame_duration_ < 3.5: i = 2
            else: i = 3
            a_fb, b_fb = self.FIREBRAND_HEIGHT_FACTORS[i]
            self.firebrand_height_from_torching_trees_ = (
                a_fb * (self.flame_duration_ ** b_fb) * self.flame_height_for_torching_trees_
                + tree_ft / 2.0)
            self.cover_height_used_for_torching_trees_ = self._critical_cover_ht(
                self.firebrand_height_from_torching_trees_, ch_ft)
            if self.cover_height_used_for_torching_trees_ > 1e-7:
                flat_mi = self._flat_dist(self.firebrand_height_from_torching_trees_,
                                          self.cover_height_used_for_torching_trees_, wind_mph)
                mtn_mi = self._mtn_dist(flat_mi, loc, rtv_mi, rtv_ft)
                self.flat_distance_from_torching_trees_ = LengthUnits.toBaseUnits(flat_mi, LengthUnits.LengthUnitsEnum.Miles)
                self.mountain_distance_from_torching_trees_ = LengthUnits.toBaseUnits(mtn_mi, LengthUnits.LengthUnitsEnum.Miles)

    def calculate_spotting_distance_from_active_crown(self):
        si = self.spot_inputs_
        loc = si.get_location()
        rtv_mi = si.get_ridge_to_valley_distance(LengthUnits.LengthUnitsEnum.Miles)
        rtv_ft = si.get_ridge_to_valley_elevation(LengthUnits.LengthUnitsEnum.Feet)
        tree_ht_m = self._tree_ht_m()
        fire_int_kw_m = si.get_crown_fireline_intensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilowattsPerMeter)
        fl_m = si.get_active_crown_flame_length(LengthUnits.LengthUnitsEnum.Meters)
        if abs(fire_int_kw_m) < 0.01 and fl_m > 0.:
            fire_int_kw_m = (fl_m / 0.0775) ** (1. / 0.46)
        wind_kmh = si.get_wind_speed_at_twenty_feet(SpeedUnits.SpeedUnitsEnum.KilometersPerHour)
        proc = CrownFirebrandProcessor(tree_ht_m, fire_int_kw_m, wind_kmh, 6.096, 0.5)
        flat_m = proc.get_firebrand_distance()
        flat_ft = LengthUnits.toBaseUnits(flat_m, LengthUnits.LengthUnitsEnum.Meters)
        self.flat_distance_from_active_crown_ = flat_ft
        flat_mi = LengthUnits.fromBaseUnits(flat_ft, LengthUnits.LengthUnitsEnum.Miles)
        mtn_mi = self._mtn_dist(flat_mi, loc, rtv_mi, rtv_ft)
        self.mountain_distance_from_active_crown_ = LengthUnits.toBaseUnits(mtn_mi, LengthUnits.LengthUnitsEnum.Miles)

    # -- Output getters --

    def get_cover_height_used_for_burning_pile(self, u):
        return LengthUnits.fromBaseUnits(self.cover_height_used_for_burning_pile_, u)
    def get_cover_height_used_for_surface_fire(self, u):
        return LengthUnits.fromBaseUnits(self.cover_height_used_for_surface_fire_, u)
    def get_cover_height_used_for_torching_trees(self, u):
        return LengthUnits.fromBaseUnits(self.cover_height_used_for_torching_trees_, u)
    def get_flame_height_for_torching_trees(self, u):
        return LengthUnits.fromBaseUnits(self.flame_height_for_torching_trees_, u)
    def get_flame_ratio_for_torching_trees(self): return self.flame_ratio_
    def get_flame_duration_for_torching_trees(self, u):
        return TimeUnits.fromBaseUnits(self.flame_duration_, u)
    def get_max_firebrand_height_from_burning_pile(self, u):
        return LengthUnits.fromBaseUnits(self.firebrand_height_from_burning_pile_, u)
    def get_max_firebrand_height_from_surface_fire(self, u):
        return LengthUnits.fromBaseUnits(self.firebrand_height_from_surface_fire_, u)
    def get_max_firebrand_height_from_torching_trees(self, u):
        return LengthUnits.fromBaseUnits(self.firebrand_height_from_torching_trees_, u)
    def get_max_flat_terrain_spotting_distance_from_burning_pile(self, u):
        return LengthUnits.fromBaseUnits(self.flat_distance_from_burning_pile_, u)
    def get_max_flat_terrain_spotting_distance_from_surface_fire(self, u):
        return LengthUnits.fromBaseUnits(self.flat_distance_from_surface_fire_, u)
    def get_max_flat_terrain_spotting_distance_from_torching_trees(self, u):
        return LengthUnits.fromBaseUnits(self.flat_distance_from_torching_trees_, u)
    def get_max_flat_terrain_spotting_distance_from_active_crown(self, u):
        return LengthUnits.fromBaseUnits(self.flat_distance_from_active_crown_, u)
    def get_max_mountainous_terrain_spotting_distance_from_burning_pile(self, u):
        return LengthUnits.fromBaseUnits(self.mountain_distance_from_burning_pile_, u)
    def get_max_mountainous_terrain_spotting_distance_from_surface_fire(self, u):
        return LengthUnits.fromBaseUnits(self.mountain_distance_from_surface_fire_, u)
    def get_max_mountainous_terrain_spotting_distance_from_torching_trees(self, u):
        return LengthUnits.fromBaseUnits(self.mountain_distance_from_torching_trees_, u)
    def get_max_mountainous_terrain_spotting_distance_from_active_crown(self, u):
        return LengthUnits.fromBaseUnits(self.mountain_distance_from_active_crown_, u)
