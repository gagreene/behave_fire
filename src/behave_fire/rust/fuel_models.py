"""FuelModels component for managing fuel model data.
Provides access to standard NFFL fuel models and custom fuel models.

Standard fuel model data follows fuelModels.cpp at firelab/behave f3b2943.
Metric model moisture-of-extinction percentages are stored as fractions here.
  - fuel_bed_depth      : feet
  - moisture_of_extinction_dead : fraction (e.g. 0.25, not 25)
  - heat_of_combustion_* : BTU/lb
  - dead_1h/10h/100h, live_herb/woody : lbs/ftÂ² (converted from tons/acre or tonnes/ha)
  - savr_1h, savr_live_herb, savr_live_woody : ftÂ²/ftÂ³
  - is_dynamic, is_reserved, is_defined : bool
"""
from ._bridge import _native
from ._serialization import NativeState as _NativeState
import copy
from .behave_units import LengthUnits, FractionUnits, HeatOfCombustionUnits, LoadingUnits, SurfaceAreaToVolumeUnits

class FuelModels(_NativeState):
    """FuelModels manages all fuel model data including standard NFFL models and custom models.
    Direct Python translation of C++ FuelModels class.
    """

    def __init__(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        self.fuel_models_ = {}
        self._initialize_standard_fuel_models()

    def _initialize_standard_fuel_models(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        self.fuel_models_ = _native.defaults('fuels')

    def get(self, fuel_model_number):
        """Dictionary-like get; returns None if not found.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return self.fuel_models_.get(fuel_model_number)

    def get_fuel_model(self, fuel_model_number):
        """Alias for get().

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return self.fuel_models_.get(fuel_model_number)

    def __getitem__(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return self.fuel_models_[fuel_model_number]

    def __contains__(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return fuel_model_number in self.fuel_models_

    def set_custom_fuel_model(
            self,
            fuel_model_number,
            code,
            name,
            fuel_bed_depth,
            length_units,
            moisture_of_extinction_dead,
            moisture_units,
            heat_of_combustion_dead,
            heat_of_combustion_live,
            heat_of_combustion_units,
            fuel_load_one_hour,
            fuel_load_ten_hour,
            fuel_load_hundred_hour,
            fuel_load_live_herbaceous,
            fuel_load_live_woody,
            loading_units,
            savr_one_hour,
            savr_live_herbaceous,
            savr_live_woody,
            savr_units,
            is_dynamic,
    ):
        """Set a custom fuel model.  Mirrors C++ setCustomFuelModel():

        - Reserved slots cannot be overwritten (returns ``False``).
        - All numeric inputs are converted to internal base units before storage
          using the supplied unit enum arguments.
        - Base units are: depth=ft, moisture=fraction, heat=BTU/lb,
          load=lb/ftÂ², SAVR=ftÂ²/ftÂ³.

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
        fuel_bed_depth = LengthUnits.toBaseUnits(fuel_bed_depth, length_units)
        moisture_of_extinction_dead = FractionUnits.toBaseUnits(moisture_of_extinction_dead, moisture_units)
        heat_of_combustion_dead = HeatOfCombustionUnits.toBaseUnits(heat_of_combustion_dead, heat_of_combustion_units)
        heat_of_combustion_live = HeatOfCombustionUnits.toBaseUnits(heat_of_combustion_live, heat_of_combustion_units)
        fuel_load_one_hour = LoadingUnits.toBaseUnits(fuel_load_one_hour, loading_units)
        fuel_load_ten_hour = LoadingUnits.toBaseUnits(fuel_load_ten_hour, loading_units)
        fuel_load_hundred_hour = LoadingUnits.toBaseUnits(fuel_load_hundred_hour, loading_units)
        fuel_load_live_herbaceous = LoadingUnits.toBaseUnits(fuel_load_live_herbaceous, loading_units)
        fuel_load_live_woody = LoadingUnits.toBaseUnits(fuel_load_live_woody, loading_units)
        savr_one_hour = SurfaceAreaToVolumeUnits.toBaseUnits(savr_one_hour, savr_units)
        savr_live_herbaceous = SurfaceAreaToVolumeUnits.toBaseUnits(savr_live_herbaceous, savr_units)
        savr_live_woody = SurfaceAreaToVolumeUnits.toBaseUnits(savr_live_woody, savr_units)
        self.fuel_models_[fuel_model_number] = {'code': code[:3], 'name': name, 'fuel_bed_depth': fuel_bed_depth, 'moisture_of_extinction_dead': moisture_of_extinction_dead, 'heat_of_combustion_dead': heat_of_combustion_dead, 'heat_of_combustion_live': heat_of_combustion_live, 'dead_1h': fuel_load_one_hour, 'dead_10h': fuel_load_ten_hour, 'dead_100h': fuel_load_hundred_hour, 'live_herb': fuel_load_live_herbaceous, 'live_woody': fuel_load_live_woody, 'savr_1h': savr_one_hour, 'savr_live_herb': savr_live_herbaceous, 'savr_live_woody': savr_live_woody, 'is_dynamic': is_dynamic, 'is_reserved': False, 'is_defined': True}
        return True

    def clear_custom_fuel_model(self, fuel_model_number):
        """Clear a custom (non-reserved) fuel model.  Mirrors C++ clearCustomFuelModel().

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        existing = self.fuel_models_.get(fuel_model_number)
        if existing and existing.get('is_reserved', False):
            return False
        if fuel_model_number in self.fuel_models_:
            del self.fuel_models_[fuel_model_number]
        return True

    def get_fuel_code(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        return m['code'] if m else ''

    def get_fuel_name(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        return m['name'] if m else ''

    def get_fuelbed_depth(self, fuel_model_number, length_units=None):
        """Return fuel bed depth.

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
        """Return dead fuel moisture of extinction.

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
        """Return heat of combustion for dead fuel.

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
        """Return heat of combustion for live fuel.

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
        """Return 1-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ftÂ² (base unit).
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
        """Return 10-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ftÂ² (base unit).
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
        """Return 100-hr dead fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ftÂ² (base unit).
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
        """Return live herbaceous fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ftÂ² (base unit).
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
        """Return live woody fuel load.

        :param fuel_model_number: Fuel model number to look up.
        :param loading_units: ``LoadingUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in lb/ftÂ² (base unit).
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
        """Return 1-hr dead fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ftÂ²/ftÂ³ (base unit).
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
        """Return live herbaceous fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ftÂ²/ftÂ³ (base unit).
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
        """Return live woody fuel surface-area-to-volume ratio (SAVR).

        :param fuel_model_number: Fuel model number to look up.
        :param savr_units: ``SurfaceAreaToVolumeUnitsEnum`` int for output units.
            ``None`` (default) returns the raw value in ftÂ²/ftÂ³ (base unit).
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
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        return m.get('is_dynamic', False) if m else False

    def is_fuel_model_defined(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        return m.get('is_defined', False) if m else False

    def is_fuel_model_reserved(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        return m.get('is_reserved', False) if m else False

    def is_all_fuel_load_zero(self, fuel_model_number):
        """Execute the explicit native operation.

        :param fuel_model_number: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        m = self.get(fuel_model_number)
        if not m:
            return True
        return not any((m.get(k, 0.0) for k in ('dead_1h', 'dead_10h', 'dead_100h', 'live_herb', 'live_woody')))

    @classmethod
    def from_python(cls, registry):
        """Execute the explicit native operation.

        :param registry: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        result = cls.__new__(cls)
        result.fuel_models_ = copy.deepcopy(registry.fuel_models_)
        return result

    def to_python(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        from ..components.fuel_models import FuelModels as PythonFuelModels
        result = PythonFuelModels.__new__(PythonFuelModels)
        result.fuel_models_ = copy.deepcopy(self.fuel_models_)
        return result
RustFuelModels = FuelModels
