//! Decode uniform numeric unit controls once; varying controls remain indexed.
use super::keys::Field;
use behave_fire::units::*;

pub(crate) struct Selector<'a, T> {
    values: Option<&'a [f64]>,
    uniform: Option<Result<T, InvalidUnit>>,
}
impl<'a, T: Copy + TryFrom<i32, Error = InvalidUnit>> Selector<'a, T> {
    pub(crate) fn new(values: Option<&'a [f64]>) -> Self {
        let uniform = values
            .filter(|values| values.len() == 1)
            .map(|values| T::try_from(values[0] as i32));
        Self { values, uniform }
    }
    pub(crate) fn at(&self, index: usize, field: Field) -> Result<T, String> {
        if let Some(unit) = self.uniform {
            return unit.map_err(|error| error.to_string());
        }
        let values = self
            .values
            .ok_or_else(|| format!("missing argument {}", field.name()))?;
        let code = values
            .get(index)
            .copied()
            .ok_or_else(|| format!("invalid length for {}", field.name()))?;
        T::try_from(code as i32).map_err(|error| error.to_string())
    }
}

pub(crate) trait UnitAccess: Copy + TryFrom<i32, Error = InvalidUnit> {
    fn cached(cache: &UnitCache<'_>, index: usize) -> Result<Self, String>;
}
macro_rules! selectors {
    ($($field:ident: $unit:ty),+ $(,)?) => {
        pub(crate) struct UnitCache<'a> { $($field: Selector<'a, $unit>),+ }
        impl<'a> UnitCache<'a> {
            pub(crate) fn new(values: Option<&'a [f64]>) -> Self { Self { $($field: Selector::new(values)),+ } }
        }
        $(impl UnitAccess for $unit {
            fn cached(cache: &UnitCache<'_>, index: usize) -> Result<Self, String> { cache.$field.at(index, Field::Units) }
        })+
    };
}
selectors! {
    area: AreaUnit, basal_area: BasalAreaUnit, length: LengthUnit, loading: LoadingUnit,
    pressure: PressureUnit, savr: SurfaceAreaToVolumeUnit, speed: SpeedUnit, fraction: FractionUnit,
    slope: SlopeUnit, density: DensityUnit, heat_combustion: HeatOfCombustionUnit,
    heat_sink: HeatSinkUnit, hpua: HeatPerUnitAreaUnit, heat_source: HeatSourceAndReactionIntensityUnit,
    intensity: FirelineIntensityUnit, temperature: TemperatureUnit, time: TimeUnit,
}
