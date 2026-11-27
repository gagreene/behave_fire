//! Unit families retain one predecoded typed selector and direct output loop.
use crate::{
    component_columns::Columns,
    operation::Operation,
    prepared::{
        primitive::{Arguments, Prepared},
        selectors::Selected,
        Field,
    },
};
use behave_fire::{units::*, ParallelError};

pub(crate) trait Convert: Copy + TryFrom<i32, Error = InvalidUnit> {
    fn convert(self, value: f64, forward: bool) -> f64;
}
macro_rules! units {
    ($($class:ident: $unit:ty),+ $(,)?) => {
        $(impl Convert for $unit {
            fn convert(self, value: f64, forward: bool) -> f64 {
                if forward { self.to_base(value) } else { self.from_base(value) }
            }
        })+
        pub(crate) fn run(operation: Operation, arguments: &Arguments, prepared: &Prepared<'_>,
            output: &mut Columns, threads: usize) -> Option<Result<(), ParallelError<String>>> {
            Some(match operation.name() {
                $(concat!("units.", stringify!($class), ".to_base") => fill::<$unit>(arguments, prepared, output, threads, true),
                  concat!("units.", stringify!($class), ".from_base") => fill::<$unit>(arguments, prepared, output, threads, false),)+
                _ => return None,
            })
        }
    };
}
units! {
    Area: AreaUnit, BasalArea: BasalAreaUnit, Length: LengthUnit, Loading: LoadingUnit,
    Pressure: PressureUnit, SurfaceAreaToVolume: SurfaceAreaToVolumeUnit, Speed: SpeedUnit,
    Fraction: FractionUnit, Slope: SlopeUnit, Density: DensityUnit, HeatOfCombustion: HeatOfCombustionUnit,
    HeatSink: HeatSinkUnit, HeatPerUnitArea: HeatPerUnitAreaUnit, HeatSourceAndReactionIntensity: HeatSourceAndReactionIntensityUnit,
    FirelineIntensity: FirelineIntensityUnit, Temperature: TemperatureUnit, Time: TimeUnit,
}
fn fill<T: Convert + Send + Sync>(
    arguments: &Arguments,
    prepared: &Prepared<'_>,
    output: &mut Columns,
    threads: usize,
    forward: bool,
) -> Result<(), ParallelError<String>> {
    prepared
        .require(&[Field::Value, Field::Units])
        .map_err(ParallelError::Work)?;
    let selector = Selected::<T>::new(
        arguments
            .codes
            .get("units")
            .ok_or_else(|| ParallelError::Work("missing units".into()))?,
    )
    .map_err(ParallelError::Work)?;
    let values = arguments
        .physical
        .get("value")
        .ok_or_else(|| ParallelError::Work("missing value".into()))?;
    output.fill_physical(threads, |row| {
        selector
            .at(row)
            .convert(values[if values.len() == 1 { 0 } else { row }], forward)
    })
}
