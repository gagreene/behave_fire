//! Native units marshalling and scalar entry-point dispatch.
use crate::dispatch::{scalar, Cell};
use crate::prepared::{Field, PreparedArguments};
use behave_fire::units::*;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "units.Area.to_base" => scalar(a.unit::<AreaUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Area.from_base" => scalar(a.unit::<AreaUnit>(i)?.from_base(a.n(Field::Value, i)?)),
        "units.BasalArea.to_base" => {
            scalar(a.unit::<BasalAreaUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        "units.BasalArea.from_base" => {
            scalar(a.unit::<BasalAreaUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.Length.to_base" => scalar(a.unit::<LengthUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Length.from_base" => {
            scalar(a.unit::<LengthUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.Loading.to_base" => scalar(a.unit::<LoadingUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Loading.from_base" => {
            scalar(a.unit::<LoadingUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.Pressure.to_base" => {
            scalar(a.unit::<PressureUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        "units.Pressure.from_base" => {
            scalar(a.unit::<PressureUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.SurfaceAreaToVolume.to_base" => scalar(
            a.unit::<SurfaceAreaToVolumeUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        "units.SurfaceAreaToVolume.from_base" => scalar(
            a.unit::<SurfaceAreaToVolumeUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.Speed.to_base" => scalar(a.unit::<SpeedUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Speed.from_base" => scalar(a.unit::<SpeedUnit>(i)?.from_base(a.n(Field::Value, i)?)),
        "units.Fraction.to_base" => {
            scalar(a.unit::<FractionUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        "units.Fraction.from_base" => {
            scalar(a.unit::<FractionUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.Slope.to_base" => scalar(a.unit::<SlopeUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Slope.from_base" => scalar(a.unit::<SlopeUnit>(i)?.from_base(a.n(Field::Value, i)?)),
        "units.Density.to_base" => scalar(a.unit::<DensityUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Density.from_base" => {
            scalar(a.unit::<DensityUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.HeatOfCombustion.to_base" => scalar(
            a.unit::<HeatOfCombustionUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        "units.HeatOfCombustion.from_base" => scalar(
            a.unit::<HeatOfCombustionUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.HeatSink.to_base" => {
            scalar(a.unit::<HeatSinkUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        "units.HeatSink.from_base" => {
            scalar(a.unit::<HeatSinkUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        "units.HeatPerUnitArea.to_base" => scalar(
            a.unit::<HeatPerUnitAreaUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        "units.HeatPerUnitArea.from_base" => scalar(
            a.unit::<HeatPerUnitAreaUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.HeatSourceAndReactionIntensity.to_base" => scalar(
            a.unit::<HeatSourceAndReactionIntensityUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        "units.HeatSourceAndReactionIntensity.from_base" => scalar(
            a.unit::<HeatSourceAndReactionIntensityUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.FirelineIntensity.to_base" => scalar(
            a.unit::<FirelineIntensityUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        "units.FirelineIntensity.from_base" => scalar(
            a.unit::<FirelineIntensityUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.Temperature.to_base" => {
            scalar(a.unit::<TemperatureUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        "units.Temperature.from_base" => scalar(
            a.unit::<TemperatureUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        "units.Time.to_base" => scalar(a.unit::<TimeUnit>(i)?.to_base(a.n(Field::Value, i)?)),
        "units.Time.from_base" => scalar(a.unit::<TimeUnit>(i)?.from_base(a.n(Field::Value, i)?)),
        _ => return Ok(None),
    }))
}
