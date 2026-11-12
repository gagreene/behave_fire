//! Native units marshalling and scalar entry-point dispatch.
use crate::column_output::RowWriter;
use crate::dispatch::scalar;
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::units::*;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::UnitsAreaToBase => {
            scalar(out, a.unit::<AreaUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsAreaFromBase => {
            scalar(out, a.unit::<AreaUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsBasalAreaToBase => scalar(
            out,
            a.unit::<BasalAreaUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsBasalAreaFromBase => scalar(
            out,
            a.unit::<BasalAreaUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsLengthToBase => {
            scalar(out, a.unit::<LengthUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsLengthFromBase => scalar(
            out,
            a.unit::<LengthUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsLoadingToBase => scalar(
            out,
            a.unit::<LoadingUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsLoadingFromBase => scalar(
            out,
            a.unit::<LoadingUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsPressureToBase => scalar(
            out,
            a.unit::<PressureUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsPressureFromBase => scalar(
            out,
            a.unit::<PressureUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsSurfaceAreaToVolumeToBase => scalar(
            out,
            a.unit::<SurfaceAreaToVolumeUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsSurfaceAreaToVolumeFromBase => scalar(
            out,
            a.unit::<SurfaceAreaToVolumeUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsSpeedToBase => {
            scalar(out, a.unit::<SpeedUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsSpeedFromBase => scalar(
            out,
            a.unit::<SpeedUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsFractionToBase => scalar(
            out,
            a.unit::<FractionUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsFractionFromBase => scalar(
            out,
            a.unit::<FractionUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsSlopeToBase => {
            scalar(out, a.unit::<SlopeUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsSlopeFromBase => scalar(
            out,
            a.unit::<SlopeUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsDensityToBase => scalar(
            out,
            a.unit::<DensityUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsDensityFromBase => scalar(
            out,
            a.unit::<DensityUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatOfCombustionToBase => scalar(
            out,
            a.unit::<HeatOfCombustionUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatOfCombustionFromBase => scalar(
            out,
            a.unit::<HeatOfCombustionUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatSinkToBase => scalar(
            out,
            a.unit::<HeatSinkUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatSinkFromBase => scalar(
            out,
            a.unit::<HeatSinkUnit>(i)?.from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatPerUnitAreaToBase => scalar(
            out,
            a.unit::<HeatPerUnitAreaUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatPerUnitAreaFromBase => scalar(
            out,
            a.unit::<HeatPerUnitAreaUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatSourceAndReactionIntensityToBase => scalar(
            out,
            a.unit::<HeatSourceAndReactionIntensityUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsHeatSourceAndReactionIntensityFromBase => scalar(
            out,
            a.unit::<HeatSourceAndReactionIntensityUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsFirelineIntensityToBase => scalar(
            out,
            a.unit::<FirelineIntensityUnit>(i)?
                .to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsFirelineIntensityFromBase => scalar(
            out,
            a.unit::<FirelineIntensityUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsTemperatureToBase => scalar(
            out,
            a.unit::<TemperatureUnit>(i)?.to_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsTemperatureFromBase => scalar(
            out,
            a.unit::<TemperatureUnit>(i)?
                .from_base(a.n(Field::Value, i)?),
        ),
        Operation::UnitsTimeToBase => {
            scalar(out, a.unit::<TimeUnit>(i)?.to_base(a.n(Field::Value, i)?))
        }
        Operation::UnitsTimeFromBase => {
            scalar(out, a.unit::<TimeUnit>(i)?.from_base(a.n(Field::Value, i)?))
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
