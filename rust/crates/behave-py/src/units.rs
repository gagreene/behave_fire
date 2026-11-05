//! Native units marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{scalar, Cell};
use behave_fire::units::*;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "units.Area.to_base" => scalar(
            AreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Area.from_base" => scalar(
            AreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.BasalArea.to_base" => scalar(
            BasalAreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.BasalArea.from_base" => scalar(
            BasalAreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Length.to_base" => scalar(
            LengthUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Length.from_base" => scalar(
            LengthUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Loading.to_base" => scalar(
            LoadingUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Loading.from_base" => scalar(
            LoadingUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Pressure.to_base" => scalar(
            PressureUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Pressure.from_base" => scalar(
            PressureUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.SurfaceAreaToVolume.to_base" => scalar(
            SurfaceAreaToVolumeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.SurfaceAreaToVolume.from_base" => scalar(
            SurfaceAreaToVolumeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Speed.to_base" => scalar(
            SpeedUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Speed.from_base" => scalar(
            SpeedUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Fraction.to_base" => scalar(
            FractionUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Fraction.from_base" => scalar(
            FractionUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Slope.to_base" => scalar(
            SlopeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Slope.from_base" => scalar(
            SlopeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Density.to_base" => scalar(
            DensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Density.from_base" => scalar(
            DensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.HeatOfCombustion.to_base" => scalar(
            HeatOfCombustionUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.HeatOfCombustion.from_base" => scalar(
            HeatOfCombustionUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.HeatSink.to_base" => scalar(
            HeatSinkUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.HeatSink.from_base" => scalar(
            HeatSinkUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.HeatPerUnitArea.to_base" => scalar(
            HeatPerUnitAreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.HeatPerUnitArea.from_base" => scalar(
            HeatPerUnitAreaUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.HeatSourceAndReactionIntensity.to_base" => scalar(
            HeatSourceAndReactionIntensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.HeatSourceAndReactionIntensity.from_base" => scalar(
            HeatSourceAndReactionIntensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.FirelineIntensity.to_base" => scalar(
            FirelineIntensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.FirelineIntensity.from_base" => scalar(
            FirelineIntensityUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Temperature.to_base" => scalar(
            TemperatureUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Temperature.from_base" => scalar(
            TemperatureUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        "units.Time.to_base" => scalar(
            TimeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .to_base(a.n("value", i)?),
        ),
        "units.Time.from_base" => scalar(
            TimeUnit::try_from(a.n("units", i)? as i32)
                .map_err(|e| e.to_string())?
                .from_base(a.n("value", i)?),
        ),
        _ => return Ok(None),
    }))
}
