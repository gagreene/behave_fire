//! Fuse normalized model-unit conversions before preparing borrowed raster views.
use super::fields::Arguments;
use super::units::Convert;
use behave_fire::units::{DensityUnit, FractionUnit, LengthUnit, SlopeUnit, SpeedUnit};

fn convert<T: Convert>(values: &mut Vec<f64>, units: &[f64]) -> Result<(), String> {
    if values.is_empty() || units.is_empty() {
        values.clear();
        return Ok(());
    }
    if units.len() == 1 {
        let unit = T::try_from(units[0] as i32).map_err(|error| error.to_string())?;
        for value in values {
            *value = unit.convert(*value, true);
        }
    } else {
        if values.len() == 1 {
            let scalar = values[0];
            values
                .try_reserve_exact(units.len() - 1)
                .map_err(|error| format!("MemoryError: {error}"))?;
            values.resize(units.len(), scalar);
        }
        if values.len() != units.len() {
            return Err("conversion selector length differs from input".to_owned());
        }
        for (value, code) in values.iter_mut().zip(units) {
            *value = T::try_from(*code as i32)
                .map_err(|error| error.to_string())?
                .convert(*value, true);
        }
    }
    Ok(())
}

pub(crate) fn apply(arguments: &mut Arguments) -> Result<(), String> {
    for (name, family) in &arguments.strings {
        let Some(field) = name.strip_prefix("_convert.") else {
            continue;
        };
        let units = arguments
            .values
            .remove(name)
            .ok_or_else(|| format!("missing conversion selectors for {field}"))?;
        let Some(values) = arguments.values.get_mut(field) else {
            continue;
        };
        match family.as_str() {
            "Density" => convert::<DensityUnit>(values, &units)?,
            "Fraction" => convert::<FractionUnit>(values, &units)?,
            "Length" => convert::<LengthUnit>(values, &units)?,
            "Slope" => convert::<SlopeUnit>(values, &units)?,
            "Speed" => convert::<SpeedUnit>(values, &units)?,
            _ => return Err(format!("unsupported pipeline conversion {family}")),
        }
    }
    if arguments.strings.contains_key("_derive_crown_ratio") {
        let height = arguments
            .values
            .get("canopy_height")
            .ok_or_else(|| "missing canopy_height".to_owned())?;
        let base = arguments
            .values
            .get("canopy_base_height")
            .ok_or_else(|| "missing canopy_base_height".to_owned())?;
        let count = if height.is_empty() || base.is_empty() {
            0
        } else {
            height.len().max(base.len())
        };
        let mut ratio = Vec::new();
        ratio
            .try_reserve_exact(count)
            .map_err(|error| format!("MemoryError: {error}"))?;
        for index in 0..count {
            let height = height
                .get(if height.len() == 1 { 0 } else { index })
                .ok_or_else(|| "invalid canopy_height length".to_owned())?;
            let base = base
                .get(if base.len() == 1 { 0 } else { index })
                .ok_or_else(|| "invalid canopy_base_height length".to_owned())?;
            ratio.push(behave_fire::canopy_crown_ratio(*height, *base));
        }
        arguments.values.insert("crown_ratio".to_owned(), ratio);
    }
    Ok(())
}
