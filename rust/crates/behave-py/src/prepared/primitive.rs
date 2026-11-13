//! Physical float64 and exact code/flag component columns, without dtype payloads.
use super::keys::Field;
use super::selectors::Selected;
use behave_fire::engine::{SurfaceOptions, WafMethod, WindHeight, WindOrientation};
use behave_fire::units::*;
use behave_fire::{GridShape, PerCell};
use std::collections::BTreeMap;

/// Attached-boundary snapshots; text keys are resolved before scientific iteration.
#[derive(Default)]
pub(crate) struct Arguments {
    pub(crate) physical: BTreeMap<String, Vec<f64>>,
    pub(crate) codes: BTreeMap<String, Vec<i64>>,
    pub(crate) flags: BTreeMap<String, Vec<bool>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    Physical,
    Code,
    Flag,
    Mode,
}

fn role(field: Field) -> Role {
    match field {
        Field::FuelModelGrid
        | Field::FuelBedTypeGrid
        | Field::ChargeType
        | Field::DryBulbI
        | Field::RhI
        | Field::SlopeI
        | Field::AspectI
        | Field::ShadingI
        | Field::MonthI
        | Field::ElevI
        | Field::TimeI
        | Field::Units
        | Field::WindSpeedUnits
        | Field::SlopeUnits
        | Field::TempUnits
        | Field::RhUnits
        | Field::OutputUnits => Role::Code,
        Field::IsCrown
        | Field::IsSurfaceFire
        | Field::IsPassiveCrown
        | Field::PIsDefined
        | Field::IbIsDefined
        | Field::ModelIsDefined
        | Field::ModelIsDynamic => Role::Flag,
        Field::WindOrientationMode | Field::WindHeightMode | Field::WafMethod => Role::Mode,
        _ => Role::Physical,
    }
}

fn resolve(name: &str, expected: Role) -> Result<Field, String> {
    if name.starts_with('_') {
        return Err(format!(
            "retired numeric metadata is not a component input: {name}"
        ));
    }
    let field = Field::from_name(name).ok_or_else(|| format!("unknown component input {name}"))?;
    if role(field) != expected {
        return Err(format!("wrong storage role for component input {name}"));
    }
    Ok(field)
}

fn column<'a, T: Copy>(
    values: &'a [T],
    shape: &GridShape,
    name: &str,
) -> Result<PerCell<'a, T>, String> {
    let column = if values.len() == 1 {
        PerCell::Scalar(values[0])
    } else {
        PerCell::Cells(values)
    };
    column.validate(shape, name).map_err(|e| e.to_string())?;
    Ok(column)
}

/// Dense, immutable views of validated scalar or per-cell primitive columns.
pub(crate) struct Prepared<'a> {
    cells: usize,
    physical: [Option<PerCell<'a, f64>>; Field::COUNT],
    codes: [Option<PerCell<'a, i64>>; Field::COUNT],
    flags: [Option<PerCell<'a, bool>>; Field::COUNT],
    options: SurfaceOptions,
    speed: Option<Selected<SpeedUnit>>,
    slope: Option<Selected<SlopeUnit>>,
    temperature: Option<Selected<TemperatureUnit>>,
    humidity: Option<Selected<FractionUnit>>,
    pressure: Option<Selected<PressureUnit>>,
}

impl<'a> Prepared<'a> {
    /// Validate every role, extent, and bounded identifier before exposing row access.
    pub(crate) fn new(arguments: &'a Arguments, shape: &GridShape) -> Result<Self, String> {
        let mut result = Self {
            cells: shape.len(),
            physical: [None; Field::COUNT],
            codes: [None; Field::COUNT],
            flags: [None; Field::COUNT],
            options: SurfaceOptions::default(),
            speed: None,
            slope: None,
            temperature: None,
            humidity: None,
            pressure: None,
        };
        for (name, values) in &arguments.physical {
            let field = resolve(name, Role::Physical)?;
            result.physical[field as usize] = Some(column(values, shape, name)?);
        }
        for (name, values) in &arguments.codes {
            let field = resolve(name, Role::Code)?;
            if values.iter().any(|&v| i32::try_from(v).is_err()) {
                return Err(format!(
                    "component control {name} is outside the int32 range"
                ));
            }
            result.codes[field as usize] = Some(column(values, shape, name)?);
        }
        for (name, values) in &arguments.flags {
            let field = resolve(name, Role::Flag)?;
            result.flags[field as usize] = Some(column(values, shape, name)?);
        }
        result.speed = arguments
            .codes
            .get("wind_speed_units")
            .map(|v| Selected::new(v))
            .transpose()?;
        result.slope = arguments
            .codes
            .get("slope_units")
            .map(|v| Selected::new(v))
            .transpose()?;
        result.temperature = arguments
            .codes
            .get("temp_units")
            .map(|v| Selected::new(v))
            .transpose()?;
        result.humidity = arguments
            .codes
            .get("rh_units")
            .map(|v| Selected::new(v))
            .transpose()?;
        result.pressure = arguments
            .codes
            .get("output_units")
            .map(|v| Selected::new(v))
            .transpose()?;
        Ok(result)
    }
    /// Attach already interpreted scalar wind configuration.
    pub(crate) fn with_options(mut self, options: SurfaceOptions) -> Self {
        self.options = options;
        self
    }
    /// Read a physical component argument using the scalar evaluator convention.
    pub(crate) fn n(&self, field: Field, row: usize) -> Result<f64, String> {
        self.physical(field, row)
    }
    /// Borrow an optional actual physical operand; absent values remain absent.
    pub(crate) fn optional(&self, field: Field, row: usize) -> Option<f64> {
        self.physical[field as usize].and_then(|v| v.get(row).copied())
    }
    /// Report structural presence without evaluating a row, including empty grids.
    pub(crate) fn has(&self, field: Field) -> bool {
        self.physical[field as usize].is_some()
            || self.codes[field as usize].is_some()
            || self.flags[field as usize].is_some()
    }
    pub(crate) fn speed_units(&self, row: usize) -> Result<SpeedUnit, String> {
        Ok(self
            .speed
            .as_ref()
            .map_or(SpeedUnit::FeetPerMinute, |v| v.at(row)))
    }
    pub(crate) fn slope_units(&self, row: usize) -> Result<SlopeUnit, String> {
        Ok(self
            .slope
            .as_ref()
            .map_or(SlopeUnit::Degrees, |v| v.at(row)))
    }
    pub(crate) fn temperature_units(&self, row: usize) -> Result<TemperatureUnit, String> {
        self.temperature
            .as_ref()
            .map(|v| v.at(row))
            .ok_or_else(|| "missing temp_units".into())
    }
    pub(crate) fn humidity_units(&self, row: usize) -> Result<FractionUnit, String> {
        self.humidity
            .as_ref()
            .map(|v| v.at(row))
            .ok_or_else(|| "missing rh_units".into())
    }
    pub(crate) fn pressure_units(&self, row: usize) -> Result<PressureUnit, String> {
        self.pressure
            .as_ref()
            .map(|v| v.at(row))
            .ok_or_else(|| "missing output_units".into())
    }
    pub(crate) fn orientation(&self) -> Result<WindOrientation, String> {
        Ok(self.options.wind_orientation)
    }
    pub(crate) fn height(&self) -> Result<WindHeight, String> {
        Ok(self.options.wind_height)
    }
    pub(crate) fn waf(&self) -> Result<WafMethod, String> {
        Ok(self.options.waf_method)
    }
    pub(crate) fn surface_options(&self) -> Result<SurfaceOptions, String> {
        Ok(self.options)
    }
    pub(crate) fn crown_options(&self) -> SurfaceOptions {
        self.options
    }
    /// Check the actual prepared program's required fields without running a fake cell.
    pub(crate) fn require(&self, fields: &[Field]) -> Result<(), String> {
        for &field in fields {
            let present = match role(field) {
                Role::Physical => self.physical[field as usize].is_some(),
                Role::Code => self.codes[field as usize].is_some(),
                Role::Flag => self.flags[field as usize].is_some(),
                Role::Mode => false,
            };
            if !present {
                return Err(format!(
                    "required component input {} is unbound",
                    field.name()
                ));
            }
        }
        Ok(())
    }
    fn read<T: Copy>(
        &self,
        field: Field,
        row: usize,
        column: Option<PerCell<'_, T>>,
    ) -> Result<T, String> {
        if row >= self.cells {
            return Err(format!(
                "component row {row} is outside {} cells",
                self.cells
            ));
        }
        column.and_then(|v| v.get(row).copied()).ok_or_else(|| {
            format!(
                "component input {} is unbound in this storage role",
                field.name()
            )
        })
    }
    /// Read a physical operand; identifiers and flags never pass through float64.
    pub(crate) fn physical(&self, field: Field, row: usize) -> Result<f64, String> {
        self.read(field, row, self.physical[field as usize])
    }
    /// Read an exact integral control, including iteration limits above 2^53.
    pub(crate) fn code(&self, field: Field, row: usize) -> Result<i64, String> {
        self.read(field, row, self.codes[field as usize])
    }
    /// Read a boolean condition without reconstructing a numeric dtype or mask.
    pub(crate) fn flag(&self, field: Field, row: usize) -> Result<bool, String> {
        self.read(field, row, self.flags[field as usize])
    }
}
