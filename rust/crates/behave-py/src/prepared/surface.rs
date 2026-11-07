//! Prepared native raster views borrow only Rust-owned numeric snapshots.
use super::fields::PreparedArguments;
use super::keys::Field;
use behave_fire::engine::SurfacePlan;
use behave_fire::raster::{SurfaceColumns, SurfaceRaster};
use behave_fire::{GridShape, PerCell};
use std::borrow::Cow;

pub(crate) struct PreparedSurface<'a> {
    shape: GridShape,
    fields: SurfaceColumns<'a>,
    grid: Vec<i32>,
    pub(super) wind: Cow<'a, [f64]>,
    slope: Cow<'a, [f64]>,
    plan: SurfacePlan,
}

pub(super) fn column(values: &[f64]) -> PerCell<'_, f64> {
    if values.len() == 1 {
        PerCell::Scalar(values[0])
    } else {
        PerCell::Cells(values)
    }
}

pub(super) fn required<'a>(
    a: &PreparedArguments<'a>,
    field: Field,
) -> Result<PerCell<'a, f64>, String> {
    a.values(field)
        .map(column)
        .ok_or_else(|| format!("missing argument {}", field.name()))
}

pub(super) fn converted<'a>(
    a: &PreparedArguments<'a>,
    field: Field,
    selector: Field,
    convert: impl Fn(usize, f64) -> Result<f64, String>,
) -> Result<Cow<'a, [f64]>, String> {
    let values = a
        .values(field)
        .ok_or_else(|| format!("missing argument {}", field.name()))?;
    let units = a
        .values(selector)
        .ok_or_else(|| format!("missing argument {}", selector.name()))?;
    if units.len() == 1 && units[0] == 0.0 {
        return Ok(Cow::Borrowed(values));
    }
    let count = if values.is_empty() || units.is_empty() {
        0
    } else {
        values.len().max(units.len())
    };
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(count)
        .map_err(|error| format!("MemoryError: {error}"))?;
    for index in 0..count {
        converted.push(convert(
            index,
            values[if values.len() == 1 { 0 } else { index }],
        )?);
    }
    Ok(Cow::Owned(converted))
}

impl<'a> PreparedSurface<'a> {
    pub(crate) fn new(a: &PreparedArguments<'a>, shape: GridShape) -> Result<Self, String> {
        let grid_values = a
            .values(Field::FuelModelGrid)
            .ok_or_else(|| "missing argument fuel_model_grid".to_owned())?;
        let mut grid = Vec::new();
        grid.try_reserve_exact(grid_values.len())
            .map_err(|error| format!("MemoryError: {error}"))?;
        grid.extend(grid_values.iter().map(|value| *value as i32));
        let wind = converted(a, Field::WindSpeed, Field::WindSpeedUnits, |i, value| {
            Ok(a.speed_units(i)?.to_base(value))
        })?;
        let slope = converted(a, Field::Slope, Field::SlopeUnits, |i, value| {
            Ok(a.slope_units(i)?.to_base(value))
        })?;
        let fields = SurfaceColumns {
            m1h: required(a, Field::M1h)?,
            m10h: required(a, Field::M10h)?,
            m100h: required(a, Field::M100h)?,
            mlh: required(a, Field::Mlh)?,
            mlw: required(a, Field::Mlw)?,
            wind_direction: required(a, Field::WindDirection)?,
            aspect: required(a, Field::Aspect)?,
            canopy_cover: required(a, Field::CanopyCover)?,
            canopy_height: required(a, Field::CanopyHeight)?,
            crown_ratio: required(a, Field::CrownRatio)?,
            user_waf: a.values(Field::UserWaf).map(column),
            ..SurfaceColumns::default()
        };
        let result = Self {
            shape,
            fields,
            grid,
            wind,
            slope,
            plan: SurfacePlan::new(a.surface_options()?),
        };
        result.raster()?;
        Ok(result)
    }

    pub(crate) fn plan(&self) -> &SurfacePlan {
        &self.plan
    }

    pub(crate) fn raster(&self) -> Result<SurfaceRaster<'_>, String> {
        SurfaceRaster::new(
            self.shape.clone(),
            SurfaceColumns {
                fuel_model_grid: if self.grid.len() == 1 {
                    PerCell::Scalar(self.grid[0])
                } else {
                    PerCell::Cells(&self.grid)
                },
                wind_speed: column(&self.wind),
                slope: column(&self.slope),
                ..self.fields
            },
        )
        .map_err(|error| error.to_string())
    }
}
