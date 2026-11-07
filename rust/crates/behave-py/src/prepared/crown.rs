//! Crown spatial columns share the native raster pipeline with typed Surface payloads.
use super::fields::PreparedArguments;
use super::keys::Field;
use super::surface::{column, converted};
use behave_fire::engine::{CrownCell, CrownPlan};
use behave_fire::raster::{CrownColumns, CrownRaster};
use behave_fire::{GridShape, PerCell};
use std::borrow::Cow;

pub(crate) struct PreparedCrown<'a> {
    arguments: &'a PreparedArguments<'a>,
    shape: GridShape,
    fields: CrownColumns<'a>,
    wind: Cow<'a, [f64]>,
    slope: Cow<'a, [f64]>,
    plan: CrownPlan,
}

fn optional<'a>(a: &PreparedArguments<'a>, field: Field, default: f64) -> PerCell<'a, f64> {
    a.values(field)
        .map(column)
        .unwrap_or(PerCell::Scalar(default))
}

impl<'a> PreparedCrown<'a> {
    pub(crate) fn new(a: &'a PreparedArguments<'a>, shape: GridShape) -> Result<Self, String> {
        let default = CrownCell::default();
        let fields = CrownColumns {
            m1h: optional(a, Field::M1h, default.m1h),
            m10h: optional(a, Field::M10h, default.m10h),
            m100h: optional(a, Field::M100h, default.m100h),
            mlh: optional(a, Field::Mlh, default.mlh),
            mlw: optional(a, Field::Mlw, default.mlw),
            wind_direction: optional(a, Field::WindDirection, default.wind_direction),
            aspect: optional(a, Field::Aspect, default.aspect),
            canopy_base_height: optional(a, Field::CanopyBaseHeight, default.canopy_base_height),
            canopy_height: optional(a, Field::CanopyHeight, default.canopy_height),
            canopy_bulk_density: optional(a, Field::CanopyBulkDensity, default.canopy_bulk_density),
            moisture_foliar: optional(a, Field::MoistureFoliar, default.moisture_foliar),
            surface_ros: optional(a, Field::SurfaceResultsSpreadRate, 0.0),
            surface_fli: optional(a, Field::SurfaceResultsFirelineIntensity, 0.0),
            surface_hpua: optional(a, Field::SurfaceResultsHeatPerUnitArea, 0.0),
            ..CrownColumns::default()
        };
        let wind = converted(a, Field::WindSpeed, Field::WindSpeedUnits, |i, value| {
            Ok(a.speed_units(i)?.to_base(value))
        })?;
        let slope = converted(a, Field::Slope, Field::SlopeUnits, |i, value| {
            Ok(a.slope_units(i)?.to_base(value))
        })?;
        let result = Self {
            arguments: a,
            shape,
            fields,
            wind,
            slope,
            plan: CrownPlan::new(a.crown_options()),
        };
        result.raster()?;
        Ok(result)
    }

    pub(crate) fn arguments(&self) -> &PreparedArguments<'a> {
        self.arguments
    }

    pub(crate) fn plan(&self) -> &CrownPlan {
        &self.plan
    }

    pub(crate) fn raster(&self) -> Result<CrownRaster<'_>, String> {
        CrownRaster::new(
            self.shape.clone(),
            CrownColumns {
                wind_speed: column(&self.wind),
                slope: column(&self.slope),
                ..self.fields
            },
        )
        .map_err(|error| error.to_string())
    }
}
