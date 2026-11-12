//! Reborrow the owned snapshot for the existing configured model kernel.
use super::super::{run_model_observed, CrownPlan, ModelBatch, SurfacePlan, SurfaceSource};
use super::*;
use crate::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, FuelbedColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};

impl GridSession {
    fn f64_column(&self, field: Field) -> Result<PerCell<'_, f64>, BehaveError> {
        let v = self.get(field)?.physical()?;
        Ok(if v.len() == 1 {
            PerCell::Scalar(v[0])
        } else {
            PerCell::Cells(v)
        })
    }
    fn code_column(&self, field: Field) -> Result<PerCell<'_, i32>, BehaveError> {
        let v = self.get(field)?.code()?;
        Ok(if v.len() == 1 {
            PerCell::Scalar(v[0])
        } else {
            PerCell::Cells(v)
        })
    }
    fn flag_column(&self, field: Field) -> Result<PerCell<'_, bool>, BehaveError> {
        let v = self.get(field)?.flag()?;
        Ok(if v.len() == 1 {
            PerCell::Scalar(v[0])
        } else {
            PerCell::Cells(v)
        })
    }
    fn override_column(&self, field: Field) -> Result<Option<PerCell<'_, f64>>, BehaveError> {
        self.slots[field as usize]
            .persistent
            .as_ref()
            .map(|_| self.f64_column(field))
            .transpose()
    }
    fn surface_tweaks(&self) -> Result<SurfaceTweakColumns<'_>, BehaveError> {
        Ok(SurfaceTweakColumns {
            reaction_intensity: self.override_column(Field::ReactionIntensity)?,
            no_wind_no_slope_spread_rate: self.override_column(Field::NoWindNoSlopeSpreadRate)?,
            midflame_wind_speed: self.override_column(Field::MidflameWindSpeed)?,
            wind_phi: self.override_column(Field::WindPhi)?,
            slope_phi: self.override_column(Field::SlopePhi)?,
            spread_rate: self.override_column(Field::SpreadRate)?,
        })
    }
    fn borrowed_run(&self) -> Result<ModelRun<'_>, BehaveError> {
        let p = |field| self.f64_column(field);
        let tweaks = self.surface_tweaks()?;
        match self.pipeline {
            Pipeline::Surface => {
                let raster = SurfaceRaster::new(
                    self.shape.clone(),
                    SurfaceColumns {
                        fuel_model_grid: self.code_column(Field::FuelModelGrid)?,
                        m1h: p(Field::M1h)?,
                        m10h: p(Field::M10h)?,
                        m100h: p(Field::M100h)?,
                        mlh: p(Field::Mlh)?,
                        mlw: p(Field::Mlw)?,
                        wind_speed: p(Field::WindSpeed)?,
                        wind_direction: p(Field::WindDirection)?,
                        slope: p(Field::Slope)?,
                        aspect: p(Field::Aspect)?,
                        canopy_cover: p(Field::CanopyCover)?,
                        canopy_height: p(Field::CanopyHeight)?,
                        crown_ratio: p(Field::CrownRatio)?,
                        user_waf: self.slots[Field::UserWaf as usize]
                            .active()
                            .map(|_| p(Field::UserWaf))
                            .transpose()?,
                        validity: Some(self.flag_column(Field::Validity)?),
                    },
                )?;
                let source = if self.supplied_fuelbed {
                    SurfaceSource::Fuelbeds(FuelbedColumns {
                        sigma: p(Field::FuelbedSigma)?,
                        bulk_density: p(Field::FuelbedBulkDensity)?,
                        packing_ratio: p(Field::FuelbedPackingRatio)?,
                        relative_packing_ratio: p(Field::FuelbedRelativePackingRatio)?,
                        heat_sink: p(Field::FuelbedHeatSink)?,
                        propagating_flux: p(Field::FuelbedPropagatingFlux)?,
                        w_heat_dead: p(Field::FuelbedWHeatDead)?,
                        w_heat_live: p(Field::FuelbedWHeatLive)?,
                        w_silica_dead: p(Field::FuelbedWSilicaDead)?,
                        w_silica_live: p(Field::FuelbedWSilicaLive)?,
                        w_moist_dead: p(Field::FuelbedWMoistDead)?,
                        w_moist_live: p(Field::FuelbedWMoistLive)?,
                        wfl_dead: p(Field::FuelbedWflDead)?,
                        wfl_live: p(Field::FuelbedWflLive)?,
                        moe_dead: p(Field::FuelbedMoeDead)?,
                        moe_live: p(Field::FuelbedMoeLive)?,
                        frac_dead: p(Field::FuelbedFracDead)?,
                        frac_live: p(Field::FuelbedFracLive)?,
                        depth: p(Field::FuelbedDepth)?,
                        is_defined: self.flag_column(Field::FuelbedDefined)?,
                    })
                } else {
                    SurfaceSource::Fuels(
                        self.fuels
                            .as_ref()
                            .ok_or_else(|| access(Field::FuelModelGrid, "fuel registry missing"))?,
                    )
                };
                Ok(ModelRun::Surface {
                    raster,
                    plan: SurfacePlan::new(self.options),
                    source,
                    tweaks,
                })
            }
            Pipeline::Crown => Ok(ModelRun::Crown {
                raster: CrownRaster::new(
                    self.shape.clone(),
                    CrownColumns {
                        m1h: p(Field::M1h)?,
                        m10h: p(Field::M10h)?,
                        m100h: p(Field::M100h)?,
                        mlh: p(Field::Mlh)?,
                        mlw: p(Field::Mlw)?,
                        wind_speed: p(Field::WindSpeed)?,
                        wind_direction: p(Field::WindDirection)?,
                        slope: p(Field::Slope)?,
                        aspect: p(Field::Aspect)?,
                        canopy_base_height: p(Field::CanopyBaseHeight)?,
                        canopy_height: p(Field::CanopyHeight)?,
                        canopy_bulk_density: p(Field::CanopyBulkDensity)?,
                        moisture_foliar: p(Field::MoistureFoliar)?,
                        surface_ros: p(Field::SurfaceRos)?,
                        surface_fli: p(Field::SurfaceFli)?,
                        surface_hpua: p(Field::SurfaceHpua)?,
                        validity: Some(self.flag_column(Field::Validity)?),
                    },
                )?,
                plan: CrownPlan::new(self.options),
                fuels: self
                    .fuels
                    .as_ref()
                    .ok_or_else(|| access(Field::FuelModelGrid, "fuel registry missing"))?,
                tweaks: CrownTweakColumns {
                    fm10: tweaks,
                    critical_surface_intensity: self
                        .override_column(Field::CrownCriticalSurfaceFireLineIntensity)?,
                    critical_crown_rate: self
                        .override_column(Field::CrownCriticalFireSpreadRate)?,
                    crown_fraction_burned: self.override_column(Field::CrownFractionBurned)?,
                },
            }),
        }
    }
    /// Restart using the configured fused kernel when outputs/overrides permit it.
    /// Custom intermediate requests or arbitrary persistent fields use a fresh field DAG.
    pub fn run_fused_observed<O: RunObserver>(
        &mut self,
        observer: &O,
    ) -> Result<(), ParallelError<BehaveError>> {
        self.reset();
        let request = OutputRequest::new(
            &self
                .request
                .fields()
                .iter()
                .chain(&self.keep)
                .copied()
                .collect::<Vec<_>>(),
        );
        let allowed = match self.pipeline {
            Pipeline::Surface => OutputRequest::surface(),
            Pipeline::Crown => OutputRequest::crown(),
        };
        let standard = [
            Field::ReactionIntensity,
            Field::NoWindNoSlopeSpreadRate,
            Field::MidflameWindSpeed,
            Field::WindPhi,
            Field::SlopePhi,
            Field::SpreadRate,
            Field::CrownCriticalSurfaceFireLineIntensity,
            Field::CrownCriticalFireSpreadRate,
            Field::CrownFractionBurned,
        ];
        if request.fields().iter().any(|f| !allowed.contains(*f))
            || Field::ALL
                .iter()
                .any(|f| self.slots[*f as usize].persistent.is_some() && !standard.contains(f))
        {
            return self.run_remaining_observed(observer);
        }
        let result = {
            let run = self.borrowed_run().map_err(ParallelError::Work)?;
            run_model_observed(&run, &request, self.threads, observer)?
        };
        let columns = match result {
            ModelBatch::Surface(v) => v.into_named_columns(),
            ModelBatch::Crown(v) => v.into_named_columns(),
        };
        for (name, values) in columns {
            let field = Field::from_name(name).expect("core output field");
            self.slots[field as usize].computed = Some(values);
        }
        Ok(())
    }
}
