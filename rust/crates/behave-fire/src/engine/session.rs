//! Owned, typed field slots with explicit injection and recomputation lifetimes.
use super::{
    Field, FieldKind, ModelRun, OutputRequest, Pipeline, PreparedFuels, Stage, SurfaceOptions,
};
use crate::output::{fill_chunks, ColumnValues, NoopObserver, RunObserver, RunPhase};
use crate::{BehaveError, GridShape, ParallelError, PerCell};
#[path = "session_crown.rs"]
mod crown_math;
#[path = "session_fused.rs"]
mod fused;
#[path = "session_surface.rs"]
mod surface_math;

/// Borrow a currently retained column without exposing mutable session storage.
#[derive(Clone, Copy, Debug)]
pub struct FieldView<'a> {
    field: Field,
    values: &'a ColumnValues,
}
impl<'a> FieldView<'a> {
    /// Borrow physical values, rejecting an accessor for another storage kind.
    pub fn physical(self) -> Result<&'a [f64], BehaveError> {
        match self.values {
            ColumnValues::Float64(values) => Ok(values),
            _ => Err(access(self.field, "expected physical storage")),
        }
    }
    /// Borrow exact category values.
    pub fn code(self) -> Result<&'a [i32], BehaveError> {
        match self.values {
            ColumnValues::Int32(values) => Ok(values),
            _ => Err(access(self.field, "expected code storage")),
        }
    }
    /// Borrow exact flag values.
    pub fn flag(self) -> Result<&'a [bool], BehaveError> {
        match self.values {
            ColumnValues::Bool(values) => Ok(values),
            _ => Err(access(self.field, "expected flag storage")),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Slot {
    input: Option<ColumnValues>,
    persistent: Option<ColumnValues>,
    transient: Option<ColumnValues>,
    computed: Option<ColumnValues>,
    freed: bool,
}
impl Slot {
    fn active(&self) -> Option<&ColumnValues> {
        self.transient
            .as_ref()
            .or(self.persistent.as_ref())
            .or(self.input.as_ref())
            .or(self.computed.as_ref())
    }
    fn pinned(&self) -> bool {
        self.transient.is_some() || self.persistent.is_some() || self.input.is_some()
    }
}

fn access(field: Field, reason: &'static str) -> BehaveError {
    BehaveError::InvalidFieldAccess {
        field: field.name().into(),
        reason,
    }
}
fn floats(column: PerCell<'_, f64>) -> ColumnValues {
    ColumnValues::Float64(match column {
        PerCell::Scalar(value) => vec![value],
        PerCell::Cells(values) => values.to_vec(),
    })
}
fn flags(column: PerCell<'_, bool>) -> ColumnValues {
    ColumnValues::Bool(match column {
        PerCell::Scalar(value) => vec![value],
        PerCell::Cells(values) => values.to_vec(),
    })
}
fn codes(column: PerCell<'_, i32>) -> ColumnValues {
    ColumnValues::Int32(match column {
        PerCell::Scalar(value) => vec![value],
        PerCell::Cells(values) => values.to_vec(),
    })
}

/// A complete owned model snapshot; only explicit staged calls allocate field storage.
#[derive(Clone, Debug)]
pub struct GridSession {
    shape: GridShape,
    pipeline: Pipeline,
    options: SurfaceOptions,
    fuels: Option<PreparedFuels>,
    slots: Vec<Slot>,
    request: OutputRequest,
    keep: Vec<Field>,
    threads: usize,
    supplied_fuelbed: bool,
}
impl GridSession {
    /// Borrow the one common logical grid shape.
    pub fn shape(&self) -> &GridShape {
        &self.shape
    }
    /// Return the explicit worker request, including zero for automatic selection.
    pub fn threads(&self) -> usize {
        self.threads
    }
    /// Return the selected scientific pipeline.
    pub fn pipeline(&self) -> Pipeline {
        self.pipeline
    }
    /// Borrow the captured scalar scientific options.
    pub fn options(&self) -> &SurfaceOptions {
        &self.options
    }
    /// Borrow the immutable fuel source, absent for supplied-fuelbed sessions.
    pub fn fuels(&self) -> Option<&PreparedFuels> {
        self.fuels.as_ref()
    }
    /// Snapshot every input, option, fuel source, and persistent model override.
    pub fn new(
        run: &ModelRun<'_>,
        request: OutputRequest,
        keep: &[Field],
        threads: usize,
    ) -> Result<Self, BehaveError> {
        let (shape, pipeline, options, fuels) = match run {
            ModelRun::Crown {
                raster,
                plan,
                fuels,
                ..
            } => (
                raster.shape().clone(),
                Pipeline::Crown,
                *plan.options(),
                Some((*fuels).clone()),
            ),
            ModelRun::Surface {
                raster,
                plan,
                source,
                ..
            } => (
                raster.shape().clone(),
                Pipeline::Surface,
                *plan.options(),
                match source {
                    super::SurfaceSource::Fuels(fuels) => Some((*fuels).clone()),
                    super::SurfaceSource::Fuelbeds(_) => None,
                },
            ),
        };
        let mut result = Self {
            shape,
            pipeline,
            options,
            fuels,
            slots: vec![Slot::default(); Field::COUNT],
            request,
            keep: keep.to_vec(),
            threads,
            supplied_fuelbed: false,
        };
        match run {
            ModelRun::Crown { raster, tweaks, .. } => {
                tweaks.validate(raster.shape())?;
                let c = raster.columns();
                macro_rules! input { ($($name:ident => $field:ident),+ $(,)?) => {$(result.input(Field::$field, floats(c.$name));)+}; }
                input!(m1h=>M1h,m10h=>M10h,m100h=>M100h,mlh=>Mlh,mlw=>Mlw,
                    wind_speed=>WindSpeed,wind_direction=>WindDirection,slope=>Slope,aspect=>Aspect,
                    canopy_base_height=>CanopyBaseHeight,canopy_height=>CanopyHeight,
                    canopy_bulk_density=>CanopyBulkDensity,moisture_foliar=>MoistureFoliar,
                    surface_ros=>SurfaceRos,surface_fli=>SurfaceFli,surface_hpua=>SurfaceHpua);
                result.input(Field::FuelModelGrid, ColumnValues::Int32(vec![10]));
                result.input(
                    Field::Validity,
                    flags(c.validity.unwrap_or(PerCell::Scalar(true))),
                );
                result.capture_surface_tweaks(&tweaks.fm10);
                for (field, value) in [
                    (
                        Field::CrownCriticalSurfaceFireLineIntensity,
                        tweaks.critical_surface_intensity,
                    ),
                    (
                        Field::CrownCriticalFireSpreadRate,
                        tweaks.critical_crown_rate,
                    ),
                    (Field::CrownFractionBurned, tweaks.crown_fraction_burned),
                ] {
                    if let Some(value) = value {
                        result.persistent(field, floats(value))?;
                    }
                }
            }
            ModelRun::Surface {
                raster,
                source,
                tweaks,
                ..
            } => {
                tweaks.validate(raster.shape())?;
                let c = raster.columns();
                macro_rules! input { ($($name:ident => $field:ident),+ $(,)?) => {$(result.input(Field::$field, floats(c.$name));)+}; }
                input!(m1h=>M1h,m10h=>M10h,m100h=>M100h,mlh=>Mlh,mlw=>Mlw,
                    wind_speed=>WindSpeed,wind_direction=>WindDirection,slope=>Slope,aspect=>Aspect,
                    canopy_cover=>CanopyCover,canopy_height=>CanopyHeight,crown_ratio=>CrownRatio);
                result.input(Field::FuelModelGrid, codes(c.fuel_model_grid));
                result.input(
                    Field::Validity,
                    flags(c.validity.unwrap_or(PerCell::Scalar(true))),
                );
                if let Some(value) = c.user_waf {
                    result.input(Field::UserWaf, floats(value));
                }
                if let super::SurfaceSource::Fuelbeds(c) = source {
                    c.validate(raster.shape())?;
                    result.supplied_fuelbed = true;
                    macro_rules! input { ($($name:ident => $field:ident),+ $(,)?) => {$(result.input(Field::$field, floats(c.$name));)+}; }
                    input!(sigma=>FuelbedSigma,bulk_density=>FuelbedBulkDensity,packing_ratio=>FuelbedPackingRatio,
                        relative_packing_ratio=>FuelbedRelativePackingRatio,heat_sink=>FuelbedHeatSink,
                        propagating_flux=>FuelbedPropagatingFlux,w_heat_dead=>FuelbedWHeatDead,w_heat_live=>FuelbedWHeatLive,
                        w_silica_dead=>FuelbedWSilicaDead,w_silica_live=>FuelbedWSilicaLive,
                        w_moist_dead=>FuelbedWMoistDead,w_moist_live=>FuelbedWMoistLive,wfl_dead=>FuelbedWflDead,
                        wfl_live=>FuelbedWflLive,moe_dead=>FuelbedMoeDead,moe_live=>FuelbedMoeLive,
                        frac_dead=>FuelbedFracDead,frac_live=>FuelbedFracLive,depth=>FuelbedDepth);
                    result.input(Field::FuelbedDefined, flags(c.is_defined));
                }
                result.capture_surface_tweaks(tweaks);
            }
        }
        result.validate_outputs(&result.request, &result.keep)?;
        Ok(result)
    }
    fn input(&mut self, field: Field, value: ColumnValues) {
        self.slots[field as usize].input = Some(value);
    }
    fn capture_surface_tweaks(&mut self, tweaks: &crate::raster::SurfaceTweakColumns<'_>) {
        for (field, value) in [
            (Field::ReactionIntensity, tweaks.reaction_intensity),
            (
                Field::NoWindNoSlopeSpreadRate,
                tweaks.no_wind_no_slope_spread_rate,
            ),
            (Field::MidflameWindSpeed, tweaks.midflame_wind_speed),
            (Field::WindPhi, tweaks.wind_phi),
            (Field::SlopePhi, tweaks.slope_phi),
            (Field::SpreadRate, tweaks.spread_rate),
        ] {
            if let Some(value) = value {
                self.slots[field as usize].persistent = Some(floats(value));
            }
        }
    }
    fn validate(&self, field: Field, value: &ColumnValues) -> Result<(), BehaveError> {
        self.validate_field(field)?;
        let (kind, len) = match value {
            ColumnValues::Float64(v) => (FieldKind::Physical, v.len()),
            ColumnValues::Int32(v) => (FieldKind::Code, v.len()),
            ColumnValues::Bool(v) => (FieldKind::Flag, v.len()),
        };
        if field.kind() != kind {
            return Err(access(field, "wrong typed column"));
        }
        if len != 1 {
            self.shape.validate_len(field.name(), len)?;
        }
        Ok(())
    }
    fn validate_field(&self, field: Field) -> Result<(), BehaveError> {
        if self.pipeline == Pipeline::Surface
            && field.producer().is_some_and(|stage| {
                matches!(
                    stage,
                    Stage::CrownSpread
                        | Stage::CrownCritical
                        | Stage::CrownHeat
                        | Stage::CrownRatios
                        | Stage::CrownFraction
                        | Stage::CrownPassive
                        | Stage::CrownFinal
                )
            })
        {
            return Err(access(field, "field belongs to the Crown pipeline"));
        }
        if self.supplied_fuelbed && field.producer() == Some(Stage::Particles) {
            return Err(access(field, "supplied fuelbeds have no particle source"));
        }
        if field.producer().is_none()
            && self.slots[field as usize].input.is_none()
            && !(self.pipeline == Pipeline::Surface && field == Field::UserWaf)
        {
            return Err(access(field, "field is not an input of this model"));
        }
        Ok(())
    }
    fn validate_outputs(&self, request: &OutputRequest, keep: &[Field]) -> Result<(), BehaveError> {
        for &field in request.fields().iter().chain(keep) {
            self.validate_field(field)?;
        }
        Ok(())
    }
    /// Replace the immutable registry snapshot and invalidate its unpinned descendants.
    pub fn set_fuels(&mut self, fuels: PreparedFuels) -> Result<(), BehaveError> {
        if self.supplied_fuelbed {
            return Err(access(
                Field::FuelModelGrid,
                "supplied fuelbeds have no registry source",
            ));
        }
        self.fuels = Some(fuels);
        self.invalidate(Field::FuelModelGrid);
        Ok(())
    }
    /// Change retained outputs atomically; existing fields remain until the next completion.
    pub fn set_outputs(
        &mut self,
        request: OutputRequest,
        keep: &[Field],
    ) -> Result<(), BehaveError> {
        self.validate_outputs(&request, keep)?;
        self.request = request;
        self.keep = keep.to_vec();
        Ok(())
    }
    /// Select workers for subsequent calls; zero delegates to available parallelism.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads;
    }
    /// Clear a transient field injection, retaining any persistent override beneath it.
    pub fn clear_injection(&mut self, field: Field) -> Result<(), BehaveError> {
        self.validate_field(field)?;
        self.slots[field as usize].transient = None;
        self.slots[field as usize].computed = None;
        self.invalidate(field);
        Ok(())
    }
    /// Clear a persistent override without discarding a transient injection.
    pub fn clear_persistent(&mut self, field: Field) -> Result<(), BehaveError> {
        self.validate_field(field)?;
        self.slots[field as usize].persistent = None;
        self.slots[field as usize].computed = None;
        self.invalidate(field);
        Ok(())
    }
    /// Borrow a retained input, injection, override, or computed field.
    pub fn get(&self, field: Field) -> Result<FieldView<'_>, BehaveError> {
        let values = self.slots[field as usize]
            .active()
            .ok_or_else(|| access(field, "field is freed or uncomputed"))?;
        Ok(FieldView { field, values })
    }
    /// Edit an input or pin a transient physical field, including an explicit NaN.
    pub fn set_physical(&mut self, field: Field, values: Vec<f64>) -> Result<(), BehaveError> {
        self.set_column(field, ColumnValues::Float64(values))
    }
    /// Edit an exact identifier or pin a derived classification.
    pub fn set_code(&mut self, field: Field, values: Vec<i32>) -> Result<(), BehaveError> {
        if field == Field::FireType && values.iter().any(|v| !(-1..=3).contains(v)) {
            return Err(access(field, "fire type must be -1, 0, 1, 2, or 3"));
        }
        if self.pipeline == Pipeline::Crown && field == Field::FuelModelGrid {
            return Err(access(
                field,
                "Crown's model-10 source is fixed; edit its registry row instead",
            ));
        }
        self.set_column(field, ColumnValues::Int32(values))
    }
    /// Edit exact validity/definition flags without scientific-number coercion.
    pub fn set_flag(&mut self, field: Field, values: Vec<bool>) -> Result<(), BehaveError> {
        self.set_column(field, ColumnValues::Bool(values))
    }
    fn set_column(&mut self, field: Field, value: ColumnValues) -> Result<(), BehaveError> {
        self.validate(field, &value)?;
        let slot = &mut self.slots[field as usize];
        if slot.input.is_some() || (self.pipeline == Pipeline::Surface && field == Field::UserWaf) {
            slot.input = Some(value);
        } else if field.producer().is_some() {
            slot.transient = Some(value);
        } else {
            return Err(access(field, "field is not an input of this model"));
        }
        slot.computed = None;
        slot.freed = false;
        self.invalidate(field);
        Ok(())
    }
    /// Retain a persistent override across explicit stage runs and resets.
    pub fn persistent(&mut self, field: Field, value: ColumnValues) -> Result<(), BehaveError> {
        self.validate(field, &value)?;
        if let ColumnValues::Int32(values) = &value {
            if field == Field::FireType && values.iter().any(|v| !(-1..=3).contains(v)) {
                return Err(access(field, "fire type must be -1, 0, 1, 2, or 3"));
            }
        }
        if field.producer().is_none() {
            return Err(access(
                field,
                "persistent overrides require a derived field; edit inputs directly",
            ));
        }
        self.slots[field as usize].persistent = Some(value);
        self.slots[field as usize].computed = None;
        self.invalidate(field);
        Ok(())
    }
    fn invalidate(&mut self, changed: Field) {
        let mut dirty = [false; Field::COUNT];
        dirty[changed as usize] = true;
        loop {
            let mut progress = false;
            for &field in Field::ALL {
                if dirty[field as usize] || self.slots[field as usize].pinned() {
                    continue;
                }
                if (changed == Field::Validity && field.producer().is_some())
                    || self.dependencies(field).iter().any(|f| dirty[*f as usize])
                {
                    dirty[field as usize] = true;
                    self.slots[field as usize].computed = None;
                    self.slots[field as usize].freed = false;
                    progress = true;
                }
            }
            if !progress {
                break;
            }
        }
    }
    /// Discard derived state and transient injections while retaining input edits and overrides.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.computed = None;
            slot.transient = None;
            slot.freed = false;
        }
    }
    /// Restart from retained inputs/overrides, clearing transient staged values.
    pub fn run_fused(&mut self) -> Result<(), ParallelError<BehaveError>> {
        self.run_fused_observed(&NoopObserver)
    }
    /// Recompute all fields produced by an explicitly selected stage.
    pub fn run_stage(&mut self, stage: Stage) -> Result<(), ParallelError<BehaveError>> {
        self.run_stage_observed(stage, &NoopObserver)
    }
    /// Run the same stage with observation of actual scientific/write scopes.
    pub fn run_stage_observed<O: RunObserver>(
        &mut self,
        stage: Stage,
        observer: &O,
    ) -> Result<(), ParallelError<BehaveError>> {
        if self.supplied_fuelbed && matches!(stage, Stage::Particles | Stage::Fuelbed) {
            return Err(ParallelError::Work(access(
                Field::FuelbedSigma,
                "supplied fuelbeds have no upstream producer",
            )));
        }
        for &field in stage.writes() {
            self.validate_field(field).map_err(ParallelError::Work)?;
        }
        for &field in stage.writes() {
            let slot = &mut self.slots[field as usize];
            slot.transient = None;
            slot.computed = None;
            self.invalidate(field);
        }
        for &field in stage.writes() {
            self.ensure(field, observer)?;
        }
        Ok(())
    }
    /// Complete requested/kept fields, treating every supplied injection as pinned.
    pub fn run_remaining(&mut self) -> Result<(), ParallelError<BehaveError>> {
        self.run_remaining_observed(&NoopObserver)
    }
    /// Complete requested fields with optional scientific/write scope observation.
    pub fn run_remaining_observed<O: RunObserver>(
        &mut self,
        observer: &O,
    ) -> Result<(), ParallelError<BehaveError>> {
        let fields = self
            .request
            .fields()
            .iter()
            .chain(&self.keep)
            .copied()
            .collect::<Vec<_>>();
        for field in fields {
            self.ensure(field, observer)?;
        }
        for &field in Field::ALL {
            let slot = &mut self.slots[field as usize];
            if !slot.pinned()
                && !self.request.contains(field)
                && !self.keep.contains(&field)
                && slot.computed.is_some()
            {
                slot.computed = None;
                slot.freed = true;
            }
        }
        Ok(())
    }
    fn ensure<O: RunObserver>(
        &mut self,
        field: Field,
        observer: &O,
    ) -> Result<(), ParallelError<BehaveError>> {
        if self.slots[field as usize].active().is_some() {
            return Ok(());
        }
        if field.producer().is_none() {
            return Err(ParallelError::Work(access(
                field,
                "required input is unbound",
            )));
        }
        for dependency in self.dependencies(field) {
            self.ensure(dependency, observer)?;
        }
        let cells = self.shape.len();
        let result = match field.kind() {
            FieldKind::Physical => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(cells)
                    .map_err(|e| ParallelError::Allocation(e.to_string()))?;
                values.resize(cells, f64::NAN);
                fill_chunks(
                    cells,
                    self.threads,
                    &mut [values.as_mut_slice()],
                    &mut [],
                    &mut [],
                    |range, out| {
                        for (local, index) in range.enumerate() {
                            let value = if self.flag_value(Field::Validity, index)? {
                                observer.observe(RunPhase::Kernel, || {
                                    self.evaluate_physical(field, index)
                                })?
                            } else {
                                f64::NAN
                            };
                            observer.observe(RunPhase::Encoder, || out.physical[0][local] = value);
                        }
                        Ok(())
                    },
                )?;
                ColumnValues::Float64(values)
            }
            FieldKind::Code => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(cells)
                    .map_err(|e| ParallelError::Allocation(e.to_string()))?;
                values.resize(cells, -1);
                fill_chunks(
                    cells,
                    self.threads,
                    &mut [],
                    &mut [values.as_mut_slice()],
                    &mut [],
                    |range, out| {
                        for (local, index) in range.enumerate() {
                            let value = if self.flag_value(Field::Validity, index)? {
                                observer.observe(RunPhase::Kernel, || {
                                    self.evaluate_code(field, index)
                                })?
                            } else {
                                -1
                            };
                            observer.observe(RunPhase::Encoder, || out.codes[0][local] = value);
                        }
                        Ok(())
                    },
                )?;
                ColumnValues::Int32(values)
            }
            FieldKind::Flag => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(cells)
                    .map_err(|e| ParallelError::Allocation(e.to_string()))?;
                values.resize(cells, false);
                fill_chunks(
                    cells,
                    self.threads,
                    &mut [],
                    &mut [],
                    &mut [values.as_mut_slice()],
                    |range, out| {
                        for (local, index) in range.enumerate() {
                            let value = if self.flag_value(Field::Validity, index)? {
                                observer.observe(RunPhase::Kernel, || {
                                    self.evaluate_flag(field, index)
                                })?
                            } else {
                                false
                            };
                            observer.observe(RunPhase::Encoder, || out.flags[0][local] = value);
                        }
                        Ok(())
                    },
                )?;
                ColumnValues::Bool(values)
            }
        };
        self.slots[field as usize].computed = Some(result);
        self.slots[field as usize].freed = false;
        Ok(())
    }
    fn physical(&self, field: Field, index: usize) -> Result<f64, BehaveError> {
        let values = self.get(field)?.physical()?;
        Ok(values[if values.len() == 1 { 0 } else { index }])
    }
    fn code_value(&self, field: Field, index: usize) -> Result<i32, BehaveError> {
        let values = self.get(field)?.code()?;
        Ok(values[if values.len() == 1 { 0 } else { index }])
    }
    fn flag_value(&self, field: Field, index: usize) -> Result<bool, BehaveError> {
        let values = self.get(field)?.flag()?;
        Ok(values[if values.len() == 1 { 0 } else { index }])
    }
    fn source_fuel(&self, index: usize) -> Result<&super::PreparedFuel, BehaveError> {
        let fuels = self.fuels.as_ref().ok_or_else(|| {
            access(
                Field::FuelModelGrid,
                "supplied fuelbeds have no particle fuel source",
            )
        })?;
        Ok(fuels.for_grid(self.code_value(Field::FuelModelGrid, index)?))
    }
    fn evaluate_code(&self, field: Field, index: usize) -> Result<i32, BehaveError> {
        match field {
            Field::FireType => Ok(crate::crown::available::classify(
                self.physical(Field::CrownFireTransitionRatio, index)?,
                self.physical(Field::CrownFireActiveRatio, index)?,
            )
            .fire_type),
            _ => Err(access(field, "no code producer")),
        }
    }
    fn evaluate_flag(&self, field: Field, index: usize) -> Result<bool, BehaveError> {
        Ok(match field {
            Field::ParticleDefined => self.source_fuel(index)?.is_defined,
            Field::FuelbedDefined => self.flag_value(Field::ParticleDefined, index)?,
            Field::IsSurfaceFire => matches!(self.code_value(Field::FireType, index)?, 0 | 2),
            Field::IsPassiveCrown => self.code_value(Field::FireType, index)? == 1,
            Field::IsActiveCrown => self.code_value(Field::FireType, index)? == 3,
            _ => return Err(access(field, "no flag producer")),
        })
    }
    fn evaluate_physical(&self, field: Field, index: usize) -> Result<f64, BehaveError> {
        if super::field::SURFACE_FIELDS.contains(&field)
            && (self.pipeline == Pipeline::Surface || field != Field::ReactionIntensity)
            && !self.flag_value(Field::FuelbedDefined, index)?
        {
            return Ok(if field == Field::FireLengthToWidthRatio {
                1.
            } else {
                0.
            });
        }
        let p = |field| self.physical(field, index);
        let number = field as usize;
        if (Field::ParticleLoadDead0 as usize..=Field::ParticleLoadDead4 as usize).contains(&number)
        {
            let fuel = self.source_fuel(index)?;
            return Ok(match number - Field::ParticleLoadDead0 as usize {
                0 => fuel.dead_1h,
                1 => fuel.dead_10h,
                2 => fuel.dead_100h,
                3 => {
                    crate::surface::transfer_herbaceous_load(
                        fuel.live_herb,
                        fuel.is_dynamic,
                        crate::math::np_max(p(Field::Mlh)?, 0.01),
                    )
                    .0
                }
                _ => 0.,
            });
        }
        if (Field::ParticleLoadLive0 as usize..=Field::ParticleLoadLive4 as usize).contains(&number)
        {
            let fuel = self.source_fuel(index)?;
            return Ok(match number - Field::ParticleLoadLive0 as usize {
                0 => {
                    crate::surface::transfer_herbaceous_load(
                        fuel.live_herb,
                        fuel.is_dynamic,
                        crate::math::np_max(p(Field::Mlh)?, 0.01),
                    )
                    .1
                }
                1 => fuel.live_woody,
                _ => 0.,
            });
        }
        if (Field::ParticleSavrDead0 as usize..=Field::ParticleSavrDead4 as usize).contains(&number)
        {
            let fuel = self.source_fuel(index)?;
            return Ok([fuel.savr_1h, 109., 30., fuel.savr_live_herb, 0.]
                [number - Field::ParticleSavrDead0 as usize]);
        }
        if (Field::ParticleSavrLive0 as usize..=Field::ParticleSavrLive4 as usize).contains(&number)
        {
            let fuel = self.source_fuel(index)?;
            return Ok([fuel.savr_live_herb, fuel.savr_live_woody, 0., 0., 0.]
                [number - Field::ParticleSavrLive0 as usize]);
        }
        if (Field::ParticleMoistureDead0 as usize..=Field::ParticleMoistureDead4 as usize)
            .contains(&number)
        {
            return Ok(match number - Field::ParticleMoistureDead0 as usize {
                0 | 3 => crate::math::np_max(p(Field::M1h)?, 0.01),
                1 => crate::math::np_max(p(Field::M10h)?, 0.01),
                2 => crate::math::np_max(p(Field::M100h)?, 0.01),
                _ => 0.,
            });
        }
        if (Field::ParticleMoistureLive0 as usize..=Field::ParticleMoistureLive4 as usize)
            .contains(&number)
        {
            return Ok(match number - Field::ParticleMoistureLive0 as usize {
                0 => crate::math::np_max(p(Field::Mlh)?, 0.01),
                1 => crate::math::np_max(p(Field::Mlw)?, 0.01),
                _ => 0.,
            });
        }
        Ok(match field {
            Field::FuelbedDepth => p(Field::ParticleDepth)?,
            Field::FuelbedMoeDead => p(Field::ParticleMoeDead)?,
            Field::FuelbedBulkDensity => crate::surface::bulk_from_loads::<true>(
                &self.five(Field::ParticleLoadDead0, index)?,
                &self.five(Field::ParticleLoadLive0, index)?,
                p(Field::ParticleDepth)?,
            ),
            Field::FuelbedPackingRatio => crate::surface::packing_from_loads::<true>(
                &self.five(Field::ParticleLoadDead0, index)?,
                &self.five(Field::ParticleLoadLive0, index)?,
                p(Field::ParticleDepth)?,
            ),
            Field::FuelbedWHeatDead => crate::surface::weighted_constant(
                &self.area_side(true, index)?.1,
                p(Field::ParticleHocDead)?,
            ),
            Field::FuelbedWHeatLive => crate::surface::weighted_constant(
                &self.area_side(false, index)?.1,
                p(Field::ParticleHocLive)?,
            ),
            Field::FuelbedWSilicaDead => {
                crate::surface::weighted_constant(&self.area_side(true, index)?.1, 0.01)
            }
            Field::FuelbedWSilicaLive => crate::surface::weighted_values(
                &self.area_side(false, index)?.1,
                &[0.01, 0.01, 0., 0., 0.],
            ),
            Field::FuelbedWMoistDead => crate::surface::weighted_values(
                &self.area_side(true, index)?.1,
                &self.five(Field::ParticleMoistureDead0, index)?,
            ),
            Field::FuelbedWMoistLive => crate::surface::weighted_values(
                &self.area_side(false, index)?.1,
                &self.five(Field::ParticleMoistureLive0, index)?,
            ),
            Field::FuelbedWflDead | Field::FuelbedWflLive => {
                let dead = field == Field::FuelbedWflDead;
                let (load, savr) = if dead {
                    (Field::ParticleLoadDead0, Field::ParticleSavrDead0)
                } else {
                    (Field::ParticleLoadLive0, Field::ParticleSavrLive0)
                };
                let net = self.five(load, index)?.map(|v| v * (1. - 0.0555));
                crate::surface::size_sorted_wfl(
                    &self.five(savr, index)?,
                    &self.area_side(dead, index)?.1,
                    &net,
                )
            }
            Field::FuelbedMoeLive => {
                let sd = self.five(Field::ParticleSavrDead0, index)?;
                let sl = self.five(Field::ParticleSavrLive0, index)?;
                let terms = crate::surface::SavrTerms::new(&sd, &sl);
                let ld = self.five(Field::ParticleLoadDead0, index)?;
                let ll = self.five(Field::ParticleLoadLive0, index)?;
                let md = self.five(Field::ParticleMoistureDead0, index)?;
                let fine_d = (0..5).map(|n| ld[n] * terms.exp_dead[n]).sum();
                let fine_dm = (0..5).map(|n| ld[n] * terms.exp_dead[n] * md[n]).sum();
                let fine_l = (0..5).map(|n| ll[n] * terms.exp_live500[n]).sum();
                crate::surface::live_extinction_from_fines::<true>(
                    fine_d,
                    fine_dm,
                    fine_l,
                    p(Field::FuelbedMoeDead)?,
                )
            }
            Field::FuelbedHeatSink => {
                let terms = crate::surface::SavrTerms::new(
                    &self.five(Field::ParticleSavrDead0, index)?,
                    &self.five(Field::ParticleSavrLive0, index)?,
                );
                let dead = crate::surface::heat_sink_side(
                    p(Field::FuelbedFracDead)?,
                    &self.area_side(true, index)?.1,
                    &self.five(Field::ParticleMoistureDead0, index)?,
                    &terms.exp_dead,
                );
                let live = crate::surface::heat_sink_side(
                    p(Field::FuelbedFracLive)?,
                    &self.area_side(false, index)?.1,
                    &self.five(Field::ParticleMoistureLive0, index)?,
                    &terms.exp_live138,
                );
                (dead + live) * p(Field::FuelbedBulkDensity)?
            }
            Field::ReactionIntensity => {
                crate::surface::reaction_from_inputs(&crate::surface::ReactionInputs {
                    sigma: p(Field::FuelbedSigma)?,
                    relative_packing_ratio: p(Field::FuelbedRelativePackingRatio)?,
                    wfl_dead: p(Field::FuelbedWflDead)?,
                    wfl_live: p(Field::FuelbedWflLive)?,
                    w_heat_dead: p(Field::FuelbedWHeatDead)?,
                    w_heat_live: p(Field::FuelbedWHeatLive)?,
                    w_silica_dead: p(Field::FuelbedWSilicaDead)?,
                    w_silica_live: p(Field::FuelbedWSilicaLive)?,
                    w_moist_dead: p(Field::FuelbedWMoistDead)?,
                    w_moist_live: p(Field::FuelbedWMoistLive)?,
                    moe_dead: p(Field::FuelbedMoeDead)?,
                    moe_live: p(Field::FuelbedMoeLive)?,
                })
            }
            Field::FuelbedFracDead => {
                let (dead, _) = self.area_side(true, index)?;
                let (live, _) = self.area_side(false, index)?;
                crate::surface::dead_area_fraction::<true>(dead, live)
            }
            Field::FuelbedFracLive => 1. - p(Field::FuelbedFracDead)?,
            Field::FuelbedSigma => {
                let (_, wd) = self.area_side(true, index)?;
                let (_, wl) = self.area_side(false, index)?;
                crate::surface::mixture_sigma(
                    p(Field::FuelbedFracDead)?,
                    p(Field::FuelbedFracLive)?,
                    &wd,
                    &self.five(Field::ParticleSavrDead0, index)?,
                    &wl,
                    &self.five(Field::ParticleSavrLive0, index)?,
                )
            }
            Field::ParticleHocDead => self.source_fuel(index)?.heat_of_combustion_dead,
            Field::ParticleHocLive => self.source_fuel(index)?.heat_of_combustion_live,
            Field::ParticleDepth => self.source_fuel(index)?.fuel_bed_depth,
            Field::ParticleMoeDead => self.source_fuel(index)?.moisture_of_extinction_dead,
            Field::FuelbedRelativePackingRatio => {
                crate::surface::relative_packing_from_sigma::<true>(
                    p(Field::FuelbedSigma)?,
                    p(Field::FuelbedPackingRatio)?,
                )
            }
            Field::FuelbedPropagatingFlux => crate::surface::propagating_flux_from_sigma::<true>(
                p(Field::FuelbedSigma)?,
                p(Field::FuelbedPackingRatio)?,
            ),
            Field::CrownFuelLoad => crate::crown::calculate_canopy_fuel_load(
                p(Field::CanopyBaseHeight)?,
                p(Field::CanopyHeight)?,
                p(Field::CanopyBulkDensity)?,
            ),
            Field::CanopyHeatPerUnitArea => {
                crate::crown::calculate_canopy_heat_per_unit_area(p(Field::CrownFuelLoad)?)
            }
            Field::CrownFireHeatPerUnitArea => {
                p(Field::SurfaceHpua)? + p(Field::CanopyHeatPerUnitArea)?
            }
            _ => return self.surface_physical(field, index),
        })
    }
    fn five(&self, start: Field, index: usize) -> Result<[f64; 5], BehaveError> {
        let mut result = [0.; 5];
        for (n, value) in result.iter_mut().enumerate() {
            *value = self.physical(Field::ALL[start as usize + n], index)?;
        }
        Ok(result)
    }
    fn area_side(&self, dead: bool, index: usize) -> Result<(f64, [f64; 5]), BehaveError> {
        let (load, savr) = if dead {
            (Field::ParticleLoadDead0, Field::ParticleSavrDead0)
        } else {
            (Field::ParticleLoadLive0, Field::ParticleSavrLive0)
        };
        let (weights, total) = crate::surface::particle_area_weights::<true>(
            &self.five(load, index)?,
            &self.five(savr, index)?,
        );
        Ok((total, weights))
    }
    fn dependencies(&self, field: Field) -> Vec<Field> {
        let mut result = match field {
            Field::Waf if self.pipeline == Pipeline::Crown => Vec::new(),
            Field::Waf
                if self.options.waf_method == super::WafMethod::UserInput
                    && self.slots[Field::UserWaf as usize].active().is_some() =>
            {
                vec![Field::UserWaf]
            }
            Field::Waf => vec![
                Field::CanopyCover,
                Field::CanopyHeight,
                Field::CrownRatio,
                Field::FuelbedDepth,
            ],
            Field::CrowningSurfaceFireSpreadRate => {
                let mut deps = field.dependencies().to_vec();
                deps.extend([Field::WindB, Field::WindC, Field::WindE]);
                deps
            }
            _ => field
                .dependencies()
                .iter()
                .copied()
                .filter(|f| {
                    *f != Field::Aspect
                        || self.options.wind_orientation == super::WindOrientation::North
                })
                .collect(),
        };
        if super::field::SURFACE_FIELDS.contains(&field)
            && (self.pipeline == Pipeline::Surface || field != Field::ReactionIntensity)
        {
            result.push(Field::FuelbedDefined);
        }
        result
    }
}
