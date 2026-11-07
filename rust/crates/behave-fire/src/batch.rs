//! Existing model batch owners filled directly through borrowed final slices.
use crate::crown::CrownResult;
use crate::engine::{Field, OutputRequest};
use crate::output::{fill_chunks, ColumnValues};
use crate::surface::SurfaceResult;
use crate::{BehaveError, GridShape, ParallelError};

trait ColumnElement: Copy {
    fn missing() -> Self;
    fn into_column(values: Vec<Self>) -> ColumnValues;
}
impl ColumnElement for f64 {
    fn missing() -> Self {
        f64::NAN
    }
    fn into_column(values: Vec<Self>) -> ColumnValues {
        ColumnValues::Float64(values)
    }
}
impl ColumnElement for i32 {
    fn missing() -> Self {
        -1
    }
    fn into_column(values: Vec<Self>) -> ColumnValues {
        ColumnValues::Int32(values)
    }
}
fn allocated<T: ColumnElement>(count: usize) -> Result<Vec<T>, BehaveError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| BehaveError::Allocation(error.to_string()))?;
    values.resize(count, T::missing());
    Ok(values)
}
fn allocation_error<E>(error: BehaveError) -> ParallelError<E> {
    match error {
        BehaveError::Allocation(message) => ParallelError::Allocation(message),
        other => ParallelError::InvalidInput(other),
    }
}
/// Measurements of removed concatenation work, retained for truthful diagnostics.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct BatchStatistics {
    /// Zero: direct final-buffer writes require no concatenation.
    pub concatenation_seconds: f64,
    /// Zero: no later-chunk buffers are appended.
    pub appended_bytes: usize,
}
/// Existing conservative model-boundary parallel threshold, pending later measurements.
pub const PIPELINE_PARALLEL_MIN_CELLS: usize = 1024;

macro_rules! borrow_column {
    (f64,$values:expr,$physical:expr,$codes:expr) => {{
        let slot = $physical.len();
        $physical.push($values.as_mut_slice());
        slot
    }};
    (i32,$values:expr,$physical:expr,$codes:expr) => {{
        let slot = $codes.len();
        $codes.push($values.as_mut_slice());
        slot
    }};
}
macro_rules! store_column {
    (f64,$out:expr,$slot:expr,$index:expr,$value:expr) => {
        $out.physical[$slot][$index] = $value
    };
    (i32,$out:expr,$slot:expr,$index:expr,$value:expr) => {
        $out.codes[$slot][$index] = $value
    };
}
macro_rules! columns {
    ($name:ident,$scalar:ty,{$($field:ident[$key:ident]:$kind:ident),+ $(,)?})=>{
        /// Owned named model columns; unrequested fields have no allocation.
        #[derive(Clone,Debug)]
        #[non_exhaustive]
        pub struct $name {
            shape:GridShape,
            request:OutputRequest,
            statistics:BatchStatistics,
            /// Optional explicit native validity, independent of masked input coercion.
            pub validity:Option<Vec<bool>>,
            $(#[doc=concat!("Owned `",stringify!($field),"` values in component base units.")]pub $field:Vec<$kind>,)+
        }
        impl $name {
            /// Allocate selected result buffers once; skipped slots remain unavailable.
            pub fn allocate(shape:GridShape,request:&OutputRequest)->Result<Self,BehaveError> {
                for &field in request.fields() {
                    if !matches!(field,$(Field::$key)|+) {
                        return Err(BehaveError::InvalidOutputField {field:field.name().to_owned(),model:stringify!($name)});
                    }
                }
                let count=shape.len();
                Ok(Self {shape,request:request.clone(),statistics:BatchStatistics::default(),validity:None,
                    $($field:if request.contains(Field::$key) {allocated::<$kind>(count)?} else {Vec::new()},)+})
            }
            /// Borrow the checked logical shape.
            pub fn shape(&self)->&GridShape {&self.shape}
            /// Borrow direct-storage diagnostics.
            pub fn statistics(&self)->&BatchStatistics {&self.statistics}
            /// Borrow requested field identities in caller order.
            pub fn requested(&self)->&[Field] {self.request.fields()}
            /// Move selected columns without widening integer categories or copying data.
            pub fn into_named_columns(mut self)->Vec<(&'static str,ColumnValues)> {
                let mut result=Vec::with_capacity(self.request.fields().len());
                for &field in self.request.fields() {
                    let values=match field {$(Field::$key=><$kind as ColumnElement>::into_column(std::mem::take(&mut self.$field)),)+_=>unreachable!("request was validated")};
                    result.push((field.name(),values));
                }
                result
            }
            /// Evaluate all default fields directly into final owned columns.
            pub fn run_with<E:Send,F>(shape:GridShape,threads:usize,operation:F)->Result<Self,ParallelError<E>>
            where F:Fn(usize)->Result<$scalar,E>+Sync {
                let request=OutputRequest::new(&[$(Field::$key),+]);
                Self::run_with_request(shape,&request,threads,operation)
            }
            /// Evaluate cells while storing only the requested fields.
            pub fn run_with_request<E:Send,F>(shape:GridShape,request:&OutputRequest,threads:usize,operation:F)->Result<Self,ParallelError<E>>
            where F:Fn(usize)->Result<$scalar,E>+Sync {
                let count=shape.len();
                let requested=if count<PIPELINE_PARALLEL_MIN_CELLS {1}else{threads};
                let mut output=Self::allocate(shape,request).map_err(allocation_error)?;
                if count==0 || request.fields().is_empty() {return Ok(output);}
                let mut physical=Vec::new();
                let mut codes=Vec::new();
                physical.try_reserve_exact(request.fields().len()).map_err(|error|ParallelError::Allocation(error.to_string()))?;
                codes.try_reserve_exact(request.fields().len()).map_err(|error|ParallelError::Allocation(error.to_string()))?;
                let mut slots=[None;[$(stringify!($field)),+].len()];
                let mut ordinal=0;
                $({if request.contains(Field::$key) {slots[ordinal]=Some(borrow_column!($kind,output.$field,physical,codes));} ordinal+=1;})+
                let _=ordinal;
                fill_chunks(count,requested,&mut physical,&mut codes,&mut [],|range,shard| {
                    for (local,index) in range.enumerate() {
                        let cell=operation(index)?;
                        let mut ordinal=0;
                        $({if let Some(slot)=slots[ordinal] {store_column!($kind,shard,slot,local,cell.$field);} ordinal+=1;})+
                        let _=ordinal;
                    }
                    Ok(())
                })?;
                Ok(output)
            }
        }
    }
}
columns!(SurfaceBatch,SurfaceResult, {spread_rate[SpreadRate]:f64,backing_spread_rate[BackingSpreadRate]:f64,flanking_spread_rate[FlankingSpreadRate]:f64,flame_length[FlameLength]:f64,fireline_intensity[FirelineIntensity]:f64,heat_per_unit_area[HeatPerUnitArea]:f64,effective_wind_speed[EffectiveWindSpeed]:f64,fire_length_to_width_ratio[FireLengthToWidthRatio]:f64,eccentricity[Eccentricity]:f64,direction_of_max_spread[DirectionOfMaxSpread]:f64,residence_time[ResidenceTime]:f64,reaction_intensity[ReactionIntensity]:f64,midflame_wind_speed[MidflameWindSpeed]:f64,no_wind_no_slope_spread_rate[NoWindNoSlopeSpreadRate]:f64});
columns!(CrownBatch,CrownResult, {crown_fire_spread_rate[CrownFireSpreadRate]:f64,crown_flame_length[CrownFlameLength]:f64,crown_fire_line_intensity[CrownFireLineIntensity]:f64,crown_fire_heat_per_unit_area[CrownFireHeatPerUnitArea]:f64,canopy_heat_per_unit_area[CanopyHeatPerUnitArea]:f64,crown_critical_surface_fire_line_intensity[CrownCriticalSurfaceFireLineIntensity]:f64,crown_critical_fire_spread_rate[CrownCriticalFireSpreadRate]:f64,crown_fire_transition_ratio[CrownFireTransitionRatio]:f64,crown_fire_active_ratio[CrownFireActiveRatio]:f64,crown_length_to_width_ratio[CrownLengthToWidthRatio]:f64,surface_fire_critical_spread_rate[SurfaceFireCriticalSpreadRate]:f64,crowning_surface_fire_spread_rate[CrowningSurfaceFireSpreadRate]:f64,crown_fraction_burned[CrownFractionBurned]:f64,passive_crown_fire_spread_rate[PassiveCrownFireSpreadRate]:f64,passive_crown_fire_heat_per_unit_area[PassiveCrownFireHeatPerUnitArea]:f64,passive_crown_fire_line_intensity[PassiveCrownFireLineIntensity]:f64,passive_crown_fire_flame_length[PassiveCrownFireFlameLength]:f64,final_spread_rate[FinalSpreadRate]:f64,final_heat_per_unit_area[FinalHeatPerUnitArea]:f64,final_fireline_intensity[FinalFirelineIntensity]:f64,final_flame_length[FinalFlameLength]:f64,fire_type[FireType]:i32});
