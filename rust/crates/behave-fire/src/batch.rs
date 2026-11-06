//! Bounded native pipelines with owned column results and checked shapes.
use crate::crown::CrownResult;
use crate::surface::SurfaceResult;
use crate::{map_chunks, GridShape, ParallelError};
fn flatten_error<E>(error: ParallelError<ParallelError<E>>) -> ParallelError<E> {
    match error {
        ParallelError::Work(error) => error,
        ParallelError::WorkerPanic => ParallelError::WorkerPanic,
        ParallelError::WorkerSpawn(message) => ParallelError::WorkerSpawn(message),
        ParallelError::Allocation(message) => ParallelError::Allocation(message),
    }
}
trait IntoColumn {
    fn into_column(self) -> Vec<f64>;
}
impl IntoColumn for Vec<f64> {
    fn into_column(self) -> Vec<f64> {
        self
    }
}
impl IntoColumn for Vec<i32> {
    fn into_column(self) -> Vec<f64> {
        self.into_iter().map(f64::from).collect()
    }
}

/// Per-call field concatenation measurements without shared or global state.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct BatchStatistics {
    /// Time spent reserving and appending chunk columns after all workers join.
    pub concatenation_seconds: f64,
    /// Bytes explicitly appended from later chunks; allocator relocation is excluded.
    pub appended_bytes: usize,
}
/// Use serial pipelines below this measured Windows crossover bound.
///
/// On the recorded host, four workers cost more at 16 and 64 cells. Surface
/// first improved at the measured 1,024-cell point. This is a conservative
/// per-call threshold, not a promise of optimal scaling on every host.
pub const PIPELINE_PARALLEL_MIN_CELLS: usize = 1_024;
macro_rules! columns {
    ($name:ident, $scalar:ty, {$($field:ident:$kind:ty),+}) => {
        /// Owned named columns in logical row-major order.
        #[derive(Clone, Debug)]
        #[non_exhaustive]
        pub struct $name {
            shape: GridShape,
            statistics: BatchStatistics,
            $(#[doc=concat!("Owned `",stringify!($field),"` column in component base units.")] pub $field:Vec<$kind>,)+
        }
        impl $name {
            /// Construct empty columns with checked output shape metadata.
            pub fn new(shape: GridShape) -> Self { Self {shape, statistics: BatchStatistics::default(), $($field:Vec::new(),)+} }
            /// Borrow per-call concatenation measurements.
            pub fn statistics(&self) -> &BatchStatistics { &self.statistics }
            /// Borrow the logical output shape.
            pub fn shape(&self) -> &GridShape { &self.shape }
            /// Move f64 buffers; convert integer columns for generic numeric adapters.
            pub fn into_named_vectors(self) -> Vec<(&'static str,Vec<f64>)> {
                vec![$((stringify!($field),self.$field.into_column())),+]
            }
            /// Evaluate cells directly into columns; closures borrow only native data.
            pub fn run_with<E: Send,F>(shape: GridShape, threads: usize, operation:F) -> Result<Self,ParallelError<E>>
            where F: Fn(usize)->Result<$scalar,E> + Sync {
                let requested=if shape.len()<PIPELINE_PARALLEL_MIN_CELLS {1}else{threads};
                let mut chunks=map_chunks(shape.len(),requested,|range| {
                    let mut output=Self::new(GridShape::new(vec![range.len()]).expect("one dimension cannot overflow"));
                    $(output.$field.try_reserve_exact(range.len()).map_err(|error|ParallelError::Allocation(error.to_string()))?;)+
                    for index in range { let cell=operation(index).map_err(ParallelError::Work)?; $(output.$field.push(cell.$field);)+ }
                    Ok::<_,ParallelError<E>>(output)
                }).map_err(flatten_error)?.into_iter();
                let mut output=match chunks.next() {Some(chunk)=>chunk,None=>Self::new(shape.clone())};
                let start=std::time::Instant::now();
                if chunks.len()>0 {$(output.$field.try_reserve_exact(shape.len()-output.$field.len()).map_err(|error|ParallelError::Allocation(error.to_string()))?;)+}
                output.shape=shape;
                for mut chunk in chunks {$(output.statistics.appended_bytes+=chunk.$field.len()*std::mem::size_of::<$kind>(); output.$field.append(&mut chunk.$field);)+}
                output.statistics.concatenation_seconds=start.elapsed().as_secs_f64();
                Ok(output)
            }
        }
    }
}
columns!(SurfaceBatch, SurfaceResult, {
spread_rate:f64, backing_spread_rate:f64, flanking_spread_rate:f64, flame_length:f64,
fireline_intensity:f64, heat_per_unit_area:f64, effective_wind_speed:f64,
fire_length_to_width_ratio:f64, eccentricity:f64, direction_of_max_spread:f64,
residence_time:f64, reaction_intensity:f64, midflame_wind_speed:f64,
no_wind_no_slope_spread_rate:f64});
columns!(CrownBatch, CrownResult, {
crown_fire_spread_rate:f64, crown_flame_length:f64, crown_fire_line_intensity:f64,
crown_fire_heat_per_unit_area:f64, canopy_heat_per_unit_area:f64,
crown_critical_surface_fire_line_intensity:f64, crown_critical_fire_spread_rate:f64,
crown_fire_transition_ratio:f64, crown_fire_active_ratio:f64, crown_length_to_width_ratio:f64,
surface_fire_critical_spread_rate:f64, crowning_surface_fire_spread_rate:f64,
crown_fraction_burned:f64, passive_crown_fire_spread_rate:f64,
passive_crown_fire_heat_per_unit_area:f64, passive_crown_fire_line_intensity:f64,
passive_crown_fire_flame_length:f64, final_spread_rate:f64, final_heat_per_unit_area:f64,
final_fireline_intensity:f64, final_flame_length:f64, fire_type:i32});
