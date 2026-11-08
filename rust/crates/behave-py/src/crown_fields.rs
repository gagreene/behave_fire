//! Compile field dependencies once; evaluate only bound primitive operands per cell.
use behave_fire::crown::{self, CrownResult};
use behave_fire::engine::{CrownPlan, CrownSpatial, Field as F, PreparedFuels, WindOrientation};
use behave_fire::PerCell;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

enum Op<'a> {
    Input(PerCell<'a, f64>),
    CanopyLoad,
    CanopyHeat,
    CriticalIntensity,
    CriticalRate,
    Spread,
    LengthWidth,
    Add(F, F),
    ActiveIntensity,
    PassiveIntensity,
    Flame(F),
    Ratio(F, F),
    SurfaceCritical,
    Crowning,
    Fraction,
    PassiveSpread,
    PassiveHeat,
    Classify,
    Select(F, F, F),
    SurfaceFlame,
}
struct Instruction<'a> {
    field: F,
    op: Op<'a>,
}

pub(crate) struct Program<'a> {
    source: &'a BTreeMap<String, Vec<f64>>,
    nodes: Vec<Instruction<'a>>,
    compiled: [bool; F::COUNT],
    requested: Vec<F>,
    fuels: &'a PreparedFuels,
    plan: CrownPlan,
}
impl<'a> Program<'a> {
    pub(crate) fn new(
        source: &'a BTreeMap<String, Vec<f64>>,
        requested: &[F],
        fuels: &'a PreparedFuels,
        plan: CrownPlan,
    ) -> PyResult<Self> {
        let mut result = Self {
            source,
            nodes: Vec::new(),
            compiled: [false; F::COUNT],
            requested: requested.to_vec(),
            fuels,
            plan,
        };
        for &field in requested {
            result.require(field)?;
        }
        Ok(result)
    }

    fn inputs(&mut self, fields: &[F]) -> PyResult<()> {
        for &field in fields {
            self.require(field)?;
        }
        Ok(())
    }

    fn require(&mut self, field: F) -> PyResult<()> {
        if self.compiled[field as usize] {
            return Ok(());
        }
        let op = if self.source.contains_key(field.name()) {
            Op::Input(crate::model_f64::column(self.source, field.name())?)
        } else {
            match field {
                F::CrownFuelLoad => {
                    self.inputs(&[F::CanopyBaseHeight, F::CanopyHeight, F::CanopyBulkDensity])?;
                    Op::CanopyLoad
                }
                F::CanopyHeatPerUnitArea => {
                    self.require(F::CrownFuelLoad)?;
                    Op::CanopyHeat
                }
                F::CrownCriticalSurfaceFireLineIntensity => {
                    self.inputs(&[F::CanopyBaseHeight, F::MoistureFoliar])?;
                    Op::CriticalIntensity
                }
                F::CrownCriticalFireSpreadRate => {
                    self.require(F::CanopyBulkDensity)?;
                    Op::CriticalRate
                }
                F::CrownFireSpreadRate | F::CrownLengthToWidthRatio => {
                    self.inputs(&[
                        F::M1h,
                        F::M10h,
                        F::M100h,
                        F::Mlh,
                        F::Mlw,
                        F::WindSpeed,
                        F::WindDirection,
                        F::Slope,
                    ])?;
                    if self.plan.options().wind_orientation == WindOrientation::North {
                        self.require(F::Aspect)?;
                    }
                    if field == F::CrownFireSpreadRate {
                        Op::Spread
                    } else {
                        Op::LengthWidth
                    }
                }
                F::CrownFireHeatPerUnitArea => {
                    self.inputs(&[F::SurfaceHpua, F::CanopyHeatPerUnitArea])?;
                    Op::Add(F::SurfaceHpua, F::CanopyHeatPerUnitArea)
                }
                F::CrownFireLineIntensity => {
                    self.inputs(&[F::CrownFireSpreadRate, F::CrownFireHeatPerUnitArea])?;
                    Op::ActiveIntensity
                }
                F::CrownFlameLength => {
                    self.require(F::CrownFireLineIntensity)?;
                    Op::Flame(F::CrownFireLineIntensity)
                }
                F::CrownFireTransitionRatio => {
                    self.inputs(&[F::SurfaceFli, F::CrownCriticalSurfaceFireLineIntensity])?;
                    Op::Ratio(F::SurfaceFli, F::CrownCriticalSurfaceFireLineIntensity)
                }
                F::CrownFireActiveRatio => {
                    self.inputs(&[F::CrownFireSpreadRate, F::CrownCriticalFireSpreadRate])?;
                    Op::Ratio(F::CrownFireSpreadRate, F::CrownCriticalFireSpreadRate)
                }
                F::FireType => {
                    self.inputs(&[F::CrownFireTransitionRatio, F::CrownFireActiveRatio])?;
                    Op::Classify
                }
                F::SurfaceFireCriticalSpreadRate => {
                    self.inputs(&[F::CrownCriticalSurfaceFireLineIntensity, F::SurfaceHpua])?;
                    Op::SurfaceCritical
                }
                F::CrowningSurfaceFireSpreadRate => {
                    self.inputs(&[
                        F::M1h,
                        F::M10h,
                        F::M100h,
                        F::Mlh,
                        F::Mlw,
                        F::CrownCriticalFireSpreadRate,
                    ])?;
                    Op::Crowning
                }
                F::CrownFractionBurned => {
                    self.inputs(&[
                        F::SurfaceRos,
                        F::SurfaceFireCriticalSpreadRate,
                        F::CrowningSurfaceFireSpreadRate,
                    ])?;
                    Op::Fraction
                }
                F::PassiveCrownFireSpreadRate => {
                    self.inputs(&[
                        F::SurfaceRos,
                        F::CrownFractionBurned,
                        F::CrownFireSpreadRate,
                    ])?;
                    Op::PassiveSpread
                }
                F::PassiveCrownFireHeatPerUnitArea => {
                    self.inputs(&[
                        F::SurfaceHpua,
                        F::CrownFractionBurned,
                        F::CanopyHeatPerUnitArea,
                    ])?;
                    Op::PassiveHeat
                }
                F::PassiveCrownFireLineIntensity => {
                    self.inputs(&[
                        F::PassiveCrownFireHeatPerUnitArea,
                        F::PassiveCrownFireSpreadRate,
                    ])?;
                    Op::PassiveIntensity
                }
                F::PassiveCrownFireFlameLength => {
                    self.require(F::PassiveCrownFireLineIntensity)?;
                    Op::Flame(F::PassiveCrownFireLineIntensity)
                }
                F::FinalSpreadRate => {
                    self.inputs(&[
                        F::FireType,
                        F::SurfaceRos,
                        F::PassiveCrownFireSpreadRate,
                        F::CrownFireSpreadRate,
                    ])?;
                    Op::Select(
                        F::SurfaceRos,
                        F::PassiveCrownFireSpreadRate,
                        F::CrownFireSpreadRate,
                    )
                }
                F::FinalHeatPerUnitArea => {
                    self.inputs(&[
                        F::FireType,
                        F::SurfaceHpua,
                        F::PassiveCrownFireHeatPerUnitArea,
                        F::CrownFireHeatPerUnitArea,
                    ])?;
                    Op::Select(
                        F::SurfaceHpua,
                        F::PassiveCrownFireHeatPerUnitArea,
                        F::CrownFireHeatPerUnitArea,
                    )
                }
                F::FinalFirelineIntensity => {
                    self.inputs(&[
                        F::FireType,
                        F::SurfaceFli,
                        F::PassiveCrownFireLineIntensity,
                        F::CrownFireLineIntensity,
                    ])?;
                    Op::Select(
                        F::SurfaceFli,
                        F::PassiveCrownFireLineIntensity,
                        F::CrownFireLineIntensity,
                    )
                }
                F::FinalFlameLength => {
                    self.inputs(&[
                        F::FireType,
                        F::SurfaceFli,
                        F::PassiveCrownFireFlameLength,
                        F::CrownFlameLength,
                    ])?;
                    Op::SurfaceFlame
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "missing/unbound required model field {}",
                        field.name()
                    )))
                }
            }
        };
        self.compiled[field as usize] = true;
        self.nodes.push(Instruction { field, op });
        Ok(())
    }

    pub(crate) fn evaluate(&self, index: usize) -> CrownResult {
        let mut values = [f64::NAN; F::COUNT];
        let mut code = -1;
        let mut spread = None;
        for node in &self.nodes {
            let v = |field: F| values[field as usize];
            let value = match &node.op {
                Op::Input(column) => *column.get(index).expect("validated bound input"),
                Op::CanopyLoad => crown::calculate_canopy_fuel_load(
                    v(F::CanopyBaseHeight),
                    v(F::CanopyHeight),
                    v(F::CanopyBulkDensity),
                ),
                Op::CanopyHeat => crown::calculate_canopy_heat_per_unit_area(v(F::CrownFuelLoad)),
                Op::CriticalIntensity => crown::calculate_critical_surface_fireline_intensity(
                    v(F::CanopyBaseHeight),
                    v(F::MoistureFoliar),
                ),
                Op::CriticalRate => crown::available::critical_rate(v(F::CanopyBulkDensity)),
                Op::Spread | Op::LengthWidth => {
                    let result = spread.get_or_insert_with(|| {
                        self.plan.prepare_spread_spatial(
                            &CrownSpatial {
                                m1h: v(F::M1h),
                                m10h: v(F::M10h),
                                m100h: v(F::M100h),
                                mlh: v(F::Mlh),
                                mlw: v(F::Mlw),
                                wind_speed: v(F::WindSpeed),
                                wind_direction: v(F::WindDirection),
                                slope: v(F::Slope),
                                aspect: if self.plan.options().wind_orientation
                                    == WindOrientation::North
                                {
                                    v(F::Aspect)
                                } else {
                                    0.
                                },
                            },
                            self.fuels.for_grid(10),
                        )
                    });
                    if matches!(node.op, Op::Spread) {
                        result.crown_fire_spread_rate
                    } else {
                        result.crown_length_to_width_ratio
                    }
                }
                Op::Add(a, b) => v(*a) + v(*b),
                Op::ActiveIntensity => {
                    (v(F::CrownFireSpreadRate) / 60.) * v(F::CrownFireHeatPerUnitArea)
                }
                Op::PassiveIntensity => crown::calculate_passive_intensity(
                    v(F::PassiveCrownFireHeatPerUnitArea),
                    v(F::PassiveCrownFireSpreadRate),
                ),
                Op::Flame(input) => crown::available::flame_length(v(*input)),
                Op::Ratio(a, b) => crown::available::ratio(v(*a), v(*b)),
                Op::SurfaceCritical => crown::available::surface_critical(
                    v(F::CrownCriticalSurfaceFireLineIntensity),
                    v(F::SurfaceHpua),
                ),
                Op::Crowning => crown::available::crowning_rate(
                    [v(F::M1h), v(F::M10h), v(F::M100h), v(F::Mlh), v(F::Mlw)],
                    self.fuels.for_grid(10),
                    v(F::CrownCriticalFireSpreadRate),
                ),
                Op::Fraction => crown::available::fraction(
                    v(F::SurfaceRos),
                    v(F::SurfaceFireCriticalSpreadRate),
                    v(F::CrowningSurfaceFireSpreadRate),
                ),
                Op::PassiveSpread => crown::calculate_passive_spread_rate(
                    v(F::SurfaceRos),
                    v(F::CrownFireSpreadRate),
                    v(F::CrownFractionBurned),
                ),
                Op::PassiveHeat => crown::calculate_passive_heat_per_unit_area(
                    v(F::SurfaceHpua),
                    v(F::CanopyHeatPerUnitArea),
                    v(F::CrownFractionBurned),
                ),
                Op::Classify => {
                    code = crown::available::classify(
                        v(F::CrownFireTransitionRatio),
                        v(F::CrownFireActiveRatio),
                    )
                    .fire_type;
                    continue;
                }
                Op::Select(surface, passive, crown) => match code {
                    0 | 2 => v(*surface),
                    1 => v(*passive),
                    3 => v(*crown),
                    _ => f64::NAN,
                },
                Op::SurfaceFlame => match code {
                    0 | 2 => crown::available::flame_length(v(F::SurfaceFli)),
                    1 => v(F::PassiveCrownFireFlameLength),
                    3 => v(F::CrownFlameLength),
                    _ => f64::NAN,
                },
            };
            values[node.field as usize] = value;
        }
        self.materialize(&values, code)
    }

    pub(crate) fn missing(&self) -> CrownResult {
        self.materialize(&[f64::NAN; F::COUNT], -1)
    }

    fn materialize(&self, values: &[f64; F::COUNT], code: i32) -> CrownResult {
        let mut result = CrownResult::new();
        for &field in &self.requested {
            let value = values[field as usize];
            match field {
                F::CrownFireSpreadRate => result.crown_fire_spread_rate = value,
                F::CrownLengthToWidthRatio => result.crown_length_to_width_ratio = value,
                F::CrownFireHeatPerUnitArea => result.crown_fire_heat_per_unit_area = value,
                F::CanopyHeatPerUnitArea => result.canopy_heat_per_unit_area = value,
                F::CrownFireLineIntensity => result.crown_fire_line_intensity = value,
                F::CrownFlameLength => result.crown_flame_length = value,
                F::CrownCriticalSurfaceFireLineIntensity => {
                    result.crown_critical_surface_fire_line_intensity = value
                }
                F::CrownCriticalFireSpreadRate => result.crown_critical_fire_spread_rate = value,
                F::CrownFireTransitionRatio => result.crown_fire_transition_ratio = value,
                F::CrownFireActiveRatio => result.crown_fire_active_ratio = value,
                F::SurfaceFireCriticalSpreadRate => {
                    result.surface_fire_critical_spread_rate = value
                }
                F::CrowningSurfaceFireSpreadRate => {
                    result.crowning_surface_fire_spread_rate = value
                }
                F::CrownFractionBurned => result.crown_fraction_burned = value,
                F::PassiveCrownFireSpreadRate => result.passive_crown_fire_spread_rate = value,
                F::PassiveCrownFireHeatPerUnitArea => {
                    result.passive_crown_fire_heat_per_unit_area = value
                }
                F::PassiveCrownFireLineIntensity => {
                    result.passive_crown_fire_line_intensity = value
                }
                F::PassiveCrownFireFlameLength => result.passive_crown_fire_flame_length = value,
                F::FinalSpreadRate => result.final_spread_rate = value,
                F::FinalHeatPerUnitArea => result.final_heat_per_unit_area = value,
                F::FinalFirelineIntensity => result.final_fireline_intensity = value,
                F::FinalFlameLength => result.final_flame_length = value,
                F::FireType => result.fire_type = code,
                _ => unreachable!("validated Crown output"),
            }
        }
        result
    }
}
