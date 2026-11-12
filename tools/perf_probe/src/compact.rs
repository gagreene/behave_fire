//! Disposable Task-3 feasibility experiment, restricted to benchmark layouts A/B.
//! This is evidence tooling, not a supported numerical API or production fast path.
use super::{Arguments, Case};
use behave_fire::crown;
use behave_fire::engine::{CrownCell, CrownPlan, PreparedFuel, PreparedFuels, SurfaceTweaks};
use behave_fire::surface::masked::{MaskedValue, Precision};

#[derive(Clone, Copy, Default)]
struct Value {
    value: f64,
    mask: bool,
    ma: bool,
}

impl Value {
    fn plain(value: f64) -> Self {
        Self {
            value,
            mask: false,
            ma: false,
        }
    }
    fn from_checked(value: MaskedValue) -> Self {
        Self {
            value: value.value,
            mask: value.mask,
            ma: value.is_masked_array,
        }
    }
    fn binary(self, rhs: Self, raw: f64) -> Self {
        let ma = self.ma || rhs.ma;
        let mask = self.mask || rhs.mask;
        Self {
            value: if ma && mask { self.value } else { raw },
            mask,
            ma,
        }
    }
    fn add(self, rhs: Self) -> Self {
        self.binary(rhs, self.value + rhs.value)
    }
    fn sub(self, rhs: Self) -> Self {
        self.binary(rhs, self.value - rhs.value)
    }
    fn mul(self, rhs: Self) -> Self {
        self.binary(rhs, self.value * rhs.value)
    }
    fn div(self, rhs: Self) -> Self {
        let raw = self.value / rhs.value;
        let mut result = self.binary(rhs, raw);
        if result.ma {
            result.mask |=
                self.value.abs() * f64::MIN_POSITIVE >= rhs.value.abs() || !raw.is_finite();
            if result.mask {
                result.value = self.value;
            }
            if result.value == 0.0 {
                result.value = if result.mask {
                    0.0 + self.value
                } else {
                    raw + 0.0 * self.value
                };
            }
        }
        result
    }
    fn pow(self, rhs: Self) -> Self {
        let mut result = self.binary(rhs, self.value.powf(rhs.value));
        if result.ma && !result.value.is_finite() {
            result.mask = true;
            result.value = 1e20;
        }
        result
    }
    fn maximum(self, rhs: Self) -> Self {
        Self {
            value: if self.value.is_nan() || self.value > rhs.value {
                self.value
            } else {
                rhs.value
            },
            mask: self.mask || rhs.mask,
            ma: self.ma || rhs.ma,
        }
    }
    fn minimum(self, rhs: Self) -> Self {
        Self {
            value: if self.value.is_nan() || self.value < rhs.value {
                self.value
            } else {
                rhs.value
            },
            mask: self.mask || rhs.mask,
            ma: self.ma || rhs.ma,
        }
    }
}

fn select(condition: bool, yes: Value, no: Value) -> Value {
    Value::plain(if condition { yes.value } else { no.value })
}

fn flame(intensity: Value) -> Value {
    select(
        intensity.value > 0.0,
        Value::plain(0.2).mul(intensity.pow(Value::plain(2.0 / 3.0))),
        Value::plain(0.0),
    )
}

struct Record {
    values: [Value; 21],
    class: i32,
}

fn evaluate(
    cell: &CrownCell,
    supplied: [Value; 3],
    f32_heat: bool,
    plan: &CrownPlan,
    fuel: &PreparedFuel,
) -> Record {
    let p = Value::plain;
    let [rate, intensity, heat] = supplied;
    let fm10 = plan.prepare_spread(cell, fuel, SurfaceTweaks::default());
    let crown_rate = p(fm10.crown_fire_spread_rate);
    let load = p(cell.canopy_bulk_density)
        .mul(p(cell.canopy_height).sub(p(cell.canopy_base_height)))
        .maximum(p(0.0));
    let canopy = load.mul(p(8000.0));
    let crown_heat = heat.add(canopy);
    let crown_intensity = crown_rate.div(p(60.0)).mul(crown_heat);
    let crown_flame = flame(crown_intensity);
    let critical_intensity = p(crown::calculate_critical_surface_fireline_intensity(
        cell.canopy_base_height,
        cell.moisture_foliar,
    ));
    let critical_rate = p(crown::calculate_critical_crown_fire_spread_rate(
        cell.canopy_bulk_density,
    ));
    let transition = select(
        critical_intensity.value > 1e-7,
        intensity.div(select(
            critical_intensity.value > 1e-7,
            critical_intensity,
            p(1.0),
        )),
        p(0.0),
    );
    let active = select(
        critical_rate.value > 1e-7,
        crown_rate.div(select(critical_rate.value > 1e-7, critical_rate, p(1.0))),
        p(0.0),
    );
    let class = if transition.value < 1.0 {
        if active.value < 1.0 {
            0
        } else {
            2
        }
    } else if active.value < 1.0 {
        1
    } else {
        3
    };
    // Layout B compares and selects its HPUA in f32 before a strong-f64 divide.
    let usable_heat = if f32_heat {
        (heat.value as f32) > (1e-7_f64 as f32)
    } else {
        heat.value > 1e-7
    };
    let safe_heat = select(usable_heat, heat, p(1.0));
    let surface_critical = select(
        usable_heat,
        p(60.0).mul(critical_intensity).div(safe_heat),
        p(0.0),
    );
    let crowning = p(crown::calculate_crowning_surface_fire_spread_rate(
        fm10.ri_crown,
        &fm10.ib_crown,
        critical_rate.value,
        cell.canopy_height,
    ));
    let denominator = crowning.sub(surface_critical);
    let fraction = select(
        denominator.value > 1e-7,
        rate.sub(surface_critical).div(denominator),
        p(0.0),
    )
    .maximum(p(0.0))
    .minimum(p(1.0));
    let passive_rate = rate.add(fraction.mul(crown_rate.sub(rate)));
    let passive_heat = heat.add(fraction.mul(canopy));
    let passive_intensity = passive_heat.mul(passive_rate).div(p(60.0));
    let passive_flame = flame(passive_intensity);
    let is_surface = class == 0 || class == 2;
    let is_passive = class == 1;
    Record {
        values: [
            crown_rate,
            crown_flame,
            crown_intensity,
            crown_heat,
            canopy,
            critical_intensity,
            critical_rate,
            transition,
            active,
            p(fm10.crown_length_to_width_ratio),
            surface_critical,
            crowning,
            fraction,
            passive_rate,
            passive_heat,
            passive_intensity,
            passive_flame,
            select(
                is_surface,
                rate,
                select(is_passive, passive_rate, crown_rate),
            ),
            select(
                is_surface,
                heat,
                select(is_passive, passive_heat, crown_heat),
            ),
            select(
                is_surface,
                intensity,
                select(is_passive, passive_intensity, crown_intensity),
            ),
            select(
                is_surface,
                flame(intensity),
                select(is_passive, passive_flame, crown_flame),
            ),
        ],
        class,
    }
}

struct Columns {
    values: [Vec<f64>; 21],
    masks: [Vec<bool>; 21],
    classes: Vec<i32>,
}

// Each logical cell indexes several independent input and output columns.
#[allow(clippy::needless_range_loop)]
pub(super) fn run(
    args: &Arguments,
    cases: &[Case<'_>],
    fuels: &PreparedFuels,
) -> Result<(), String> {
    let plan = CrownPlan::default();
    let fuel = fuels.for_grid(10);
    let f32_heat = args.layout == "b";
    let data = super::raster_probe::CrownData::new(args, cases);
    // Preserve input quantization and independently typed mask/class inputs.
    let supplied: [Vec<Value>; 3] = std::array::from_fn(|field| {
        (0..args.cells)
            .map(|i| {
                let fields = &cases[i % cases.len()].typed;
                Value::from_checked(
                    [fields.surface_ros, fields.surface_fli, fields.surface_hpua][field],
                )
            })
            .collect()
    });
    let reference = cases
        .iter()
        .map(|case| {
            crown::run_with_surface_inputs(&case.typed, &case.crown, fuel, plan.options())
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Strict metadata and raw-data gate over the complete weather/fuel/mask period.
    for (index, case) in cases.iter().enumerate() {
        let result = evaluate(
            &case.crown,
            [supplied[0][index], supplied[1][index], supplied[2][index]],
            f32_heat,
            &plan,
            fuel,
        );
        if result.class != reference[index].fire_type {
            return Err(format!("compact class/{index}"));
        }
        for ((name, expected), actual) in reference[index]
            .named_values()
            .into_iter()
            .zip(result.values)
        {
            let restored = MaskedValue::new(actual.value, actual.mask, actual.ma, Precision::F64);
            if !super::raster_probe::check_value(actual.value, expected.value)
                || super::raster_probe::schema(restored) != super::raster_probe::schema(expected)
                || actual.mask != expected.mask
            {
                return Err(format!(
                    "compact metadata/raw mismatch {name}/{index}: {} != {}",
                    actual.value, expected.value
                ));
            }
        }
    }
    let run = || {
        let mut output = Columns {
            values: std::array::from_fn(|_| Vec::with_capacity(args.cells)),
            masks: std::array::from_fn(|_| Vec::with_capacity(args.cells)),
            classes: Vec::with_capacity(args.cells),
        };
        for index in 0..args.cells {
            let result = evaluate(
                &data.cell(index),
                [supplied[0][index], supplied[1][index], supplied[2][index]],
                f32_heat,
                &plan,
                fuel,
            );
            for (field, value) in result.values.into_iter().enumerate() {
                output.values[field].push(value.value);
                output.masks[field].push(value.mask);
            }
            output.classes.push(result.class);
        }
        Ok(output)
    };
    super::raster_probe::measure(args, run, |output| {
        super::raster_probe::dump_typed(
            &output.values,
            &output.masks,
            &output.classes,
            &reference[0],
        )?;
        for index in 0..args.cells {
            let expected = &reference[index % reference.len()];
            if output.classes[index] != expected.fire_type {
                return Err(format!("compact class/{index}"));
            }
            for (field, (name, value)) in expected.named_values().into_iter().enumerate() {
                if !super::raster_probe::check_value(output.values[field][index], value.value)
                    || output.masks[field][index] != value.mask
                {
                    return Err(format!("compact stored mismatch {name}/{index}"));
                }
            }
        }
        Ok(())
    })
}
