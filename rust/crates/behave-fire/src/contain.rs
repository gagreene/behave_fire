//! Current fixed-step ellipse-growth containment model. This is not RK4.
use std::f64::consts::PI;

/// Reported status, before simulation starts.
pub const STATUS_REPORTED: i32 = 1;
/// Contained status.
pub const STATUS_CONTAINED: i32 = 3;
/// Overrun status.
pub const STATUS_OVERRUN: i32 = 4;
/// Reserved exhausted status, retained from the Python API.
pub const STATUS_EXHAUSTED: i32 = 5;
/// Iteration limit reached.
pub const STATUS_TIME_LIMIT: i32 = 8;

/// Per-cell containment inputs. Report size only sets the overrun threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContainInput {
    /// Reported spread rate in feet per minute.
    pub report_spread_rate: f64,
    /// Length-to-width ratio.
    pub lw_ratio: f64,
    /// Reported acres, used only as a threshold.
    pub report_size: f64,
    /// Line production in chains per hour.
    pub production_rate: f64,
    /// Attack delay in minutes.
    pub attack_time: f64,
    /// Maximum iteration count; zero and negative counts perform no steps.
    pub max_steps: i64,
    /// Distance step in chains.
    pub dist_step: f64,
}
impl Default for ContainInput {
    /// Default integration controls are 500 steps and 0.01 chains.
    fn default() -> Self {
        Self {
            report_spread_rate: 0.0,
            lw_ratio: 1.0,
            report_size: 0.0,
            production_rate: 0.0,
            attack_time: 0.0,
            max_steps: 500,
            dist_step: 0.01,
        }
    }
}

/// Containment output, with the exact Python dictionary names.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ContainResult {
    /// Final status integer.
    pub status: i32,
    /// Minutes to containment; zero for other terminal statuses.
    pub contained_time: f64,
    /// Final perimeter in chains.
    pub final_perimeter: f64,
    /// Final area in acres.
    pub final_area: f64,
}
impl ContainResult {
    /// Numeric output values, with status represented exactly as an f64.
    pub fn named_values(self) -> [(&'static str, f64); 4] {
        [
            ("status", f64::from(self.status)),
            ("contained_time", self.contained_time),
            ("final_perimeter", self.final_perimeter),
            ("final_area", self.final_area),
        ]
    }
}

/// Run one scalar cell with the current fixed-step model.
pub fn run(input: ContainInput) -> ContainResult {
    run_contain_sim_array(
        input.report_spread_rate,
        input.lw_ratio,
        input.report_size,
        input.production_rate,
        input.attack_time,
        input.max_steps,
        input.dist_step,
    )
}

/// Return unavailable terminal behavior when a required step condition is unknown.
fn unavailable() -> ContainResult {
    ContainResult {
        status: -1,
        contained_time: f64::NAN,
        final_perimeter: f64::NAN,
        final_area: f64::NAN,
    }
}

/// Run one cell; terminal cells freeze at the exact step of their transition.
pub fn run_contain_sim_array(
    report_spread_rate: f64,
    lw_ratio: f64,
    report_size: f64,
    production_rate: f64,
    attack_time: f64,
    max_steps: i64,
    dist_step: f64,
) -> ContainResult {
    let fire_ros = report_spread_rate / 66.0;
    let mut status = STATUS_REPORTED;
    let mut time = 0.0;
    let mut perimeter = 0.0;
    let mut area = 0.0;
    let mut line_built = 0.0;
    let mut active = true;
    for _ in 0..max_steps {
        if !active {
            break;
        }
        if fire_ros.is_nan() || lw_ratio.is_nan() || attack_time.is_nan() || dist_step.is_nan() {
            return unavailable();
        }
        let dt = dist_step / if fire_ros > 1e-7 { fire_ros } else { 1.0 };
        time += dt;
        let safe_lwr = if lw_ratio > 1e-7 { lw_ratio } else { 1.0 };
        let head_dist = fire_ros * time;
        perimeter = PI * head_dist * (1.0 + 1.0 / safe_lwr);
        area = PI * head_dist.powi(2) / (4.0 * safe_lwr) / 10.0;
        let line_rate = if time >= attack_time {
            production_rate / 60.0
        } else {
            0.0
        };
        line_built += line_rate * dt;
        if time.is_nan() || perimeter.is_nan() || area.is_nan() || line_built.is_nan() {
            return unavailable();
        }
        let contained_now = line_built >= perimeter && time >= attack_time;
        let overrun_now = if time < attack_time {
            if report_size.is_nan() {
                return unavailable();
            }
            area > report_size * 10.0
        } else {
            false
        };
        if contained_now {
            status = STATUS_CONTAINED;
        }
        if overrun_now {
            status = STATUS_OVERRUN;
        }
        active = !contained_now && !overrun_now;
    }
    if active {
        status = STATUS_TIME_LIMIT;
    }
    ContainResult {
        status,
        contained_time: if status == STATUS_CONTAINED {
            time
        } else {
            0.0
        },
        final_perimeter: perimeter,
        final_area: area,
    }
}
