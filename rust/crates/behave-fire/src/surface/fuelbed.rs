//! Fuelbed weighting follows the Python five-particle sums and guarded divisions.
use super::{FuelbedIntermediates, Particles};
use crate::math::np_max;

/// Calculate all twenty Python fuelbed dictionary fields for one cell, in base units.
pub fn calculate_fuelbed_intermediates(p: &Particles) -> FuelbedIntermediates {
    calculate_fuelbed_with_terms(p, &super::SavrTerms::new(&p.savr_dead, &p.savr_live))
}

/// Evaluate moisture-dependent fuelbed terms with validated prepared SAVR invariants.
pub(crate) fn calculate_fuelbed_with_terms(
    p: &Particles,
    terms: &super::SavrTerms,
) -> FuelbedIntermediates {
    calculate_fuelbed_selected::<true>(p, terms)
}

/// Share finite reductions while preserving unknown threshold conditions.
pub(crate) fn calculate_fuelbed_available_with_terms(
    p: &Particles,
    terms: &super::SavrTerms,
) -> FuelbedIntermediates {
    calculate_fuelbed_selected::<true>(p, terms)
}

fn calculate_fuelbed_selected<const STRICT: bool>(
    p: &Particles,
    terms: &super::SavrTerms,
) -> FuelbedIntermediates {
    let (fp_d, tsa_d) = particle_area_weights::<STRICT>(&p.load_dead, &p.savr_dead);
    let (fp_l, tsa_l) = particle_area_weights::<STRICT>(&p.load_live, &p.savr_live);
    let frac_dead = dead_area_fraction::<STRICT>(tsa_d, tsa_l);
    let frac_live = 1.0 - frac_dead;
    let exp_d = terms.exp_dead;
    let exp_l500 = terms.exp_live500;
    let fine_d: f64 = (0..5).map(|n| p.load_dead[n] * exp_d[n]).sum();
    let fine_dm: f64 = (0..5)
        .map(|n| p.load_dead[n] * exp_d[n] * p.moisture_dead[n])
        .sum();
    let fine_l: f64 = (0..5).map(|n| p.load_live[n] * exp_l500[n]).sum();
    let moe_dead = p.moe_dead;
    let moe_live = live_extinction_from_fines::<STRICT>(fine_d, fine_dm, fine_l, moe_dead);
    let w_heat_dead = weighted_constant(&fp_d, p.hoc_dead);
    let w_heat_live = weighted_constant(&fp_l, p.hoc_live);
    let w_silica_dead = weighted_constant(&fp_d, 0.01);
    let w_silica_live = (0..5)
        .map(|n| fp_l[n] * if n < 2 { 0.01 } else { 0.0 })
        .sum();
    let w_moist_dead = weighted_values(&fp_d, &p.moisture_dead);
    let w_moist_live = weighted_values(&fp_l, &p.moisture_live);
    let sigma = mixture_sigma(
        frac_dead,
        frac_live,
        &fp_d,
        &p.savr_dead,
        &fp_l,
        &p.savr_live,
    );
    let wn_d = std::array::from_fn(|n| p.load_dead[n] * (1.0 - 0.0555));
    let wn_l = std::array::from_fn(|n| p.load_live[n] * (1.0 - 0.0555));
    let wfl_dead = super::size_sorted_prepared(&terms.bins_dead, &fp_d, &wn_d);
    let wfl_live = super::size_sorted_prepared(&terms.bins_live, &fp_l, &wn_l);
    let depth = p.depth;
    let bulk_density = bulk_from_loads::<STRICT>(&p.load_dead, &p.load_live, depth);
    let packing_ratio = packing_from_loads::<STRICT>(&p.load_dead, &p.load_live, depth);
    let relative_packing_ratio = relative_packing_from_sigma::<STRICT>(sigma, packing_ratio);
    let hs_d = heat_sink_side(frac_dead, &fp_d, &p.moisture_dead, &exp_d);
    let hs_l = heat_sink_side(frac_live, &fp_l, &p.moisture_live, &terms.exp_live138);
    let heat_sink = (hs_d + hs_l) * bulk_density;
    let propagating_flux = propagating_flux_from_sigma::<STRICT>(sigma, packing_ratio);
    FuelbedIntermediates {
        sigma,
        bulk_density,
        packing_ratio,
        relative_packing_ratio,
        heat_sink,
        propagating_flux,
        w_heat_dead,
        w_heat_live,
        w_silica_dead,
        w_silica_live,
        w_moist_dead,
        w_moist_live,
        wfl_dead,
        wfl_live,
        moe_dead,
        moe_live,
        frac_dead,
        frac_live,
        depth,
        is_defined: p.is_defined,
    }
}

/// Sum actual particle values in the existing weighted operation order.
#[inline]
pub(crate) fn weighted_values(weights: &[f64; 5], values: &[f64; 5]) -> f64 {
    (0..5).map(|n| weights[n] * values[n]).sum()
}

/// Apply the same constant separately to each weight, preserving summation order.
#[inline]
pub(crate) fn weighted_constant(weights: &[f64; 5], value: f64) -> f64 {
    (0..5).map(|n| weights[n] * value).sum()
}

/// Resolve live extinction from actual fine-fuel totals and dead extinction.
#[inline]
pub(crate) fn live_extinction_from_fines<const STRICT: bool>(
    fine_d: f64,
    fine_dm: f64,
    fine_l: f64,
    dead: f64,
) -> f64 {
    let safe_d = if fine_d > 1e-7 { fine_d } else { 1. };
    let moisture = if STRICT && fine_d.is_nan() {
        f64::NAN
    } else if fine_d > 1e-7 {
        fine_dm / safe_d
    } else {
        0.
    };
    let safe_l = if fine_l > 1e-7 { fine_l } else { 1. };
    let ratio = if STRICT && fine_l.is_nan() {
        f64::NAN
    } else if fine_l > 1e-7 {
        fine_d / safe_l
    } else {
        0.
    };
    let safe_m = if STRICT && dead.is_nan() {
        f64::NAN
    } else if dead > 1e-7 {
        dead
    } else {
        1.
    };
    np_max(2.9 * ratio * (1. - moisture / safe_m) - 0.226, dead)
}

/// Calculate bulk density using actual load totals and the three-valued depth guard.
#[inline]
pub(crate) fn bulk_from_loads<const STRICT: bool>(
    dead: &[f64; 5],
    live: &[f64; 5],
    depth: f64,
) -> f64 {
    let total = dead.iter().sum::<f64>() + live.iter().sum::<f64>();
    let safe = if depth > 1e-7 { depth } else { 1. };
    if STRICT && depth.is_nan() {
        f64::NAN
    } else if depth > 1e-7 {
        total / safe
    } else {
        0.
    }
}

/// Retain particle-by-particle packing division, rather than reordering it algebraically.
#[inline]
pub(crate) fn packing_from_loads<const STRICT: bool>(
    dead: &[f64; 5],
    live: &[f64; 5],
    depth: f64,
) -> f64 {
    let safe = if depth > 1e-7 { depth } else { 1. };
    if STRICT && depth.is_nan() {
        f64::NAN
    } else if depth > 1e-7 {
        dead.iter().map(|v| v / (safe * 32.)).sum::<f64>()
            + live.iter().map(|v| v / (safe * 32.)).sum::<f64>()
    } else {
        0.
    }
}

/// Calculate one actual moisture/area side of the heat-sink sum.
#[inline]
pub(crate) fn heat_sink_side(
    fraction: f64,
    weights: &[f64; 5],
    moisture: &[f64; 5],
    exponent: &[f64; 5],
) -> f64 {
    (0..5)
        .map(|n| fraction * weights[n] * (250. + 1116. * moisture[n]) * exponent[n])
        .sum()
}

/// Return exact area weights and their sum for one actual particle side.
#[inline]
pub(crate) fn particle_area_weights<const STRICT: bool>(
    load: &[f64; 5],
    savr: &[f64; 5],
) -> ([f64; 5], f64) {
    let area: [f64; 5] = std::array::from_fn(|n| load[n] * savr[n] / 32.);
    let total: f64 = area.iter().sum();
    let safe = if STRICT && total.is_nan() {
        f64::NAN
    } else if total > 1e-7 {
        total
    } else {
        1.
    };
    (std::array::from_fn(|n| area[n] / safe), total)
}

/// Resolve the dead area fraction without converting an unknown denominator to zero.
#[inline]
pub(crate) fn dead_area_fraction<const STRICT: bool>(dead: f64, live: f64) -> f64 {
    let total = dead + live;
    let safe = if STRICT && total.is_nan() {
        f64::NAN
    } else if total > 1e-7 {
        total
    } else {
        1.
    };
    if STRICT && total.is_nan() {
        f64::NAN
    } else if total > 1e-7 {
        dead / safe
    } else {
        0.
    }
}

/// Calculate characteristic SAVR from the actual, potentially injected mixture fractions.
#[inline]
pub(crate) fn mixture_sigma(
    dead: f64,
    live: f64,
    wd: &[f64; 5],
    sd: &[f64; 5],
    wl: &[f64; 5],
    sl: &[f64; 5],
) -> f64 {
    let dead_savr: f64 = (0..5).map(|n| wd[n] * sd[n]).sum();
    let live_savr: f64 = (0..5).map(|n| wl[n] * sl[n]).sum();
    dead * dead_savr + live * live_savr
}

/// Share the actual sigma/packing dependency without rebuilding unrelated fuelbed fields.
#[inline]
pub(crate) fn relative_packing_from_sigma<const STRICT: bool>(sigma: f64, packing: f64) -> f64 {
    let safe_sigma = if sigma > 1e-7 { sigma } else { 1.0 };
    let opt = if STRICT && sigma.is_nan() {
        f64::NAN
    } else if sigma > 1e-7 {
        3.348 / safe_sigma.powf(0.8189)
    } else {
        1.0
    };
    let safe_opt = if opt > 1e-7 { opt } else { 1.0 };
    if STRICT && crate::engine::missing::all_above(&[(sigma, 1e-7), (opt, 1e-7)]).is_none() {
        f64::NAN
    } else if sigma > 1e-7 && opt > 1e-7 {
        packing / safe_opt
    } else {
        0.0
    }
}

/// Share the propagating-flux equation with field-level staged evaluation.
#[inline]
pub(crate) fn propagating_flux_from_sigma<const STRICT: bool>(sigma: f64, packing: f64) -> f64 {
    if STRICT && sigma.is_nan() {
        f64::NAN
    } else if sigma > 1e-7 {
        ((0.792 + 0.681 * sigma.sqrt()) * (packing + 0.1)).exp() / (192.0 + 0.2595 * sigma)
    } else {
        0.0
    }
}

/// Static fuelbed coefficients; moisture-dependent fields are rebuilt for every cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PreparedFuelbed {
    pub template: FuelbedIntermediates,
    fp_dead: [f64; 5],
    fp_live: [f64; 5],
    fine_dead: [f64; 5],
    fine_dead_total: f64,
    fine_live: f64,
    heat_dead: [f64; 5],
    heat_live: [f64; 5],
    exp_dead: [f64; 5],
    exp_live: [f64; 5],
}
impl PreparedFuelbed {
    /// Retain fast cached finite invariants; rebuild missing static thresholds strictly.
    pub(crate) fn run_available(
        &self,
        p: &Particles,
        terms: &super::SavrTerms,
    ) -> FuelbedIntermediates {
        if p.depth.is_nan() || self.template.sigma.is_nan() {
            return calculate_fuelbed_available_with_terms(p, terms);
        }
        self.run(p)
    }
    /// Prepare physical terms only after dynamic-transfer eligibility is checked.
    pub(crate) fn new(p: &Particles, terms: &super::SavrTerms) -> Self {
        let template = calculate_fuelbed_with_terms(p, terms);
        let sa_dead: [f64; 5] = std::array::from_fn(|i| p.load_dead[i] * p.savr_dead[i] / 32.0);
        let sa_live: [f64; 5] = std::array::from_fn(|i| p.load_live[i] * p.savr_live[i] / 32.0);
        let total_dead: f64 = sa_dead.iter().sum();
        let total_live: f64 = sa_live.iter().sum();
        let fp_dead = sa_dead.map(|v| v / if total_dead > 1e-7 { total_dead } else { 1.0 });
        let fp_live = sa_live.map(|v| v / if total_live > 1e-7 { total_live } else { 1.0 });
        let fine_dead = std::array::from_fn(|i| p.load_dead[i] * terms.exp_dead[i]);
        Self {
            template,
            fp_dead,
            fp_live,
            fine_dead,
            fine_dead_total: fine_dead.iter().sum(),
            fine_live: (0..5).map(|i| p.load_live[i] * terms.exp_live500[i]).sum(),
            heat_dead: fp_dead.map(|f| template.frac_dead * f),
            heat_live: fp_live.map(|f| template.frac_live * f),
            exp_dead: terms.exp_dead,
            exp_live: terms.exp_live138,
        }
    }

    /// Restore exactly the moisture-dependent reductions in their reference order.
    pub(crate) fn run(&self, p: &Particles) -> FuelbedIntermediates {
        let mut result = self.template;
        let fine_dm: f64 = (0..5).map(|i| self.fine_dead[i] * p.moisture_dead[i]).sum();
        let safe_fd = if self.fine_dead_total > 1e-7 {
            self.fine_dead_total
        } else {
            1.0
        };
        let fine_moisture = if self.fine_dead_total > 1e-7 {
            fine_dm / safe_fd
        } else {
            0.0
        };
        let safe_fl = if self.fine_live > 1e-7 {
            self.fine_live
        } else {
            1.0
        };
        let fdol = if self.fine_live > 1e-7 {
            self.fine_dead_total / safe_fl
        } else {
            0.0
        };
        let safe_md = if p.moe_dead > 1e-7 { p.moe_dead } else { 1.0 };
        result.moe_live = np_max(
            2.9 * fdol * (1.0 - fine_moisture / safe_md) - 0.226,
            p.moe_dead,
        );
        result.w_moist_dead = (0..5).map(|i| self.fp_dead[i] * p.moisture_dead[i]).sum();
        result.w_moist_live = (0..5).map(|i| self.fp_live[i] * p.moisture_live[i]).sum();
        let hs_dead: f64 = (0..5)
            .map(|i| self.heat_dead[i] * (250.0 + 1116.0 * p.moisture_dead[i]) * self.exp_dead[i])
            .sum();
        let hs_live: f64 = (0..5)
            .map(|i| self.heat_live[i] * (250.0 + 1116.0 * p.moisture_live[i]) * self.exp_live[i])
            .sum();
        result.heat_sink = (hs_dead + hs_live) * result.bulk_density;
        result
    }
}
