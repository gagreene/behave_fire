//! Immutable physical wind terms shared by forward spread and Crown backsolve.
use super::{FuelbedIntermediates, WindCoefficients};
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PreparedWind {
    /// Sigma-dependent coefficients, computed only when physical fuel changes.
    pub coefficients: WindCoefficients,
    sigma: f64,
    packing: f64,
    positive: f64,
    negative: f64,
}
impl PreparedWind {
    /// Prepare both powers separately to retain the reference's rounding order.
    pub fn new(bed: &FuelbedIntermediates) -> Self {
        let c = super::calculate_surface_wind_coefficients(bed.sigma);
        let packing = if bed.relative_packing_ratio > 1e-7 {
            bed.relative_packing_ratio
        } else {
            1.
        };
        Self {
            coefficients: c,
            sigma: bed.sigma,
            packing: bed.relative_packing_ratio,
            positive: packing.powf(c.wind_e),
            negative: packing.powf(-c.wind_e),
        }
    }
    /// Reject stale terms after supplied fuelbed or physical fuel edits.
    pub fn matches(&self, bed: &FuelbedIntermediates) -> bool {
        self.sigma.to_bits() == bed.sigma.to_bits()
            && self.packing.to_bits() == bed.relative_packing_ratio.to_bits()
    }
    /// Evaluate the ordinary wind equation with an owned invariant packing power.
    pub fn wind_phi(&self, mid: f64) -> f64 {
        if self.sigma.is_nan() || mid.is_nan() || self.packing.is_nan() {
            return super::calculate_wind_phi(
                self.sigma,
                mid,
                self.coefficients.wind_b,
                self.coefficients.wind_c,
                self.packing,
                self.coefficients.wind_e,
            );
        }
        if self.sigma > 1e-7 && mid > 1e-7 {
            super::wind::wind_phi_product(
                mid,
                self.coefficients.wind_b,
                self.coefficients.wind_c,
                self.negative,
            )
        } else {
            0.
        }
    }
    /// Evaluate effective wind with the same threshold and product order as the ordinary helper.
    pub fn effective(&self, spread: f64, r0: f64) -> f64 {
        let c = self.coefficients;
        if [spread, r0, c.wind_b, c.wind_c, self.packing, c.wind_e]
            .iter()
            .any(|v| v.is_nan())
        {
            return super::calculate_effective_wind_speed(
                spread,
                r0,
                c.wind_b,
                c.wind_c,
                self.packing,
                c.wind_e,
            );
        }
        let r0_safe = if r0 > 1e-7 { r0 } else { 1. };
        let phi = if r0 > 1e-7 { spread / r0_safe - 1. } else { 0. };
        if phi.is_nan() {
            return super::calculate_effective_wind_speed(
                spread,
                r0,
                c.wind_b,
                c.wind_c,
                self.packing,
                c.wind_e,
            );
        }
        let wc = if c.wind_c > 1e-7 { c.wind_c } else { 1. };
        let wb = if c.wind_b > 1e-7 { c.wind_b } else { 1. };
        let fpm = if phi > 0. && c.wind_b > 1e-7 && c.wind_c > 1e-7 && self.packing > 1e-7 {
            super::wind::effective_wind_power(phi, wb, wc, self.positive)
        } else {
            0.
        };
        fpm * 0.01136363636
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{self, FuelbedIntermediates};

    fn same(actual: f64, expected: f64) {
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn prepared_wind_matches_ordinary_equations_at_thresholds_and_missing_values() {
        for sigma in [-0., 0., 1e-7, 1633., f64::NAN, f64::INFINITY] {
            for packing in [-0., 1e-7, 0.31, f64::NAN, f64::INFINITY] {
                let bed = FuelbedIntermediates {
                    sigma,
                    relative_packing_ratio: packing,
                    ..FuelbedIntermediates::default()
                };
                let prepared = PreparedWind::new(&bed);
                let wc = surface::calculate_surface_wind_coefficients(sigma);
                for mid in [-0., 1e-7, 37., f64::NAN, f64::INFINITY] {
                    same(
                        prepared.wind_phi(mid),
                        surface::calculate_wind_phi(
                            sigma, mid, wc.wind_b, wc.wind_c, packing, wc.wind_e,
                        ),
                    );
                }
                for r0 in [-0., 1e-7, 3., f64::NAN, f64::INFINITY] {
                    for spread in [-0., 1e-7, 17., f64::NAN, f64::INFINITY] {
                        same(
                            prepared.effective(spread, r0),
                            surface::calculate_effective_wind_speed(
                                spread, r0, wc.wind_b, wc.wind_c, packing, wc.wind_e,
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_wind_rejects_changed_dependencies_and_keeps_zero_signs_distinct() {
        let mut bed = FuelbedIntermediates {
            sigma: 1600.,
            relative_packing_ratio: 0.3,
            ..FuelbedIntermediates::default()
        };
        let prepared = PreparedWind::new(&bed);
        assert!(prepared.matches(&bed));
        bed.sigma = 1601.;
        assert!(!prepared.matches(&bed));
        bed.sigma = 1600.;
        bed.relative_packing_ratio = 0.4;
        assert!(!prepared.matches(&bed));
        bed.relative_packing_ratio = 0.;
        let zero = PreparedWind::new(&bed);
        bed.relative_packing_ratio = -0.;
        assert!(!zero.matches(&bed));
    }
}
