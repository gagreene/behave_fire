//! Fuelbed weighting follows the Python five-particle sums and guarded divisions.
use super::{size_sorted_wfl, FuelbedIntermediates, Particles};
use crate::math::np_max;

/// Float32 particle arithmetic, with f64 silica constants and size-bin accumulation as in Python.
pub(super) mod single {
    use super::{size_sorted_wfl, FuelbedIntermediates, Particles};
    use crate::math::single::np_max;
    struct SingleParticles {
        load_dead: [f32; 5],
        load_live: [f32; 5],
        savr_dead: [f32; 5],
        savr_live: [f32; 5],
        moisture_dead: [f32; 5],
        moisture_live: [f32; 5],
        hoc_dead: f32,
        hoc_live: f32,
        depth: f32,
        moe_dead: f32,
        is_defined: bool,
    }
    /// Compute each float32 particle operation natively, widening only the final transport fields.
    pub fn calculate_fuelbed_intermediates(p: &Particles) -> FuelbedIntermediates {
        let p = SingleParticles {
            load_dead: std::array::from_fn(|n| p.load_dead[n] as f32),
            load_live: std::array::from_fn(|n| p.load_live[n] as f32),
            savr_dead: std::array::from_fn(|n| p.savr_dead[n] as f32),
            savr_live: std::array::from_fn(|n| p.savr_live[n] as f32),
            moisture_dead: std::array::from_fn(|n| p.moisture_dead[n] as f32),
            moisture_live: std::array::from_fn(|n| p.moisture_live[n] as f32),
            hoc_dead: p.hoc_dead as f32,
            hoc_live: p.hoc_live as f32,
            depth: p.depth as f32,
            moe_dead: p.moe_dead as f32,
            is_defined: p.is_defined,
        };
        let sa_d: [f32; 5] = std::array::from_fn(|n| p.load_dead[n] * p.savr_dead[n] / 32.0);
        let sa_l: [f32; 5] = std::array::from_fn(|n| p.load_live[n] * p.savr_live[n] / 32.0);
        let tsa_d: f32 = sa_d.iter().sum();
        let tsa_l: f32 = sa_l.iter().sum();
        let tsa_total = tsa_d + tsa_l;
        let safe_tsa = if tsa_total > 1e-7 { tsa_total } else { 1.0 };
        let frac_dead = if tsa_total > 1e-7 {
            tsa_d / safe_tsa
        } else {
            0.0
        };
        let frac_live = 1.0 - frac_dead;
        let safe_d = if tsa_d > 1e-7 { tsa_d } else { 1.0 };
        let safe_l = if tsa_l > 1e-7 { tsa_l } else { 1.0 };
        let fp_d: [f32; 5] = std::array::from_fn(|n| sa_d[n] / safe_d);
        let fp_l: [f32; 5] = std::array::from_fn(|n| sa_l[n] / safe_l);
        let exp_d: [f32; 5] = std::array::from_fn(|n| {
            if p.savr_dead[n] > 1e-7 {
                (-138.0 / p.savr_dead[n]).exp()
            } else {
                0.0
            }
        });
        let exp_l500: [f32; 5] = std::array::from_fn(|n| {
            if p.savr_live[n] > 1e-7 {
                (-500.0 / p.savr_live[n]).exp()
            } else {
                0.0
            }
        });
        let fine_d: f32 = (0..5).map(|n| p.load_dead[n] * exp_d[n]).sum();
        let fine_dm: f32 = (0..5)
            .map(|n| p.load_dead[n] * exp_d[n] * p.moisture_dead[n])
            .sum();
        let fine_l: f32 = (0..5).map(|n| p.load_live[n] * exp_l500[n]).sum();
        let safe_fd = if fine_d > 1e-7 { fine_d } else { 1.0 };
        let fine_dead_moist = if fine_d > 1e-7 {
            fine_dm / safe_fd
        } else {
            0.0
        };
        let safe_fl = if fine_l > 1e-7 { fine_l } else { 1.0 };
        let fdol = if fine_l > 1e-7 { fine_d / safe_fl } else { 0.0 };
        let moe_dead = p.moe_dead;
        let safe_md = if moe_dead > 1e-7 { moe_dead } else { 1.0 };
        let moe_live = np_max(
            2.9 * fdol * (1.0 - fine_dead_moist / safe_md) - 0.226,
            moe_dead,
        );
        let w_heat_dead: f32 = (0..5).map(|n| fp_d[n] * p.hoc_dead).sum();
        let w_heat_live: f32 = (0..5).map(|n| fp_l[n] * p.hoc_live).sum();
        let w_silica_dead: f64 = (0..5).map(|n| (fp_d[n] as f64) * 0.01).sum();
        let w_silica_live: f64 = (0..5)
            .map(|n| (fp_l[n] as f64) * if n < 2 { 0.01 } else { 0.0 })
            .sum();
        let w_moist_dead: f32 = (0..5).map(|n| fp_d[n] * p.moisture_dead[n]).sum();
        let w_moist_live: f32 = (0..5).map(|n| fp_l[n] * p.moisture_live[n]).sum();
        let w_savr_dead: f32 = (0..5).map(|n| fp_d[n] * p.savr_dead[n]).sum();
        let w_savr_live: f32 = (0..5).map(|n| fp_l[n] * p.savr_live[n]).sum();
        let sigma = frac_dead * w_savr_dead + frac_live * w_savr_live;
        // Python evaluates the silica subtraction as f64 before the f32 array multiplication.
        let wn_d: [f32; 5] =
            std::array::from_fn(|n| p.load_dead[n] * ((1.0f64 - 0.0555f64) as f32));
        let wn_l: [f32; 5] =
            std::array::from_fn(|n| p.load_live[n] * ((1.0f64 - 0.0555f64) as f32));
        let wfl_dead = size_sorted_wfl(
            &p.savr_dead.map(f64::from),
            &fp_d.map(f64::from),
            &wn_d.map(f64::from),
        );
        let wfl_live = size_sorted_wfl(
            &p.savr_live.map(f64::from),
            &fp_l.map(f64::from),
            &wn_l.map(f64::from),
        );
        let total_load = p.load_dead.iter().sum::<f32>() + p.load_live.iter().sum::<f32>();
        let depth = p.depth;
        let safe_depth = if depth > 1e-7 { depth } else { 1.0 };
        let bulk_density = if depth > 1e-7 {
            total_load / safe_depth
        } else {
            0.0
        };
        let packing_ratio = if depth > 1e-7 {
            p.load_dead
                .iter()
                .map(|v| v / (safe_depth * 32.0))
                .sum::<f32>()
                + p.load_live
                    .iter()
                    .map(|v| v / (safe_depth * 32.0))
                    .sum::<f32>()
        } else {
            0.0
        };
        let safe_sigma = if sigma > 1e-7 { sigma } else { 1.0 };
        let opt = if sigma > 1e-7 {
            3.348 / safe_sigma.powf(0.8189)
        } else {
            1.0
        };
        let safe_opt = if opt > 1e-7 { opt } else { 1.0 };
        let relative_packing_ratio = if sigma > 1e-7 && opt > 1e-7 {
            packing_ratio / safe_opt
        } else {
            0.0
        };
        let hs_d: f32 = (0..5)
            .map(|n| frac_dead * fp_d[n] * (250.0 + 1116.0 * p.moisture_dead[n]) * exp_d[n])
            .sum();
        let hs_l: f32 = (0..5)
            .map(|n| {
                let exp = if p.savr_live[n] > 1e-7 {
                    (-138.0 / p.savr_live[n]).exp()
                } else {
                    0.0
                };
                frac_live * fp_l[n] * (250.0 + 1116.0 * p.moisture_live[n]) * exp
            })
            .sum();
        let heat_sink = (hs_d + hs_l) * bulk_density;
        let propagating_flux = if sigma > 1e-7 {
            ((0.792 + 0.681 * safe_sigma.sqrt()) * (packing_ratio + 0.1)).exp()
                / (192.0 + 0.2595 * safe_sigma)
        } else {
            0.0
        };
        FuelbedIntermediates {
            sigma: sigma as f64,
            bulk_density: bulk_density as f64,
            packing_ratio: packing_ratio as f64,
            relative_packing_ratio: relative_packing_ratio as f64,
            heat_sink: heat_sink as f64,
            propagating_flux: propagating_flux as f64,
            w_heat_dead: w_heat_dead as f64,
            w_heat_live: w_heat_live as f64,
            w_silica_dead,
            w_silica_live,
            w_moist_dead: w_moist_dead as f64,
            w_moist_live: w_moist_live as f64,
            wfl_dead: wfl_dead as f64,
            wfl_live: wfl_live as f64,
            moe_dead: moe_dead as f64,
            moe_live: moe_live as f64,
            frac_dead: frac_dead as f64,
            frac_live: frac_live as f64,
            depth: depth as f64,
            is_defined: p.is_defined,
        }
    }
}

/// Calculate all twenty Python fuelbed dictionary fields for one cell, in base units.
pub fn calculate_fuelbed_intermediates(p: &Particles) -> FuelbedIntermediates {
    let sa_d: [f64; 5] = std::array::from_fn(|n| p.load_dead[n] * p.savr_dead[n] / 32.0);
    let sa_l: [f64; 5] = std::array::from_fn(|n| p.load_live[n] * p.savr_live[n] / 32.0);
    let tsa_d: f64 = sa_d.iter().sum();
    let tsa_l: f64 = sa_l.iter().sum();
    let tsa_total = tsa_d + tsa_l;
    let safe_tsa = if tsa_total > 1e-7 { tsa_total } else { 1.0 };
    let frac_dead = if tsa_total > 1e-7 {
        tsa_d / safe_tsa
    } else {
        0.0
    };
    let frac_live = 1.0 - frac_dead;
    let safe_d = if tsa_d > 1e-7 { tsa_d } else { 1.0 };
    let safe_l = if tsa_l > 1e-7 { tsa_l } else { 1.0 };
    let fp_d: [f64; 5] = std::array::from_fn(|n| sa_d[n] / safe_d);
    let fp_l: [f64; 5] = std::array::from_fn(|n| sa_l[n] / safe_l);
    let exp_d: [f64; 5] = std::array::from_fn(|n| {
        if p.savr_dead[n] > 1e-7 {
            (-138.0 / p.savr_dead[n]).exp()
        } else {
            0.0
        }
    });
    let exp_l500: [f64; 5] = std::array::from_fn(|n| {
        if p.savr_live[n] > 1e-7 {
            (-500.0 / p.savr_live[n]).exp()
        } else {
            0.0
        }
    });
    let fine_d: f64 = (0..5).map(|n| p.load_dead[n] * exp_d[n]).sum();
    let fine_dm: f64 = (0..5)
        .map(|n| p.load_dead[n] * exp_d[n] * p.moisture_dead[n])
        .sum();
    let fine_l: f64 = (0..5).map(|n| p.load_live[n] * exp_l500[n]).sum();
    let safe_fd = if fine_d > 1e-7 { fine_d } else { 1.0 };
    let fine_dead_moist = if fine_d > 1e-7 {
        fine_dm / safe_fd
    } else {
        0.0
    };
    let safe_fl = if fine_l > 1e-7 { fine_l } else { 1.0 };
    let fdol = if fine_l > 1e-7 { fine_d / safe_fl } else { 0.0 };
    let moe_dead = p.moe_dead;
    let safe_md = if moe_dead > 1e-7 { moe_dead } else { 1.0 };
    let moe_live = np_max(
        2.9 * fdol * (1.0 - fine_dead_moist / safe_md) - 0.226,
        moe_dead,
    );
    let w_heat_dead = (0..5).map(|n| fp_d[n] * p.hoc_dead).sum();
    let w_heat_live = (0..5).map(|n| fp_l[n] * p.hoc_live).sum();
    let w_silica_dead = (0..5).map(|n| fp_d[n] * 0.01).sum();
    let w_silica_live = (0..5)
        .map(|n| fp_l[n] * if n < 2 { 0.01 } else { 0.0 })
        .sum();
    let w_moist_dead = (0..5).map(|n| fp_d[n] * p.moisture_dead[n]).sum();
    let w_moist_live = (0..5).map(|n| fp_l[n] * p.moisture_live[n]).sum();
    let w_savr_dead: f64 = (0..5).map(|n| fp_d[n] * p.savr_dead[n]).sum();
    let w_savr_live: f64 = (0..5).map(|n| fp_l[n] * p.savr_live[n]).sum();
    let sigma = frac_dead * w_savr_dead + frac_live * w_savr_live;
    let wn_d = std::array::from_fn(|n| p.load_dead[n] * (1.0 - 0.0555));
    let wn_l = std::array::from_fn(|n| p.load_live[n] * (1.0 - 0.0555));
    let wfl_dead = size_sorted_wfl(&p.savr_dead, &fp_d, &wn_d);
    let wfl_live = size_sorted_wfl(&p.savr_live, &fp_l, &wn_l);
    let total_load = p.load_dead.iter().sum::<f64>() + p.load_live.iter().sum::<f64>();
    let depth = p.depth;
    let safe_depth = if depth > 1e-7 { depth } else { 1.0 };
    let bulk_density = if depth > 1e-7 {
        total_load / safe_depth
    } else {
        0.0
    };
    let packing_ratio = if depth > 1e-7 {
        p.load_dead
            .iter()
            .map(|v| v / (safe_depth * 32.0))
            .sum::<f64>()
            + p.load_live
                .iter()
                .map(|v| v / (safe_depth * 32.0))
                .sum::<f64>()
    } else {
        0.0
    };
    let safe_sigma = if sigma > 1e-7 { sigma } else { 1.0 };
    let opt = if sigma > 1e-7 {
        3.348 / safe_sigma.powf(0.8189)
    } else {
        1.0
    };
    let safe_opt = if opt > 1e-7 { opt } else { 1.0 };
    let relative_packing_ratio = if sigma > 1e-7 && opt > 1e-7 {
        packing_ratio / safe_opt
    } else {
        0.0
    };
    let hs_d: f64 = (0..5)
        .map(|n| frac_dead * fp_d[n] * (250.0 + 1116.0 * p.moisture_dead[n]) * exp_d[n])
        .sum();
    let hs_l: f64 = (0..5)
        .map(|n| {
            let exp = if p.savr_live[n] > 1e-7 {
                (-138.0 / p.savr_live[n]).exp()
            } else {
                0.0
            };
            frac_live * fp_l[n] * (250.0 + 1116.0 * p.moisture_live[n]) * exp
        })
        .sum();
    let heat_sink = (hs_d + hs_l) * bulk_density;
    let propagating_flux = if sigma > 1e-7 {
        ((0.792 + 0.681 * safe_sigma.sqrt()) * (packing_ratio + 0.1)).exp()
            / (192.0 + 0.2595 * safe_sigma)
    } else {
        0.0
    };
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
