//! Fixed five-particle construction with dynamic herb transfer.
use super::Particles;
use crate::{engine::PreparedFuel, fuel_models::FuelModels, math::np_max};

/// Build the Python particle arrays for one grid cell; all moisture inputs are fractions.
pub fn build_particle_arrays(
    fuels: &FuelModels,
    fuel_model_grid: i32,
    m1h: f64,
    m10h: f64,
    m100h: f64,
    mlh: f64,
    mlw: f64,
) -> Particles {
    build_particle_arrays_from_prepared(
        &PreparedFuel::from_model(fuels.for_grid(fuel_model_grid)),
        m1h,
        m10h,
        m100h,
        mlh,
        mlw,
    )
}

/// Build particles from a raw lookup row without filtering its defined flag or scientific properties.
///
/// Moisture inputs are fractions. Dynamic transfer and moisture clamping match
/// the grid entry point. A false defined flag is preserved for the final surface
/// output mask, while nonzero raw lookup properties still participate in stages.
pub fn build_particle_arrays_from_prepared(
    fuel: &PreparedFuel,
    m1h: f64,
    m10h: f64,
    m100h: f64,
    mlh: f64,
    mlw: f64,
) -> Particles {
    let f = fuel;
    let m1h = np_max(m1h, 0.01);
    let m10h = np_max(m10h, 0.01);
    let m100h = np_max(m100h, 0.01);
    let mlh = np_max(mlh, 0.01);
    let mlw = np_max(mlw, 0.01);
    let mut p = Particles {
        load_dead: [f.dead_1h, f.dead_10h, f.dead_100h, 0.0, 0.0],
        load_live: [f.live_herb, f.live_woody, 0.0, 0.0, 0.0],
        savr_dead: [f.savr_1h, 109.0, 30.0, f.savr_live_herb, 0.0],
        savr_live: [f.savr_live_herb, f.savr_live_woody, 0.0, 0.0, 0.0],
        moisture_dead: [m1h, m10h, m100h, m1h, 0.0],
        moisture_live: [mlh, mlw, 0.0, 0.0, 0.0],
        hoc_dead: f.heat_of_combustion_dead,
        hoc_live: f.heat_of_combustion_live,
        depth: f.fuel_bed_depth,
        moe_dead: f.moisture_of_extinction_dead,
        is_defined: f.is_defined,
    };
    if f.is_dynamic && mlh < 0.30 {
        p.load_dead[3] = p.load_live[0];
        p.load_live[0] = 0.0;
    } else if f.is_dynamic && (0.30..=1.20).contains(&mlh) {
        let transferred = p.load_live[0] * (1.333 - 1.11 * mlh);
        p.load_dead[3] = transferred;
        p.load_live[0] -= transferred;
    }
    p
}

/// Size-class weighted net loading (lb/ft²), retaining the Albini bin boundaries exactly.
pub fn size_sorted_wfl(savr_arr: &[f64; 5], frac_arr: &[f64; 5], wn_arr: &[f64; 5]) -> f64 {
    let bounds = [
        (1200.0, f64::INFINITY),
        (192.0, 1200.0),
        (96.0, 192.0),
        (48.0, 96.0),
        (16.0, 48.0),
    ];
    let mut summed = [0.0; 5];
    for p in 0..5 {
        for (b, (lo, hi)) in bounds.iter().enumerate() {
            if savr_arr[p] >= *lo && (b == 0 || savr_arr[p] < *hi) {
                summed[b] += frac_arr[p];
            }
        }
    }
    let mut wfl = 0.0;
    for p in 0..5 {
        let mut assigned = 0.0;
        for (b, (lo, hi)) in bounds.iter().enumerate() {
            if savr_arr[p] >= *lo && (b == 0 || savr_arr[p] < *hi) {
                assigned += summed[b];
            }
        }
        wfl += assigned * wn_arr[p];
    }
    wfl
}
