//! Typed masked particle/fuelbed dictionaries and translated reduction stages.
use super::{
    c, finish, validate_condition_and, MaskedCondition, MaskedError, MaskedValue, Precision,
};

impl Default for MaskedValue {
    /// Construct an ordinary zero-valued f64 array element.
    fn default() -> Self {
        Self::plain(0.0)
    }
}
/// Particle arrays and per-cell properties with independent mask/dtype/rank metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct MaskedParticles {
    /// Whether the spatial shape is scalar; particle arrays still have a leading five-particle axis.
    pub scalar_cell: bool,
    /// Python `load_dead` field with its original masked-array metadata.
    pub load_dead: [MaskedValue; 5],
    /// Python `load_live` field with its original masked-array metadata.
    pub load_live: [MaskedValue; 5],
    /// Python `savr_dead` field with its original masked-array metadata.
    pub savr_dead: [MaskedValue; 5],
    /// Python `savr_live` field with its original masked-array metadata.
    pub savr_live: [MaskedValue; 5],
    /// Python `moisture_dead` field with its original masked-array metadata.
    pub moisture_dead: [MaskedValue; 5],
    /// Python `moisture_live` field with its original masked-array metadata.
    pub moisture_live: [MaskedValue; 5],
    /// Python `hoc_dead` field with its original masked-array metadata.
    pub hoc_dead: MaskedValue,
    /// Python `hoc_live` field with its original masked-array metadata.
    pub hoc_live: MaskedValue,
    /// Python `depth` field with its original masked-array metadata.
    pub depth: MaskedValue,
    /// Python `moe_dead` field with its original masked-array metadata.
    pub moe_dead: MaskedValue,
    /// Python `is_defined` field with its original masked-array metadata.
    pub is_defined: MaskedValue,
}
impl MaskedParticles {
    /// Construct an empty masked-particle transport record with ordinary f64 defaults.
    pub fn new() -> Self {
        Self::default()
    }
}
/// Every fuelbed dictionary field with independent masked-array metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct MaskedFuelbedIntermediates {
    /// Python `sigma` field with its original masked-array metadata.
    pub sigma: MaskedValue,
    /// Python `bulk_density` field with its original masked-array metadata.
    pub bulk_density: MaskedValue,
    /// Python `packing_ratio` field with its original masked-array metadata.
    pub packing_ratio: MaskedValue,
    /// Python `relative_packing_ratio` field with its original masked-array metadata.
    pub relative_packing_ratio: MaskedValue,
    /// Python `heat_sink` field with its original masked-array metadata.
    pub heat_sink: MaskedValue,
    /// Python `propagating_flux` field with its original masked-array metadata.
    pub propagating_flux: MaskedValue,
    /// Python `w_heat_dead` field with its original masked-array metadata.
    pub w_heat_dead: MaskedValue,
    /// Python `w_heat_live` field with its original masked-array metadata.
    pub w_heat_live: MaskedValue,
    /// Python `w_silica_dead` field with its original masked-array metadata.
    pub w_silica_dead: MaskedValue,
    /// Python `w_silica_live` field with its original masked-array metadata.
    pub w_silica_live: MaskedValue,
    /// Python `w_moist_dead` field with its original masked-array metadata.
    pub w_moist_dead: MaskedValue,
    /// Python `w_moist_live` field with its original masked-array metadata.
    pub w_moist_live: MaskedValue,
    /// Python `wfl_dead` field with its original masked-array metadata.
    pub wfl_dead: MaskedValue,
    /// Python `wfl_live` field with its original masked-array metadata.
    pub wfl_live: MaskedValue,
    /// Python `moe_dead` field with its original masked-array metadata.
    pub moe_dead: MaskedValue,
    /// Python `moe_live` field with its original masked-array metadata.
    pub moe_live: MaskedValue,
    /// Python `frac_dead` field with its original masked-array metadata.
    pub frac_dead: MaskedValue,
    /// Python `frac_live` field with its original masked-array metadata.
    pub frac_live: MaskedValue,
    /// Python `depth` field with its original masked-array metadata.
    pub depth: MaskedValue,
    /// Python `is_defined` field with its original masked-array metadata.
    pub is_defined: MaskedValue,
}
impl MaskedFuelbedIntermediates {
    /// Construct an empty ordinary f64 transport record.
    pub fn new() -> Self {
        Self::default()
    }
    /// Convert an ordinary f64 cell record whose source arrays have at least one dimension.
    pub fn from_plain(value: &super::super::FuelbedIntermediates) -> Self {
        Self {
            sigma: MaskedValue::plain(value.sigma),
            bulk_density: MaskedValue::plain(value.bulk_density),
            packing_ratio: MaskedValue::plain(value.packing_ratio),
            relative_packing_ratio: MaskedValue::plain(value.relative_packing_ratio),
            heat_sink: MaskedValue::plain(value.heat_sink),
            propagating_flux: MaskedValue::plain(value.propagating_flux),
            w_heat_dead: MaskedValue::plain(value.w_heat_dead),
            w_heat_live: MaskedValue::plain(value.w_heat_live),
            w_silica_dead: MaskedValue::plain(value.w_silica_dead),
            w_silica_live: MaskedValue::plain(value.w_silica_live),
            w_moist_dead: MaskedValue::plain(value.w_moist_dead),
            w_moist_live: MaskedValue::plain(value.w_moist_live),
            wfl_dead: MaskedValue::plain(value.wfl_dead),
            wfl_live: MaskedValue::plain(value.wfl_live),
            moe_dead: MaskedValue::plain(value.moe_dead),
            moe_live: MaskedValue::plain(value.moe_live),
            frac_dead: MaskedValue::plain(value.frac_dead),
            frac_live: MaskedValue::plain(value.frac_live),
            depth: MaskedValue::plain(value.depth),
            is_defined: MaskedValue::plain(f64::from(u8::from(value.is_defined))),
        }
    }
    /// Return all fields under their exact Python dictionary keys, including the boolean defined flag as 0/1.
    pub fn named_values(&self) -> [(&'static str, MaskedValue); 20] {
        [
            ("sigma", self.sigma),
            ("bulk_density", self.bulk_density),
            ("packing_ratio", self.packing_ratio),
            ("relative_packing_ratio", self.relative_packing_ratio),
            ("heat_sink", self.heat_sink),
            ("propagating_flux", self.propagating_flux),
            ("w_heat_dead", self.w_heat_dead),
            ("w_heat_live", self.w_heat_live),
            ("w_silica_dead", self.w_silica_dead),
            ("w_silica_live", self.w_silica_live),
            ("w_moist_dead", self.w_moist_dead),
            ("w_moist_live", self.w_moist_live),
            ("wfl_dead", self.wfl_dead),
            ("wfl_live", self.wfl_live),
            ("moe_dead", self.moe_dead),
            ("moe_live", self.moe_live),
            ("frac_dead", self.frac_dead),
            ("frac_live", self.frac_live),
            ("depth", self.depth),
            ("is_defined", self.is_defined),
        ]
    }
}

/// Sum one leading particle axis, filling masked values with zero and masking an all-masked reduction.
fn sum(values: [MaskedValue; 5], scalar_cell: bool) -> MaskedValue {
    if values.iter().any(|x| x.object_result) {
        return super::objects::sum(values, scalar_cell);
    }
    if let Some(error) = values.iter().find_map(|x| x.error) {
        return super::integers::failure(values[0], error);
    }
    if values.iter().any(|x| super::integers::is_integer(*x)) {
        return super::integers::sum(values, scalar_cell);
    }
    let ma = values.iter().any(|x| x.is_masked_array);
    let mask = ma && values.iter().all(|x| x.mask);
    let precision = if values.iter().any(|x| x.precision == Precision::F64) {
        Precision::F64
    } else {
        Precision::F32
    };
    let total = match precision {
        Precision::F32 => values.iter().fold(0.0f32, |a, x| {
            a + if ma && x.mask { 0.0 } else { x.value as f32 }
        }) as f64,
        Precision::F64 => values
            .iter()
            .fold(0.0, |a, x| a + if ma && x.mask { 0.0 } else { x.value }),
    };
    finish(
        MaskedValue::new(total, mask, ma, precision)
            .with_scalar_array(scalar_cell)
            .with_numpy_scalar(scalar_cell && !ma)
            .with_math_profile(values[0].math_profile)
            // NumPy's reduction constructs a fresh result with a lazy default fill value.
            .with_fill_value(None),
        true,
    )
}
/// Extract one particle from its leading array axis, including NumPy's scalar MaskedConstant behavior.
fn particle(value: MaskedValue, scalar_cell: bool) -> MaskedValue {
    if scalar_cell {
        finish(
            value
                .with_scalar_array(true)
                .with_numpy_scalar(!value.is_masked_array),
            true,
        )
    } else {
        value
    }
}
/// Broadcast a per-cell scalar back over the leading particle axis.
fn broad(value: MaskedValue) -> MaskedValue {
    value.with_scalar_array(false).with_numpy_scalar(false)
}
/// Compute the private size-bin helper, which deliberately accumulates fractions in ordinary f64 arrays.
fn size_sorted(
    savr: [MaskedValue; 5],
    fractions: [MaskedValue; 5],
    net: [MaskedValue; 5],
    scalar: bool,
) -> MaskedValue {
    let bounds = [
        (1200., f64::INFINITY),
        (192., 1200.),
        (96., 192.),
        (48., 96.),
        (16., 48.),
    ];
    let zero = MaskedValue::plain(0.0).with_scalar_array(scalar);
    let mut bins = [zero; 5];
    for n in 0..5 {
        let sv = particle(savr[n], scalar);
        let frac = particle(fractions[n], scalar);
        for b in 0..5 {
            let inside = MaskedCondition::new(
                sv.greater_equal(c(bounds[b].0)) && (b == 0 || sv.less(c(bounds[b].1))),
                sv.is_scalar_array,
            );
            bins[b] = bins[b].add(MaskedValue::select_with_condition(inside, frac, c(0.0)));
        }
    }
    let mut weighted = zero;
    for n in 0..5 {
        let sv = particle(savr[n], scalar);
        let mut assigned = zero;
        for b in 0..5 {
            let inside = MaskedCondition::new(
                sv.greater_equal(c(bounds[b].0)) && (b == 0 || sv.less(c(bounds[b].1))),
                sv.is_scalar_array,
            );
            assigned = assigned.add(MaskedValue::select_with_condition(inside, bins[b], c(0.0)));
        }
        // Python's in-place addition writes into the ordinary np.zeros buffer,
        // discarding the product's mask while retaining its raw operator data.
        weighted =
            MaskedValue::plain(weighted.value + assigned.mul(particle(net[n], scalar)).value)
                .with_scalar_array(scalar);
    }
    weighted
}

/// Calculate moisture damping, retaining the source's masked division before np.where normalization.
fn eta_m(moisture: MaskedValue, moe: MaskedValue) -> Result<MaskedValue, MaskedError> {
    validate_condition_and(&[moe])?;
    let safe = MaskedValue::select_with_condition(moe.greater_condition(c(1e-7)), moe, c(1.0));
    let rm = MaskedValue::select_with_condition(
        moe.greater_condition(c(1e-7)),
        moisture.div(safe),
        c(1.0),
    );
    let extinguished = moe
        .less_condition(c(1e-7))
        .or(rm.greater_equal_condition(c(1.0)));
    Ok(MaskedValue::select_with_condition(
        extinguished,
        c(0.0),
        c(1.0)
            .sub(c(2.59).mul(rm))
            .add(c(5.11).mul(rm.pow(MaskedValue::python_int(2))))
            .sub(c(3.52).mul(rm.pow(MaskedValue::python_int(3)))),
    ))
}
/// Calculate mineral damping with masked comparison/where normalization.
fn eta_s(silica: MaskedValue) -> MaskedValue {
    let safe =
        MaskedValue::select_with_condition(silica.greater_condition(c(1e-7)), silica, c(1.0));
    MaskedValue::select_with_condition(
        silica.greater_condition(c(1e-7)),
        c(0.174).div(safe.pow(c(0.19))).minimum(c(1.0)),
        c(0.0),
    )
}
/// Calculate reaction intensity from independently masked fuelbed fields; reject NumPy-invalid scalar boolean combinations.
pub fn try_calculate_reaction_intensity(
    ib: &MaskedFuelbedIntermediates,
) -> Result<MaskedValue, MaskedError> {
    validate_condition_and(&[ib.sigma, ib.relative_packing_ratio])?;
    let valid = ib
        .sigma
        .greater_condition(c(1e-7))
        .and(ib.relative_packing_ratio.greater_condition(c(0.0)));
    let sigma = MaskedValue::select_with_condition(valid, ib.sigma, c(1.0));
    let rpr = MaskedValue::select_with_condition(valid, ib.relative_packing_ratio, c(1.0));
    let aa = c(133.).div(sigma.pow(c(0.7913)));
    let gamma_max = sigma
        .pow(c(1.5))
        .div(c(495.).add(c(0.0594).mul(sigma.pow(c(1.5)))));
    let gamma = gamma_max.mul(rpr.pow(aa)).mul(aa.mul(c(1.).sub(rpr)).exp());
    let dead = gamma
        .mul(ib.wfl_dead)
        .mul(ib.w_heat_dead)
        .mul(eta_m(ib.w_moist_dead, ib.moe_dead)?)
        .mul(eta_s(ib.w_silica_dead));
    let live = gamma
        .mul(ib.wfl_live)
        .mul(ib.w_heat_live)
        .mul(eta_m(ib.w_moist_live, ib.moe_live)?)
        .mul(eta_s(ib.w_silica_live));
    let result = MaskedValue::select_with_condition(valid, dead.add(live), c(0.0));
    if let Some(error) = result.error {
        Err(error)
    } else {
        Ok(result)
    }
}

/// Calculate all weighted fuelbed fields from independently masked particle arrays.
pub fn try_calculate_fuelbed_intermediates(
    p: &MaskedParticles,
) -> Result<MaskedFuelbedIntermediates, MaskedError> {
    let scalar = p.scalar_cell;
    if scalar {
        validate_condition_and(&p.savr_dead.map(|x| particle(x, true)))?;
        validate_condition_and(&p.savr_live.map(|x| particle(x, true)))?;
    }
    let sa_d = std::array::from_fn(|n| p.load_dead[n].mul(p.savr_dead[n]).div(c(32.0)));
    let sa_l = std::array::from_fn(|n| p.load_live[n].mul(p.savr_live[n]).div(c(32.0)));
    let tsa_d = sum(sa_d, scalar);
    let tsa_l = sum(sa_l, scalar);
    let tsa = tsa_d.add(tsa_l);
    let safe_tsa = MaskedValue::select_with_condition(tsa.greater_condition(c(1e-7)), tsa, c(1.0));
    let frac_dead = MaskedValue::select_with_condition(
        tsa.greater_condition(c(1e-7)),
        tsa_d.div(safe_tsa),
        c(0.0),
    );
    let frac_live = c(1.0).sub(frac_dead);
    let safe_d =
        MaskedValue::select_with_condition(tsa_d.greater_condition(c(1e-7)), tsa_d, c(1.0));
    let safe_l =
        MaskedValue::select_with_condition(tsa_l.greater_condition(c(1e-7)), tsa_l, c(1.0));
    let fp_d: [MaskedValue; 5] = std::array::from_fn(|n| sa_d[n].div(broad(safe_d)));
    let fp_l: [MaskedValue; 5] = std::array::from_fn(|n| sa_l[n].div(broad(safe_l)));
    let exp_d: [MaskedValue; 5] = std::array::from_fn(|n| {
        let sv = p.savr_dead[n];
        let safe = MaskedValue::select_with_condition(sv.greater_condition(c(1e-7)), sv, c(1.0));
        MaskedValue::select_with_condition(
            sv.greater_condition(c(1e-7)),
            c(-138.0).div(safe).exp(),
            c(0.0),
        )
    });
    let exp_l500: [MaskedValue; 5] = std::array::from_fn(|n| {
        let sv = p.savr_live[n];
        let safe = MaskedValue::select_with_condition(sv.greater_condition(c(1e-7)), sv, c(1.0));
        MaskedValue::select_with_condition(
            sv.greater_condition(c(1e-7)),
            c(-500.0).div(safe).exp(),
            c(0.0),
        )
    });
    let fine_d = sum(
        std::array::from_fn(|n| p.load_dead[n].mul(exp_d[n])),
        scalar,
    );
    let fine_dm = sum(
        std::array::from_fn(|n| p.load_dead[n].mul(exp_d[n]).mul(p.moisture_dead[n])),
        scalar,
    );
    let fine_l = sum(
        std::array::from_fn(|n| p.load_live[n].mul(exp_l500[n])),
        scalar,
    );
    let safe_fd =
        MaskedValue::select_with_condition(fine_d.greater_condition(c(1e-7)), fine_d, c(1.0));
    let fine_dead_moist = MaskedValue::select_with_condition(
        fine_d.greater_condition(c(1e-7)),
        fine_dm.div(safe_fd),
        c(0.0),
    );
    let safe_fl =
        MaskedValue::select_with_condition(fine_l.greater_condition(c(1e-7)), fine_l, c(1.0));
    let fdol = MaskedValue::select_with_condition(
        fine_l.greater_condition(c(1e-7)),
        fine_d.div(safe_fl),
        c(0.0),
    );
    let moe_dead = p.moe_dead;
    let safe_md =
        MaskedValue::select_with_condition(moe_dead.greater_condition(c(1e-7)), moe_dead, c(1.0));
    let moe_live = c(2.9)
        .mul(fdol)
        .mul(c(1.0).sub(fine_dead_moist.div(safe_md)))
        .sub(c(0.226))
        .maximum(moe_dead);
    let w_heat_dead = sum(
        std::array::from_fn(|n| fp_d[n].mul(broad(p.hoc_dead))),
        scalar,
    );
    let w_heat_live = sum(
        std::array::from_fn(|n| fp_l[n].mul(broad(p.hoc_live))),
        scalar,
    );
    let w_silica_dead = sum(
        std::array::from_fn(|n| fp_d[n].mul(MaskedValue::plain(0.01))),
        scalar,
    );
    let w_silica_live = sum(
        std::array::from_fn(|n| fp_l[n].mul(MaskedValue::plain(if n < 2 { 0.01 } else { 0.0 }))),
        scalar,
    );
    let w_moist_dead = sum(
        std::array::from_fn(|n| fp_d[n].mul(p.moisture_dead[n])),
        scalar,
    );
    let w_moist_live = sum(
        std::array::from_fn(|n| fp_l[n].mul(p.moisture_live[n])),
        scalar,
    );
    let w_savr_dead = sum(std::array::from_fn(|n| fp_d[n].mul(p.savr_dead[n])), scalar);
    let w_savr_live = sum(std::array::from_fn(|n| fp_l[n].mul(p.savr_live[n])), scalar);
    let sigma = frac_dead.mul(w_savr_dead).add(frac_live.mul(w_savr_live));
    let wn_d = std::array::from_fn(|n| p.load_dead[n].mul(c(1.0 - 0.0555)));
    let wn_l = std::array::from_fn(|n| p.load_live[n].mul(c(1.0 - 0.0555)));
    let wfl_dead = size_sorted(p.savr_dead, fp_d, wn_d, scalar);
    let wfl_live = size_sorted(p.savr_live, fp_l, wn_l, scalar);
    let total_load = sum(p.load_dead, scalar).add(sum(p.load_live, scalar));
    let depth = p.depth;
    let safe_depth =
        MaskedValue::select_with_condition(depth.greater_condition(c(1e-7)), depth, c(1.0));
    let bulk_density = MaskedValue::select_with_condition(
        depth.greater_condition(c(1e-7)),
        total_load.div(safe_depth),
        c(0.0),
    );
    let packing_ratio = MaskedValue::select_with_condition(
        depth.greater_condition(c(1e-7)),
        sum(
            std::array::from_fn(|n| p.load_dead[n].div(broad(safe_depth).mul(c(32.0)))),
            scalar,
        )
        .add(sum(
            std::array::from_fn(|n| p.load_live[n].div(broad(safe_depth).mul(c(32.0)))),
            scalar,
        )),
        c(0.0),
    );
    let safe_sigma =
        MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), sigma, c(1.0));
    let opt = MaskedValue::select_with_condition(
        sigma.greater_condition(c(1e-7)),
        c(3.348).div(safe_sigma.pow(c(0.8189))),
        c(1.0),
    );
    let safe_opt = MaskedValue::select_with_condition(opt.greater_condition(c(1e-7)), opt, c(1.0));
    validate_condition_and(&[sigma, opt])?;
    let relative_packing_ratio = MaskedValue::select_with_condition(
        sigma
            .greater_condition(c(1e-7))
            .and(opt.greater_condition(c(1e-7))),
        packing_ratio.div(safe_opt),
        c(0.0),
    );
    let heat_d = sum(
        std::array::from_fn(|n| {
            broad(frac_dead)
                .mul(fp_d[n])
                .mul(c(250.0).add(c(1116.0).mul(p.moisture_dead[n])))
                .mul(exp_d[n])
        }),
        scalar,
    );
    let heat_l = sum(
        std::array::from_fn(|n| {
            let sv = p.savr_live[n];
            let safe =
                MaskedValue::select_with_condition(sv.greater_condition(c(1e-7)), sv, c(1.0));
            let ex = MaskedValue::select_with_condition(
                sv.greater_condition(c(1e-7)),
                c(-138.0).div(safe).exp(),
                c(0.0),
            );
            broad(frac_live)
                .mul(fp_l[n])
                .mul(c(250.0).add(c(1116.0).mul(p.moisture_live[n])))
                .mul(ex)
        }),
        scalar,
    );
    let heat_sink = heat_d.add(heat_l).mul(bulk_density);
    let propagating_flux = MaskedValue::select_with_condition(
        sigma.greater_condition(c(1e-7)),
        c(0.792)
            .add(c(0.681).mul(safe_sigma.sqrt()))
            .mul(packing_ratio.add(c(0.1)))
            .exp()
            .div(c(192.0).add(c(0.2595).mul(safe_sigma))),
        c(0.0),
    );
    let result = MaskedFuelbedIntermediates {
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
    };
    if let Some(error) = result.named_values().iter().find_map(|(_, x)| x.error) {
        Err(error)
    } else {
        Ok(result)
    }
}

/// Rank metadata for the normalized numerical surface input arguments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceInputShape {
    /// Whether wind speed has shape ().
    pub wind_speed: bool,
    /// Whether wind direction has shape ().
    pub wind_direction: bool,
    /// Whether slope has shape ().
    pub slope: bool,
    /// Whether aspect has shape ().
    pub aspect: bool,
    /// Whether canopy cover has shape ().
    pub canopy_cover: bool,
    /// Whether canopy height has shape ().
    pub canopy_height: bool,
    /// Whether crown ratio has shape ().
    pub crown_ratio: bool,
    /// Whether the supplied user WAF has shape ().
    pub user_waf: bool,
}
impl SurfaceInputShape {
    /// Set all normalized input arguments to the same scalar/vector rank convention.
    pub fn uniform(scalar: bool) -> Self {
        Self {
            wind_speed: scalar,
            wind_direction: scalar,
            slope: scalar,
            aspect: scalar,
            canopy_cover: scalar,
            canopy_height: scalar,
            crown_ratio: scalar,
            user_waf: scalar,
        }
    }
}

/// Run spread/geometry using the common spatial rank of the reaction and sigma fields.
pub fn try_run_surface_fire_with_intermediates(
    ri: MaskedValue,
    ib: &MaskedFuelbedIntermediates,
    input: &super::super::SurfaceInput,
) -> Result<super::super::SurfaceResult, MaskedError> {
    try_run_surface_fire_with_intermediates_and_shape(
        ri,
        ib,
        input,
        &SurfaceInputShape::uniform(ri.is_scalar_array && ib.sigma.is_scalar_array),
    )
}

/// Run spread/geometry from independently masked fields and explicit normalized-argument rank metadata.
pub fn try_run_surface_fire_with_intermediates_and_shape(
    ri: MaskedValue,
    ib: &MaskedFuelbedIntermediates,
    input: &super::super::SurfaceInput,
    shape: &SurfaceInputShape,
) -> Result<super::super::SurfaceResult, MaskedError> {
    if let Some(error) = ri
        .error
        .or_else(|| ib.named_values().iter().find_map(|(_, x)| x.error))
    {
        return Err(error);
    }
    let plain = MaskedValue::plain;
    let sigma = ib.sigma;
    let safe_sigma =
        MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), sigma, c(1.0));
    let wc = c(7.47).mul(c(-0.133).mul(safe_sigma.pow(c(0.55))).exp());
    let wb = c(0.02526).mul(safe_sigma.pow(c(0.54)));
    let we = c(0.715).mul(c(-0.000359).mul(safe_sigma).exp());
    let r0 = super::calculate_no_wind_no_slope_spread_rate(ri, ib.propagating_flux, ib.heat_sink);
    if let Some(error) = r0.error {
        return Err(error);
    }
    let ws = super::super::calculate_twenty_foot_wind_speed(
        input.wind_speed,
        input.wind_speed_units,
        &input.wind_height_mode,
    );
    let waf = super::super::calculate_surface_wind_adjustment_factor(
        input.canopy_cover,
        input.canopy_height,
        input.crown_ratio,
        ib.depth.value,
        &input.waf_method,
        input.user_waf,
    );
    let midflame = super::super::calculate_midflame_wind_speed(waf, ws);
    let user_mode = input
        .waf_method
        .bytes()
        .filter(|b| *b != b'_')
        .map(|b| b.to_ascii_lowercase())
        .eq(b"userinput".iter().copied())
        && input.user_waf.is_some();
    let waf_scalar = if user_mode {
        shape.user_waf
    } else {
        shape.canopy_cover && shape.canopy_height && shape.crown_ratio && ib.depth.is_scalar_array
    };
    validate_condition_and(&[sigma])?;
    let wp = super::calculate_wind_phi(
        sigma,
        plain(midflame).with_scalar_array(shape.wind_speed && waf_scalar),
        wb,
        wc,
        ib.relative_packing_ratio,
        we,
    );
    if let Some(error) = wp.error {
        return Err(error);
    }
    validate_condition_and(&[ib.packing_ratio])?;
    let sp = super::calculate_slope_phi(
        plain(input.slope).with_scalar_array(shape.slope),
        input.slope_units,
        ib.packing_ratio,
        ri,
    );
    if let Some(error) = sp.error {
        return Err(error);
    }
    let vector = super::calculate_surface_rate_vector(
        r0,
        wp,
        sp,
        plain(input.wind_direction).with_scalar_array(shape.wind_direction),
        &input.wind_orientation_mode,
        plain(input.aspect).with_scalar_array(shape.aspect),
    );
    if let Some(error) = vector.named_values().iter().find_map(|(_, x)| x.error) {
        return Err(error);
    }
    let spread_rate =
        super::super::calculate_forward_spread_rate(r0.value, vector.rate_vector.value);
    validate_condition_and(&[wb, wc, ib.relative_packing_ratio])?;
    let effective_wind_speed = super::calculate_effective_wind_speed(
        plain(spread_rate)
            .with_scalar_array(r0.is_scalar_array && vector.rate_vector.is_scalar_array),
        r0,
        wb,
        wc,
        ib.relative_packing_ratio,
        we,
    )
    .value;
    let fire_length_to_width_ratio =
        super::super::calculate_fire_length_to_width_ratio(effective_wind_speed);
    let eccentricity = super::super::calculate_eccentricity(fire_length_to_width_ratio);
    let backing_spread_rate =
        super::super::calculate_backing_spread_rate(spread_rate, eccentricity);
    let flanking_spread_rate = super::super::calculate_flanking_spread_rate(
        spread_rate,
        backing_spread_rate,
        fire_length_to_width_ratio,
    );
    let safe_res =
        MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), sigma, c(1.0));
    let residence_time = MaskedValue::select_with_condition(
        sigma.greater_condition(c(1e-7)),
        c(384.0).div(safe_res),
        c(0.0),
    )
    .value;
    let heat_per_unit_area = super::super::calculate_heat_per_unit_area(ri.value, residence_time);
    let fireline_intensity =
        super::super::calculate_fireline_intensity(spread_rate, heat_per_unit_area);
    let flame_length = super::super::calculate_flame_length(fireline_intensity);
    let direction_of_max_spread = super::calculate_direction_of_max_spread(
        vector.x_component,
        vector.y_component,
        vector.rate_vector,
        &input.wind_orientation_mode,
        plain(input.aspect).with_scalar_array(shape.aspect),
    )
    .value;
    if ib.is_defined.value == 0.0 {
        return Ok(super::super::SurfaceResult {
            fire_length_to_width_ratio: 1.0,
            ..super::super::SurfaceResult::default()
        });
    }
    Ok(super::super::SurfaceResult {
        spread_rate,
        backing_spread_rate,
        flanking_spread_rate,
        flame_length,
        fireline_intensity,
        heat_per_unit_area,
        effective_wind_speed,
        fire_length_to_width_ratio,
        eccentricity,
        direction_of_max_spread,
        residence_time,
        reaction_intensity: ri.value,
        midflame_wind_speed: midflame,
        no_wind_no_slope_spread_rate: r0.value,
    })
}
