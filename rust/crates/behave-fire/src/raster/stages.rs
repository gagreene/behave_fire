//! Explicit raster stage inputs and override columns, separate from science execution.
use super::{validate_optional, value};
use crate::engine::{CrownTweaks, SurfaceTweaks};
use crate::surface::FuelbedIntermediates;
use crate::{BehaveError, GridShape, PerCell};

macro_rules! surface_tweaks {
    ($($field:ident),+) => {
        /// Optional scalar or cell-length Surface stage replacements.
        #[derive(Clone,Copy,Debug,Default)]
        pub struct SurfaceTweakColumns<'a> {
            $(#[doc=concat!("Optional `",stringify!($field),"` stage replacement in SurfaceTweaks units.")]
              pub $field:Option<PerCell<'a,f64>>,)+
        }
        impl SurfaceTweakColumns<'_> {
            pub(crate) fn validate(&self,shape:&GridShape)->Result<(),BehaveError> {
                $(validate_optional(&self.$field,shape,stringify!($field))?;)+
                Ok(())
            }
            pub(crate) fn cell(&self,index:usize)->SurfaceTweaks {
                SurfaceTweaks {$($field:self.$field.map(|column|value(&column,index)),)+}
            }
        }
    }
}
surface_tweaks!(
    reaction_intensity,
    no_wind_no_slope_spread_rate,
    midflame_wind_speed,
    wind_phi,
    slope_phi,
    spread_rate
);

/// Optional native Crown replacements applied before dependent stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrownTweakColumns<'a> {
    /// Overrides for the internal model-10 Surface calculation.
    pub fm10: SurfaceTweakColumns<'a>,
    /// Critical Surface intensity in BTU/ft/s.
    pub critical_surface_intensity: Option<PerCell<'a, f64>>,
    /// Critical active-Crown rate in ft/min.
    pub critical_crown_rate: Option<PerCell<'a, f64>>,
    /// Crown fraction burned, before passive/final assignments.
    pub crown_fraction_burned: Option<PerCell<'a, f64>>,
}

impl CrownTweakColumns<'_> {
    pub(crate) fn validate(&self, shape: &GridShape) -> Result<(), BehaveError> {
        self.fm10.validate(shape)?;
        validate_optional(
            &self.critical_surface_intensity,
            shape,
            "critical_surface_intensity",
        )?;
        validate_optional(&self.critical_crown_rate, shape, "critical_crown_rate")?;
        validate_optional(&self.crown_fraction_burned, shape, "crown_fraction_burned")
    }
    pub(crate) fn cell(&self, index: usize) -> CrownTweaks {
        CrownTweaks {
            fm10: self.fm10.cell(index),
            critical_surface_intensity: self
                .critical_surface_intensity
                .map(|column| value(&column, index)),
            critical_crown_rate: self.critical_crown_rate.map(|column| value(&column, index)),
            crown_fraction_burned: self
                .crown_fraction_burned
                .map(|column| value(&column, index)),
        }
    }
}

macro_rules! fuelbed_columns {
    ($($field:ident),+) => {
        /// Borrowed editable fuelbed stage fields, in FuelbedIntermediates units.
        #[derive(Clone,Copy,Debug)]
        pub struct FuelbedColumns<'a> {
            $(#[doc=concat!("Supplied `",stringify!($field),"` scalar or cell-length stage values.")]
              pub $field:PerCell<'a,f64>,)+
            /// Defined-fuel flag used only by the final scientific output mask.
            pub is_defined:PerCell<'a,bool>,
        }
        impl FuelbedColumns<'_> {
            /// Broadcast one independently supplied fuelbed record across a raster.
            pub fn uniform(value:FuelbedIntermediates)->Self {
                Self {$($field:PerCell::Scalar(value.$field),)+is_defined:PerCell::Scalar(value.is_defined)}
            }
            pub(crate) fn validate(&self,shape:&GridShape)->Result<(),BehaveError> {
                $(self.$field.validate(shape,stringify!($field))?;)+
                self.is_defined.validate(shape,"is_defined")
            }
            pub(crate) fn cell(&self,index:usize)->FuelbedIntermediates {
                let mut state=FuelbedIntermediates::new();
                $(state.$field=value(&self.$field,index);)+
                state.is_defined=value(&self.is_defined,index);
                state
            }
        }
    }
}
fuelbed_columns!(
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
    depth
);
