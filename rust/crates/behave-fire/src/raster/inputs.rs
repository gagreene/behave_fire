//! Single field lists keep packing and one-time length checks consistent.
use super::{validate_optional, value};
use crate::crown::SurfaceFireInputs;
use crate::engine::{CrownCell, SurfaceCell};
use crate::{BehaveError, GridShape, PerCell};

macro_rules! surface_columns {
    ($($field:ident),+) => {
        /// Borrowed per-field Surface inputs in component base units.
        #[derive(Clone,Copy,Debug)]
        pub struct SurfaceColumns<'a> {
            /// Scalar model ID or one ID per logical cell.
            pub fuel_model_grid: PerCell<'a,i32>,
            $(#[doc=concat!("Scalar or cell-length `",stringify!($field),"` values in the SurfaceCell units.")]
              pub $field:PerCell<'a,f64>,)+
            /// Optional user WAF, used only in the plan's UserInput mode.
            pub user_waf:Option<PerCell<'a,f64>>,
            /// Explicit native validity; false pixels skip calculation.
            pub validity:Option<PerCell<'a,bool>>,
        }
        impl Default for SurfaceColumns<'_> {
            fn default()->Self {
                let cell=SurfaceCell::default();
                Self {fuel_model_grid:PerCell::Scalar(cell.fuel_model_grid),
                      $($field:PerCell::Scalar(cell.$field),)+ user_waf:None,validity:None}
            }
        }
        impl SurfaceColumns<'_> {
            pub(crate) fn validate(&self,shape:&GridShape)->Result<(),BehaveError> {
                self.fuel_model_grid.validate(shape,"fuel_model_grid")?;
                $(self.$field.validate(shape,stringify!($field))?;)+
                validate_optional(&self.user_waf,shape,"user_waf")?;
                validate_optional(&self.validity,shape,"validity")
            }
            pub(crate) fn cell(&self,index:usize)->SurfaceCell {
                SurfaceCell {fuel_model_grid:value(&self.fuel_model_grid,index),
                             $($field:value(&self.$field,index),)+
                             user_waf:self.user_waf.map(|column|value(&column,index))}
            }
            pub(crate) fn valid(&self,index:usize)->bool {
                self.validity.map_or(true,|column|value(&column,index))
            }
        }
    }
}
surface_columns!(
    m1h,
    m10h,
    m100h,
    mlh,
    mlw,
    wind_speed,
    wind_direction,
    slope,
    aspect,
    canopy_cover,
    canopy_height,
    crown_ratio
);

macro_rules! crown_columns {
    ($($field:ident),+) => {
        /// Borrowed base-unit Crown inputs and independent supplied Surface fields.
        /// Model-10 fuel comes from the explicit prepared registry, not a redundant ID column.
        #[derive(Clone,Copy,Debug)]
        pub struct CrownColumns<'a> {
            $(#[doc=concat!("Scalar or cell-length `",stringify!($field),"` values in the CrownCell units.")]
              pub $field:PerCell<'a,f64>,)+
            /// Supplied Surface spread rate in ft/min.
            pub surface_ros:PerCell<'a,f64>,
            /// Supplied Surface intensity in BTU/ft/s.
            pub surface_fli:PerCell<'a,f64>,
            /// Supplied Surface heat in BTU/ft².
            pub surface_hpua:PerCell<'a,f64>,
            /// Explicit native validity; false pixels skip calculation.
            pub validity:Option<PerCell<'a,bool>>,
        }
        impl Default for CrownColumns<'_> {
            fn default()->Self {
                let cell=CrownCell::default();
                Self {$($field:PerCell::Scalar(cell.$field),)+surface_ros:PerCell::Scalar(0.0),
                      surface_fli:PerCell::Scalar(0.0),surface_hpua:PerCell::Scalar(0.0),validity:None}
            }
        }
        impl CrownColumns<'_> {
            pub(crate) fn validate(&self,shape:&GridShape)->Result<(),BehaveError> {
                $(self.$field.validate(shape,stringify!($field))?;)+
                self.surface_ros.validate(shape,"surface_ros")?;
                self.surface_fli.validate(shape,"surface_fli")?;
                self.surface_hpua.validate(shape,"surface_hpua")?;
                validate_optional(&self.validity,shape,"validity")
            }
            pub(crate) fn cell(&self,index:usize)->CrownCell {
                CrownCell {$($field:value(&self.$field,index),)+}
            }
            pub(crate) fn surface(&self,index:usize)->SurfaceFireInputs<f64> {
                let mut supplied=SurfaceFireInputs::<f64>::new();
                supplied.surface_ros=value(&self.surface_ros,index);
                supplied.surface_fli=value(&self.surface_fli,index);
                supplied.surface_hpua=value(&self.surface_hpua,index);
                supplied
            }
            pub(crate) fn valid(&self,index:usize)->bool {
                self.validity.map_or(true,|column|value(&column,index))
            }
        }
    }
}
crown_columns!(
    m1h,
    m10h,
    m100h,
    mlh,
    mlw,
    wind_speed,
    wind_direction,
    slope,
    aspect,
    canopy_base_height,
    canopy_height,
    canopy_bulk_density,
    moisture_foliar
);
