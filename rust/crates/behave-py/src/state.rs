//! Checked stateful tools: snapshots are copied before detached execution.
use behave_fire::units::*;
use behave_fire::{
    fire_size::{FireSize, NumericKind},
    slope::SlopeTool,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
fn numeric_kind(code: i32) -> NumericKind {
    match code {
        2 => NumericKind::NumpyFloat32,
        1 => NumericKind::NumpyFloat64,
        _ => NumericKind::PythonFloat,
    }
}
fn kind_code(kind: NumericKind) -> i32 {
    match kind {
        NumericKind::NumpyFloat32 => 2,
        NumericKind::NumpyFloat64 => 1,
        NumericKind::PythonFloat => 0,
    }
}
#[pyfunction]
pub fn state_call<'py>(
    py: Python<'py>,
    domain: &str,
    method: &str,
    state: &Bound<'py, PyDict>,
    arguments: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, PyDict>, Option<f64>, Option<String>)> {
    let result = PyDict::new(py);
    match domain {
        "fire_size" => {
            let mut value = FireSize::default();
            if let Some(item) = state.get_item("effective_wind_numeric_kind_")? {
                value.effective_wind_numeric_kind_ = numeric_kind(item.extract()?);
            }
            if let Some(item) = state.get_item("ratio_numeric_kind_")? {
                value.ratio_numeric_kind_ = numeric_kind(item.extract()?);
            }
            if let Some(item) = state.get_item("eccentricity_numeric_kind_")? {
                value.eccentricity_numeric_kind_ = numeric_kind(item.extract()?);
            }
            if let Some(item) = state.get_item("spread_rates_are_numpy_")? {
                value.spread_rates_are_numpy_ = item.extract()?;
            }
            if let Some(item) = state.get_item("effective_wind_is_numpy_")? {
                value.effective_wind_is_numpy_ = item.extract()?;
            }
            if let Some(item) = state.get_item("ratio_is_numpy_")? {
                value.ratio_is_numpy_ = item.extract()?;
            }
            if let Some(item) = state.get_item("effective_wind_speed_")? {
                value.effective_wind_speed_ = item.extract()?;
            }
            if let Some(item) = state.get_item("forward_spread_rate_")? {
                value.forward_spread_rate_ = item.extract()?;
            }
            if let Some(item) = state.get_item("elapsed_time_")? {
                value.elapsed_time_ = item.extract()?;
            }
            if let Some(item) = state.get_item("elliptical_a_")? {
                value.elliptical_a_ = item.extract()?;
            }
            if let Some(item) = state.get_item("elliptical_b_")? {
                value.elliptical_b_ = item.extract()?;
            }
            if let Some(item) = state.get_item("elliptical_c_")? {
                value.elliptical_c_ = item.extract()?;
            }
            if let Some(item) = state.get_item("eccentricity_")? {
                value.eccentricity_ = item.extract()?;
            }
            if let Some(item) = state.get_item("backing_spread_rate_")? {
                value.backing_spread_rate_ = item.extract()?;
            }
            if let Some(item) = state.get_item("flanking_spread_rate_")? {
                value.flanking_spread_rate_ = item.extract()?;
            }
            if let Some(item) = state.get_item("fire_length_to_width_ratio_")? {
                value.fire_length_to_width_ratio_ = item.extract()?;
            }
            if let Some(item) = state.get_item("heading_to_backing_ratio_")? {
                value.heading_to_backing_ratio_ = item.extract()?;
            }
            let (output, error) = match method {
                "__init__" => (None, None),
                "calculate_fire_basic_dimensions" => {
                    let is_crown: bool = arguments
                        .get_item("is_crown")?
                        .ok_or_else(|| PyValueError::new_err("missing is_crown"))?
                        .extract()?;
                    let effective_wind_speed: f64 = arguments
                        .get_item("effective_wind_speed")?
                        .ok_or_else(|| PyValueError::new_err("missing effective_wind_speed"))?
                        .extract()?;
                    let wind_speed_units: i32 = arguments
                        .get_item("wind_speed_units")?
                        .ok_or_else(|| PyValueError::new_err("missing wind_speed_units"))?
                        .extract()?;
                    let wind_speed_units = SpeedUnit::try_from(wind_speed_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let forward_spread_rate: f64 = arguments
                        .get_item("forward_spread_rate")?
                        .ok_or_else(|| PyValueError::new_err("missing forward_spread_rate"))?
                        .extract()?;
                    let spread_rate_units: i32 = arguments
                        .get_item("spread_rate_units")?
                        .ok_or_else(|| PyValueError::new_err("missing spread_rate_units"))?
                        .extract()?;
                    let spread_rate_units = SpeedUnit::try_from(spread_rate_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let wind_kind = numeric_kind(
                        arguments
                            .get_item("_wind_kind")?
                            .ok_or_else(|| PyValueError::new_err("missing numeric kind"))?
                            .extract()?,
                    );
                    py.detach(|| {
                        match value.calculate_fire_basic_dimensions_with_wind_kind(
                            is_crown,
                            effective_wind_speed,
                            wind_speed_units,
                            forward_spread_rate,
                            spread_rate_units,
                            wind_kind,
                        ) {
                            Ok(_output) => (None, None),
                            Err(error) => (None, Some(format!("{error:?}"))),
                        }
                    })
                }
                "calculate_surface_fire_length_to_width_ratio" => {
                    py.detach(
                        || match value.calculate_surface_fire_length_to_width_ratio() {
                            Ok(_output) => (None, None),
                            Err(error) => (None, Some(format!("{error:?}"))),
                        },
                    )
                }
                "calculate_crown_fire_length_to_width_ratio" => py.detach(|| {
                    value.calculate_crown_fire_length_to_width_ratio();
                    (None, None)
                }),
                "calculate_fire_eccentricity" => py.detach(|| {
                    value.calculate_fire_eccentricity();
                    (None, None)
                }),
                "calculate_elliptical_dimensions" => {
                    py.detach(|| match value.calculate_elliptical_dimensions() {
                        Ok(_output) => (None, None),
                        Err(error) => (None, Some(format!("{error:?}"))),
                    })
                }
                "calculate_backing_spread_rate" => {
                    py.detach(|| match value.calculate_backing_spread_rate() {
                        Ok(_output) => (None, None),
                        Err(error) => (None, Some(format!("{error:?}"))),
                    })
                }
                "calculate_flanking_spread_rate" => {
                    py.detach(|| match value.calculate_flanking_spread_rate() {
                        Ok(_output) => (None, None),
                        Err(error) => (None, Some(format!("{error:?}"))),
                    })
                }
                "get_fire_length_to_width_ratio" => {
                    py.detach(|| (Some(value.get_fire_length_to_width_ratio()), None))
                }
                "get_eccentricity" => py.detach(|| (Some(value.get_eccentricity()), None)),
                "get_heading_to_backing_ratio" => {
                    py.detach(|| (Some(value.get_heading_to_backing_ratio()), None))
                }
                "get_backing_spread_rate" => {
                    let spread_rate_units: i32 = arguments
                        .get_item("spread_rate_units")?
                        .ok_or_else(|| PyValueError::new_err("missing spread_rate_units"))?
                        .extract()?;
                    let spread_rate_units = SpeedUnit::try_from(spread_rate_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| (Some(value.get_backing_spread_rate(spread_rate_units)), None))
                }
                "get_flanking_spread_rate" => {
                    let spread_rate_units: i32 = arguments
                        .get_item("spread_rate_units")?
                        .ok_or_else(|| PyValueError::new_err("missing spread_rate_units"))?
                        .extract()?;
                    let spread_rate_units = SpeedUnit::try_from(spread_rate_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_flanking_spread_rate(spread_rate_units)),
                            None,
                        )
                    })
                }
                "get_elliptical_a" => {
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_elliptical_a(length_units, elapsed_time, time_units)),
                            None,
                        )
                    })
                }
                "get_elliptical_b" => {
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_elliptical_b(length_units, elapsed_time, time_units)),
                            None,
                        )
                    })
                }
                "get_elliptical_c" => {
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_elliptical_c(length_units, elapsed_time, time_units)),
                            None,
                        )
                    })
                }
                "get_fire_length" => {
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_fire_length(length_units, elapsed_time, time_units)),
                            None,
                        )
                    })
                }
                "get_max_fire_width" => {
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_max_fire_width(length_units, elapsed_time, time_units)),
                            None,
                        )
                    })
                }
                "get_fire_perimeter" => {
                    let is_crown: bool = arguments
                        .get_item("is_crown")?
                        .ok_or_else(|| PyValueError::new_err("missing is_crown"))?
                        .extract()?;
                    let length_units: i32 = arguments
                        .get_item("length_units")?
                        .ok_or_else(|| PyValueError::new_err("missing length_units"))?
                        .extract()?;
                    let length_units = LengthUnit::try_from(length_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        match value.get_fire_perimeter(
                            is_crown,
                            length_units,
                            elapsed_time,
                            time_units,
                        ) {
                            Ok(output) => (Some(output), None),
                            Err(error) => (None, Some(format!("{error:?}"))),
                        }
                    })
                }
                "get_fire_area" => {
                    let is_crown: bool = arguments
                        .get_item("is_crown")?
                        .ok_or_else(|| PyValueError::new_err("missing is_crown"))?
                        .extract()?;
                    let area_units: i32 = arguments
                        .get_item("area_units")?
                        .ok_or_else(|| PyValueError::new_err("missing area_units"))?
                        .extract()?;
                    let area_units = AreaUnit::try_from(area_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let elapsed_time: f64 = arguments
                        .get_item("elapsed_time")?
                        .ok_or_else(|| PyValueError::new_err("missing elapsed_time"))?
                        .extract()?;
                    let time_units: i32 = arguments
                        .get_item("time_units")?
                        .ok_or_else(|| PyValueError::new_err("missing time_units"))?
                        .extract()?;
                    let time_units = TimeUnit::try_from(time_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_fire_area(
                                is_crown,
                                area_units,
                                elapsed_time,
                                time_units,
                            )),
                            None,
                        )
                    })
                }
                _ => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        method.to_owned(),
                    ))
                }
            };
            result.set_item(
                "effective_wind_numeric_kind_",
                kind_code(value.effective_wind_numeric_kind_),
            )?;
            result.set_item("ratio_numeric_kind_", kind_code(value.ratio_numeric_kind_))?;
            result.set_item(
                "eccentricity_numeric_kind_",
                kind_code(value.eccentricity_numeric_kind_),
            )?;
            result.set_item("spread_rates_are_numpy_", value.spread_rates_are_numpy_)?;
            result.set_item("effective_wind_is_numpy_", value.effective_wind_is_numpy_)?;
            result.set_item("ratio_is_numpy_", value.ratio_is_numpy_)?;
            result.set_item("effective_wind_speed_", value.effective_wind_speed_)?;
            result.set_item("forward_spread_rate_", value.forward_spread_rate_)?;
            result.set_item("elapsed_time_", value.elapsed_time_)?;
            result.set_item("elliptical_a_", value.elliptical_a_)?;
            result.set_item("elliptical_b_", value.elliptical_b_)?;
            result.set_item("elliptical_c_", value.elliptical_c_)?;
            result.set_item("eccentricity_", value.eccentricity_)?;
            result.set_item("backing_spread_rate_", value.backing_spread_rate_)?;
            result.set_item("flanking_spread_rate_", value.flanking_spread_rate_)?;
            result.set_item(
                "fire_length_to_width_ratio_",
                value.fire_length_to_width_ratio_,
            )?;
            result.set_item("heading_to_backing_ratio_", value.heading_to_backing_ratio_)?;
            Ok((result, output, error))
        }
        "slope_tool" => {
            let mut value = SlopeTool::default();
            if let Some(item) = state.get_item("horizontal_distances_")? {
                value.horizontal_distances_ = item.extract()?;
            }
            if let Some(item) = state.get_item("max_slope_in_degrees_")? {
                value.max_slope_in_degrees_ = item.extract()?;
            }
            if let Some(item) = state.get_item("max_slope_in_percent_")? {
                value.max_slope_in_percent_ = item.extract()?;
            }
            if let Some(item) = state.get_item("slope_from_map_measurements_")? {
                value.slope_from_map_measurements_ = item.extract()?;
            }
            if let Some(item) = state.get_item("slope_horizontal_distance_")? {
                value.slope_horizontal_distance_ = item.extract()?;
            }
            if let Some(item) = state.get_item("slope_elevation_change_")? {
                value.slope_elevation_change_ = item.extract()?;
            }
            if let Some(item) = state.get_item("representative_fractions_")? {
                value.representative_fractions_ = item.extract()?;
            }
            if let Some(item) = state.get_item("inches_per_mile_")? {
                value.inches_per_mile_ = item.extract()?;
            }
            if let Some(item) = state.get_item("miles_per_inch_")? {
                value.miles_per_inch_ = item.extract()?;
            }
            if let Some(item) = state.get_item("centimeters_per_kilometer_")? {
                value.centimeters_per_kilometer_ = item.extract()?;
            }
            if let Some(item) = state.get_item("kilometers_per_centimeter_")? {
                value.kilometers_per_centimeter_ = item.extract()?;
            }
            let (output, error) = match method {
                "__init__" => (None, None),
                "calculate_horizontal_distance" => {
                    let calculated_map_distance: f64 = arguments
                        .get_item("calculated_map_distance")?
                        .ok_or_else(|| PyValueError::new_err("missing calculated_map_distance"))?
                        .extract()?;
                    let distance_units: i32 = arguments
                        .get_item("distance_units")?
                        .ok_or_else(|| PyValueError::new_err("missing distance_units"))?
                        .extract()?;
                    let distance_units = LengthUnit::try_from(distance_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let max_slope_steepness: f64 = arguments
                        .get_item("max_slope_steepness")?
                        .ok_or_else(|| PyValueError::new_err("missing max_slope_steepness"))?
                        .extract()?;
                    let slope_units: i32 = arguments
                        .get_item("slope_units")?
                        .ok_or_else(|| PyValueError::new_err("missing slope_units"))?
                        .extract()?;
                    let slope_units = SlopeUnit::try_from(slope_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        match value.calculate_horizontal_distance(
                            calculated_map_distance,
                            distance_units,
                            max_slope_steepness,
                            slope_units,
                        ) {
                            Ok(_output) => (None, None),
                            Err(error) => (None, Some(format!("{error:?}"))),
                        }
                    })
                }
                "calculate_slope_from_map_measurements" => {
                    let map_representative_fraction: f64 = arguments
                        .get_item("map_representative_fraction")?
                        .ok_or_else(|| {
                            PyValueError::new_err("missing map_representative_fraction")
                        })?
                        .extract()?;
                    let map_distance: f64 = arguments
                        .get_item("map_distance")?
                        .ok_or_else(|| PyValueError::new_err("missing map_distance"))?
                        .extract()?;
                    let distance_units: i32 = arguments
                        .get_item("distance_units")?
                        .ok_or_else(|| PyValueError::new_err("missing distance_units"))?
                        .extract()?;
                    let distance_units = LengthUnit::try_from(distance_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let contour_interval: f64 = arguments
                        .get_item("contour_interval")?
                        .ok_or_else(|| PyValueError::new_err("missing contour_interval"))?
                        .extract()?;
                    let number_of_contours: f64 = arguments
                        .get_item("number_of_contours")?
                        .ok_or_else(|| PyValueError::new_err("missing number_of_contours"))?
                        .extract()?;
                    let contour_units: i32 = arguments
                        .get_item("contour_units")?
                        .ok_or_else(|| PyValueError::new_err("missing contour_units"))?
                        .extract()?;
                    let contour_units = LengthUnit::try_from(contour_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        value.calculate_slope_from_map_measurements(
                            map_representative_fraction,
                            map_distance,
                            distance_units,
                            contour_interval,
                            number_of_contours,
                            contour_units,
                        );
                        (None, None)
                    })
                }
                "get_number_of_horizontal_distances" => py.detach(|| {
                    (
                        Some(value.get_number_of_horizontal_distances() as f64),
                        None,
                    )
                }),
                "get_horizontal_distance_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    let map_distance_units: i32 = arguments
                        .get_item("map_distance_units")?
                        .ok_or_else(|| PyValueError::new_err("missing map_distance_units"))?
                        .extract()?;
                    let map_distance_units = LengthUnit::try_from(map_distance_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_horizontal_distance_at_index(index, map_distance_units)),
                            None,
                        )
                    })
                }
                "get_slope_from_map_measurements" => {
                    let slope_units: i32 = arguments
                        .get_item("slope_units")?
                        .ok_or_else(|| PyValueError::new_err("missing slope_units"))?
                        .extract()?;
                    let slope_units = SlopeUnit::try_from(slope_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_slope_from_map_measurements(slope_units)),
                            None,
                        )
                    })
                }
                "get_slope_from_map_measurements_in_percent" => py.detach(|| {
                    (
                        Some(value.get_slope_from_map_measurements_in_percent()),
                        None,
                    )
                }),
                "get_slope_from_map_measurements_in_degrees" => py.detach(|| {
                    (
                        Some(value.get_slope_from_map_measurements_in_degrees()),
                        None,
                    )
                }),
                "get_slope_horizontal_distance_from_map_measurements" => {
                    let distance_units: i32 = arguments
                        .get_item("distance_units")?
                        .ok_or_else(|| PyValueError::new_err("missing distance_units"))?
                        .extract()?;
                    let distance_units = LengthUnit::try_from(distance_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(value.get_slope_horizontal_distance_from_map_measurements(
                                distance_units,
                            )),
                            None,
                        )
                    })
                }
                "get_slope_elevation_change_from_map_measurements" => {
                    let elevation_units: i32 = arguments
                        .get_item("elevation_units")?
                        .ok_or_else(|| PyValueError::new_err("missing elevation_units"))?
                        .extract()?;
                    let elevation_units = LengthUnit::try_from(elevation_units)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    py.detach(|| {
                        (
                            Some(
                                value.get_slope_elevation_change_from_map_measurements(
                                    elevation_units,
                                ),
                            ),
                            None,
                        )
                    })
                }
                "get_number_of_representative_fractions" => py.detach(|| {
                    (
                        Some(value.get_number_of_representative_fractions() as f64),
                        None,
                    )
                }),
                "get_representative_fraction_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    py.detach(|| {
                        (
                            Some(value.get_representative_fraction_at_index(index) as f64),
                            None,
                        )
                    })
                }
                "get_inches_per_mile_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    py.detach(|| (Some(value.get_inches_per_mile_at_index(index)), None))
                }
                "get_miles_per_inch_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    py.detach(|| (Some(value.get_miles_per_inch_at_index(index)), None))
                }
                "get_centimeters_per_kilometer_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    py.detach(|| {
                        (
                            Some(value.get_centimeters_per_kilometer_at_index(index)),
                            None,
                        )
                    })
                }
                "get_kilometers_per_centimeter_at_index" => {
                    let index: i64 = arguments
                        .get_item("index")?
                        .ok_or_else(|| PyValueError::new_err("missing index"))?
                        .extract()?;
                    py.detach(|| {
                        (
                            Some(value.get_kilometers_per_centimeter_at_index(index)),
                            None,
                        )
                    })
                }
                _ => {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        method.to_owned(),
                    ))
                }
            };
            result.set_item("horizontal_distances_", value.horizontal_distances_)?;
            result.set_item("max_slope_in_degrees_", value.max_slope_in_degrees_)?;
            result.set_item("max_slope_in_percent_", value.max_slope_in_percent_)?;
            result.set_item(
                "slope_from_map_measurements_",
                value.slope_from_map_measurements_,
            )?;
            result.set_item(
                "slope_horizontal_distance_",
                value.slope_horizontal_distance_,
            )?;
            result.set_item("slope_elevation_change_", value.slope_elevation_change_)?;
            result.set_item("representative_fractions_", value.representative_fractions_)?;
            result.set_item("inches_per_mile_", value.inches_per_mile_)?;
            result.set_item("miles_per_inch_", value.miles_per_inch_)?;
            result.set_item(
                "centimeters_per_kilometer_",
                value.centimeters_per_kilometer_,
            )?;
            result.set_item(
                "kilometers_per_centimeter_",
                value.kilometers_per_centimeter_,
            )?;
            Ok((result, output, error))
        }
        _ => Err(PyValueError::new_err("unknown stateful domain")),
    }
}
