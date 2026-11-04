//! Native default-table export; every returned mapping owns its data.
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
#[pyfunction]
pub fn defaults(py: Python<'_>, kind: &str) -> PyResult<Py<PyAny>> {
    match kind {
        "fuels" => {
            let result = PyDict::new(py);
            for (number, record) in behave_fire::FuelModels::default().records() {
                let row = PyDict::new(py);
                row.set_item("code", &record.code)?;
                row.set_item("name", &record.name)?;
                row.set_item("fuel_bed_depth", record.fuel_bed_depth)?;
                row.set_item(
                    "moisture_of_extinction_dead",
                    record.moisture_of_extinction_dead,
                )?;
                row.set_item("heat_of_combustion_dead", record.heat_of_combustion_dead)?;
                row.set_item("heat_of_combustion_live", record.heat_of_combustion_live)?;
                row.set_item("dead_1h", record.dead_1h)?;
                row.set_item("dead_10h", record.dead_10h)?;
                row.set_item("dead_100h", record.dead_100h)?;
                row.set_item("live_herb", record.live_herb)?;
                row.set_item("live_woody", record.live_woody)?;
                row.set_item("savr_1h", record.savr_1h)?;
                row.set_item("savr_live_herb", record.savr_live_herb)?;
                row.set_item("savr_live_woody", record.savr_live_woody)?;
                row.set_item("is_dynamic", record.is_dynamic)?;
                row.set_item("is_reserved", record.is_reserved)?;
                row.set_item("is_defined", record.is_defined)?;
                result.set_item(number, row)?;
            }
            Ok(result.into_any().unbind())
        }
        "species" => {
            let result = PyList::empty(py);
            for record in behave_fire::SpeciesMasterTable::default().records {
                let row = PyDict::new(py);
                row.set_item("species_code", &record.species_code)?;
                row.set_item("scientific_name", &record.scientific_name)?;
                row.set_item("common_name", &record.common_name)?;
                row.set_item(
                    "mortality_equation_number",
                    record.mortality_equation_number,
                )?;
                row.set_item("bark_equation_number", record.bark_equation_number)?;
                row.set_item("crown_coefficient_code", record.crown_coefficient_code)?;
                row.set_item("alaska", record.alaska)?;
                row.set_item("california", record.california)?;
                row.set_item("eastern_area", record.eastern_area)?;
                row.set_item("great_basin", record.great_basin)?;
                row.set_item("northern_rockies", record.northern_rockies)?;
                row.set_item("northwest", record.northwest)?;
                row.set_item("rocky_mountain", record.rocky_mountain)?;
                row.set_item("southern_area", record.southern_area)?;
                row.set_item("southwest", record.southwest)?;
                row.set_item("equation_type", record.equation_type.code())?;
                row.set_item(
                    "crown_damage_equation_code",
                    record.crown_damage_equation_code.code(),
                )?;
                result.append(row)?;
            }
            Ok(result.into_any().unbind())
        }
        "mortality" => {
            let result = PyList::new(py, behave_fire::mortality::build_mortality_lookup(None))?;
            Ok(result.into_any().unbind())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err("unknown table")),
    }
}
#[pyfunction]
// Keep defaults for metadata unrelated to coefficient diagnostics.
#[allow(clippy::field_reassign_with_default)]
pub fn missing_mortality(py: Python<'_>, records: Vec<(String, i32)>) -> Vec<(String, i32)> {
    py.detach(move || {
        let mut table = behave_fire::SpeciesMasterTable::default();
        table.records = records
            .into_iter()
            .map(|(code, equation)| {
                let mut row = behave_fire::SpeciesRecord::default();
                row.species_code = code;
                row.mortality_equation_number = equation;
                row
            })
            .collect();
        behave_fire::mortality::missing_coefficient_equations(&table)
            .into_iter()
            .map(|(code, equation)| (code.to_owned(), equation))
            .collect()
    })
}
