//! Scorch and logistic mortality, preserving explicit Python reference 7933789.
use crate::{math::np_clip, species::SpeciesMasterTable};

/// Mortality coefficients indexed by equation number, including undefined rows.
pub type MortalityLookup = [[f64; 3]; 121];

/// Return the standalone coefficient table. Validation diagnostics are separate.
pub fn build_mortality_lookup(
    _species_master_table: Option<&SpeciesMasterTable>,
) -> MortalityLookup {
    let mut coefficients = [[0.0; 3]; 121];
    let crown = [
        (-2.4027, 0.9568),
        (-2.2471, 0.8093),
        (-3.1789, 1.5416),
        (-2.3327, 0.9577),
        (-2.3327, 0.9577),
        (-1.4667, 0.7401),
        (-2.4027, 0.9568),
        (-3.3701, 1.4804),
        (-3.3701, 1.4804),
        (-2.9986, 1.4804),
        (-2.9986, 1.4804),
        (-2.4027, 0.9568),
        (-3.0471, 1.2668),
        (-2.3327, 0.9577),
        (-2.7735, 1.0070),
        (-3.3701, 1.4804),
        (-2.2471, 0.8093),
        (-2.4027, 0.9568),
        (-2.2471, 0.8093),
        (-3.3701, 1.4804),
    ];
    for (index, (b1, b3)) in crown.into_iter().enumerate() {
        coefficients[index + 1] = [b1, 0.0, b3];
    }
    coefficients[100..110].copy_from_slice(&[
        [2.3014, -0.3267, 1.1137],
        [-0.8727, -0.1814, 4.1947],
        [2.7899, -0.5511, 1.2888],
        [1.9438, -0.4602, 1.6352],
        [-1.8137, -0.0603, 0.8666],
        [-1.6262, -0.0339, 0.6901],
        [0.3714, -0.1005, 1.5577],
        [-1.4416, -0.1469, 1.3159],
        [0.1122, -0.1287, 1.2612],
        [1.6779, -1.0299, 10.2855],
    ]);
    coefficients
}

/// Return one diagnostic per uncovered species record, preserving record order.
pub fn missing_coefficient_equations(table: &SpeciesMasterTable) -> Vec<(&str, i32)> {
    table
        .records
        .iter()
        .filter_map(|r| {
            let eq = r.mortality_equation_number;
            (eq > 0 && !(1..=20).contains(&eq) && !(100..=109).contains(&eq))
                .then_some((r.species_code.as_str(), eq))
        })
        .collect()
}

/// Per-cell crown scorch and mortality fractions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MortalityResult {
    /// Fraction of crown length scorched.
    pub crown_length_scorch: f64,
    /// Fraction of crown volume scorched.
    pub crown_volume_scorch: f64,
    /// Logistic mortality probability.
    pub probability_mortality: f64,
}
impl MortalityResult {
    /// Output names matching the Python dictionary.
    pub fn named_values(self) -> [(&'static str, f64); 3] {
        [
            ("crown_length_scorch", self.crown_length_scorch),
            ("crown_volume_scorch", self.crown_volume_scorch),
            ("probability_mortality", self.probability_mortality),
        ]
    }
}

/// Calculate crown scorch mortality with clamped equation indexing.
pub fn calculate_crown_scorch_mortality(
    scorch_height_ft: f64,
    tree_height_ft: f64,
    crown_ratio: f64,
    dbh_inches: f64,
    equation_number_grid: i32,
    coeffs: &MortalityLookup,
) -> MortalityResult {
    calculate_crown_scorch_mortality_with_coefficients(
        scorch_height_ft,
        tree_height_ft,
        crown_ratio,
        dbh_inches,
        coeffs[equation_number_grid.clamp(0, 120) as usize],
    )
}

/// Calculate scorch fractions from their three actual tree and scorch inputs.
pub fn calculate_crown_scorch_fractions(
    scorch_height_ft: f64,
    tree_height_ft: f64,
    crown_ratio: f64,
) -> (f64, f64) {
    let crown_length = crown_ratio * tree_height_ft;
    let crown_base = tree_height_ft - crown_length;
    let cls = if crown_length.is_nan() {
        f64::NAN
    } else if crown_length > 1e-7 {
        np_clip((scorch_height_ft - crown_base) / crown_length, 0.0, 1.0)
    } else {
        0.0
    };
    let cvs = cls.powi(2) * (3.0 - 2.0 * cls);
    (cls, cvs)
}

/// Calculate mortality from an actual selected coefficient row.
/// A zero coefficient triple suppresses mortality; equation numbers add no gate.
pub fn calculate_crown_scorch_mortality_with_coefficients(
    scorch_height_ft: f64,
    tree_height_ft: f64,
    crown_ratio: f64,
    dbh_inches: f64,
    coefficients: [f64; 3],
) -> MortalityResult {
    let (cls, cvs) =
        calculate_crown_scorch_fractions(scorch_height_ft, tree_height_ft, crown_ratio);
    let [b1, b2, b3] = coefficients;
    let logit = b1 + b2 * dbh_inches + b3 * cvs;
    let probability = if b1 == 0.0 && b2 == 0.0 && b3 == 0.0 {
        0.0
    } else {
        1.0 / (1.0 + (-logit).exp())
    };
    MortalityResult {
        crown_length_scorch: cls,
        crown_volume_scorch: cvs,
        probability_mortality: np_clip(probability, 0.0, 1.0),
    }
}
/// Calculate scorch height in feet with the reference near-zero denominator rule.
pub fn calculate_scorch_height(
    fireline_intensity_btu_ft_s: f64,
    midflame_wind_mph: f64,
    air_temp_f: f64,
) -> f64 {
    let fi = fireline_intensity_btu_ft_s;
    let denom = 140.0 - air_temp_f;
    let safe_denom = if denom.is_nan() {
        f64::NAN
    } else if denom.abs() > 1e-7 {
        denom
    } else {
        1.0
    };
    if fi.is_nan() {
        f64::NAN
    } else if fi > 1e-7 {
        (63.0 / safe_denom) * fi.powf(1.166667) / (fi + midflame_wind_mph.powi(3)).sqrt()
    } else {
        0.0
    }
}
