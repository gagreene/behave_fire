//! Prepared numeric fuel snapshots without names, dictionaries, or heap-owned fields.
use crate::{FuelModel, FuelModels};

/// Numerical fuel properties consumed by particle and fire calculations.
#[derive(Clone, Copy, Debug, Default)]
pub struct PreparedFuel {
    /// Fuel bed depth in feet.
    pub fuel_bed_depth: f64,
    /// Dead moisture of extinction as a fraction.
    pub moisture_of_extinction_dead: f64,
    /// Dead heat of combustion in BTU/lb.
    pub heat_of_combustion_dead: f64,
    /// Live heat of combustion in BTU/lb.
    pub heat_of_combustion_live: f64,
    /// One-hour dead loading in lb/ft².
    pub dead_1h: f64,
    /// Ten-hour dead loading in lb/ft².
    pub dead_10h: f64,
    /// Hundred-hour dead loading in lb/ft².
    pub dead_100h: f64,
    /// Live herbaceous loading in lb/ft².
    pub live_herb: f64,
    /// Live woody loading in lb/ft².
    pub live_woody: f64,
    /// One-hour surface area to volume ratio in inverse feet.
    pub savr_1h: f64,
    /// Live herbaceous surface area to volume ratio in inverse feet.
    pub savr_live_herb: f64,
    /// Live woody surface area to volume ratio in inverse feet.
    pub savr_live_woody: f64,
    /// Whether live herbaceous load transfers as its moisture decreases.
    pub is_dynamic: bool,
    /// Whether final Surface outputs represent a defined fuel.
    pub is_defined: bool,
    savr_cache: Option<([f64; 3], crate::surface::SavrTerms)>,
    static_cache: Option<([f64; 14], StaticPhysics)>,
}

/// Moisture-independent coefficients retained outside ordinary cell loops.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StaticPhysics {
    fuelbed: crate::surface::PreparedFuelbed,
    reaction: crate::surface::ReactionCoefficients,
    wind: crate::surface::PreparedWind,
}

impl PartialEq for PreparedFuel {
    /// Compare public physical properties independently of private cache history.
    fn eq(&self, other: &Self) -> bool {
        self.fuel_bed_depth == other.fuel_bed_depth
            && self.moisture_of_extinction_dead == other.moisture_of_extinction_dead
            && self.heat_of_combustion_dead == other.heat_of_combustion_dead
            && self.heat_of_combustion_live == other.heat_of_combustion_live
            && self.dead_1h == other.dead_1h
            && self.dead_10h == other.dead_10h
            && self.dead_100h == other.dead_100h
            && self.live_herb == other.live_herb
            && self.live_woody == other.live_woody
            && self.savr_1h == other.savr_1h
            && self.savr_live_herb == other.savr_live_herb
            && self.savr_live_woody == other.savr_live_woody
            && self.is_dynamic == other.is_dynamic
            && self.is_defined == other.is_defined
    }
}

impl PreparedFuel {
    /// Capture every physical property that determines static cached coefficients.
    fn physics_key(&self) -> [f64; 14] {
        [
            self.fuel_bed_depth,
            self.moisture_of_extinction_dead,
            self.heat_of_combustion_dead,
            self.heat_of_combustion_live,
            self.dead_1h,
            self.dead_10h,
            self.dead_100h,
            self.live_herb,
            self.live_woody,
            self.savr_1h,
            self.savr_live_herb,
            self.savr_live_woody,
            self.is_dynamic as u8 as f64,
            self.is_defined as u8 as f64,
        ]
    }
    /// Borrow static coefficients only while their exact source properties match.
    fn physics(&self) -> Option<&StaticPhysics> {
        self.static_cache
            .as_ref()
            .filter(|(key, _)| {
                key.iter()
                    .zip(self.physics_key())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            })
            .map(|(_, physics)| physics)
    }
    /// Evaluate a fuelbed using validated cached terms or the ordinary equations.
    pub(crate) fn prepare_fuelbed(
        &self,
        particles: &crate::surface::Particles,
    ) -> crate::surface::FuelbedIntermediates {
        match self.physics() {
            Some(physics) => physics.fuelbed.run(particles),
            None => crate::surface::calculate_fuelbed_with_terms(particles, &self.savr_terms()),
        }
    }
    /// Prepare a fuelbed without losing missing selected physical prerequisites.
    pub fn prepare_fuelbed_available(
        &self,
        particles: &crate::surface::Particles,
    ) -> crate::surface::FuelbedIntermediates {
        match self.physics() {
            Some(physics) => physics.fuelbed.run_available(particles, &self.savr_terms()),
            None => crate::surface::calculate_fuelbed_available_with_terms(
                particles,
                &self.savr_terms(),
            ),
        }
    }
    /// Evaluate reaction intensity using validated cached coefficients when available.
    pub(crate) fn reaction_intensity(&self, bed: &crate::surface::FuelbedIntermediates) -> f64 {
        match self.physics() {
            Some(physics) => physics.reaction.run(bed),
            None => crate::surface::calculate_reaction_intensity(bed),
        }
    }
    /// Apply strict availability through cached or ordinary reaction coefficients.
    pub fn reaction_intensity_available(&self, bed: &crate::surface::FuelbedIntermediates) -> f64 {
        match self.physics() {
            Some(physics) => physics.reaction.run_available(bed),
            None => crate::surface::calculate_reaction_intensity_available(bed),
        }
    }
    /// Return validated static wind physics without caching dynamic moisture transfers.
    pub(crate) fn wind_terms(&self) -> Option<crate::surface::PreparedWind> {
        self.physics().map(|p| p.wind)
    }
    /// Capture the physical ratios that determine particle exponentials and bins.
    fn savr_key(&self) -> [f64; 3] {
        [self.savr_1h, self.savr_live_herb, self.savr_live_woody]
    }

    /// Reuse particle terms only while the original physical SAVR values match.
    pub(crate) fn savr_terms(&self) -> crate::surface::SavrTerms {
        if let Some((key, terms)) = self.savr_cache {
            if key
                .iter()
                .zip(self.savr_key())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                return terms;
            }
        }
        crate::surface::SavrTerms::new(
            &[self.savr_1h, 109.0, 30.0, self.savr_live_herb, 0.0],
            &[self.savr_live_herb, self.savr_live_woody, 0.0, 0.0, 0.0],
        )
    }
    /// Prepare a numeric row after direct construction or edits, without registry names.
    pub fn prepared(mut self) -> Self {
        self.savr_cache = None;
        self.static_cache = None;
        self.savr_cache = Some((self.savr_key(), self.savr_terms()));
        if !self.is_dynamic {
            let particles = crate::surface::build_particle_arrays_from_prepared(
                &self, 0.01, 0.01, 0.01, 0.01, 0.01,
            );
            let fuelbed = crate::surface::PreparedFuelbed::new(&particles, &self.savr_terms());
            let reaction = crate::surface::ReactionCoefficients::new(&fuelbed.template);
            self.static_cache = Some((
                self.physics_key(),
                StaticPhysics {
                    wind: crate::surface::PreparedWind::new(&fuelbed.template),
                    fuelbed,
                    reaction,
                },
            ));
        }
        self
    }
    /// Copy physical properties without normalizing undefined rows or cloning names.
    pub fn from_model(model: &FuelModel) -> Self {
        let result = Self {
            fuel_bed_depth: model.fuel_bed_depth,
            moisture_of_extinction_dead: model.moisture_of_extinction_dead,
            heat_of_combustion_dead: model.heat_of_combustion_dead,
            heat_of_combustion_live: model.heat_of_combustion_live,
            dead_1h: model.dead_1h,
            dead_10h: model.dead_10h,
            dead_100h: model.dead_100h,
            live_herb: model.live_herb,
            live_woody: model.live_woody,
            savr_1h: model.savr_1h,
            savr_live_herb: model.savr_live_herb,
            savr_live_woody: model.savr_live_woody,
            is_dynamic: model.is_dynamic,
            is_defined: model.is_defined,
            savr_cache: None,
            static_cache: None,
        };
        result.prepared()
    }
}

#[cfg(test)]
mod wind_cache_tests {
    use super::*;

    #[test]
    fn wind_terms_invalidate_after_fuel_edits_and_exclude_dynamic_transfers() {
        let registry = FuelModels::default();
        let mut row = PreparedFuel::from_model(registry.for_grid(10));
        assert!(row.wind_terms().is_some());
        row.savr_1h += 1.;
        assert!(row.wind_terms().is_none());
        row = row.prepared();
        let refreshed = row.wind_terms().unwrap();
        let cell = crate::engine::SurfaceCell::new(10);
        let bed = crate::engine::SurfacePlan::default().prepare_fuelbed(&cell, &row);
        assert!(refreshed.matches(&bed));
        row.is_dynamic = true;
        assert!(row.wind_terms().is_none());
        assert!(row.prepared().wind_terms().is_none());
        for id in [101, 124] {
            assert!(PreparedFuel::from_model(registry.for_grid(id))
                .wind_terms()
                .is_none());
        }
    }
}

/// Immutable dense lookup snapshot prepared once for repeated calculations.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFuels {
    rows: Box<[PreparedFuel]>,
}

impl PreparedFuels {
    /// Snapshot every clamped grid slot using the registry's existing fallback rules.
    pub fn from_registry(fuels: &FuelModels) -> Self {
        Self {
            rows: (0..261)
                .map(|id| PreparedFuel::from_model(fuels.for_grid(id)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    /// Prepare numeric lookup rows without constructing named registry objects.
    pub fn from_rows(rows: Vec<PreparedFuel>) -> Result<Self, crate::BehaveError> {
        if rows.is_empty() {
            return Err(crate::BehaveError::LengthMismatch {
                field: "fuel registry".into(),
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self {
            rows: rows
                .into_iter()
                .map(PreparedFuel::prepared)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    /// Borrow a physical record, clamping the identifier to the available numeric rows.
    pub fn for_grid(&self, id: i32) -> &PreparedFuel {
        &self.rows[(id.max(0) as usize).min(self.rows.len() - 1)]
    }
}

#[cfg(test)]
mod preparation_tests {
    use super::*;
    #[test]
    fn savr_terms_are_recomputed_after_raw_edits() {
        let mut fuel = PreparedFuel::from_model(FuelModels::default().for_grid(10));
        let original = fuel.savr_terms();
        fuel.savr_1h = 1200.0;
        fuel.savr_live_herb = 48.0;
        let changed = fuel.savr_terms();
        assert_eq!(changed.exp_dead[0], (-138.0_f64 / 1200.0).exp());
        assert_eq!(changed.exp_live500[0], (-500.0_f64 / 48.0).exp());
        assert_eq!(changed.bins_dead[0], 0);
        assert_eq!(changed.bins_dead[3], 3);
        assert_ne!(original.exp_dead[0], changed.exp_dead[0]);
    }
}

#[cfg(test)]
mod static_preparation_tests {
    use super::*;
    #[test]
    fn static_physical_coefficients_do_not_leak_into_dynamic_or_edited_fuelbeds() {
        let registry = FuelModels::default();
        let mut model = registry.for_grid(10).clone();
        model.live_herb = 0.31;
        let original = PreparedFuel::from_model(&model);
        assert!(original.static_cache.is_some());
        let cell = crate::engine::SurfaceCell {
            m1h: 0.19,
            m10h: 0.23,
            mlh: 0.41,
            ..crate::engine::SurfaceCell::new(10)
        };
        let plan = crate::engine::SurfacePlan::default();
        for edit in 0..4 {
            let mut fuel = original;
            if edit == 1 {
                fuel.dead_1h *= 1.3;
                fuel.fuel_bed_depth *= 0.7;
            }
            if edit == 2 {
                fuel.is_dynamic = true;
            }
            if edit == 3 {
                fuel.is_defined = false;
                fuel.heat_of_combustion_dead = 17.5;
            }
            let p = crate::surface::build_particle_arrays_from_prepared(
                &fuel, cell.m1h, cell.m10h, cell.m100h, cell.mlh, cell.mlw,
            );
            let expected = crate::surface::calculate_fuelbed_intermediates(&p);
            assert_eq!(plan.prepare_fuelbed(&cell, &fuel), expected);
            let cached = fuel.reaction_intensity(&expected);
            let ordinary = crate::surface::calculate_reaction_intensity(&expected);
            assert_eq!(cached.to_bits(), ordinary.to_bits());
        }
        assert!(PreparedFuel::from_model(registry.for_grid(101))
            .static_cache
            .is_none());
    }
}

#[cfg(test)]
mod zero_sign_tests {
    use super::*;
    #[test]
    fn physical_cache_preserves_changed_zero_signs() {
        let mut fuel = PreparedFuel::from_model(FuelModels::default().for_grid(0));
        fuel.fuel_bed_depth = -0.0;
        let plan = crate::engine::SurfacePlan::default();
        let bed = plan.prepare_fuelbed(&crate::engine::SurfaceCell::new(0), &fuel);
        assert_eq!(bed.depth.to_bits(), (-0.0_f64).to_bits());
    }
}

#[cfg(test)]
mod numeric_rows_tests {
    use super::*;
    #[test]
    fn numeric_registry_rows_prepare_without_registry_names_and_refresh_edits() {
        let registry = FuelModels::default();
        let mut row = PreparedFuel::from_model(registry.for_grid(10));
        row.fuel_bed_depth *= 1.3;
        let row = row.prepared();
        assert!(row.physics().is_some());
        let fuels = PreparedFuels::from_rows(vec![PreparedFuel::default(), row]).unwrap();
        assert_eq!(fuels.for_grid(-5).fuel_bed_depth, 0.0);
        assert_eq!(fuels.for_grid(500).fuel_bed_depth, row.fuel_bed_depth);
        assert!(PreparedFuels::from_rows(Vec::new()).is_err());
    }
}
