//! Scalar and raster injection must use the same numerical stages.
use behave_fire::engine::{PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks};
use behave_fire::raster::{FuelbedColumns, SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn crown_override_columns_and_tiles_match_scalar_stages() {
    use behave_fire::crown::SurfaceFireInputs;
    use behave_fire::engine::{CrownCell, CrownPlan, CrownTweaks};
    use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
    let rate = [1.0, 3.0, 9.0];
    let fraction = [0.0, 0.5, 1.0];
    let raster = CrownRaster::new(
        GridShape::new(vec![3]).unwrap(),
        CrownColumns {
            surface_ros: PerCell::Scalar(4.0),
            surface_fli: PerCell::Scalar(60.0),
            surface_hpua: PerCell::Scalar(900.0),
            wind_speed: PerCell::Scalar(440.0),
            canopy_base_height: PerCell::Scalar(6.0),
            canopy_height: PerCell::Scalar(30.0),
            canopy_bulk_density: PerCell::Scalar(0.02),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let plan = CrownPlan::new(SurfaceOptions::default());
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let tweaks = CrownTweakColumns {
        fm10: SurfaceTweakColumns {
            spread_rate: Some(PerCell::Cells(&rate)),
            ..SurfaceTweakColumns::default()
        },
        critical_surface_intensity: Some(PerCell::Scalar(1.0)),
        critical_crown_rate: Some(PerCell::Scalar(1e6)),
        crown_fraction_burned: Some(PerCell::Cells(&fraction)),
    };
    let output = raster.run(&plan, &fuels, &tweaks, 1).unwrap();
    for index in 0..3 {
        let cell = CrownCell {
            wind_speed: 440.0,
            canopy_base_height: 6.0,
            canopy_height: 30.0,
            canopy_bulk_density: 0.02,
            ..CrownCell::default()
        };
        let mut supplied = SurfaceFireInputs::<f64>::new();
        supplied.surface_ros = 4.0;
        supplied.surface_fli = 60.0;
        supplied.surface_hpua = 900.0;
        let scalar = plan.run_cell(
            &cell,
            &supplied,
            fuels.for_grid(10),
            CrownTweaks {
                fm10: SurfaceTweaks {
                    spread_rate: Some(rate[index]),
                    ..SurfaceTweaks::default()
                },
                critical_surface_intensity: Some(1.0),
                critical_crown_rate: Some(1e6),
                crown_fraction_burned: Some(fraction[index]),
            },
        );
        assert_eq!(output.final_spread_rate[index], scalar.final_spread_rate);
        assert_eq!(output.fire_type[index], scalar.fire_type);
    }
    let tile = raster.run_tile(1..3, &plan, &fuels, &tweaks, 1).unwrap();
    assert_eq!(tile.final_spread_rate, output.final_spread_rate[1..]);
}

#[test]
fn per_cell_overrides_match_scalar_composition_and_do_not_persist() {
    let reactions = [100.0, 200.0, 400.0, 800.0];
    let midflame = [0.0, 10.0, 100.0, 1000.0];
    let rates = [1.0, 2.0, 4.0, 8.0];
    let raster = SurfaceRaster::new(
        GridShape::new(vec![2, 2]).unwrap(),
        SurfaceColumns {
            fuel_model_grid: PerCell::Scalar(10),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let before = raster
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    let tweaks = SurfaceTweakColumns {
        reaction_intensity: Some(PerCell::Cells(&reactions)),
        midflame_wind_speed: Some(PerCell::Cells(&midflame)),
        spread_rate: Some(PerCell::Cells(&rates)),
        ..SurfaceTweakColumns::default()
    };
    let output = raster.run(&plan, &fuels, &tweaks, 1).unwrap();
    for index in 0..4 {
        let scalar = plan.run_cell(
            &SurfaceCell::new(10),
            fuels.for_grid(10),
            SurfaceTweaks {
                reaction_intensity: Some(reactions[index]),
                midflame_wind_speed: Some(midflame[index]),
                spread_rate: Some(rates[index]),
                ..SurfaceTweaks::default()
            },
        );
        assert_eq!(output.spread_rate[index], scalar.spread_rate);
        assert_eq!(output.fireline_intensity[index], scalar.fireline_intensity);
        assert_eq!(
            output.midflame_wind_speed[index],
            scalar.midflame_wind_speed
        );
    }
    let after = raster
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    assert_eq!(before.spread_rate, after.spread_rate);
}

#[test]
fn edited_fuelbed_columns_resume_without_particle_reconstruction() {
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let cell = SurfaceCell::new(10);
    let ib = plan.prepare_fuelbed(&cell, fuels.for_grid(10));
    let flux = [
        ib.propagating_flux,
        ib.propagating_flux * 2.0,
        ib.propagating_flux * 4.0,
    ];
    let mut columns = FuelbedColumns::uniform(ib);
    columns.propagating_flux = PerCell::Cells(&flux);
    let raster = SurfaceRaster::new(
        GridShape::new(vec![3]).unwrap(),
        SurfaceColumns {
            fuel_model_grid: PerCell::Scalar(0),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let output = raster
        .run_from_fuelbeds(&plan, &columns, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    for (index, value) in flux.into_iter().enumerate() {
        let mut supplied = ib;
        supplied.propagating_flux = value;
        let scalar = plan.run_from_fuelbed(&cell, &supplied, SurfaceTweaks::default());
        assert_eq!(output.spread_rate[index], scalar.spread_rate);
        assert!(output.spread_rate[index] > 0.0);
    }
}
