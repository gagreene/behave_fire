//! The new model path stores every requested f64 value and exact typed code.
use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    run_model, CrownCell, CrownPlan, CrownTweaks, Field, ModelBatch, ModelRun, OutputRequest,
    PreparedFuel, PreparedFuels, SurfaceOptions, WindHeight, WindOrientation,
};
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn fused_crown_matches_scalar_f64_composition_across_options_and_numeric_fuels() {
    let models = FuelModels::default();
    for scenario in 0..3 {
        let mut rows = (0..=260)
            .map(|id| PreparedFuel::from_model(models.for_grid(id)))
            .collect::<Vec<_>>();
        if scenario == 1 {
            rows[10].is_dynamic = true;
            rows[10].live_herb = 0.1;
            rows[10].savr_live_herb = 1800.;
        }
        if scenario == 2 {
            rows[10].is_defined = false;
        }
        let fuels = PreparedFuels::from_rows(rows).unwrap();
        for orientation in [WindOrientation::Upslope, WindOrientation::North] {
            for height in [WindHeight::TwentyFoot, WindHeight::TenMetre] {
                for herb in [0.3, 0.8, 2.] {
                    let plan = CrownPlan::new(SurfaceOptions {
                        wind_orientation: orientation,
                        wind_height: height,
                        ..SurfaceOptions::default()
                    });
                    let cell = CrownCell {
                        wind_speed: 551.,
                        wind_direction: 271.,
                        slope: 27.,
                        aspect: 113.,
                        mlh: herb,
                        canopy_base_height: 4.,
                        canopy_height: 30.,
                        canopy_bulk_density: 0.01,
                        ..CrownCell::default()
                    };
                    let mut supplied = SurfaceFireInputs::<f64>::new();
                    supplied.surface_ros = 2.;
                    supplied.surface_fli = 200.;
                    supplied.surface_hpua = 1000.;
                    let expected =
                        plan.run_cell(&cell, &supplied, fuels.for_grid(10), CrownTweaks::default());
                    let inputs = CrownColumns {
                        wind_speed: PerCell::Scalar(cell.wind_speed),
                        wind_direction: PerCell::Scalar(cell.wind_direction),
                        slope: PerCell::Scalar(cell.slope),
                        aspect: PerCell::Scalar(cell.aspect),
                        mlh: PerCell::Scalar(herb),
                        canopy_base_height: PerCell::Scalar(4.),
                        canopy_height: PerCell::Scalar(30.),
                        canopy_bulk_density: PerCell::Scalar(0.01),
                        surface_ros: PerCell::Scalar(2.),
                        surface_fli: PerCell::Scalar(200.),
                        surface_hpua: PerCell::Scalar(1000.),
                        ..CrownColumns::default()
                    };
                    let run = ModelRun::Crown {
                        raster: CrownRaster::new(GridShape::new(vec![1031]).unwrap(), inputs)
                            .unwrap(),
                        plan,
                        fuels: &fuels,
                        tweaks: CrownTweakColumns::default(),
                    };
                    let actual = match run_model(&run, &OutputRequest::crown(), 4).unwrap() {
                        ModelBatch::Crown(actual) => actual,
                        _ => panic!("wrong model"),
                    };
                    assert!(actual
                        .fire_type
                        .iter()
                        .all(|code| *code == expected.fire_type));
                    for (name, values) in actual.into_named_columns() {
                        if let behave_fire::output::ColumnValues::Float64(values) = values {
                            let wanted = expected
                                .named_values()
                                .into_iter()
                                .find(|(key, _)| *key == name)
                                .unwrap()
                                .1;
                            assert!(
                                values
                                    .iter()
                                    .all(|value| value.to_bits() == wanted.to_bits()
                                        || (value.is_nan() && wanted.is_nan())),
                                "{name}, fuel scenario{scenario}"
                            );
                        }
                    }
                    let selected = match run_model(
                        &run,
                        &OutputRequest::new(&[Field::FinalSpreadRate, Field::FireType]),
                        1,
                    )
                    .unwrap()
                    {
                        ModelBatch::Crown(actual) => actual,
                        _ => panic!("wrong model"),
                    };
                    assert_eq!(
                        selected.requested(),
                        [Field::FinalSpreadRate, Field::FireType]
                    );
                    assert_eq!(selected.final_spread_rate.len(), 1031);
                    assert!(selected.crown_fire_spread_rate.is_empty());
                    assert_eq!(selected.statistics().appended_bytes, 0);
                }
            }
        }
    }
}
