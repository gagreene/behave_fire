//! The f64 runner retains options, overrides, and independent field dependencies.
use behave_fire::batch::SurfaceBatch;
use behave_fire::engine::{
    run_model, ModelBatch, ModelRun, OutputRequest, PreparedFuels, SurfaceCell, SurfaceOptions,
    SurfacePlan, SurfaceSource, SurfaceTweaks, WafMethod, WindHeight, WindOrientation,
};
use behave_fire::raster::{FuelbedColumns, SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

fn surface(columns: SurfaceColumns<'_>) -> SurfaceBatch {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = SurfaceRaster::new(GridShape::new(vec![1]).unwrap(), columns).unwrap();
    match run_model(
        &ModelRun::Surface {
            raster,
            plan: SurfacePlan::default(),
            source: SurfaceSource::Fuels(&fuels),
            tweaks: SurfaceTweakColumns::default(),
        },
        &OutputRequest::surface(),
        1,
    )
    .unwrap()
    {
        ModelBatch::Surface(result) => result,
        _ => panic!("wrong result"),
    }
}

#[test]
fn missing_slope_is_not_a_zero_slope_and_keeps_reaction_available() {
    let output = surface(SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        wind_speed: PerCell::Scalar(88.),
        slope: PerCell::Scalar(f64::NAN),
        ..SurfaceColumns::default()
    });
    assert!(output.reaction_intensity[0].is_finite());
    assert!(output.residence_time[0].is_finite());
    assert!(output.midflame_wind_speed[0].is_finite());
    assert!(output.spread_rate[0].is_nan());
    assert!(output.effective_wind_speed[0].is_nan());
    assert!(output.fire_length_to_width_ratio[0].is_nan());
    assert!(output.flame_length[0].is_nan());
}

#[test]
fn no_canopy_branch_ignores_missing_sheltered_inputs() {
    let output = surface(SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        wind_speed: PerCell::Scalar(88.),
        canopy_cover: PerCell::Scalar(0.),
        canopy_height: PerCell::Scalar(f64::NAN),
        crown_ratio: PerCell::Scalar(f64::NAN),
        aspect: PerCell::Scalar(f64::NAN),
        ..SurfaceColumns::default()
    });
    assert!(output.spread_rate[0].is_finite());
    assert!(output.midflame_wind_speed[0].is_finite());
    assert!(output.direction_of_max_spread[0].is_finite());
}

#[test]
fn nondefault_options_and_persistent_overrides_survive_the_run_bundle() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let options = SurfaceOptions {
        wind_orientation: WindOrientation::North,
        wind_height: WindHeight::TenMetre,
        waf_method: WafMethod::UserInput,
    };
    let plan = SurfacePlan::new(options);
    let mut cell = SurfaceCell::new(124);
    cell.wind_speed = 551.;
    cell.wind_direction = 271.;
    cell.slope = 27.;
    cell.aspect = 113.;
    cell.user_waf = Some(0.4);
    let tweaks = SurfaceTweaks {
        reaction_intensity: Some(913.),
        no_wind_no_slope_spread_rate: Some(3.1),
        midflame_wind_speed: Some(61.),
        wind_phi: Some(2.7),
        slope_phi: Some(0.9),
        spread_rate: Some(13.7),
    };
    let columns = SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        wind_speed: PerCell::Scalar(551.),
        wind_direction: PerCell::Scalar(271.),
        slope: PerCell::Scalar(27.),
        aspect: PerCell::Scalar(113.),
        user_waf: Some(PerCell::Scalar(0.4)),
        ..SurfaceColumns::default()
    };
    let raster = SurfaceRaster::new(GridShape::new(vec![1]).unwrap(), columns).unwrap();
    let overrides = SurfaceTweakColumns {
        reaction_intensity: Some(PerCell::Scalar(913.)),
        no_wind_no_slope_spread_rate: Some(PerCell::Scalar(3.1)),
        midflame_wind_speed: Some(PerCell::Scalar(61.)),
        wind_phi: Some(PerCell::Scalar(2.7)),
        slope_phi: Some(PerCell::Scalar(0.9)),
        spread_rate: Some(PerCell::Scalar(13.7)),
    };
    let expected = plan.run_cell(&cell, fuels.for_grid(124), tweaks);
    let result = match run_model(
        &ModelRun::Surface {
            raster,
            plan,
            source: SurfaceSource::Fuels(&fuels),
            tweaks: overrides,
        },
        &OutputRequest::surface(),
        4,
    )
    .unwrap()
    {
        ModelBatch::Surface(result) => result,
        _ => panic!("wrong result"),
    };
    for (name, column) in result.into_named_columns() {
        if let behave_fire::output::ColumnValues::Float64(values) = column {
            let wanted = expected
                .named_values()
                .into_iter()
                .find(|(key, _)| *key == name)
                .unwrap()
                .1;
            assert_eq!(values[0].to_bits(), wanted.to_bits(), "{name}");
        }
    }
}

#[test]
fn supplied_fuelbeds_keep_their_typed_defined_flag() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::default();
    let mut bed = plan.prepare_fuelbed(&SurfaceCell::new(124), fuels.for_grid(124));
    bed.is_defined = false;
    let raster =
        SurfaceRaster::new(GridShape::new(vec![1]).unwrap(), SurfaceColumns::default()).unwrap();
    let result = match run_model(
        &ModelRun::Surface {
            raster,
            plan,
            source: SurfaceSource::Fuelbeds(FuelbedColumns::uniform(bed)),
            tweaks: SurfaceTweakColumns::default(),
        },
        &OutputRequest::surface(),
        1,
    )
    .unwrap()
    {
        ModelBatch::Surface(result) => result,
        _ => panic!("wrong result"),
    };
    assert_eq!(result.spread_rate, vec![0.]);
    assert_eq!(result.fire_length_to_width_ratio, vec![1.]);
}
#[test]
fn unknown_canopy_condition_does_not_become_a_sheltered_default() {
    let output = surface(SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        wind_speed: PerCell::Scalar(88.),
        canopy_cover: PerCell::Scalar(f64::NAN),
        canopy_height: PerCell::Scalar(30.),
        crown_ratio: PerCell::Scalar(1.),
        ..SurfaceColumns::default()
    });
    assert!(output.midflame_wind_speed[0].is_nan());
    assert!(output.spread_rate[0].is_nan());
}

#[test]
fn static_surface_wind_context_matches_uncached_composition_with_options_and_overrides() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    for id in [1, 4, 10, 101, 124] {
        for orientation in [WindOrientation::North, WindOrientation::Upslope] {
            for height in [WindHeight::TwentyFoot, WindHeight::TenMetre] {
                for injected in [false, true] {
                    let plan = SurfacePlan::new(SurfaceOptions {
                        wind_orientation: orientation,
                        wind_height: height,
                        waf_method: WafMethod::UserInput,
                    });
                    let cell = SurfaceCell {
                        wind_speed: 551.,
                        wind_direction: 271.,
                        slope: 27.,
                        aspect: 113.,
                        user_waf: Some(0.4),
                        ..SurfaceCell::new(id)
                    };
                    let tweaks = if injected {
                        SurfaceTweaks {
                            reaction_intensity: Some(913.),
                            no_wind_no_slope_spread_rate: Some(3.1),
                            midflame_wind_speed: Some(61.),
                            wind_phi: Some(2.7),
                            slope_phi: Some(0.9),
                            spread_rate: Some(13.7),
                        }
                    } else {
                        SurfaceTweaks::default()
                    };
                    let expected = plan.run_cell(&cell, fuels.for_grid(id), tweaks);
                    let raster = SurfaceRaster::new(
                        GridShape::new(vec![1031]).unwrap(),
                        SurfaceColumns {
                            fuel_model_grid: PerCell::Scalar(id),
                            wind_speed: PerCell::Scalar(551.),
                            wind_direction: PerCell::Scalar(271.),
                            slope: PerCell::Scalar(27.),
                            aspect: PerCell::Scalar(113.),
                            user_waf: Some(PerCell::Scalar(0.4)),
                            ..SurfaceColumns::default()
                        },
                    )
                    .unwrap();
                    let columns = SurfaceTweakColumns {
                        reaction_intensity: tweaks.reaction_intensity.map(PerCell::Scalar),
                        no_wind_no_slope_spread_rate: tweaks
                            .no_wind_no_slope_spread_rate
                            .map(PerCell::Scalar),
                        midflame_wind_speed: tweaks.midflame_wind_speed.map(PerCell::Scalar),
                        wind_phi: tweaks.wind_phi.map(PerCell::Scalar),
                        slope_phi: tweaks.slope_phi.map(PerCell::Scalar),
                        spread_rate: tweaks.spread_rate.map(PerCell::Scalar),
                    };
                    let run = ModelRun::Surface {
                        raster,
                        plan,
                        source: SurfaceSource::Fuels(&fuels),
                        tweaks: columns,
                    };
                    let ModelBatch::Surface(actual) =
                        run_model(&run, &OutputRequest::surface(), 4).unwrap()
                    else {
                        unreachable!()
                    };
                    for (name, column) in actual.into_named_columns() {
                        let behave_fire::output::ColumnValues::Float64(values) = column else {
                            panic!("wrong Surface column type")
                        };
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
                            "{id}/{name}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn missing_dead_moisture_keeps_independent_wind_and_residence() {
    let output = surface(SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        m1h: PerCell::Scalar(f64::NAN),
        wind_speed: PerCell::Scalar(88.),
        ..SurfaceColumns::default()
    });
    assert!(output.reaction_intensity[0].is_nan());
    assert!(output.no_wind_no_slope_spread_rate[0].is_nan());
    assert!(output.spread_rate[0].is_nan());
    assert!(output.heat_per_unit_area[0].is_nan());
    assert!(output.flame_length[0].is_nan());
    assert!(output.midflame_wind_speed[0].is_finite());
    assert!(output.residence_time[0].is_finite());
}

#[test]
fn missing_direction_is_not_a_zero_direction_or_valid_flame() {
    let output = surface(SurfaceColumns {
        fuel_model_grid: PerCell::Scalar(124),
        wind_speed: PerCell::Scalar(88.),
        wind_direction: PerCell::Scalar(f64::NAN),
        ..SurfaceColumns::default()
    });
    assert!(output.direction_of_max_spread[0].is_nan());
    assert!(output.spread_rate[0].is_nan());
    assert!(output.flame_length[0].is_nan());
    assert!(output.reaction_intensity[0].is_finite());
    assert!(output.midflame_wind_speed[0].is_finite());
}
