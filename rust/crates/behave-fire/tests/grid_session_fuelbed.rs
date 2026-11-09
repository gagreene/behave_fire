use behave_fire::engine::{
    Field, GridSession, ModelRun, OutputRequest, PreparedFuels, SurfacePlan, SurfaceSource,
};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape};

#[test]
fn injected_sigma_and_packing_drive_their_actual_descendants() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster =
        SurfaceRaster::new(GridShape::new(vec![2]).unwrap(), SurfaceColumns::default()).unwrap();
    let run = ModelRun::Surface {
        raster,
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    let mut session = GridSession::new(
        &run,
        OutputRequest::new(&[
            Field::FuelbedRelativePackingRatio,
            Field::FuelbedPropagatingFlux,
        ]),
        &[],
        1,
    )
    .unwrap();
    session
        .set_physical(Field::FuelbedSigma, vec![1500.])
        .unwrap();
    session
        .set_physical(Field::FuelbedPackingRatio, vec![0.02])
        .unwrap();
    session.run_remaining().unwrap();
    let first = session
        .get(Field::FuelbedRelativePackingRatio)
        .unwrap()
        .physical()
        .unwrap()[0];
    let flux = session
        .get(Field::FuelbedPropagatingFlux)
        .unwrap()
        .physical()
        .unwrap()[0];
    assert!(first.is_finite() && first > 0. && flux.is_finite());
    assert!(session.get(Field::ParticleLoadDead0).is_err());
    session
        .set_physical(Field::FuelbedPackingRatio, vec![0.04])
        .unwrap();
    assert!(session.get(Field::FuelbedPropagatingFlux).is_err());
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::FuelbedRelativePackingRatio)
            .unwrap()
            .physical()
            .unwrap()[0],
        first * 2.
    );
    assert_ne!(
        session
            .get(Field::FuelbedPropagatingFlux)
            .unwrap()
            .physical()
            .unwrap()[0],
        flux
    );
    session
        .set_physical(Field::FuelbedSigma, vec![f64::NAN])
        .unwrap();
    session.run_remaining().unwrap();
    assert!(session
        .get(Field::FuelbedRelativePackingRatio)
        .unwrap()
        .physical()
        .unwrap()[0]
        .is_nan());
    assert!(session
        .get(Field::FuelbedPropagatingFlux)
        .unwrap()
        .physical()
        .unwrap()[0]
        .is_nan());
}

#[test]
fn injected_dead_fraction_updates_live_fraction_and_characteristic_sigma() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster =
        SurfaceRaster::new(GridShape::new(vec![2]).unwrap(), SurfaceColumns::default()).unwrap();
    let run = ModelRun::Surface {
        raster,
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    let mut session = GridSession::new(
        &run,
        OutputRequest::new(&[Field::FuelbedFracLive, Field::FuelbedSigma]),
        &[],
        1,
    )
    .unwrap();
    session
        .set_physical(Field::FuelbedFracDead, vec![0.])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::FuelbedFracLive)
            .unwrap()
            .physical()
            .unwrap(),
        &[1.; 2]
    );
    assert_eq!(
        session
            .get(Field::FuelbedSigma)
            .unwrap()
            .physical()
            .unwrap(),
        &[0.; 2]
    );
    session
        .set_physical(Field::FuelbedFracDead, vec![1.])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::FuelbedFracLive)
            .unwrap()
            .physical()
            .unwrap(),
        &[0.; 2]
    );
    assert!(
        session
            .get(Field::FuelbedSigma)
            .unwrap()
            .physical()
            .unwrap()[0]
            > 0.
    );
}
