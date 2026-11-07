//! Flat f64 stage state distinguishes absent data from known unavailable values.
use behave_fire::engine::{CellState, Field, FieldKind, Stage};

#[test]
fn state_keeps_nan_presence_and_typed_controls_separate() {
    let mut state = CellState::new();
    assert!(state.physical(Field::ReactionIntensity).is_err());
    state
        .set_physical(Field::ReactionIntensity, f64::NAN)
        .unwrap();
    assert!(state.is_present(Field::ReactionIntensity));
    assert!(!state.is_available(Field::ReactionIntensity));
    assert!(state.physical(Field::ReactionIntensity).unwrap().is_nan());
    state.set_code(Field::FireType, 3).unwrap();
    assert_eq!(state.code(Field::FireType).unwrap(), 3);
    assert!(state.set_physical(Field::FireType, 3.).is_err());
    state.set_flag(Field::FuelbedDefined, false).unwrap();
    assert!(!state.flag(Field::FuelbedDefined).unwrap());
    assert!(state.is_available(Field::FuelbedDefined));
    assert_eq!(Field::FireType.kind(), FieldKind::Code);
    state.reset_cell();
    assert!(!state.is_present(Field::FireType));
    assert!(state.code(Field::FireType).is_err());
}

#[test]
fn stage_metadata_has_one_producer_and_no_derived_cycles() {
    fn visit(field: Field, active: &mut Vec<Field>, done: &mut Vec<Field>) {
        if done.contains(&field) {
            return;
        }
        assert!(!active.contains(&field), "cycle through {field:?}");
        active.push(field);
        for &dependency in field.dependencies() {
            visit(dependency, active, done);
        }
        active.pop();
        done.push(field);
    }
    let mut done = Vec::new();
    for &field in Field::ALL {
        let producers = Stage::ALL
            .iter()
            .filter(|stage| stage.writes().contains(&field))
            .count();
        assert!(producers <= 1, "multiple producers for {field:?}");
        if let Some(stage) = field.producer() {
            assert!(stage.writes().contains(&field));
        }
        visit(field, &mut Vec::new(), &mut done);
    }
    assert!(Field::FinalFirelineIntensity
        .dependencies()
        .contains(&Field::FireType));
    assert!(Field::CrownFireHeatPerUnitArea
        .dependencies()
        .contains(&Field::CanopyHeatPerUnitArea));
    assert!(Field::PassiveCrownFireLineIntensity
        .dependencies()
        .contains(&Field::PassiveCrownFireSpreadRate));
}
