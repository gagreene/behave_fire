use behave_fire::fuel_models::{CustomFuelModel, FuelModel, FuelModels};
use behave_fire::species::{
    CrownDamageEquationCode, EquationType, Gacc, SpeciesMasterTable, SpeciesRecord,
};
use behave_fire::units::*;

fn close(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else if expected.is_infinite() {
        assert_eq!(actual, expected);
    } else if expected == 0.0 {
        assert_eq!(actual.to_bits(), expected.to_bits());
    } else {
        assert!(
            (actual - expected).abs() <= 1e-12_f64.max(expected.abs() * 1e-12),
            "actual={actual}, expected={expected}"
        );
    }
}

fn custom() -> CustomFuelModel {
    CustomFuelModel {
        model: FuelModel {
            code: "🔥éABC".into(),
            name: "Metric custom".into(),
            fuel_bed_depth: 1.0,
            moisture_of_extinction_dead: 24.0,
            heat_of_combustion_dead: 19000.0,
            heat_of_combustion_live: 20000.0,
            dead_1h: 1.0,
            dead_10h: 2.0,
            dead_100h: 3.0,
            live_herb: 4.0,
            live_woody: 5.0,
            savr_1h: 6000.0,
            savr_live_herb: 7000.0,
            savr_live_woody: 8000.0,
            is_dynamic: true,
            ..FuelModel::default()
        },
        length_units: LengthUnit::Meters,
        moisture_units: FractionUnit::Percent,
        heat_of_combustion_units: HeatOfCombustionUnit::KilojoulesPerKilogram,
        loading_units: LoadingUnit::TonnesPerHectare,
        savr_units: SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters,
    }
}

#[test]
fn custom_fuel_registration_clear_and_snapshot() {
    let mut fuels = FuelModels::default();
    assert!(!fuels.set_custom_fuel_model(1, custom()));
    assert!(!fuels.set_custom_fuel_model(112, custom()));
    assert!(!fuels.clear_custom_fuel_model(1));
    for number in [-2, 0, 14, 96, 260, 999] {
        assert!(fuels.set_custom_fuel_model(number, custom()));
        let model = fuels.get(number).unwrap();
        assert_eq!(model.code, "🔥éA");
        close(model.fuel_bed_depth, 3.2808398950131);
        close(model.moisture_of_extinction_dead, 0.24);
        close(model.heat_of_combustion_dead, 8162.248);
        close(model.dead_1h, 0.10197162129);
        close(model.dead_10h, 0.20394324258);
        close(model.dead_100h, 0.30591486387);
        close(model.live_herb, 0.40788648516);
        close(model.live_woody, 0.50985810645);
        close(model.heat_of_combustion_live, 8591.84);
        close(model.savr_1h, 1828.8);
        close(model.savr_live_herb, 2133.6);
        close(model.savr_live_woody, 2438.4);
        assert!(!model.is_reserved);
        assert!(model.is_defined && model.is_dynamic);
    }
    assert_eq!(fuels.for_grid(999).code, fuels.get(260).unwrap().code);
    assert!(!fuels.for_grid(-2).is_defined);
    assert!(!fuels.for_grid(0).is_defined);
    let snapshot = fuels.snapshot();
    assert!(fuels.clear_custom_fuel_model(14));
    assert!(fuels.clear_custom_fuel_model(14));
    assert!(snapshot.get(14).unwrap().is_defined);
    assert!(fuels.get(14).is_none());
    assert!(!fuels.for_grid(14).is_defined);
    fuels.get_mut(260).unwrap().is_defined = false;
    assert_eq!(fuels.for_grid(260).fuel_bed_depth, 0.0);
    assert_eq!(fuels.get_fuel_code(999), "🔥éA");
    assert_eq!(fuels.get_fuel_code(14), "");
    assert!(fuels.is_all_fuel_load_zero(14));
    assert!(!snapshot.is_all_fuel_load_zero(14));
    let base = CustomFuelModel::default();
    assert_eq!(base.length_units, LengthUnit::Feet);
    assert_eq!(base.moisture_units, FractionUnit::Fraction);
    assert_eq!(base.loading_units, LoadingUnit::PoundsPerSquareFoot);
    assert_eq!(
        base.heat_of_combustion_units,
        HeatOfCombustionUnit::BtusPerPound
    );
    assert_eq!(
        base.savr_units,
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet
    );
    assert!(fuels.set_custom_fuel_model(20, base));
    assert!(fuels.is_all_fuel_load_zero(20));
    close(
        snapshot.get_fuelbed_depth(14, Some(LengthUnit::Meters)),
        1.0,
    );
    close(
        snapshot.get_moisture_of_extinction_dead(14, Some(FractionUnit::Percent)),
        24.0,
    );
}

#[test]
fn default_fuel_records_match_every_field() {
    let fuels = FuelModels::default();
    let lines: Vec<_> = include_str!("data/fuels.tsv").lines().collect();
    assert_eq!(fuels.records().len(), lines.len());
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        let number: i32 = fields[0].parse().unwrap();
        let model = fuels.get(number).unwrap();
        assert_eq!(model.code, fields[1]);
        assert_eq!(model.name, fields[2]);
        let values = [
            model.fuel_bed_depth,
            model.moisture_of_extinction_dead,
            model.heat_of_combustion_dead,
            model.heat_of_combustion_live,
            model.dead_1h,
            model.dead_10h,
            model.dead_100h,
            model.live_herb,
            model.live_woody,
            model.savr_1h,
            model.savr_live_herb,
            model.savr_live_woody,
        ];
        for (value, expected) in values.into_iter().zip(&fields[3..15]) {
            assert_eq!(value, expected.parse::<f64>().unwrap(), "fuel {number}");
        }
        assert_eq!(model.is_dynamic, fields[15] == "true");
        assert_eq!(model.is_reserved, fields[16] == "true");
        assert_eq!(model.is_defined, fields[17] == "true");
        assert!(fuels.contains(number));
        assert_eq!(fuels.get_fuel_model(number), Some(model));
        assert_eq!(fuels.get_is_dynamic(number), model.is_dynamic);
        assert_eq!(fuels.is_fuel_model_reserved(number), model.is_reserved);
        assert_eq!(fuels.is_fuel_model_defined(number), model.is_defined);
    }
    assert_eq!(fuels.get(1).unwrap().dead_1h, 0.034);
    assert_eq!(fuels.get(110).unwrap().moisture_of_extinction_dead, 0.24);
    assert_eq!(fuels.get(158).unwrap().moisture_of_extinction_dead, 0.24);
    assert_eq!(fuels.get(1).unwrap().savr_1h, 3500.0);
    assert!(fuels.get(112).unwrap().is_reserved);
    assert!(!fuels.get(112).unwrap().is_defined);
}

#[test]
fn every_fuel_getter_supports_all_output_units_and_missing_slots() {
    let mut fuels = FuelModels::default();
    assert!(fuels.set_custom_fuel_model(14, custom()));
    for number in [-123, 1, 14, 91, 112, 999] {
        let absent = FuelModel::default();
        let record = fuels.get(number).unwrap_or(&absent);
        assert_eq!(fuels.get_fuel_name(number), record.name);
        for unit in LengthUnit::ALL {
            close(
                fuels.get_fuelbed_depth(number, Some(*unit)),
                unit.from_base(record.fuel_bed_depth),
            );
        }
        for unit in FractionUnit::ALL {
            close(
                fuels.get_moisture_of_extinction_dead(number, Some(*unit)),
                unit.from_base(record.moisture_of_extinction_dead),
            );
        }
        for unit in HeatOfCombustionUnit::ALL {
            close(
                fuels.get_heat_of_combustion_dead(number, Some(*unit)),
                unit.from_base(record.heat_of_combustion_dead),
            );
            close(
                fuels.get_heat_of_combustion_live(number, Some(*unit)),
                unit.from_base(record.heat_of_combustion_live),
            );
        }
        for unit in LoadingUnit::ALL {
            close(
                fuels.get_fuel_load_one_hour(number, Some(*unit)),
                unit.from_base(record.dead_1h),
            );
            close(
                fuels.get_fuel_load_ten_hour(number, Some(*unit)),
                unit.from_base(record.dead_10h),
            );
            close(
                fuels.get_fuel_load_hundred_hour(number, Some(*unit)),
                unit.from_base(record.dead_100h),
            );
            close(
                fuels.get_fuel_load_live_herbaceous(number, Some(*unit)),
                unit.from_base(record.live_herb),
            );
            close(
                fuels.get_fuel_load_live_woody(number, Some(*unit)),
                unit.from_base(record.live_woody),
            );
        }
        for unit in SurfaceAreaToVolumeUnit::ALL {
            close(
                fuels.get_savr_one_hour(number, Some(*unit)),
                unit.from_base(record.savr_1h),
            );
            close(
                fuels.get_savr_live_herbaceous(number, Some(*unit)),
                unit.from_base(record.savr_live_herb),
            );
            close(
                fuels.get_savr_live_woody(number, Some(*unit)),
                unit.from_base(record.savr_live_woody),
            );
        }
        assert_eq!(fuels.get_fuelbed_depth(number, None), record.fuel_bed_depth);
        assert_eq!(
            fuels.get_heat_of_combustion_dead(number, None),
            record.heat_of_combustion_dead
        );
        assert_eq!(
            fuels.get_heat_of_combustion_live(number, None),
            record.heat_of_combustion_live
        );
        assert_eq!(
            fuels.get_moisture_of_extinction_dead(number, None),
            record.moisture_of_extinction_dead
        );
        assert_eq!(fuels.get_fuel_load_one_hour(number, None), record.dead_1h);
        assert_eq!(fuels.get_fuel_load_ten_hour(number, None), record.dead_10h);
        assert_eq!(
            fuels.get_fuel_load_hundred_hour(number, None),
            record.dead_100h
        );
        assert_eq!(
            fuels.get_fuel_load_live_herbaceous(number, None),
            record.live_herb
        );
        assert_eq!(
            fuels.get_fuel_load_live_woody(number, None),
            record.live_woody
        );
        assert_eq!(fuels.get_savr_one_hour(number, None), record.savr_1h);
        assert_eq!(
            fuels.get_savr_live_herbaceous(number, None),
            record.savr_live_herb
        );
        assert_eq!(
            fuels.get_savr_live_woody(number, None),
            record.savr_live_woody
        );
    }
    assert_eq!(fuels[1], *fuels.get(1).unwrap());
    let mut input = custom();
    input.model.dead_1h = f64::NAN;
    assert!(fuels.set_custom_fuel_model(14, input));
    assert!(fuels.get(14).unwrap().dead_1h.is_nan());
    assert!(!fuels.is_all_fuel_load_zero(14));
}

#[test]
fn moisture_tables_match_every_cell() {
    use behave_fire::generated::fine_dead_fuel_moisture::{CORR, REF};
    for line in include_str!("data/moisture.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let row: usize = fields[1].parse().unwrap();
        let col: usize = fields[2].parse().unwrap();
        let actual = if fields[0] == "REF" {
            REF[row][col]
        } else {
            CORR[row][col]
        };
        assert_eq!(actual, fields[3].parse::<f64>().unwrap());
    }
    assert_eq!(REF[0][20], 0.14);
    assert_eq!(CORR[35][17], 0.06);
}

#[test]
fn species_custom_lookup_filters_and_sentinel() {
    let mut table = SpeciesMasterTable::default();
    assert_eq!(table.get_number_of_records(), 24);
    assert_eq!(
        table
            .get_species_record("pipo")
            .unwrap()
            .bark_equation_number,
        31
    );
    assert!(!table.contains_species_code("unknown"));
    assert!(table.get_species_record_at_index(-1).is_none());
    assert!(table.get_species_record_at_index(24).is_none());
    assert_eq!(
        table.get_species_table_index_from_species_code("unknown"),
        -1
    );
    let mut duplicate = table.get_species_record("PIPO").unwrap().clone();
    duplicate.equation_type = EquationType::CrownDamage;
    table.insert_record(duplicate);
    assert_eq!(
        table
            .get_species_record_by_code_and_type("pipo", EquationType::CrownDamage)
            .unwrap()
            .equation_type,
        EquationType::CrownDamage
    );
    assert!(table.remove_species_record("PIPO"));
    assert_eq!(
        table.get_species_record("PIPO").unwrap().equation_type,
        EquationType::CrownDamage
    );
    assert!(!table.remove_species_record("missing"));
    table.add_species_record(SpeciesRecord::default());
    table.add_species_record(SpeciesRecord {
        species_code: "AFTER".into(),
        ..SpeciesRecord::default()
    });
    assert!(!table.contains_species_code("AFTER"));
    assert_eq!(
        table.get_all_records().last().unwrap().species_code,
        "AFTER"
    );
    table.initialize_master_table();
    assert_eq!(table.get_number_of_records(), 24);
    for region in Gacc::ALL {
        let regional = table.get_species_records_for_gacc_region(*region);
        assert_eq!(
            regional,
            table
                .records
                .iter()
                .filter(|record| record.get_gacc_availability(*region))
                .collect::<Vec<_>>()
        );
        for eq in EquationType::ALL {
            let records = table.get_species_records_for_gacc_region_and_equation_type(*region, *eq);
            assert!(
                records
                    .iter()
                    .all(|record| record.get_gacc_availability(*region)
                        && record.equation_type == *eq)
            );
            assert_eq!(
                records.len(),
                table
                    .records
                    .iter()
                    .filter(|record| record.get_gacc_availability(*region)
                        && record.equation_type == *eq)
                    .count()
            );
        }
    }
    assert_eq!(CrownDamageEquationCode::NotSet as i32, -1);
}

#[test]
fn species_records_match_every_field() {
    let table = SpeciesMasterTable::default();
    for (index, line) in include_str!("data/species.tsv").lines().enumerate() {
        let fields: Vec<_> = line.split('\t').collect();
        let record = table.get_species_record_at_index(index as i32).unwrap();
        assert_eq!(
            [
                record.species_code.as_str(),
                record.scientific_name.as_str(),
                record.common_name.as_str()
            ],
            fields[..3]
        );
        assert_eq!(
            record.mortality_equation_number,
            fields[3].parse::<i32>().unwrap()
        );
        assert_eq!(
            record.bark_equation_number,
            fields[4].parse::<i32>().unwrap()
        );
        assert_eq!(
            record.crown_coefficient_code,
            fields[5].parse::<i32>().unwrap()
        );
        for (region, expected) in Gacc::ALL[1..].iter().zip(&fields[6..15]) {
            assert_eq!(record.get_gacc_availability(*region), *expected == "true");
        }
        assert_eq!(
            record.equation_type as i32,
            fields[15].parse::<i32>().unwrap()
        );
        assert_eq!(
            record.crown_damage_equation_code as i32,
            fields[16].parse::<i32>().unwrap()
        );
        assert_eq!(
            table.get_species_table_index_from_species_code(&record.species_code),
            index as i32
        );
        assert_eq!(
            table.get_species_table_index_from_species_code_and_equation_type(
                &record.species_code,
                record.equation_type
            ),
            index as i32
        );
    }
}

#[test]
fn species_enum_integer_codes_and_empty_defaults() {
    for (value, expected) in Gacc::ALL.iter().zip([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
        assert_eq!(value.code(), expected);
        assert_eq!(Gacc::try_from(expected), Ok(*value));
    }
    for (value, expected) in EquationType::ALL.iter().zip([-1, 0, 1, 2]) {
        assert_eq!(value.code(), expected);
        assert_eq!(EquationType::try_from(expected), Ok(*value));
    }
    assert_eq!(
        CrownDamageEquationCode::try_from(-1),
        Ok(CrownDamageEquationCode::NotSet)
    );
    assert_eq!(CrownDamageEquationCode::NotSet.code(), -1);
    assert!(CrownDamageEquationCode::try_from(0).is_err());
    assert!(Gacc::try_from(0).is_err());
    assert!(Gacc::try_from(10).is_err());
    assert!(EquationType::try_from(3).is_err());
    let empty = SpeciesRecord::default();
    assert_eq!(empty.mortality_equation_number, -1);
    assert_eq!(empty.bark_equation_number, -1);
    assert_eq!(empty.crown_coefficient_code, 0);
    assert_eq!(empty.equation_type, EquationType::NotSet);
    for region in Gacc::ALL {
        assert!(!empty.get_gacc_availability(*region));
    }
    assert_eq!(
        LengthUnit::try_from(99).unwrap_err().to_string(),
        "invalid LengthUnit unit code 99"
    );
    assert_eq!(
        Gacc::try_from(99).unwrap_err().to_string(),
        "invalid Gacc code 99"
    );
}

#[test]
fn units_have_independently_pinned_anchors() {
    assert_eq!(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(1.0),
        0.429592
    );
    assert_eq!(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(1.0),
        2.32779
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(1.0),
        0.3048
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(1.0),
        30.48
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(1.0),
        0.528
    );
    assert_eq!(TemperatureUnit::Celsius.to_base(0.0), 32.0);
    close(SlopeUnit::Percent.to_base(100.0), 45.0);
}

// exhaustive_units fixture test is appended by the one-time explicit reference exporter.

#[test]
fn exhaustive_units_and_integer_codes_match_explicit_python() {
    assert_eq!(AreaUnit::ALL.len(), 6);
    assert_eq!(AreaUnit::SquareFeet.code(), 0);
    assert_eq!(AreaUnit::try_from(0), Ok(AreaUnit::SquareFeet));
    close(AreaUnit::SquareFeet.to_base(-100.0), -100.0);
    close(AreaUnit::SquareFeet.from_base(-100.0), -100.0);
    close(AreaUnit::SquareFeet.to_base(-1.0), -1.0);
    close(AreaUnit::SquareFeet.from_base(-1.0), -1.0);
    close(AreaUnit::SquareFeet.to_base(-0.0), -0.0);
    close(AreaUnit::SquareFeet.from_base(-0.0), -0.0);
    close(AreaUnit::SquareFeet.to_base(0.0), 0.0);
    close(AreaUnit::SquareFeet.from_base(0.0), 0.0);
    close(AreaUnit::SquareFeet.to_base(1.0), 1.0);
    close(AreaUnit::SquareFeet.from_base(1.0), 1.0);
    close(AreaUnit::SquareFeet.to_base(12.345), 12.345);
    close(AreaUnit::SquareFeet.from_base(12.345), 12.345);
    close(AreaUnit::SquareFeet.to_base(100.0), 100.0);
    close(AreaUnit::SquareFeet.from_base(100.0), 100.0);
    close(AreaUnit::SquareFeet.to_base(f64::INFINITY), f64::INFINITY);
    close(AreaUnit::SquareFeet.from_base(f64::INFINITY), f64::INFINITY);
    close(
        AreaUnit::SquareFeet.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::SquareFeet.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::SquareFeet.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::SquareFeet.from_base(f64::NAN), f64::NAN);
    assert_eq!(AreaUnit::Acres.code(), 1);
    assert_eq!(AreaUnit::try_from(1), Ok(AreaUnit::Acres));
    close(AreaUnit::Acres.to_base(-100.0), -4356000.216057611);
    close(AreaUnit::Acres.from_base(-100.0), -0.002295684);
    close(AreaUnit::Acres.to_base(-1.0), -43560.00216057611);
    close(AreaUnit::Acres.from_base(-1.0), -2.295684e-05);
    close(AreaUnit::Acres.to_base(-0.0), -0.0);
    close(AreaUnit::Acres.from_base(-0.0), -0.0);
    close(AreaUnit::Acres.to_base(0.0), 0.0);
    close(AreaUnit::Acres.from_base(0.0), 0.0);
    close(AreaUnit::Acres.to_base(1.0), 43560.00216057611);
    close(AreaUnit::Acres.from_base(1.0), 2.295684e-05);
    close(AreaUnit::Acres.to_base(12.345), 537748.226672312);
    close(AreaUnit::Acres.from_base(12.345), 0.0002834021898);
    close(AreaUnit::Acres.to_base(100.0), 4356000.216057611);
    close(AreaUnit::Acres.from_base(100.0), 0.002295684);
    close(AreaUnit::Acres.to_base(f64::INFINITY), f64::INFINITY);
    close(AreaUnit::Acres.from_base(f64::INFINITY), f64::INFINITY);
    close(
        AreaUnit::Acres.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::Acres.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::Acres.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::Acres.from_base(f64::NAN), f64::NAN);
    assert_eq!(AreaUnit::Hectares.code(), 2);
    assert_eq!(AreaUnit::try_from(2), Ok(AreaUnit::Hectares));
    close(AreaUnit::Hectares.to_base(-100.0), -10763910.416709723);
    close(AreaUnit::Hectares.from_base(-100.0), -0.0009290303599999999);
    close(AreaUnit::Hectares.to_base(-1.0), -107639.10416709723);
    close(AreaUnit::Hectares.from_base(-1.0), -9.2903036e-06);
    close(AreaUnit::Hectares.to_base(-0.0), -0.0);
    close(AreaUnit::Hectares.from_base(-0.0), -0.0);
    close(AreaUnit::Hectares.to_base(0.0), 0.0);
    close(AreaUnit::Hectares.from_base(0.0), 0.0);
    close(AreaUnit::Hectares.to_base(1.0), 107639.10416709723);
    close(AreaUnit::Hectares.from_base(1.0), 9.2903036e-06);
    close(AreaUnit::Hectares.to_base(12.345), 1328804.7409428153);
    close(AreaUnit::Hectares.from_base(12.345), 0.000114688797942);
    close(AreaUnit::Hectares.to_base(100.0), 10763910.416709723);
    close(AreaUnit::Hectares.from_base(100.0), 0.0009290303599999999);
    close(AreaUnit::Hectares.to_base(f64::INFINITY), f64::INFINITY);
    close(AreaUnit::Hectares.from_base(f64::INFINITY), f64::INFINITY);
    close(
        AreaUnit::Hectares.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::Hectares.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::Hectares.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::Hectares.from_base(f64::NAN), f64::NAN);
    assert_eq!(AreaUnit::SquareMeters.code(), 3);
    assert_eq!(AreaUnit::try_from(3), Ok(AreaUnit::SquareMeters));
    close(AreaUnit::SquareMeters.to_base(-100.0), -1076.391041671);
    close(AreaUnit::SquareMeters.from_base(-100.0), -9.29030353835);
    close(AreaUnit::SquareMeters.to_base(-1.0), -10.76391041671);
    close(AreaUnit::SquareMeters.from_base(-1.0), -0.0929030353835);
    close(AreaUnit::SquareMeters.to_base(-0.0), -0.0);
    close(AreaUnit::SquareMeters.from_base(-0.0), -0.0);
    close(AreaUnit::SquareMeters.to_base(0.0), 0.0);
    close(AreaUnit::SquareMeters.from_base(0.0), 0.0);
    close(AreaUnit::SquareMeters.to_base(1.0), 10.76391041671);
    close(AreaUnit::SquareMeters.from_base(1.0), 0.0929030353835);
    close(AreaUnit::SquareMeters.to_base(12.345), 132.88047409428498);
    close(AreaUnit::SquareMeters.from_base(12.345), 1.1468879718093077);
    close(AreaUnit::SquareMeters.to_base(100.0), 1076.391041671);
    close(AreaUnit::SquareMeters.from_base(100.0), 9.29030353835);
    close(AreaUnit::SquareMeters.to_base(f64::INFINITY), f64::INFINITY);
    close(
        AreaUnit::SquareMeters.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        AreaUnit::SquareMeters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::SquareMeters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::SquareMeters.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::SquareMeters.from_base(f64::NAN), f64::NAN);
    assert_eq!(AreaUnit::SquareMiles.code(), 4);
    assert_eq!(AreaUnit::try_from(4), Ok(AreaUnit::SquareMiles));
    close(AreaUnit::SquareMiles.to_base(-100.0), -2787840000.0);
    close(AreaUnit::SquareMiles.from_base(-100.0), -3.5870064279e-06);
    close(AreaUnit::SquareMiles.to_base(-1.0), -27878400.0);
    close(AreaUnit::SquareMiles.from_base(-1.0), -3.5870064279e-08);
    close(AreaUnit::SquareMiles.to_base(-0.0), -0.0);
    close(AreaUnit::SquareMiles.from_base(-0.0), -0.0);
    close(AreaUnit::SquareMiles.to_base(0.0), 0.0);
    close(AreaUnit::SquareMiles.from_base(0.0), 0.0);
    close(AreaUnit::SquareMiles.to_base(1.0), 27878400.0);
    close(AreaUnit::SquareMiles.from_base(1.0), 3.5870064279e-08);
    close(AreaUnit::SquareMiles.to_base(12.345), 344158848.0);
    close(
        AreaUnit::SquareMiles.from_base(12.345),
        4.42815943524255e-07,
    );
    close(AreaUnit::SquareMiles.to_base(100.0), 2787840000.0);
    close(AreaUnit::SquareMiles.from_base(100.0), 3.5870064279e-06);
    close(AreaUnit::SquareMiles.to_base(f64::INFINITY), f64::INFINITY);
    close(
        AreaUnit::SquareMiles.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        AreaUnit::SquareMiles.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::SquareMiles.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::SquareMiles.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::SquareMiles.from_base(f64::NAN), f64::NAN);
    assert_eq!(AreaUnit::SquareKilometers.code(), 5);
    assert_eq!(AreaUnit::try_from(5), Ok(AreaUnit::SquareKilometers));
    close(
        AreaUnit::SquareKilometers.to_base(-100.0),
        -1076391041.670972,
    );
    close(
        AreaUnit::SquareKilometers.from_base(-100.0),
        -9.290303999999999e-06,
    );
    close(
        AreaUnit::SquareKilometers.to_base(-1.0),
        -10763910.416709721,
    );
    close(AreaUnit::SquareKilometers.from_base(-1.0), -9.290304e-08);
    close(AreaUnit::SquareKilometers.to_base(-0.0), -0.0);
    close(AreaUnit::SquareKilometers.from_base(-0.0), -0.0);
    close(AreaUnit::SquareKilometers.to_base(0.0), 0.0);
    close(AreaUnit::SquareKilometers.from_base(0.0), 0.0);
    close(AreaUnit::SquareKilometers.to_base(1.0), 10763910.416709721);
    close(AreaUnit::SquareKilometers.from_base(1.0), 9.290304e-08);
    close(
        AreaUnit::SquareKilometers.to_base(12.345),
        132880474.09428151,
    );
    close(
        AreaUnit::SquareKilometers.from_base(12.345),
        1.1468880288e-06,
    );
    close(AreaUnit::SquareKilometers.to_base(100.0), 1076391041.670972);
    close(
        AreaUnit::SquareKilometers.from_base(100.0),
        9.290303999999999e-06,
    );
    close(
        AreaUnit::SquareKilometers.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        AreaUnit::SquareKilometers.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        AreaUnit::SquareKilometers.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        AreaUnit::SquareKilometers.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(AreaUnit::SquareKilometers.to_base(f64::NAN), f64::NAN);
    close(AreaUnit::SquareKilometers.from_base(f64::NAN), f64::NAN);
    assert!(AreaUnit::try_from(-1).is_err());
    assert!(AreaUnit::try_from(6).is_err());
    assert_eq!(BasalAreaUnit::ALL.len(), 2);
    assert_eq!(BasalAreaUnit::SquareFeetPerAcre.code(), 0);
    assert_eq!(
        BasalAreaUnit::try_from(0),
        Ok(BasalAreaUnit::SquareFeetPerAcre)
    );
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(-100.0), -100.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(-100.0), -100.0);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(-1.0), -1.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(-1.0), -1.0);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(-0.0), -0.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(-0.0), -0.0);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(0.0), 0.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(0.0), 0.0);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(1.0), 1.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(1.0), 1.0);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(12.345), 12.345);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(12.345), 12.345);
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(100.0), 100.0);
    close(BasalAreaUnit::SquareFeetPerAcre.from_base(100.0), 100.0);
    close(
        BasalAreaUnit::SquareFeetPerAcre.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        BasalAreaUnit::SquareFeetPerAcre.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        BasalAreaUnit::SquareFeetPerAcre.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        BasalAreaUnit::SquareFeetPerAcre.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(BasalAreaUnit::SquareFeetPerAcre.to_base(f64::NAN), f64::NAN);
    close(
        BasalAreaUnit::SquareFeetPerAcre.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(BasalAreaUnit::SquareMetersPerHectare.code(), 1);
    assert_eq!(
        BasalAreaUnit::try_from(1),
        Ok(BasalAreaUnit::SquareMetersPerHectare)
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(-100.0),
        -435.88899999999995,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(-100.0),
        -22.94162045841946,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(-1.0),
        -4.35889,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(-1.0),
        -0.22941620458419462,
    );
    close(BasalAreaUnit::SquareMetersPerHectare.to_base(-0.0), -0.0);
    close(BasalAreaUnit::SquareMetersPerHectare.from_base(-0.0), -0.0);
    close(BasalAreaUnit::SquareMetersPerHectare.to_base(0.0), 0.0);
    close(BasalAreaUnit::SquareMetersPerHectare.from_base(0.0), 0.0);
    close(BasalAreaUnit::SquareMetersPerHectare.to_base(1.0), 4.35889);
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(1.0),
        0.22941620458419462,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(12.345),
        53.81049705,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(12.345),
        2.8321430455918826,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(100.0),
        435.88899999999995,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(100.0),
        22.94162045841946,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        BasalAreaUnit::SquareMetersPerHectare.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(BasalAreaUnit::try_from(-1).is_err());
    assert!(BasalAreaUnit::try_from(2).is_err());
    assert_eq!(LengthUnit::ALL.len(), 8);
    assert_eq!(LengthUnit::Feet.code(), 0);
    assert_eq!(LengthUnit::try_from(0), Ok(LengthUnit::Feet));
    close(LengthUnit::Feet.to_base(-100.0), -100.0);
    close(LengthUnit::Feet.from_base(-100.0), -100.0);
    close(LengthUnit::Feet.to_base(-1.0), -1.0);
    close(LengthUnit::Feet.from_base(-1.0), -1.0);
    close(LengthUnit::Feet.to_base(-0.0), -0.0);
    close(LengthUnit::Feet.from_base(-0.0), -0.0);
    close(LengthUnit::Feet.to_base(0.0), 0.0);
    close(LengthUnit::Feet.from_base(0.0), 0.0);
    close(LengthUnit::Feet.to_base(1.0), 1.0);
    close(LengthUnit::Feet.from_base(1.0), 1.0);
    close(LengthUnit::Feet.to_base(12.345), 12.345);
    close(LengthUnit::Feet.from_base(12.345), 12.345);
    close(LengthUnit::Feet.to_base(100.0), 100.0);
    close(LengthUnit::Feet.from_base(100.0), 100.0);
    close(LengthUnit::Feet.to_base(f64::INFINITY), f64::INFINITY);
    close(LengthUnit::Feet.from_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Feet.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Feet.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Feet.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Feet.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Inches.code(), 1);
    assert_eq!(LengthUnit::try_from(1), Ok(LengthUnit::Inches));
    close(LengthUnit::Inches.to_base(-100.0), -8.333333333333);
    close(LengthUnit::Inches.from_base(-100.0), -1200.0);
    close(LengthUnit::Inches.to_base(-1.0), -0.08333333333333);
    close(LengthUnit::Inches.from_base(-1.0), -12.0);
    close(LengthUnit::Inches.to_base(-0.0), -0.0);
    close(LengthUnit::Inches.from_base(-0.0), -0.0);
    close(LengthUnit::Inches.to_base(0.0), 0.0);
    close(LengthUnit::Inches.from_base(0.0), 0.0);
    close(LengthUnit::Inches.to_base(1.0), 0.08333333333333);
    close(LengthUnit::Inches.from_base(1.0), 12.0);
    close(LengthUnit::Inches.to_base(12.345), 1.028749999999959);
    close(LengthUnit::Inches.from_base(12.345), 148.14000000000001);
    close(LengthUnit::Inches.to_base(100.0), 8.333333333333);
    close(LengthUnit::Inches.from_base(100.0), 1200.0);
    close(LengthUnit::Inches.to_base(f64::INFINITY), f64::INFINITY);
    close(LengthUnit::Inches.from_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Inches.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Inches.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Inches.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Inches.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Millimeters.code(), 2);
    assert_eq!(LengthUnit::try_from(2), Ok(LengthUnit::Millimeters));
    close(LengthUnit::Millimeters.to_base(-100.0), -0.3280839895);
    close(
        LengthUnit::Millimeters.from_base(-100.0),
        -30480.00000012192,
    );
    close(LengthUnit::Millimeters.to_base(-1.0), -0.003280839895);
    close(LengthUnit::Millimeters.from_base(-1.0), -304.8000000012192);
    close(LengthUnit::Millimeters.to_base(-0.0), -0.0);
    close(LengthUnit::Millimeters.from_base(-0.0), -0.0);
    close(LengthUnit::Millimeters.to_base(0.0), 0.0);
    close(LengthUnit::Millimeters.from_base(0.0), 0.0);
    close(LengthUnit::Millimeters.to_base(1.0), 0.003280839895);
    close(LengthUnit::Millimeters.from_base(1.0), 304.8000000012192);
    close(LengthUnit::Millimeters.to_base(12.345), 0.04050196850377501);
    close(LengthUnit::Millimeters.from_base(12.345), 3762.756000015051);
    close(LengthUnit::Millimeters.to_base(100.0), 0.3280839895);
    close(LengthUnit::Millimeters.from_base(100.0), 30480.00000012192);
    close(
        LengthUnit::Millimeters.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LengthUnit::Millimeters.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LengthUnit::Millimeters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Millimeters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Millimeters.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Millimeters.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Centimeters.code(), 3);
    assert_eq!(LengthUnit::try_from(3), Ok(LengthUnit::Centimeters));
    close(LengthUnit::Centimeters.to_base(-100.0), -3.2808398949999997);
    close(LengthUnit::Centimeters.from_base(-100.0), -3048.0);
    close(LengthUnit::Centimeters.to_base(-1.0), -0.03280839895);
    close(LengthUnit::Centimeters.from_base(-1.0), -30.48);
    close(LengthUnit::Centimeters.to_base(-0.0), -0.0);
    close(LengthUnit::Centimeters.from_base(-0.0), -0.0);
    close(LengthUnit::Centimeters.to_base(0.0), 0.0);
    close(LengthUnit::Centimeters.from_base(0.0), 0.0);
    close(LengthUnit::Centimeters.to_base(1.0), 0.03280839895);
    close(LengthUnit::Centimeters.from_base(1.0), 30.48);
    close(LengthUnit::Centimeters.to_base(12.345), 0.40501968503775);
    close(LengthUnit::Centimeters.from_base(12.345), 376.2756);
    close(LengthUnit::Centimeters.to_base(100.0), 3.2808398949999997);
    close(LengthUnit::Centimeters.from_base(100.0), 3048.0);
    close(
        LengthUnit::Centimeters.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LengthUnit::Centimeters.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LengthUnit::Centimeters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Centimeters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Centimeters.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Centimeters.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Meters.code(), 4);
    assert_eq!(LengthUnit::try_from(4), Ok(LengthUnit::Meters));
    close(LengthUnit::Meters.to_base(-100.0), -328.08398950131);
    close(LengthUnit::Meters.from_base(-100.0), -30.48);
    close(LengthUnit::Meters.to_base(-1.0), -3.2808398950131);
    close(LengthUnit::Meters.from_base(-1.0), -0.3048);
    close(LengthUnit::Meters.to_base(-0.0), -0.0);
    close(LengthUnit::Meters.from_base(-0.0), -0.0);
    close(LengthUnit::Meters.to_base(0.0), 0.0);
    close(LengthUnit::Meters.from_base(0.0), 0.0);
    close(LengthUnit::Meters.to_base(1.0), 3.2808398950131);
    close(LengthUnit::Meters.from_base(1.0), 0.3048);
    close(LengthUnit::Meters.to_base(12.345), 40.50196850393672);
    close(LengthUnit::Meters.from_base(12.345), 3.7627560000000004);
    close(LengthUnit::Meters.to_base(100.0), 328.08398950131);
    close(LengthUnit::Meters.from_base(100.0), 30.48);
    close(LengthUnit::Meters.to_base(f64::INFINITY), f64::INFINITY);
    close(LengthUnit::Meters.from_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Meters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Meters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Meters.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Meters.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Chains.code(), 5);
    assert_eq!(LengthUnit::try_from(5), Ok(LengthUnit::Chains));
    close(LengthUnit::Chains.to_base(-100.0), -6600.0);
    close(LengthUnit::Chains.from_base(-100.0), -1.5151515151500001);
    close(LengthUnit::Chains.to_base(-1.0), -66.0);
    close(LengthUnit::Chains.from_base(-1.0), -0.0151515151515);
    close(LengthUnit::Chains.to_base(-0.0), -0.0);
    close(LengthUnit::Chains.from_base(-0.0), -0.0);
    close(LengthUnit::Chains.to_base(0.0), 0.0);
    close(LengthUnit::Chains.from_base(0.0), 0.0);
    close(LengthUnit::Chains.to_base(1.0), 66.0);
    close(LengthUnit::Chains.from_base(1.0), 0.0151515151515);
    close(LengthUnit::Chains.to_base(12.345), 814.7700000000001);
    close(LengthUnit::Chains.from_base(12.345), 0.18704545454526753);
    close(LengthUnit::Chains.to_base(100.0), 6600.0);
    close(LengthUnit::Chains.from_base(100.0), 1.5151515151500001);
    close(LengthUnit::Chains.to_base(f64::INFINITY), f64::INFINITY);
    close(LengthUnit::Chains.from_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Chains.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Chains.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Chains.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Chains.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Miles.code(), 6);
    assert_eq!(LengthUnit::try_from(6), Ok(LengthUnit::Miles));
    close(LengthUnit::Miles.to_base(-100.0), -528000.0);
    close(LengthUnit::Miles.from_base(-100.0), -0.01893939393939394);
    close(LengthUnit::Miles.to_base(-1.0), -5280.0);
    close(LengthUnit::Miles.from_base(-1.0), -0.0001893939393939394);
    close(LengthUnit::Miles.to_base(-0.0), -0.0);
    close(LengthUnit::Miles.from_base(-0.0), -0.0);
    close(LengthUnit::Miles.to_base(0.0), 0.0);
    close(LengthUnit::Miles.from_base(0.0), 0.0);
    close(LengthUnit::Miles.to_base(1.0), 5280.0);
    close(LengthUnit::Miles.from_base(1.0), 0.0001893939393939394);
    close(LengthUnit::Miles.to_base(12.345), 65181.600000000006);
    close(LengthUnit::Miles.from_base(12.345), 0.002338068181818182);
    close(LengthUnit::Miles.to_base(100.0), 528000.0);
    close(LengthUnit::Miles.from_base(100.0), 0.01893939393939394);
    close(LengthUnit::Miles.to_base(f64::INFINITY), f64::INFINITY);
    close(LengthUnit::Miles.from_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Miles.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Miles.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Miles.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Miles.from_base(f64::NAN), f64::NAN);
    assert_eq!(LengthUnit::Kilometers.code(), 7);
    assert_eq!(LengthUnit::try_from(7), Ok(LengthUnit::Kilometers));
    close(LengthUnit::Kilometers.to_base(-100.0), -328083.98950131);
    close(
        LengthUnit::Kilometers.from_base(-100.0),
        -0.030479999999999997,
    );
    close(LengthUnit::Kilometers.to_base(-1.0), -3280.8398950131);
    close(LengthUnit::Kilometers.from_base(-1.0), -0.0003048);
    close(LengthUnit::Kilometers.to_base(-0.0), -0.0);
    close(LengthUnit::Kilometers.from_base(-0.0), -0.0);
    close(LengthUnit::Kilometers.to_base(0.0), 0.0);
    close(LengthUnit::Kilometers.from_base(0.0), 0.0);
    close(LengthUnit::Kilometers.to_base(1.0), 3280.8398950131);
    close(LengthUnit::Kilometers.from_base(1.0), 0.0003048);
    close(LengthUnit::Kilometers.to_base(12.345), 40501.96850393672);
    close(LengthUnit::Kilometers.from_base(12.345), 0.003762756);
    close(LengthUnit::Kilometers.to_base(100.0), 328083.98950131);
    close(
        LengthUnit::Kilometers.from_base(100.0),
        0.030479999999999997,
    );
    close(LengthUnit::Kilometers.to_base(f64::INFINITY), f64::INFINITY);
    close(
        LengthUnit::Kilometers.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LengthUnit::Kilometers.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LengthUnit::Kilometers.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LengthUnit::Kilometers.to_base(f64::NAN), f64::NAN);
    close(LengthUnit::Kilometers.from_base(f64::NAN), f64::NAN);
    assert!(LengthUnit::try_from(-1).is_err());
    assert!(LengthUnit::try_from(8).is_err());
    assert_eq!(LoadingUnit::ALL.len(), 4);
    assert_eq!(LoadingUnit::PoundsPerSquareFoot.code(), 0);
    assert_eq!(
        LoadingUnit::try_from(0),
        Ok(LoadingUnit::PoundsPerSquareFoot)
    );
    close(LoadingUnit::PoundsPerSquareFoot.to_base(-100.0), -100.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(-100.0), -100.0);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(-1.0), -1.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(-1.0), -1.0);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(-0.0), -0.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(-0.0), -0.0);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(0.0), 0.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(0.0), 0.0);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(1.0), 1.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(1.0), 1.0);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(12.345), 12.345);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(12.345), 12.345);
    close(LoadingUnit::PoundsPerSquareFoot.to_base(100.0), 100.0);
    close(LoadingUnit::PoundsPerSquareFoot.from_base(100.0), 100.0);
    close(
        LoadingUnit::PoundsPerSquareFoot.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::PoundsPerSquareFoot.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::PoundsPerSquareFoot.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LoadingUnit::PoundsPerSquareFoot.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LoadingUnit::PoundsPerSquareFoot.to_base(f64::NAN), f64::NAN);
    close(
        LoadingUnit::PoundsPerSquareFoot.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(LoadingUnit::TonsPerAcre.code(), 1);
    assert_eq!(LoadingUnit::try_from(1), Ok(LoadingUnit::TonsPerAcre));
    close(LoadingUnit::TonsPerAcre.to_base(-100.0), -2.296841643);
    close(
        LoadingUnit::TonsPerAcre.from_base(-100.0),
        -4353.804725927289,
    );
    close(LoadingUnit::TonsPerAcre.to_base(-1.0), -0.02296841643);
    close(LoadingUnit::TonsPerAcre.from_base(-1.0), -43.53804725927289);
    close(LoadingUnit::TonsPerAcre.to_base(-0.0), -0.0);
    close(LoadingUnit::TonsPerAcre.from_base(-0.0), -0.0);
    close(LoadingUnit::TonsPerAcre.to_base(0.0), 0.0);
    close(LoadingUnit::TonsPerAcre.from_base(0.0), 0.0);
    close(LoadingUnit::TonsPerAcre.to_base(1.0), 0.02296841643);
    close(LoadingUnit::TonsPerAcre.from_base(1.0), 43.53804725927289);
    close(LoadingUnit::TonsPerAcre.to_base(12.345), 0.28354510082835);
    close(LoadingUnit::TonsPerAcre.from_base(12.345), 537.477193415724);
    close(LoadingUnit::TonsPerAcre.to_base(100.0), 2.296841643);
    close(LoadingUnit::TonsPerAcre.from_base(100.0), 4353.804725927289);
    close(
        LoadingUnit::TonsPerAcre.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::TonsPerAcre.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::TonsPerAcre.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LoadingUnit::TonsPerAcre.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LoadingUnit::TonsPerAcre.to_base(f64::NAN), f64::NAN);
    close(LoadingUnit::TonsPerAcre.from_base(f64::NAN), f64::NAN);
    assert_eq!(LoadingUnit::TonnesPerHectare.code(), 2);
    assert_eq!(LoadingUnit::try_from(2), Ok(LoadingUnit::TonnesPerHectare));
    close(
        LoadingUnit::TonnesPerHectare.to_base(-100.0),
        -10.197162128999999,
    );
    close(
        LoadingUnit::TonnesPerHectare.from_base(-100.0),
        -980.6650000749439,
    );
    close(LoadingUnit::TonnesPerHectare.to_base(-1.0), -0.10197162129);
    close(
        LoadingUnit::TonnesPerHectare.from_base(-1.0),
        -9.806650000749439,
    );
    close(LoadingUnit::TonnesPerHectare.to_base(-0.0), -0.0);
    close(LoadingUnit::TonnesPerHectare.from_base(-0.0), -0.0);
    close(LoadingUnit::TonnesPerHectare.to_base(0.0), 0.0);
    close(LoadingUnit::TonnesPerHectare.from_base(0.0), 0.0);
    close(LoadingUnit::TonnesPerHectare.to_base(1.0), 0.10197162129);
    close(
        LoadingUnit::TonnesPerHectare.from_base(1.0),
        9.806650000749439,
    );
    close(
        LoadingUnit::TonnesPerHectare.to_base(12.345),
        1.25883966482505,
    );
    close(
        LoadingUnit::TonnesPerHectare.from_base(12.345),
        121.06309425925183,
    );
    close(
        LoadingUnit::TonnesPerHectare.to_base(100.0),
        10.197162128999999,
    );
    close(
        LoadingUnit::TonnesPerHectare.from_base(100.0),
        980.6650000749439,
    );
    close(
        LoadingUnit::TonnesPerHectare.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::TonnesPerHectare.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::TonnesPerHectare.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LoadingUnit::TonnesPerHectare.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(LoadingUnit::TonnesPerHectare.to_base(f64::NAN), f64::NAN);
    close(LoadingUnit::TonnesPerHectare.from_base(f64::NAN), f64::NAN);
    assert_eq!(LoadingUnit::KilogramsPerSquareMeter.code(), 3);
    assert_eq!(
        LoadingUnit::try_from(3),
        Ok(LoadingUnit::KilogramsPerSquareMeter)
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(-100.0),
        -20.481754075,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(-100.0),
        -488.23943317462175,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(-1.0),
        -0.20481754075,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(-1.0),
        -4.882394331746218,
    );
    close(LoadingUnit::KilogramsPerSquareMeter.to_base(-0.0), -0.0);
    close(LoadingUnit::KilogramsPerSquareMeter.from_base(-0.0), -0.0);
    close(LoadingUnit::KilogramsPerSquareMeter.to_base(0.0), 0.0);
    close(LoadingUnit::KilogramsPerSquareMeter.from_base(0.0), 0.0);
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(1.0),
        0.20481754075,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(1.0),
        4.882394331746218,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(12.345),
        2.52847254055875,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(12.345),
        60.273158025407064,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(100.0),
        20.481754075,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(100.0),
        488.23943317462175,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        LoadingUnit::KilogramsPerSquareMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(LoadingUnit::try_from(-1).is_err());
    assert!(LoadingUnit::try_from(4).is_err());
    assert_eq!(PressureUnit::ALL.len(), 9);
    assert_eq!(PressureUnit::Pascal.code(), 0);
    assert_eq!(PressureUnit::try_from(0), Ok(PressureUnit::Pascal));
    close(PressureUnit::Pascal.to_base(-100.0), -100.0);
    close(PressureUnit::Pascal.from_base(-100.0), -100.0);
    close(PressureUnit::Pascal.to_base(-1.0), -1.0);
    close(PressureUnit::Pascal.from_base(-1.0), -1.0);
    close(PressureUnit::Pascal.to_base(-0.0), -0.0);
    close(PressureUnit::Pascal.from_base(-0.0), -0.0);
    close(PressureUnit::Pascal.to_base(0.0), 0.0);
    close(PressureUnit::Pascal.from_base(0.0), 0.0);
    close(PressureUnit::Pascal.to_base(1.0), 1.0);
    close(PressureUnit::Pascal.from_base(1.0), 1.0);
    close(PressureUnit::Pascal.to_base(12.345), 12.345);
    close(PressureUnit::Pascal.from_base(12.345), 12.345);
    close(PressureUnit::Pascal.to_base(100.0), 100.0);
    close(PressureUnit::Pascal.from_base(100.0), 100.0);
    close(PressureUnit::Pascal.to_base(f64::INFINITY), f64::INFINITY);
    close(PressureUnit::Pascal.from_base(f64::INFINITY), f64::INFINITY);
    close(
        PressureUnit::Pascal.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::Pascal.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::Pascal.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::Pascal.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::HectoPascal.code(), 1);
    assert_eq!(PressureUnit::try_from(1), Ok(PressureUnit::HectoPascal));
    close(PressureUnit::HectoPascal.to_base(-100.0), -10000.0);
    close(PressureUnit::HectoPascal.from_base(-100.0), -1.0);
    close(PressureUnit::HectoPascal.to_base(-1.0), -100.0);
    close(PressureUnit::HectoPascal.from_base(-1.0), -0.01);
    close(PressureUnit::HectoPascal.to_base(-0.0), -0.0);
    close(PressureUnit::HectoPascal.from_base(-0.0), -0.0);
    close(PressureUnit::HectoPascal.to_base(0.0), 0.0);
    close(PressureUnit::HectoPascal.from_base(0.0), 0.0);
    close(PressureUnit::HectoPascal.to_base(1.0), 100.0);
    close(PressureUnit::HectoPascal.from_base(1.0), 0.01);
    close(PressureUnit::HectoPascal.to_base(12.345), 1234.5);
    close(PressureUnit::HectoPascal.from_base(12.345), 0.12345);
    close(PressureUnit::HectoPascal.to_base(100.0), 10000.0);
    close(PressureUnit::HectoPascal.from_base(100.0), 1.0);
    close(
        PressureUnit::HectoPascal.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::HectoPascal.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::HectoPascal.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::HectoPascal.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::HectoPascal.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::HectoPascal.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::KiloPascal.code(), 2);
    assert_eq!(PressureUnit::try_from(2), Ok(PressureUnit::KiloPascal));
    close(PressureUnit::KiloPascal.to_base(-100.0), -100000.0);
    close(PressureUnit::KiloPascal.from_base(-100.0), -0.1);
    close(PressureUnit::KiloPascal.to_base(-1.0), -1000.0);
    close(PressureUnit::KiloPascal.from_base(-1.0), -0.001);
    close(PressureUnit::KiloPascal.to_base(-0.0), -0.0);
    close(PressureUnit::KiloPascal.from_base(-0.0), -0.0);
    close(PressureUnit::KiloPascal.to_base(0.0), 0.0);
    close(PressureUnit::KiloPascal.from_base(0.0), 0.0);
    close(PressureUnit::KiloPascal.to_base(1.0), 1000.0);
    close(PressureUnit::KiloPascal.from_base(1.0), 0.001);
    close(PressureUnit::KiloPascal.to_base(12.345), 12345.0);
    close(PressureUnit::KiloPascal.from_base(12.345), 0.012345);
    close(PressureUnit::KiloPascal.to_base(100.0), 100000.0);
    close(PressureUnit::KiloPascal.from_base(100.0), 0.1);
    close(
        PressureUnit::KiloPascal.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::KiloPascal.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::KiloPascal.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::KiloPascal.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::KiloPascal.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::KiloPascal.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::MegaPascal.code(), 3);
    assert_eq!(PressureUnit::try_from(3), Ok(PressureUnit::MegaPascal));
    close(PressureUnit::MegaPascal.to_base(-100.0), -100000000.0);
    close(
        PressureUnit::MegaPascal.from_base(-100.0),
        -9.999999999999999e-05,
    );
    close(PressureUnit::MegaPascal.to_base(-1.0), -1000000.0);
    close(PressureUnit::MegaPascal.from_base(-1.0), -1e-06);
    close(PressureUnit::MegaPascal.to_base(-0.0), -0.0);
    close(PressureUnit::MegaPascal.from_base(-0.0), -0.0);
    close(PressureUnit::MegaPascal.to_base(0.0), 0.0);
    close(PressureUnit::MegaPascal.from_base(0.0), 0.0);
    close(PressureUnit::MegaPascal.to_base(1.0), 1000000.0);
    close(PressureUnit::MegaPascal.from_base(1.0), 1e-06);
    close(PressureUnit::MegaPascal.to_base(12.345), 12345000.0);
    close(PressureUnit::MegaPascal.from_base(12.345), 1.2345e-05);
    close(PressureUnit::MegaPascal.to_base(100.0), 100000000.0);
    close(
        PressureUnit::MegaPascal.from_base(100.0),
        9.999999999999999e-05,
    );
    close(
        PressureUnit::MegaPascal.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::MegaPascal.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::MegaPascal.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::MegaPascal.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::MegaPascal.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::MegaPascal.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::GigaPascal.code(), 4);
    assert_eq!(PressureUnit::try_from(4), Ok(PressureUnit::GigaPascal));
    close(PressureUnit::GigaPascal.to_base(-100.0), -100000000000.0);
    close(
        PressureUnit::GigaPascal.from_base(-100.0),
        -1.0000000000000001e-07,
    );
    close(PressureUnit::GigaPascal.to_base(-1.0), -1000000000.0);
    close(PressureUnit::GigaPascal.from_base(-1.0), -1e-09);
    close(PressureUnit::GigaPascal.to_base(-0.0), -0.0);
    close(PressureUnit::GigaPascal.from_base(-0.0), -0.0);
    close(PressureUnit::GigaPascal.to_base(0.0), 0.0);
    close(PressureUnit::GigaPascal.from_base(0.0), 0.0);
    close(PressureUnit::GigaPascal.to_base(1.0), 1000000000.0);
    close(PressureUnit::GigaPascal.from_base(1.0), 1e-09);
    close(PressureUnit::GigaPascal.to_base(12.345), 12345000000.0);
    close(PressureUnit::GigaPascal.from_base(12.345), 1.2345e-08);
    close(PressureUnit::GigaPascal.to_base(100.0), 100000000000.0);
    close(
        PressureUnit::GigaPascal.from_base(100.0),
        1.0000000000000001e-07,
    );
    close(
        PressureUnit::GigaPascal.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::GigaPascal.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::GigaPascal.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::GigaPascal.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::GigaPascal.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::GigaPascal.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::Bar.code(), 5);
    assert_eq!(PressureUnit::try_from(5), Ok(PressureUnit::Bar));
    close(PressureUnit::Bar.to_base(-100.0), -10000000.0);
    close(PressureUnit::Bar.from_base(-100.0), -0.001);
    close(PressureUnit::Bar.to_base(-1.0), -100000.0);
    close(PressureUnit::Bar.from_base(-1.0), -1e-05);
    close(PressureUnit::Bar.to_base(-0.0), -0.0);
    close(PressureUnit::Bar.from_base(-0.0), -0.0);
    close(PressureUnit::Bar.to_base(0.0), 0.0);
    close(PressureUnit::Bar.from_base(0.0), 0.0);
    close(PressureUnit::Bar.to_base(1.0), 100000.0);
    close(PressureUnit::Bar.from_base(1.0), 1e-05);
    close(PressureUnit::Bar.to_base(12.345), 1234500.0);
    close(PressureUnit::Bar.from_base(12.345), 0.00012345000000000002);
    close(PressureUnit::Bar.to_base(100.0), 10000000.0);
    close(PressureUnit::Bar.from_base(100.0), 0.001);
    close(PressureUnit::Bar.to_base(f64::INFINITY), f64::INFINITY);
    close(PressureUnit::Bar.from_base(f64::INFINITY), f64::INFINITY);
    close(
        PressureUnit::Bar.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::Bar.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::Bar.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::Bar.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::Atmosphere.code(), 6);
    assert_eq!(PressureUnit::try_from(6), Ok(PressureUnit::Atmosphere));
    close(PressureUnit::Atmosphere.to_base(-100.0), -10132500.0);
    close(
        PressureUnit::Atmosphere.from_base(-100.0),
        -0.0009869232667160128,
    );
    close(PressureUnit::Atmosphere.to_base(-1.0), -101325.0);
    close(
        PressureUnit::Atmosphere.from_base(-1.0),
        -9.869232667160129e-06,
    );
    close(PressureUnit::Atmosphere.to_base(-0.0), -0.0);
    close(PressureUnit::Atmosphere.from_base(-0.0), -0.0);
    close(PressureUnit::Atmosphere.to_base(0.0), 0.0);
    close(PressureUnit::Atmosphere.from_base(0.0), 0.0);
    close(PressureUnit::Atmosphere.to_base(1.0), 101325.0);
    close(
        PressureUnit::Atmosphere.from_base(1.0),
        9.869232667160129e-06,
    );
    close(PressureUnit::Atmosphere.to_base(12.345), 1250857.125);
    close(
        PressureUnit::Atmosphere.from_base(12.345),
        0.00012183567727609179,
    );
    close(PressureUnit::Atmosphere.to_base(100.0), 10132500.0);
    close(
        PressureUnit::Atmosphere.from_base(100.0),
        0.0009869232667160128,
    );
    close(
        PressureUnit::Atmosphere.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::Atmosphere.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::Atmosphere.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::Atmosphere.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::Atmosphere.to_base(f64::NAN), f64::NAN);
    close(PressureUnit::Atmosphere.from_base(f64::NAN), f64::NAN);
    assert_eq!(PressureUnit::TechnicalAtmosphere.code(), 7);
    assert_eq!(
        PressureUnit::try_from(7),
        Ok(PressureUnit::TechnicalAtmosphere)
    );
    close(
        PressureUnit::TechnicalAtmosphere.to_base(-100.0),
        -9806650.0,
    );
    close(
        PressureUnit::TechnicalAtmosphere.from_base(-100.0),
        -0.0010197162129779282,
    );
    close(PressureUnit::TechnicalAtmosphere.to_base(-1.0), -98066.5);
    close(
        PressureUnit::TechnicalAtmosphere.from_base(-1.0),
        -1.0197162129779282e-05,
    );
    close(PressureUnit::TechnicalAtmosphere.to_base(-0.0), -0.0);
    close(PressureUnit::TechnicalAtmosphere.from_base(-0.0), -0.0);
    close(PressureUnit::TechnicalAtmosphere.to_base(0.0), 0.0);
    close(PressureUnit::TechnicalAtmosphere.from_base(0.0), 0.0);
    close(PressureUnit::TechnicalAtmosphere.to_base(1.0), 98066.5);
    close(
        PressureUnit::TechnicalAtmosphere.from_base(1.0),
        1.0197162129779282e-05,
    );
    close(
        PressureUnit::TechnicalAtmosphere.to_base(12.345),
        1210630.9425000001,
    );
    close(
        PressureUnit::TechnicalAtmosphere.from_base(12.345),
        0.00012588396649212524,
    );
    close(PressureUnit::TechnicalAtmosphere.to_base(100.0), 9806650.0);
    close(
        PressureUnit::TechnicalAtmosphere.from_base(100.0),
        0.0010197162129779282,
    );
    close(
        PressureUnit::TechnicalAtmosphere.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::TechnicalAtmosphere.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::TechnicalAtmosphere.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::TechnicalAtmosphere.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::TechnicalAtmosphere.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        PressureUnit::TechnicalAtmosphere.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(PressureUnit::PoundPerSquareInch.code(), 8);
    assert_eq!(
        PressureUnit::try_from(8),
        Ok(PressureUnit::PoundPerSquareInch)
    );
    close(PressureUnit::PoundPerSquareInch.to_base(-100.0), -689475.7);
    close(
        PressureUnit::PoundPerSquareInch.from_base(-100.0),
        -0.014503774389728311,
    );
    close(PressureUnit::PoundPerSquareInch.to_base(-1.0), -6894.757);
    close(
        PressureUnit::PoundPerSquareInch.from_base(-1.0),
        -0.00014503774389728312,
    );
    close(PressureUnit::PoundPerSquareInch.to_base(-0.0), -0.0);
    close(PressureUnit::PoundPerSquareInch.from_base(-0.0), -0.0);
    close(PressureUnit::PoundPerSquareInch.to_base(0.0), 0.0);
    close(PressureUnit::PoundPerSquareInch.from_base(0.0), 0.0);
    close(PressureUnit::PoundPerSquareInch.to_base(1.0), 6894.757);
    close(
        PressureUnit::PoundPerSquareInch.from_base(1.0),
        0.00014503774389728312,
    );
    close(
        PressureUnit::PoundPerSquareInch.to_base(12.345),
        85115.775165,
    );
    close(
        PressureUnit::PoundPerSquareInch.from_base(12.345),
        0.0017904909484119601,
    );
    close(PressureUnit::PoundPerSquareInch.to_base(100.0), 689475.7);
    close(
        PressureUnit::PoundPerSquareInch.from_base(100.0),
        0.014503774389728311,
    );
    close(
        PressureUnit::PoundPerSquareInch.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::PoundPerSquareInch.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        PressureUnit::PoundPerSquareInch.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        PressureUnit::PoundPerSquareInch.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(PressureUnit::PoundPerSquareInch.to_base(f64::NAN), f64::NAN);
    close(
        PressureUnit::PoundPerSquareInch.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(PressureUnit::try_from(-1).is_err());
    assert!(PressureUnit::try_from(9).is_err());
    assert_eq!(SurfaceAreaToVolumeUnit::ALL.len(), 4);
    assert_eq!(SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.code(), 0);
    assert_eq!(
        SurfaceAreaToVolumeUnit::try_from(0),
        Ok(SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet)
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(-100.0),
        -100.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(-100.0),
        -100.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(-1.0),
        -1.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(-1.0),
        -1.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(1.0),
        1.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(1.0),
        1.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(12.345),
        12.345,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(12.345),
        12.345,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(100.0),
        100.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(100.0),
        100.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.code(),
        1
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::try_from(1),
        Ok(SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters)
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(-100.0),
        -30.48,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(-100.0),
        -328.0839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(-1.0),
        -0.3048,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(-1.0),
        -3.280839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(1.0),
        0.3048,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(1.0),
        3.280839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(12.345),
        3.7627560000000004,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(12.345),
        40.50196850393701,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(100.0),
        30.48,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(100.0),
        328.0839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareMetersOverCubicMeters.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.code(),
        2
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::try_from(2),
        Ok(SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches)
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(-100.0),
        -1200.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(-100.0),
        -8.333333333333332,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(-1.0),
        -12.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(-1.0),
        -0.08333333333333333,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(1.0),
        12.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(1.0),
        0.08333333333333333,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(12.345),
        148.14000000000001,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(12.345),
        1.02875,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(100.0),
        1200.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(100.0),
        8.333333333333332,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareInchesOverCubicInches.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.code(),
        3
    );
    assert_eq!(
        SurfaceAreaToVolumeUnit::try_from(3),
        Ok(SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters)
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(-100.0),
        -3048.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(-100.0),
        -3.2808398950131235,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(-1.0),
        -30.48,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(-1.0),
        -0.03280839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(-0.0),
        -0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(0.0),
        0.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(1.0),
        30.48,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(1.0),
        0.03280839895013123,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(12.345),
        376.2756,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(12.345),
        0.4050196850393701,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(100.0),
        3048.0,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(100.0),
        3.2808398950131235,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        SurfaceAreaToVolumeUnit::SquareCentimetersOverCubicCentimeters.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(SurfaceAreaToVolumeUnit::try_from(-1).is_err());
    assert!(SurfaceAreaToVolumeUnit::try_from(4).is_err());
    assert_eq!(SpeedUnit::ALL.len(), 7);
    assert_eq!(SpeedUnit::FeetPerMinute.code(), 0);
    assert_eq!(SpeedUnit::try_from(0), Ok(SpeedUnit::FeetPerMinute));
    close(SpeedUnit::FeetPerMinute.to_base(-100.0), -100.0);
    close(SpeedUnit::FeetPerMinute.from_base(-100.0), -100.0);
    close(SpeedUnit::FeetPerMinute.to_base(-1.0), -1.0);
    close(SpeedUnit::FeetPerMinute.from_base(-1.0), -1.0);
    close(SpeedUnit::FeetPerMinute.to_base(-0.0), -0.0);
    close(SpeedUnit::FeetPerMinute.from_base(-0.0), -0.0);
    close(SpeedUnit::FeetPerMinute.to_base(0.0), 0.0);
    close(SpeedUnit::FeetPerMinute.from_base(0.0), 0.0);
    close(SpeedUnit::FeetPerMinute.to_base(1.0), 1.0);
    close(SpeedUnit::FeetPerMinute.from_base(1.0), 1.0);
    close(SpeedUnit::FeetPerMinute.to_base(12.345), 12.345);
    close(SpeedUnit::FeetPerMinute.from_base(12.345), 12.345);
    close(SpeedUnit::FeetPerMinute.to_base(100.0), 100.0);
    close(SpeedUnit::FeetPerMinute.from_base(100.0), 100.0);
    close(
        SpeedUnit::FeetPerMinute.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::FeetPerMinute.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::FeetPerMinute.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::FeetPerMinute.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::FeetPerMinute.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::FeetPerMinute.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::ChainsPerHour.code(), 1);
    assert_eq!(SpeedUnit::try_from(1), Ok(SpeedUnit::ChainsPerHour));
    close(
        SpeedUnit::ChainsPerHour.to_base(-100.0),
        -110.00000000000001,
    );
    close(
        SpeedUnit::ChainsPerHour.from_base(-100.0),
        -90.9090909090909,
    );
    close(SpeedUnit::ChainsPerHour.to_base(-1.0), -1.1);
    close(
        SpeedUnit::ChainsPerHour.from_base(-1.0),
        -0.9090909090909091,
    );
    close(SpeedUnit::ChainsPerHour.to_base(-0.0), -0.0);
    close(SpeedUnit::ChainsPerHour.from_base(-0.0), -0.0);
    close(SpeedUnit::ChainsPerHour.to_base(0.0), 0.0);
    close(SpeedUnit::ChainsPerHour.from_base(0.0), 0.0);
    close(SpeedUnit::ChainsPerHour.to_base(1.0), 1.1);
    close(SpeedUnit::ChainsPerHour.from_base(1.0), 0.9090909090909091);
    close(SpeedUnit::ChainsPerHour.to_base(12.345), 13.579500000000001);
    close(
        SpeedUnit::ChainsPerHour.from_base(12.345),
        11.222727272727273,
    );
    close(SpeedUnit::ChainsPerHour.to_base(100.0), 110.00000000000001);
    close(SpeedUnit::ChainsPerHour.from_base(100.0), 90.9090909090909);
    close(
        SpeedUnit::ChainsPerHour.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::ChainsPerHour.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::ChainsPerHour.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::ChainsPerHour.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::ChainsPerHour.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::ChainsPerHour.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::MetersPerSecond.code(), 2);
    assert_eq!(SpeedUnit::try_from(2), Ok(SpeedUnit::MetersPerSecond));
    close(
        SpeedUnit::MetersPerSecond.to_base(-100.0),
        -19685.039370000002,
    );
    close(SpeedUnit::MetersPerSecond.from_base(-100.0), -0.508);
    close(SpeedUnit::MetersPerSecond.to_base(-1.0), -196.8503937);
    close(SpeedUnit::MetersPerSecond.from_base(-1.0), -0.00508);
    close(SpeedUnit::MetersPerSecond.to_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerSecond.from_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerSecond.to_base(0.0), 0.0);
    close(SpeedUnit::MetersPerSecond.from_base(0.0), 0.0);
    close(SpeedUnit::MetersPerSecond.to_base(1.0), 196.8503937);
    close(SpeedUnit::MetersPerSecond.from_base(1.0), 0.00508);
    close(
        SpeedUnit::MetersPerSecond.to_base(12.345),
        2430.1181102265004,
    );
    close(
        SpeedUnit::MetersPerSecond.from_base(12.345),
        0.06271260000000001,
    );
    close(
        SpeedUnit::MetersPerSecond.to_base(100.0),
        19685.039370000002,
    );
    close(SpeedUnit::MetersPerSecond.from_base(100.0), 0.508);
    close(
        SpeedUnit::MetersPerSecond.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerSecond.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerSecond.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::MetersPerSecond.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::MetersPerSecond.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::MetersPerSecond.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::MetersPerMinute.code(), 3);
    assert_eq!(SpeedUnit::try_from(3), Ok(SpeedUnit::MetersPerMinute));
    close(SpeedUnit::MetersPerMinute.to_base(-100.0), -328.084);
    close(SpeedUnit::MetersPerMinute.from_base(-100.0), -30.48);
    close(SpeedUnit::MetersPerMinute.to_base(-1.0), -3.28084);
    close(SpeedUnit::MetersPerMinute.from_base(-1.0), -0.3048);
    close(SpeedUnit::MetersPerMinute.to_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerMinute.from_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerMinute.to_base(0.0), 0.0);
    close(SpeedUnit::MetersPerMinute.from_base(0.0), 0.0);
    close(SpeedUnit::MetersPerMinute.to_base(1.0), 3.28084);
    close(SpeedUnit::MetersPerMinute.from_base(1.0), 0.3048);
    close(
        SpeedUnit::MetersPerMinute.to_base(12.345),
        40.501969800000005,
    );
    close(
        SpeedUnit::MetersPerMinute.from_base(12.345),
        3.7627560000000004,
    );
    close(SpeedUnit::MetersPerMinute.to_base(100.0), 328.084);
    close(SpeedUnit::MetersPerMinute.from_base(100.0), 30.48);
    close(
        SpeedUnit::MetersPerMinute.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerMinute.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerMinute.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::MetersPerMinute.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::MetersPerMinute.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::MetersPerMinute.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::MetersPerHour.code(), 4);
    assert_eq!(SpeedUnit::try_from(4), Ok(SpeedUnit::MetersPerHour));
    close(SpeedUnit::MetersPerHour.to_base(-100.0), -5.47);
    close(SpeedUnit::MetersPerHour.from_base(-100.0), -1828.8);
    close(SpeedUnit::MetersPerHour.to_base(-1.0), -0.0547);
    close(SpeedUnit::MetersPerHour.from_base(-1.0), -18.288);
    close(SpeedUnit::MetersPerHour.to_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerHour.from_base(-0.0), -0.0);
    close(SpeedUnit::MetersPerHour.to_base(0.0), 0.0);
    close(SpeedUnit::MetersPerHour.from_base(0.0), 0.0);
    close(SpeedUnit::MetersPerHour.to_base(1.0), 0.0547);
    close(SpeedUnit::MetersPerHour.from_base(1.0), 18.288);
    close(SpeedUnit::MetersPerHour.to_base(12.345), 0.6752715);
    close(
        SpeedUnit::MetersPerHour.from_base(12.345),
        225.76536000000002,
    );
    close(SpeedUnit::MetersPerHour.to_base(100.0), 5.47);
    close(SpeedUnit::MetersPerHour.from_base(100.0), 1828.8);
    close(
        SpeedUnit::MetersPerHour.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerHour.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MetersPerHour.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::MetersPerHour.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::MetersPerHour.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::MetersPerHour.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::MilesPerHour.code(), 5);
    assert_eq!(SpeedUnit::try_from(5), Ok(SpeedUnit::MilesPerHour));
    close(SpeedUnit::MilesPerHour.to_base(-100.0), -8800.0);
    close(SpeedUnit::MilesPerHour.from_base(-100.0), -1.136363636);
    close(SpeedUnit::MilesPerHour.to_base(-1.0), -88.0);
    close(SpeedUnit::MilesPerHour.from_base(-1.0), -0.01136363636);
    close(SpeedUnit::MilesPerHour.to_base(-0.0), -0.0);
    close(SpeedUnit::MilesPerHour.from_base(-0.0), -0.0);
    close(SpeedUnit::MilesPerHour.to_base(0.0), 0.0);
    close(SpeedUnit::MilesPerHour.from_base(0.0), 0.0);
    close(SpeedUnit::MilesPerHour.to_base(1.0), 88.0);
    close(SpeedUnit::MilesPerHour.from_base(1.0), 0.01136363636);
    close(SpeedUnit::MilesPerHour.to_base(12.345), 1086.3600000000001);
    close(
        SpeedUnit::MilesPerHour.from_base(12.345),
        0.14028409086420002,
    );
    close(SpeedUnit::MilesPerHour.to_base(100.0), 8800.0);
    close(SpeedUnit::MilesPerHour.from_base(100.0), 1.136363636);
    close(
        SpeedUnit::MilesPerHour.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MilesPerHour.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::MilesPerHour.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::MilesPerHour.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::MilesPerHour.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::MilesPerHour.from_base(f64::NAN), f64::NAN);
    assert_eq!(SpeedUnit::KilometersPerHour.code(), 6);
    assert_eq!(SpeedUnit::try_from(6), Ok(SpeedUnit::KilometersPerHour));
    close(SpeedUnit::KilometersPerHour.to_base(-100.0), -5468.0665);
    close(
        SpeedUnit::KilometersPerHour.from_base(-100.0),
        -1.8287999999999998,
    );
    close(SpeedUnit::KilometersPerHour.to_base(-1.0), -54.680665);
    close(SpeedUnit::KilometersPerHour.from_base(-1.0), -0.018288);
    close(SpeedUnit::KilometersPerHour.to_base(-0.0), -0.0);
    close(SpeedUnit::KilometersPerHour.from_base(-0.0), -0.0);
    close(SpeedUnit::KilometersPerHour.to_base(0.0), 0.0);
    close(SpeedUnit::KilometersPerHour.from_base(0.0), 0.0);
    close(SpeedUnit::KilometersPerHour.to_base(1.0), 54.680665);
    close(SpeedUnit::KilometersPerHour.from_base(1.0), 0.018288);
    close(SpeedUnit::KilometersPerHour.to_base(12.345), 675.032809425);
    close(SpeedUnit::KilometersPerHour.from_base(12.345), 0.22576536);
    close(SpeedUnit::KilometersPerHour.to_base(100.0), 5468.0665);
    close(
        SpeedUnit::KilometersPerHour.from_base(100.0),
        1.8287999999999998,
    );
    close(
        SpeedUnit::KilometersPerHour.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::KilometersPerHour.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        SpeedUnit::KilometersPerHour.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SpeedUnit::KilometersPerHour.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SpeedUnit::KilometersPerHour.to_base(f64::NAN), f64::NAN);
    close(SpeedUnit::KilometersPerHour.from_base(f64::NAN), f64::NAN);
    assert!(SpeedUnit::try_from(-1).is_err());
    assert!(SpeedUnit::try_from(7).is_err());
    assert_eq!(FractionUnit::ALL.len(), 2);
    assert_eq!(FractionUnit::Fraction.code(), 0);
    assert_eq!(FractionUnit::try_from(0), Ok(FractionUnit::Fraction));
    close(FractionUnit::Fraction.to_base(-100.0), -100.0);
    close(FractionUnit::Fraction.from_base(-100.0), -100.0);
    close(FractionUnit::Fraction.to_base(-1.0), -1.0);
    close(FractionUnit::Fraction.from_base(-1.0), -1.0);
    close(FractionUnit::Fraction.to_base(-0.0), -0.0);
    close(FractionUnit::Fraction.from_base(-0.0), -0.0);
    close(FractionUnit::Fraction.to_base(0.0), 0.0);
    close(FractionUnit::Fraction.from_base(0.0), 0.0);
    close(FractionUnit::Fraction.to_base(1.0), 1.0);
    close(FractionUnit::Fraction.from_base(1.0), 1.0);
    close(FractionUnit::Fraction.to_base(12.345), 12.345);
    close(FractionUnit::Fraction.from_base(12.345), 12.345);
    close(FractionUnit::Fraction.to_base(100.0), 100.0);
    close(FractionUnit::Fraction.from_base(100.0), 100.0);
    close(FractionUnit::Fraction.to_base(f64::INFINITY), f64::INFINITY);
    close(
        FractionUnit::Fraction.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FractionUnit::Fraction.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FractionUnit::Fraction.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(FractionUnit::Fraction.to_base(f64::NAN), f64::NAN);
    close(FractionUnit::Fraction.from_base(f64::NAN), f64::NAN);
    assert_eq!(FractionUnit::Percent.code(), 1);
    assert_eq!(FractionUnit::try_from(1), Ok(FractionUnit::Percent));
    close(FractionUnit::Percent.to_base(-100.0), -1.0);
    close(FractionUnit::Percent.from_base(-100.0), -10000.0);
    close(FractionUnit::Percent.to_base(-1.0), -0.01);
    close(FractionUnit::Percent.from_base(-1.0), -100.0);
    close(FractionUnit::Percent.to_base(-0.0), -0.0);
    close(FractionUnit::Percent.from_base(-0.0), -0.0);
    close(FractionUnit::Percent.to_base(0.0), 0.0);
    close(FractionUnit::Percent.from_base(0.0), 0.0);
    close(FractionUnit::Percent.to_base(1.0), 0.01);
    close(FractionUnit::Percent.from_base(1.0), 100.0);
    close(FractionUnit::Percent.to_base(12.345), 0.12345);
    close(FractionUnit::Percent.from_base(12.345), 1234.5);
    close(FractionUnit::Percent.to_base(100.0), 1.0);
    close(FractionUnit::Percent.from_base(100.0), 10000.0);
    close(FractionUnit::Percent.to_base(f64::INFINITY), f64::INFINITY);
    close(
        FractionUnit::Percent.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FractionUnit::Percent.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FractionUnit::Percent.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(FractionUnit::Percent.to_base(f64::NAN), f64::NAN);
    close(FractionUnit::Percent.from_base(f64::NAN), f64::NAN);
    assert!(FractionUnit::try_from(-1).is_err());
    assert!(FractionUnit::try_from(2).is_err());
    assert_eq!(SlopeUnit::ALL.len(), 2);
    assert_eq!(SlopeUnit::Degrees.code(), 0);
    assert_eq!(SlopeUnit::try_from(0), Ok(SlopeUnit::Degrees));
    close(SlopeUnit::Degrees.to_base(-100.0), -100.0);
    close(SlopeUnit::Degrees.from_base(-100.0), -100.0);
    close(SlopeUnit::Degrees.to_base(-1.0), -1.0);
    close(SlopeUnit::Degrees.from_base(-1.0), -1.0);
    close(SlopeUnit::Degrees.to_base(-0.0), -0.0);
    close(SlopeUnit::Degrees.from_base(-0.0), -0.0);
    close(SlopeUnit::Degrees.to_base(0.0), 0.0);
    close(SlopeUnit::Degrees.from_base(0.0), 0.0);
    close(SlopeUnit::Degrees.to_base(1.0), 1.0);
    close(SlopeUnit::Degrees.from_base(1.0), 1.0);
    close(SlopeUnit::Degrees.to_base(12.345), 12.345);
    close(SlopeUnit::Degrees.from_base(12.345), 12.345);
    close(SlopeUnit::Degrees.to_base(100.0), 100.0);
    close(SlopeUnit::Degrees.from_base(100.0), 100.0);
    close(SlopeUnit::Degrees.to_base(f64::INFINITY), f64::INFINITY);
    close(SlopeUnit::Degrees.from_base(f64::INFINITY), f64::INFINITY);
    close(
        SlopeUnit::Degrees.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        SlopeUnit::Degrees.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(SlopeUnit::Degrees.to_base(f64::NAN), f64::NAN);
    close(SlopeUnit::Degrees.from_base(f64::NAN), f64::NAN);
    assert_eq!(SlopeUnit::Percent.code(), 1);
    assert_eq!(SlopeUnit::try_from(1), Ok(SlopeUnit::Percent));
    close(SlopeUnit::Percent.to_base(-100.0), -45.0);
    close(SlopeUnit::Percent.from_base(-100.0), 567.1281819617711);
    close(SlopeUnit::Percent.to_base(-1.0), -0.5729386976834859);
    close(SlopeUnit::Percent.from_base(-1.0), -1.7455064928217585);
    close(SlopeUnit::Percent.to_base(-0.0), -0.0);
    close(SlopeUnit::Percent.from_base(-0.0), -0.0);
    close(SlopeUnit::Percent.to_base(0.0), 0.0);
    close(SlopeUnit::Percent.from_base(0.0), 0.0);
    close(SlopeUnit::Percent.to_base(1.0), 0.5729386976834859);
    close(SlopeUnit::Percent.from_base(1.0), 1.7455064928217585);
    close(SlopeUnit::Percent.to_base(12.345), 7.037557558097611);
    close(SlopeUnit::Percent.from_base(12.345), 21.88581370143567);
    close(SlopeUnit::Percent.to_base(100.0), 45.0);
    close(SlopeUnit::Percent.from_base(100.0), -567.1281819617711);
    close(SlopeUnit::Percent.to_base(f64::INFINITY), 90.0);
    close(SlopeUnit::Percent.from_base(f64::INFINITY), f64::NAN);
    close(SlopeUnit::Percent.to_base(f64::NEG_INFINITY), -90.0);
    close(SlopeUnit::Percent.from_base(f64::NEG_INFINITY), f64::NAN);
    close(SlopeUnit::Percent.to_base(f64::NAN), f64::NAN);
    close(SlopeUnit::Percent.from_base(f64::NAN), f64::NAN);
    assert!(SlopeUnit::try_from(-1).is_err());
    assert!(SlopeUnit::try_from(2).is_err());
    assert_eq!(DensityUnit::ALL.len(), 2);
    assert_eq!(DensityUnit::PoundsPerCubicFoot.code(), 0);
    assert_eq!(
        DensityUnit::try_from(0),
        Ok(DensityUnit::PoundsPerCubicFoot)
    );
    close(DensityUnit::PoundsPerCubicFoot.to_base(-100.0), -100.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(-100.0), -100.0);
    close(DensityUnit::PoundsPerCubicFoot.to_base(-1.0), -1.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(-1.0), -1.0);
    close(DensityUnit::PoundsPerCubicFoot.to_base(-0.0), -0.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(-0.0), -0.0);
    close(DensityUnit::PoundsPerCubicFoot.to_base(0.0), 0.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(0.0), 0.0);
    close(DensityUnit::PoundsPerCubicFoot.to_base(1.0), 1.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(1.0), 1.0);
    close(DensityUnit::PoundsPerCubicFoot.to_base(12.345), 12.345);
    close(DensityUnit::PoundsPerCubicFoot.from_base(12.345), 12.345);
    close(DensityUnit::PoundsPerCubicFoot.to_base(100.0), 100.0);
    close(DensityUnit::PoundsPerCubicFoot.from_base(100.0), 100.0);
    close(
        DensityUnit::PoundsPerCubicFoot.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        DensityUnit::PoundsPerCubicFoot.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        DensityUnit::PoundsPerCubicFoot.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        DensityUnit::PoundsPerCubicFoot.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(DensityUnit::PoundsPerCubicFoot.to_base(f64::NAN), f64::NAN);
    close(
        DensityUnit::PoundsPerCubicFoot.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(DensityUnit::KilogramsPerCubicMeter.code(), 1);
    assert_eq!(
        DensityUnit::try_from(1),
        Ok(DensityUnit::KilogramsPerCubicMeter)
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(-100.0),
        -6.2427961,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(-100.0),
        -1601.846326520259,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(-1.0),
        -0.062427961,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(-1.0),
        -16.01846326520259,
    );
    close(DensityUnit::KilogramsPerCubicMeter.to_base(-0.0), -0.0);
    close(DensityUnit::KilogramsPerCubicMeter.from_base(-0.0), -0.0);
    close(DensityUnit::KilogramsPerCubicMeter.to_base(0.0), 0.0);
    close(DensityUnit::KilogramsPerCubicMeter.from_base(0.0), 0.0);
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(1.0),
        0.062427961,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(1.0),
        16.01846326520259,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(12.345),
        0.7706731785450001,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(12.345),
        197.747929008926,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(100.0),
        6.2427961,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(100.0),
        1601.846326520259,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        DensityUnit::KilogramsPerCubicMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(DensityUnit::try_from(-1).is_err());
    assert!(DensityUnit::try_from(2).is_err());
    assert_eq!(HeatOfCombustionUnit::ALL.len(), 2);
    assert_eq!(HeatOfCombustionUnit::BtusPerPound.code(), 0);
    assert_eq!(
        HeatOfCombustionUnit::try_from(0),
        Ok(HeatOfCombustionUnit::BtusPerPound)
    );
    close(HeatOfCombustionUnit::BtusPerPound.to_base(-100.0), -100.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(-100.0), -100.0);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(-1.0), -1.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(-1.0), -1.0);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(-0.0), -0.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(-0.0), -0.0);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(0.0), 0.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(0.0), 0.0);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(1.0), 1.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(1.0), 1.0);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(12.345), 12.345);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(12.345), 12.345);
    close(HeatOfCombustionUnit::BtusPerPound.to_base(100.0), 100.0);
    close(HeatOfCombustionUnit::BtusPerPound.from_base(100.0), 100.0);
    close(
        HeatOfCombustionUnit::BtusPerPound.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatOfCombustionUnit::BtusPerPound.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatOfCombustionUnit::BtusPerPound.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatOfCombustionUnit::BtusPerPound.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatOfCombustionUnit::BtusPerPound.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatOfCombustionUnit::BtusPerPound.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(HeatOfCombustionUnit::KilojoulesPerKilogram.code(), 1);
    assert_eq!(
        HeatOfCombustionUnit::try_from(1),
        Ok(HeatOfCombustionUnit::KilojoulesPerKilogram)
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(-100.0),
        -42.959199999999996,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(-100.0),
        -232.77899999999997,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(-1.0),
        -0.429592,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(-1.0),
        -2.32779,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(-0.0),
        -0.0,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(-0.0),
        -0.0,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(0.0),
        0.0,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(0.0),
        0.0,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(1.0),
        0.429592,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(1.0),
        2.32779,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(12.345),
        5.30331324,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(12.345),
        28.73656755,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(100.0),
        42.959199999999996,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(100.0),
        232.77899999999997,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatOfCombustionUnit::KilojoulesPerKilogram.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(HeatOfCombustionUnit::try_from(-1).is_err());
    assert!(HeatOfCombustionUnit::try_from(2).is_err());
    assert_eq!(HeatSinkUnit::ALL.len(), 2);
    assert_eq!(HeatSinkUnit::BtusPerCubicFoot.code(), 0);
    assert_eq!(
        HeatSinkUnit::try_from(0),
        Ok(HeatSinkUnit::BtusPerCubicFoot)
    );
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(-100.0), -100.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(-100.0), -100.0);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(-1.0), -1.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(-1.0), -1.0);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(-0.0), -0.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(-0.0), -0.0);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(0.0), 0.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(0.0), 0.0);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(1.0), 1.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(1.0), 1.0);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(12.345), 12.345);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(12.345), 12.345);
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(100.0), 100.0);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(100.0), 100.0);
    close(
        HeatSinkUnit::BtusPerCubicFoot.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSinkUnit::BtusPerCubicFoot.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSinkUnit::BtusPerCubicFoot.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSinkUnit::BtusPerCubicFoot.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(HeatSinkUnit::BtusPerCubicFoot.to_base(f64::NAN), f64::NAN);
    close(HeatSinkUnit::BtusPerCubicFoot.from_base(f64::NAN), f64::NAN);
    assert_eq!(HeatSinkUnit::KilojoulesPerCubicMeter.code(), 1);
    assert_eq!(
        HeatSinkUnit::try_from(1),
        Ok(HeatSinkUnit::KilojoulesPerCubicMeter)
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(-100.0),
        -2.6839,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(-100.0),
        -3725.921234025113,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(-1.0),
        -0.026839,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(-1.0),
        -37.25921234025113,
    );
    close(HeatSinkUnit::KilojoulesPerCubicMeter.to_base(-0.0), -0.0);
    close(HeatSinkUnit::KilojoulesPerCubicMeter.from_base(-0.0), -0.0);
    close(HeatSinkUnit::KilojoulesPerCubicMeter.to_base(0.0), 0.0);
    close(HeatSinkUnit::KilojoulesPerCubicMeter.from_base(0.0), 0.0);
    close(HeatSinkUnit::KilojoulesPerCubicMeter.to_base(1.0), 0.026839);
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(1.0),
        37.25921234025113,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(12.345),
        0.331327455,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(12.345),
        459.9649763404002,
    );
    close(HeatSinkUnit::KilojoulesPerCubicMeter.to_base(100.0), 2.6839);
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(100.0),
        3725.921234025113,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSinkUnit::KilojoulesPerCubicMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(HeatSinkUnit::try_from(-1).is_err());
    assert!(HeatSinkUnit::try_from(2).is_err());
    assert_eq!(HeatPerUnitAreaUnit::ALL.len(), 3);
    assert_eq!(HeatPerUnitAreaUnit::BtusPerSquareFoot.code(), 0);
    assert_eq!(
        HeatPerUnitAreaUnit::try_from(0),
        Ok(HeatPerUnitAreaUnit::BtusPerSquareFoot)
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(-100.0),
        -100.0,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(-100.0),
        -100.0,
    );
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(-1.0), -1.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(-1.0), -1.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(-0.0), -0.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(-0.0), -0.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(0.0), 0.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(0.0), 0.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(1.0), 1.0);
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(1.0), 1.0);
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(12.345),
        12.345,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(12.345),
        12.345,
    );
    close(HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(100.0), 100.0);
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(100.0),
        100.0,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatPerUnitAreaUnit::BtusPerSquareFoot.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.code(), 1);
    assert_eq!(
        HeatPerUnitAreaUnit::try_from(1),
        Ok(HeatPerUnitAreaUnit::KilojoulesPerSquareMeter)
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(-100.0),
        -8.799999999999999,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(-100.0),
        -1136.3636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(-1.0),
        -0.088,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(-1.0),
        -11.363636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(-0.0),
        -0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(-0.0),
        -0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(0.0),
        0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(0.0),
        0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(1.0),
        0.088,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(1.0),
        11.363636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(12.345),
        1.08636,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(12.345),
        140.28409090909093,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(100.0),
        8.799999999999999,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(100.0),
        1136.3636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatPerUnitAreaUnit::KilojoulesPerSquareMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.code(), 2);
    assert_eq!(
        HeatPerUnitAreaUnit::try_from(2),
        Ok(HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter)
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(-100.0),
        -8.799999999999999,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(-100.0),
        -1136.3636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(-1.0),
        -0.088,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(-1.0),
        -11.363636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(-0.0),
        -0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(-0.0),
        -0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(0.0),
        0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(0.0),
        0.0,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(1.0),
        0.088,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(1.0),
        11.363636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(12.345),
        1.08636,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(12.345),
        140.28409090909093,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(100.0),
        8.799999999999999,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(100.0),
        1136.3636363636365,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatPerUnitAreaUnit::KilowattSecondsPerSquareMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(HeatPerUnitAreaUnit::try_from(-1).is_err());
    assert!(HeatPerUnitAreaUnit::try_from(3).is_err());
    assert_eq!(HeatSourceAndReactionIntensityUnit::ALL.len(), 5);
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.code(),
        0
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::try_from(0),
        Ok(HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute)
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(-100.0),
        -100.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(-100.0),
        -100.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(-1.0),
        -1.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(-1.0),
        -1.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(1.0),
        1.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(1.0),
        1.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(12.345),
        12.345,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(12.345),
        12.345,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(100.0),
        100.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(100.0),
        100.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerMinute.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.code(),
        1
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::try_from(1),
        Ok(HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond)
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(-100.0),
        -6000.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(-100.0),
        -1.6666666666666667,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(-1.0),
        -60.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(-1.0),
        -0.016666666666666666,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(1.0),
        60.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(1.0),
        0.016666666666666666,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(12.345),
        740.7,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(12.345),
        0.20575000000000002,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(100.0),
        6000.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(100.0),
        1.6666666666666667,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSourceAndReactionIntensityUnit::BtusPerSquareFootPerSecond.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.code(),
        2
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::try_from(2),
        Ok(HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond)
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(-100.0),
        -528.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(-100.0),
        -18.939393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(-1.0),
        -5.28,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(-1.0),
        -0.1893939393939394,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(1.0),
        5.28,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(1.0),
        0.1893939393939394,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(12.345),
        65.1816,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(12.345),
        2.338068181818182,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(100.0),
        528.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(100.0),
        18.939393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond
            .to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond
            .from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond
            .to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond
            .from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerSecond.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.code(),
        3
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::try_from(3),
        Ok(HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute)
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(-100.0),
        -8.799999999999999,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(-100.0),
        -1136.3636363636365,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(-1.0),
        -0.088,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(-1.0),
        -11.363636363636365,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(1.0),
        0.088,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(1.0),
        11.363636363636365,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(12.345),
        1.08636,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(12.345),
        140.28409090909093,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(100.0),
        8.799999999999999,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(100.0),
        1136.3636363636365,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute
            .to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute
            .from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute
            .to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute
            .from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilojoulesPerSquareMeterPerMinute.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.code(),
        4
    );
    assert_eq!(
        HeatSourceAndReactionIntensityUnit::try_from(4),
        Ok(HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter)
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(-100.0),
        -52.800000000000004,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(-100.0),
        -189.39393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(-1.0),
        -0.528,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(-1.0),
        -1.8939393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(-0.0),
        -0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(0.0),
        0.0,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(1.0),
        0.528,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(1.0),
        1.8939393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(12.345),
        6.518160000000001,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(12.345),
        23.380681818181817,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(100.0),
        52.800000000000004,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(100.0),
        189.39393939393938,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        HeatSourceAndReactionIntensityUnit::KilowattsPerSquareMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(HeatSourceAndReactionIntensityUnit::try_from(-1).is_err());
    assert!(HeatSourceAndReactionIntensityUnit::try_from(5).is_err());
    assert_eq!(FirelineIntensityUnit::ALL.len(), 5);
    assert_eq!(FirelineIntensityUnit::BtusPerFootPerSecond.code(), 0);
    assert_eq!(
        FirelineIntensityUnit::try_from(0),
        Ok(FirelineIntensityUnit::BtusPerFootPerSecond)
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(-100.0),
        -100.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(-100.0),
        -100.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(-1.0),
        -1.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(-1.0),
        -1.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(1.0),
        1.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(1.0),
        1.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(12.345),
        12.345,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(12.345),
        12.345,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(100.0),
        100.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(100.0),
        100.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerSecond.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(FirelineIntensityUnit::BtusPerFootPerMinute.code(), 1);
    assert_eq!(
        FirelineIntensityUnit::try_from(1),
        Ok(FirelineIntensityUnit::BtusPerFootPerMinute)
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(-100.0),
        -1.666666666666667,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(-100.0),
        -6000.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(-1.0),
        -0.01666666666666667,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(-1.0),
        -60.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(1.0),
        0.01666666666666667,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(1.0),
        60.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(12.345),
        0.20575000000000004,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(12.345),
        740.7,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(100.0),
        1.666666666666667,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(100.0),
        6000.0,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        FirelineIntensityUnit::BtusPerFootPerMinute.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(FirelineIntensityUnit::KilojoulesPerMeterPerSecond.code(), 2);
    assert_eq!(
        FirelineIntensityUnit::try_from(2),
        Ok(FirelineIntensityUnit::KilojoulesPerMeterPerSecond)
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(-100.0),
        -28.867189999999997,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(-100.0),
        -346.41404190000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(-1.0),
        -0.2886719,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(-1.0),
        -3.464140419,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(1.0),
        0.2886719,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(1.0),
        3.464140419,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(12.345),
        3.5636546055,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(12.345),
        42.764813472555005,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(100.0),
        28.867189999999997,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(100.0),
        346.41404190000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerSecond.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(FirelineIntensityUnit::KilojoulesPerMeterPerMinute.code(), 3);
    assert_eq!(
        FirelineIntensityUnit::try_from(3),
        Ok(FirelineIntensityUnit::KilojoulesPerMeterPerMinute)
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(-100.0),
        -0.48112081900000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(-100.0),
        -20784.800000000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(-1.0),
        -0.00481120819,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(-1.0),
        -207.848,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(-0.0),
        -0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(0.0),
        0.0,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(1.0),
        0.00481120819,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(1.0),
        207.848,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(12.345),
        0.059394365105550004,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(12.345),
        2565.88356,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(100.0),
        0.48112081900000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(100.0),
        20784.800000000003,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        FirelineIntensityUnit::KilojoulesPerMeterPerMinute.from_base(f64::NAN),
        f64::NAN,
    );
    assert_eq!(FirelineIntensityUnit::KilowattsPerMeter.code(), 4);
    assert_eq!(
        FirelineIntensityUnit::try_from(4),
        Ok(FirelineIntensityUnit::KilowattsPerMeter)
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(-100.0),
        -28.867189999999997,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(-100.0),
        -346.41404190000003,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(-1.0),
        -0.2886719,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(-1.0),
        -3.464140419,
    );
    close(FirelineIntensityUnit::KilowattsPerMeter.to_base(-0.0), -0.0);
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(-0.0),
        -0.0,
    );
    close(FirelineIntensityUnit::KilowattsPerMeter.to_base(0.0), 0.0);
    close(FirelineIntensityUnit::KilowattsPerMeter.from_base(0.0), 0.0);
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(1.0),
        0.2886719,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(1.0),
        3.464140419,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(12.345),
        3.5636546055,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(12.345),
        42.764813472555005,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(100.0),
        28.867189999999997,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(100.0),
        346.41404190000003,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.to_base(f64::NAN),
        f64::NAN,
    );
    close(
        FirelineIntensityUnit::KilowattsPerMeter.from_base(f64::NAN),
        f64::NAN,
    );
    assert!(FirelineIntensityUnit::try_from(-1).is_err());
    assert!(FirelineIntensityUnit::try_from(5).is_err());
    assert_eq!(TemperatureUnit::ALL.len(), 3);
    assert_eq!(TemperatureUnit::Fahrenheit.code(), 0);
    assert_eq!(
        TemperatureUnit::try_from(0),
        Ok(TemperatureUnit::Fahrenheit)
    );
    close(TemperatureUnit::Fahrenheit.to_base(-100.0), -100.0);
    close(TemperatureUnit::Fahrenheit.from_base(-100.0), -100.0);
    close(TemperatureUnit::Fahrenheit.to_base(-1.0), -1.0);
    close(TemperatureUnit::Fahrenheit.from_base(-1.0), -1.0);
    close(TemperatureUnit::Fahrenheit.to_base(-0.0), -0.0);
    close(TemperatureUnit::Fahrenheit.from_base(-0.0), -0.0);
    close(TemperatureUnit::Fahrenheit.to_base(0.0), 0.0);
    close(TemperatureUnit::Fahrenheit.from_base(0.0), 0.0);
    close(TemperatureUnit::Fahrenheit.to_base(1.0), 1.0);
    close(TemperatureUnit::Fahrenheit.from_base(1.0), 1.0);
    close(TemperatureUnit::Fahrenheit.to_base(12.345), 12.345);
    close(TemperatureUnit::Fahrenheit.from_base(12.345), 12.345);
    close(TemperatureUnit::Fahrenheit.to_base(100.0), 100.0);
    close(TemperatureUnit::Fahrenheit.from_base(100.0), 100.0);
    close(
        TemperatureUnit::Fahrenheit.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Fahrenheit.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Fahrenheit.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TemperatureUnit::Fahrenheit.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TemperatureUnit::Fahrenheit.to_base(f64::NAN), f64::NAN);
    close(TemperatureUnit::Fahrenheit.from_base(f64::NAN), f64::NAN);
    assert_eq!(TemperatureUnit::Celsius.code(), 1);
    assert_eq!(TemperatureUnit::try_from(1), Ok(TemperatureUnit::Celsius));
    close(TemperatureUnit::Celsius.to_base(-100.0), -148.0);
    close(
        TemperatureUnit::Celsius.from_base(-100.0),
        -73.33333333333334,
    );
    close(TemperatureUnit::Celsius.to_base(-1.0), 30.2);
    close(
        TemperatureUnit::Celsius.from_base(-1.0),
        -18.333333333333336,
    );
    close(TemperatureUnit::Celsius.to_base(-0.0), 32.0);
    close(TemperatureUnit::Celsius.from_base(-0.0), -17.77777777777778);
    close(TemperatureUnit::Celsius.to_base(0.0), 32.0);
    close(TemperatureUnit::Celsius.from_base(0.0), -17.77777777777778);
    close(TemperatureUnit::Celsius.to_base(1.0), 33.8);
    close(TemperatureUnit::Celsius.from_base(1.0), -17.22222222222222);
    close(TemperatureUnit::Celsius.to_base(12.345), 54.221000000000004);
    close(
        TemperatureUnit::Celsius.from_base(12.345),
        -10.919444444444446,
    );
    close(TemperatureUnit::Celsius.to_base(100.0), 212.0);
    close(TemperatureUnit::Celsius.from_base(100.0), 37.77777777777778);
    close(
        TemperatureUnit::Celsius.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Celsius.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Celsius.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TemperatureUnit::Celsius.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TemperatureUnit::Celsius.to_base(f64::NAN), f64::NAN);
    close(TemperatureUnit::Celsius.from_base(f64::NAN), f64::NAN);
    assert_eq!(TemperatureUnit::Kelvin.code(), 2);
    assert_eq!(TemperatureUnit::try_from(2), Ok(TemperatureUnit::Kelvin));
    close(TemperatureUnit::Kelvin.to_base(-100.0), -639.67);
    close(
        TemperatureUnit::Kelvin.from_base(-100.0),
        199.81666666666663,
    );
    close(TemperatureUnit::Kelvin.to_base(-1.0), -461.46999999999997);
    close(TemperatureUnit::Kelvin.from_base(-1.0), 254.81666666666663);
    close(TemperatureUnit::Kelvin.to_base(-0.0), -459.66999999999996);
    close(TemperatureUnit::Kelvin.from_base(-0.0), 255.3722222222222);
    close(TemperatureUnit::Kelvin.to_base(0.0), -459.66999999999996);
    close(TemperatureUnit::Kelvin.from_base(0.0), 255.3722222222222);
    close(TemperatureUnit::Kelvin.to_base(1.0), -457.86999999999995);
    close(TemperatureUnit::Kelvin.from_base(1.0), 255.92777777777775);
    close(TemperatureUnit::Kelvin.to_base(12.345), -437.4489999999999);
    close(
        TemperatureUnit::Kelvin.from_base(12.345),
        262.23055555555555,
    );
    close(TemperatureUnit::Kelvin.to_base(100.0), -279.66999999999996);
    close(TemperatureUnit::Kelvin.from_base(100.0), 310.92777777777775);
    close(
        TemperatureUnit::Kelvin.to_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Kelvin.from_base(f64::INFINITY),
        f64::INFINITY,
    );
    close(
        TemperatureUnit::Kelvin.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TemperatureUnit::Kelvin.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TemperatureUnit::Kelvin.to_base(f64::NAN), f64::NAN);
    close(TemperatureUnit::Kelvin.from_base(f64::NAN), f64::NAN);
    assert!(TemperatureUnit::try_from(-1).is_err());
    assert!(TemperatureUnit::try_from(3).is_err());
    assert_eq!(TimeUnit::ALL.len(), 5);
    assert_eq!(TimeUnit::Minutes.code(), 0);
    assert_eq!(TimeUnit::try_from(0), Ok(TimeUnit::Minutes));
    close(TimeUnit::Minutes.to_base(-100.0), -100.0);
    close(TimeUnit::Minutes.from_base(-100.0), -100.0);
    close(TimeUnit::Minutes.to_base(-1.0), -1.0);
    close(TimeUnit::Minutes.from_base(-1.0), -1.0);
    close(TimeUnit::Minutes.to_base(-0.0), -0.0);
    close(TimeUnit::Minutes.from_base(-0.0), -0.0);
    close(TimeUnit::Minutes.to_base(0.0), 0.0);
    close(TimeUnit::Minutes.from_base(0.0), 0.0);
    close(TimeUnit::Minutes.to_base(1.0), 1.0);
    close(TimeUnit::Minutes.from_base(1.0), 1.0);
    close(TimeUnit::Minutes.to_base(12.345), 12.345);
    close(TimeUnit::Minutes.from_base(12.345), 12.345);
    close(TimeUnit::Minutes.to_base(100.0), 100.0);
    close(TimeUnit::Minutes.from_base(100.0), 100.0);
    close(TimeUnit::Minutes.to_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Minutes.from_base(f64::INFINITY), f64::INFINITY);
    close(
        TimeUnit::Minutes.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TimeUnit::Minutes.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TimeUnit::Minutes.to_base(f64::NAN), f64::NAN);
    close(TimeUnit::Minutes.from_base(f64::NAN), f64::NAN);
    assert_eq!(TimeUnit::Seconds.code(), 1);
    assert_eq!(TimeUnit::try_from(1), Ok(TimeUnit::Seconds));
    close(TimeUnit::Seconds.to_base(-100.0), -1.6666666666666667);
    close(TimeUnit::Seconds.from_base(-100.0), -6000.0);
    close(TimeUnit::Seconds.to_base(-1.0), -0.016666666666666666);
    close(TimeUnit::Seconds.from_base(-1.0), -60.0);
    close(TimeUnit::Seconds.to_base(-0.0), -0.0);
    close(TimeUnit::Seconds.from_base(-0.0), -0.0);
    close(TimeUnit::Seconds.to_base(0.0), 0.0);
    close(TimeUnit::Seconds.from_base(0.0), 0.0);
    close(TimeUnit::Seconds.to_base(1.0), 0.016666666666666666);
    close(TimeUnit::Seconds.from_base(1.0), 60.0);
    close(TimeUnit::Seconds.to_base(12.345), 0.20575000000000002);
    close(TimeUnit::Seconds.from_base(12.345), 740.7);
    close(TimeUnit::Seconds.to_base(100.0), 1.6666666666666667);
    close(TimeUnit::Seconds.from_base(100.0), 6000.0);
    close(TimeUnit::Seconds.to_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Seconds.from_base(f64::INFINITY), f64::INFINITY);
    close(
        TimeUnit::Seconds.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TimeUnit::Seconds.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TimeUnit::Seconds.to_base(f64::NAN), f64::NAN);
    close(TimeUnit::Seconds.from_base(f64::NAN), f64::NAN);
    assert_eq!(TimeUnit::Hours.code(), 2);
    assert_eq!(TimeUnit::try_from(2), Ok(TimeUnit::Hours));
    close(TimeUnit::Hours.to_base(-100.0), -6000.0);
    close(TimeUnit::Hours.from_base(-100.0), -1.6666666666666667);
    close(TimeUnit::Hours.to_base(-1.0), -60.0);
    close(TimeUnit::Hours.from_base(-1.0), -0.016666666666666666);
    close(TimeUnit::Hours.to_base(-0.0), -0.0);
    close(TimeUnit::Hours.from_base(-0.0), -0.0);
    close(TimeUnit::Hours.to_base(0.0), 0.0);
    close(TimeUnit::Hours.from_base(0.0), 0.0);
    close(TimeUnit::Hours.to_base(1.0), 60.0);
    close(TimeUnit::Hours.from_base(1.0), 0.016666666666666666);
    close(TimeUnit::Hours.to_base(12.345), 740.7);
    close(TimeUnit::Hours.from_base(12.345), 0.20575000000000002);
    close(TimeUnit::Hours.to_base(100.0), 6000.0);
    close(TimeUnit::Hours.from_base(100.0), 1.6666666666666667);
    close(TimeUnit::Hours.to_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Hours.from_base(f64::INFINITY), f64::INFINITY);
    close(
        TimeUnit::Hours.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TimeUnit::Hours.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TimeUnit::Hours.to_base(f64::NAN), f64::NAN);
    close(TimeUnit::Hours.from_base(f64::NAN), f64::NAN);
    assert_eq!(TimeUnit::Days.code(), 3);
    assert_eq!(TimeUnit::try_from(3), Ok(TimeUnit::Days));
    close(TimeUnit::Days.to_base(-100.0), -144000.0);
    close(TimeUnit::Days.from_base(-100.0), -0.06944444444444445);
    close(TimeUnit::Days.to_base(-1.0), -1440.0);
    close(TimeUnit::Days.from_base(-1.0), -0.0006944444444444445);
    close(TimeUnit::Days.to_base(-0.0), -0.0);
    close(TimeUnit::Days.from_base(-0.0), -0.0);
    close(TimeUnit::Days.to_base(0.0), 0.0);
    close(TimeUnit::Days.from_base(0.0), 0.0);
    close(TimeUnit::Days.to_base(1.0), 1440.0);
    close(TimeUnit::Days.from_base(1.0), 0.0006944444444444445);
    close(TimeUnit::Days.to_base(12.345), 17776.8);
    close(TimeUnit::Days.from_base(12.345), 0.008572916666666668);
    close(TimeUnit::Days.to_base(100.0), 144000.0);
    close(TimeUnit::Days.from_base(100.0), 0.06944444444444445);
    close(TimeUnit::Days.to_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Days.from_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Days.to_base(f64::NEG_INFINITY), f64::NEG_INFINITY);
    close(
        TimeUnit::Days.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TimeUnit::Days.to_base(f64::NAN), f64::NAN);
    close(TimeUnit::Days.from_base(f64::NAN), f64::NAN);
    assert_eq!(TimeUnit::Years.code(), 4);
    assert_eq!(TimeUnit::try_from(4), Ok(TimeUnit::Years));
    close(TimeUnit::Years.to_base(-100.0), -52596000.0);
    close(TimeUnit::Years.from_base(-100.0), -0.00019012852688417368);
    close(TimeUnit::Years.to_base(-1.0), -525960.0);
    close(TimeUnit::Years.from_base(-1.0), -1.901285268841737e-06);
    close(TimeUnit::Years.to_base(-0.0), -0.0);
    close(TimeUnit::Years.from_base(-0.0), -0.0);
    close(TimeUnit::Years.to_base(0.0), 0.0);
    close(TimeUnit::Years.from_base(0.0), 0.0);
    close(TimeUnit::Years.to_base(1.0), 525960.0);
    close(TimeUnit::Years.from_base(1.0), 1.901285268841737e-06);
    close(TimeUnit::Years.to_base(12.345), 6492976.2);
    close(TimeUnit::Years.from_base(12.345), 2.3471366643851245e-05);
    close(TimeUnit::Years.to_base(100.0), 52596000.0);
    close(TimeUnit::Years.from_base(100.0), 0.00019012852688417368);
    close(TimeUnit::Years.to_base(f64::INFINITY), f64::INFINITY);
    close(TimeUnit::Years.from_base(f64::INFINITY), f64::INFINITY);
    close(
        TimeUnit::Years.to_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(
        TimeUnit::Years.from_base(f64::NEG_INFINITY),
        f64::NEG_INFINITY,
    );
    close(TimeUnit::Years.to_base(f64::NAN), f64::NAN);
    close(TimeUnit::Years.from_base(f64::NAN), f64::NAN);
    assert!(TimeUnit::try_from(-1).is_err());
    assert!(TimeUnit::try_from(5).is_err());
}
