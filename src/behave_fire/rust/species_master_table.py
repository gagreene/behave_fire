"""Native registry metadata backed by standalone Rust tables."""
from ._bridge import _native
from ._serialization import NativeState as _NativeState
import copy
from enum import IntEnum

class GACC(IntEnum):
    """Geographic Area Coordination Centers"""
    NotSet = -1
    Alaska = 1
    California = 2
    EasternArea = 3
    GreatBasin = 4
    NorthernRockies = 5
    Northwest = 6
    RockyMountain = 7
    SouthernArea = 8
    Southwest = 9

class EquationType(IntEnum):
    """Equation type for mortality calculations"""
    NotSet = -1
    CrownScorch = 0
    BoleChar = 1
    CrownDamage = 2

class CrownDamageEquationCode(IntEnum):
    """Crown damage equation codes"""
    NotSet = -1

class SpeciesMasterTableRecord(_NativeState):
    """Record structure for a single species in the master table"""

    def __init__(
            self,
            species_code='',
            scientific_name='',
            common_name='',
            mortality_equation_number=-1,
            bark_equation_number=-1,
            crown_coefficient_code=0,
            alaska=False,
            california=False,
            eastern_area=False,
            great_basin=False,
            northern_rockies=False,
            northwest=False,
            rocky_mountain=False,
            southern_area=False,
            southwest=False,
            equation_type=EquationType.NotSet,
            crown_damage_equation_code=CrownDamageEquationCode.NotSet,
    ):
        """Initialize a species master table record.

        :param species_code: Input matching the public API.
        :param scientific_name: Input matching the public API.
        :param common_name: Input matching the public API.
        :param mortality_equation_number: Input matching the public API.
        :param bark_equation_number: Input matching the public API.
        :param crown_coefficient_code: Input matching the public API.
        :param alaska: Input matching the public API.
        :param california: Input matching the public API.
        :param eastern_area: Input matching the public API.
        :param great_basin: Input matching the public API.
        :param northern_rockies: Input matching the public API.
        :param northwest: Input matching the public API.
        :param rocky_mountain: Input matching the public API.
        :param southern_area: Input matching the public API.
        :param southwest: Input matching the public API.
        :param equation_type: Input matching the public API.
        :param crown_damage_equation_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        self.species_code = species_code
        self.scientific_name = scientific_name
        self.common_name = common_name
        self.mortality_equation_number = mortality_equation_number
        self.bark_equation_number = bark_equation_number
        self.crown_coefficient_code = crown_coefficient_code
        self.alaska = alaska
        self.california = california
        self.eastern_area = eastern_area
        self.great_basin = great_basin
        self.northern_rockies = northern_rockies
        self.northwest = northwest
        self.rocky_mountain = rocky_mountain
        self.southern_area = southern_area
        self.southwest = southwest
        self.equation_type = equation_type
        self.crown_damage_equation_code = crown_damage_equation_code

    def get_gacc_availability(self, gacc):
        """Check if species is available in a specific GACC region.

        :param gacc: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        gacc_map = {GACC.Alaska: self.alaska, GACC.California: self.california, GACC.EasternArea: self.eastern_area, GACC.GreatBasin: self.great_basin, GACC.NorthernRockies: self.northern_rockies, GACC.Northwest: self.northwest, GACC.RockyMountain: self.rocky_mountain, GACC.SouthernArea: self.southern_area, GACC.Southwest: self.southwest}
        return gacc_map.get(gacc, False)

    def __repr__(self):
        """Return string representation of the record


        :return: Calculation result or updated receiver state.
        """
        return f'SpeciesMasterTableRecord(code={self.species_code}, name={self.common_name}, type={self.equation_type})'

class SpeciesMasterTable(_NativeState):
    """Master table containing all species records for Behave fire behavior model"""

    def __init__(self):
        """Initialize the species master table.


        :return: Calculation result or updated receiver state.
        """
        self.records = []
        self.initialize_master_table()

    def initialize_master_table(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        self.records = [SpeciesMasterTableRecord(**dict(row, equation_type=EquationType(row['equation_type']), crown_damage_equation_code=CrownDamageEquationCode(row['crown_damage_equation_code']))) for row in _native.defaults('species')]

    def insert_record(
            self,
            species_code,
            scientific_name,
            common_name,
            mortality_equation,
            bark_equation,
            crown_coefficient_code,
            alaska,
            california,
            eastern_area,
            great_basin,
            northern_rockies,
            northwest,
            rocky_mountain,
            southern_area,
            southwest,
            equation_type,
            crown_damage_equation_code,
    ):
        """Insert a new species record into the master table.

        :param species_code: Input matching the public API.
        :param scientific_name: Input matching the public API.
        :param common_name: Input matching the public API.
        :param mortality_equation: Input matching the public API.
        :param bark_equation: Input matching the public API.
        :param crown_coefficient_code: Input matching the public API.
        :param alaska: Input matching the public API.
        :param california: Input matching the public API.
        :param eastern_area: Input matching the public API.
        :param great_basin: Input matching the public API.
        :param northern_rockies: Input matching the public API.
        :param northwest: Input matching the public API.
        :param rocky_mountain: Input matching the public API.
        :param southern_area: Input matching the public API.
        :param southwest: Input matching the public API.
        :param equation_type: Input matching the public API.
        :param crown_damage_equation_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        record = SpeciesMasterTableRecord(species_code=species_code, scientific_name=scientific_name, common_name=common_name, mortality_equation_number=mortality_equation, bark_equation_number=bark_equation, crown_coefficient_code=crown_coefficient_code, alaska=alaska, california=california, eastern_area=eastern_area, great_basin=great_basin, northern_rockies=northern_rockies, northwest=northwest, rocky_mountain=rocky_mountain, southern_area=southern_area, southwest=southwest, equation_type=equation_type, crown_damage_equation_code=crown_damage_equation_code)
        self.records.append(record)

    def get_species_record(self, species_code):
        """Get the first species record matching the given species code.

        :param species_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        index = self.get_species_table_index_from_species_code(species_code)
        if index >= 0:
            return self.records[index]
        return None

    def get_species_record_by_code_and_type(self, species_code, equation_type):
        """Get a species record matching the given species code and equation type.

        :param species_code: Input matching the public API.
        :param equation_type: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        index = self.get_species_table_index_from_species_code_and_equation_type(species_code, equation_type)
        if index >= 0:
            return self.records[index]
        return None

    def get_species_record_at_index(self, index):
        """Get the species record at the given index.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        if 0 <= index < len(self.records):
            return self.records[index]
        return None

    def get_species_table_index_from_species_code(self, species_code):
        """Get the table index of a species by species code.

        :param species_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        species_code_upper = species_code.upper()
        for i, record in enumerate(self.records):
            if record.species_code == '':
                break
            if record.species_code.upper() == species_code_upper:
                return i
        return -1

    def get_species_table_index_from_species_code_and_equation_type(
            self,
            species_code,
            equation_type,
    ):
        """Get the table index of a species by species code and equation type.

        :param species_code: Input matching the public API.
        :param equation_type: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        species_code_upper = species_code.upper()
        for i, record in enumerate(self.records):
            if record.species_code == '':
                break
            if record.species_code.upper() == species_code_upper and record.equation_type == equation_type:
                return i
        return -1

    def add_species_record(self, record):
        """Add a new species master table record.

        :param record: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        if isinstance(record, SpeciesMasterTableRecord):
            self.records.append(record)

    def remove_species_record(self, species_code):
        """Remove a species master table record by species code.

        :param species_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        index = self.get_species_table_index_from_species_code(species_code)
        if index >= 0:
            del self.records[index]
            return True
        return False

    def get_all_records(self):
        """Get all species records.


        :return: Calculation result or updated receiver state.
        """
        return self.records

    def get_species_records_for_gacc_region(self, gacc):
        """Get all species records available in a specific GACC region.

        :param gacc: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return [record for record in self.records if record.get_gacc_availability(gacc)]

    def get_species_records_for_gacc_region_and_equation_type(self, gacc, equation_type):
        """Get all species records available in a specific GACC region with a specific equation type.

        :param gacc: Input matching the public API.
        :param equation_type: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return [record for record in self.records if record.get_gacc_availability(gacc) and record.equation_type == equation_type]

    def get_number_of_records(self):
        """Get the total number of species records in the table.


        :return: Calculation result or updated receiver state.
        """
        return len(self.records)

    def contains_species_code(self, species_code):
        """Check if a species code exists in the table.

        :param species_code: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return self.get_species_table_index_from_species_code(species_code) >= 0
