# SpeciesMasterTable component - Complete implementation
# This module defines the species master table for Behave fire behavior model

from enum import IntEnum

# Enums for geographic regions
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
    CrownScorch = 0  # "CRNSCH" - Crown Scorch Equations
    BoleChar = 1     # "BOLCHR" - Bole Char Equations
    CrownDamage = 2  # "CRCABE" - Crown Damage / Cambium / Beetle Equations


class CrownDamageEquationCode(IntEnum):
    """Crown damage equation codes"""
    NotSet = -1


class SpeciesMasterTableRecord:
    """Record structure for a single species in the master table"""
    
    def __init__(self, species_code="", scientific_name="", common_name="", 
                 mortality_equation_number=-1, bark_equation_number=-1, 
                 crown_coefficient_code=0, alaska=False, california=False, 
                 eastern_area=False, great_basin=False, northern_rockies=False,
                 northwest=False, rocky_mountain=False, southern_area=False, 
                 southwest=False, equation_type=EquationType.NotSet,
                 crown_damage_equation_code=CrownDamageEquationCode.NotSet):
        """
        Initialize a species master table record.
        
        Args:
            species_code: Species code (e.g., "ABAM")
            scientific_name: Scientific name of the species
            common_name: Common name of the species
            mortality_equation_number: Mortality equation number
            bark_equation_number: Bark equation number
            crown_coefficient_code: Crown coefficient code for canopy cover calculation
            alaska: Available in Alaska GACC (True/False)
            california: Available in California GACC (True/False)
            eastern_area: Available in Eastern Area GACC (True/False)
            great_basin: Available in Great Basin GACC (True/False)
            northern_rockies: Available in Northern Rockies GACC (True/False)
            northwest: Available in Northwest GACC (True/False)
            rocky_mountain: Available in Rocky Mountain GACC (True/False)
            southern_area: Available in Southern Area GACC (True/False)
            southwest: Available in Southwest GACC (True/False)
            equation_type: Type of equation (crown_scorch, bole_char, or crown_damage)
            crown_damage_equation_code: Crown damage equation code
        """
        self.species_code = species_code
        self.scientific_name = scientific_name
        self.common_name = common_name
        self.mortality_equation_number = mortality_equation_number
        self.bark_equation_number = bark_equation_number
        self.crown_coefficient_code = crown_coefficient_code
        
        # GACC region availability flags
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
        """
        Check if species is available in a specific GACC region.
        
        Args:
            gacc: GACC region enum value
            
        Returns:
            True if available in the region, False otherwise
        """
        gacc_map = {
            GACC.Alaska: self.alaska,
            GACC.California: self.california,
            GACC.EasternArea: self.eastern_area,
            GACC.GreatBasin: self.great_basin,
            GACC.NorthernRockies: self.northern_rockies,
            GACC.Northwest: self.northwest,
            GACC.RockyMountain: self.rocky_mountain,
            GACC.SouthernArea: self.southern_area,
            GACC.Southwest: self.southwest,
        }
        return gacc_map.get(gacc, False)
    
    def __repr__(self):
        """Return string representation of the record"""
        return f"SpeciesMasterTableRecord(code={self.species_code}, name={self.common_name}, type={self.equation_type})"


class SpeciesMasterTable:
    """Master table containing all species records for Behave fire behavior model"""
    
    def __init__(self):
        """Initialize the species master table."""
        self.records = []
        self.initialize_master_table()
    
    def initialize_master_table(self):
        """Initialize the master table with all species records from the database."""
        # Clear existing records
        self.records = []
        
        # Insert all species records from the Behave database
        # Format: (species_code, scientific_name, common_name, mort_eq, bark_eq, crown_coeff,
        #          alaska, california, eastern_area, great_basin, northern_rockies, northwest,
        #          rocky_mountain, southern_area, southwest, equation_type, crown_damage_code)
        
        species_data = [
            ("ABAM", "Abies amabilis", "Pacific silver fir", 1, 26, 1, True, False, False, False, False, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABBA", "Abies balsamea", "balsam fir", 1, 10, 2, False, False, True, False, False, False, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABCO", "Abies concolor", "white fir", 10, 27, 2, False, True, True, True, False, True, True, False, True, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABGR", "Abies grandis", "grand fir", 11, 25, 3, False, True, False, True, True, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABLA", "Abies lasiocarpa", "corkbark fir", 11, 20, 4, True, True, False, True, True, True, True, False, True, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABMA", "Abies magnifica", "California red fir", 16, 18, 5, False, True, False, True, False, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ABPR", "Abies procera", "noble fir", 1, 24, 7, False, True, False, False, False, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("ACRU", "Acer rubrum", "red maple", 100, 7, 21, False, False, True, False, False, False, True, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("COFL2", "Cornus florida", "flowering dogwood", 101, 20, 34, False, False, True, False, False, False, True, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("LAOC", "Larix occidentalis", "western larch", 14, 36, 14, False, True, False, True, True, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("NYSY", "Nyssa sylvatica", "blackgum", 102, 32, 39, False, False, True, False, False, False, False, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("OXAR", "Oxydendrum arboreum", "sourwood", 103, 15, 39, False, False, True, False, False, False, False, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("PIEN", "Picea engelmannii", "Engelmann spruce", 15, 15, 1, False, True, False, True, True, True, True, False, True, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("PIGL", "Picea glauca", "white spruce", 3, 4, 1, True, False, True, False, True, False, True, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("PIPO", "Pinus ponderosa", "ponderosa pine", 2, 31, 12, False, True, False, True, True, True, False, False, True, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("PSME", "Pseudotsuga menziesii", "Douglas-fir", 13, 22, 16, False, True, False, True, True, True, True, False, True, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("QUMA3", "Quercus marilandica", "blackjack oak", 106, 16, 28, False, False, True, False, False, False, True, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("QUEMON", "Quercus montana", "chestnut oak", 107, 28, 28, False, False, True, False, False, False, False, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("QUVE", "Quercus velutina", "black oak", 108, 24, 28, False, False, True, False, False, False, True, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("SAAL5", "Sassafras albidum", "sassafras", 109, 14, 39, False, False, True, False, False, False, False, True, False, EquationType.BoleChar, CrownDamageEquationCode.NotSet),
            ("TSCA", "Tsuga canadensis", "eastern hemlock", 1, 18, 19, False, False, True, False, False, False, False, True, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("TSHE", "Tsuga heterophylla", "western hemlock", 1, 19, 19, True, True, False, False, True, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("PIAL", "Pinus albicaulis", "whitebark pine", 1, 39, 11, False, False, False, True, True, True, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
            ("PIBA", "Pinus balfouriana", "foxtail pine", 1, 39, 15, False, True, False, False, False, False, False, False, False, EquationType.CrownScorch, CrownDamageEquationCode.NotSet),
        ]
        
        # Insert each species record
        for data in species_data:
            self.insert_record(*data)
    
    def insert_record(self, species_code, scientific_name, common_name, 
                     mortality_equation, bark_equation, crown_coefficient_code,
                     alaska, california, eastern_area, great_basin, northern_rockies,
                     northwest, rocky_mountain, southern_area, southwest,
                     equation_type, crown_damage_equation_code):
        """
        Insert a new species record into the master table.
        
        Args:
            species_code: Species code
            scientific_name: Scientific name
            common_name: Common name
            mortality_equation: Mortality equation number
            bark_equation: Bark equation number
            crown_coefficient_code: Crown coefficient code
            alaska: Alaska GACC availability
            california: California GACC availability
            eastern_area: Eastern Area GACC availability
            great_basin: Great Basin GACC availability
            northern_rockies: Northern Rockies GACC availability
            northwest: Northwest GACC availability
            rocky_mountain: Rocky Mountain GACC availability
            southern_area: Southern Area GACC availability
            southwest: Southwest GACC availability
            equation_type: Type of equation
            crown_damage_equation_code: Crown damage equation code
        """
        record = SpeciesMasterTableRecord(
            species_code=species_code,
            scientific_name=scientific_name,
            common_name=common_name,
            mortality_equation_number=mortality_equation,
            bark_equation_number=bark_equation,
            crown_coefficient_code=crown_coefficient_code,
            alaska=alaska,
            california=california,
            eastern_area=eastern_area,
            great_basin=great_basin,
            northern_rockies=northern_rockies,
            northwest=northwest,
            rocky_mountain=rocky_mountain,
            southern_area=southern_area,
            southwest=southwest,
            equation_type=equation_type,
            crown_damage_equation_code=crown_damage_equation_code
        )
        self.records.append(record)
    
    def get_species_record(self, species_code):
        """
        Get the first species record matching the given species code.
        
        Args:
            species_code: Species code to search for (case-insensitive)
            
        Returns:
            SpeciesMasterTableRecord if found, None otherwise
        """
        index = self.get_species_table_index_from_species_code(species_code)
        if index >= 0:
            return self.records[index]
        return None
    
    def get_species_record_by_code_and_type(self, species_code, equation_type):
        """
        Get a species record matching the given species code and equation type.
        
        Args:
            species_code: Species code to search for (case-insensitive)
            equation_type: EquationType to match
            
        Returns:
            SpeciesMasterTableRecord if found, None otherwise
        """
        index = self.get_species_table_index_from_species_code_and_equation_type(species_code, equation_type)
        if index >= 0:
            return self.records[index]
        return None
    
    def get_species_record_at_index(self, index):
        """
        Get the species record at the given index.
        
        Args:
            index: Index into the records list
            
        Returns:
            SpeciesMasterTableRecord at the index, or None if index is out of bounds
        """
        if 0 <= index < len(self.records):
            return self.records[index]
        return None
    
    def get_species_table_index_from_species_code(self, species_code):
        """
        Get the table index of a species by species code.
        
        Args:
            species_code: Species code to search for (case-insensitive)
            
        Returns:
            Index of the species in the records list, or -1 if not found
        """
        species_code_upper = species_code.upper()
        
        for i, record in enumerate(self.records):
            if record.species_code == "":
                break
            if record.species_code.upper() == species_code_upper:
                return i
        
        return -1
    
    def get_species_table_index_from_species_code_and_equation_type(self, species_code, equation_type):
        """
        Get the table index of a species by species code and equation type.
        
        Args:
            species_code: Species code to search for (case-insensitive)
            equation_type: EquationType to match
            
        Returns:
            Index of the species in the records list, or -1 if not found
        """
        species_code_upper = species_code.upper()
        
        for i, record in enumerate(self.records):
            if record.species_code == "":
                break
            if (record.species_code.upper() == species_code_upper and 
                record.equation_type == equation_type):
                return i
        
        return -1
    
    def add_species_record(self, record):
        """
        Add a new species master table record.
        
        Args:
            record: SpeciesMasterTableRecord to add
        """
        if isinstance(record, SpeciesMasterTableRecord):
            self.records.append(record)
    
    def remove_species_record(self, species_code):
        """
        Remove a species master table record by species code.
        
        Args:
            species_code: Species code to remove (case-insensitive)
            
        Returns:
            True if record was removed, False if not found
        """
        index = self.get_species_table_index_from_species_code(species_code)
        if index >= 0:
            del self.records[index]
            return True
        return False
    
    def get_all_records(self):
        """
        Get all species records.
        
        Returns:
            List of all SpeciesMasterTableRecord objects
        """
        return self.records
    
    def get_species_records_for_gacc_region(self, gacc):
        """
        Get all species records available in a specific GACC region.
        
        Args:
            gacc: GACC region (enum value)
            
        Returns:
            List of SpeciesMasterTableRecord objects available in the region
        """
        return [record for record in self.records if record.get_gacc_availability(gacc)]
    
    def get_species_records_for_gacc_region_and_equation_type(self, gacc, equation_type):
        """
        Get all species records available in a specific GACC region with a specific equation type.
        
        Args:
            gacc: GACC region (enum value)
            equation_type: EquationType to match
            
        Returns:
            List of SpeciesMasterTableRecord objects matching the criteria
        """
        return [record for record in self.records 
                if record.get_gacc_availability(gacc) and record.equation_type == equation_type]
    
    def get_number_of_records(self):
        """
        Get the total number of species records in the table.
        
        Returns:
            Number of records in the table
        """
        return len(self.records)
    
    def contains_species_code(self, species_code):
        """
        Check if a species code exists in the table.
        
        Args:
            species_code: Species code to check (case-insensitive)
            
        Returns:
            True if the species code is found, False otherwise
        """
        return self.get_species_table_index_from_species_code(species_code) >= 0


