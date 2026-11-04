"""Explicit native mortality adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native
from .species_master_table import SpeciesMasterTable

def build_mortality_lookup(species_master_table: Optional['SpeciesMasterTable']=None) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param species_master_table: Input matching the reference API.
    :return: Owned native calculation result.
    """
    if species_master_table is not None:
        for code, equation in _native.missing_mortality([(record.species_code, record.mortality_equation_number) for record in species_master_table.records]):
            warnings.warn(f"SpeciesMasterTable species '{code}' references mortality equation {equation} which has no B-coefficients in the hardcoded tables. Probability of mortality will be 0 for cells assigned this equation number.", UserWarning, stacklevel=2)
    return _call('mortality.build_mortality_lookup', locals())

def calculate_crown_scorch_mortality(
        scorch_height_ft: Union[float, np.ndarray],
        tree_height_ft: Union[float, np.ndarray],
        crown_ratio: Union[float, np.ndarray],
        dbh_inches: Union[float, np.ndarray],
        equation_number_grid: Union[int, np.ndarray],
        coeffs: np.ndarray,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param scorch_height_ft: Input matching the reference API.
    :param tree_height_ft: Input matching the reference API.
    :param crown_ratio: Input matching the reference API.
    :param dbh_inches: Input matching the reference API.
    :param equation_number_grid: Input matching the reference API.
    :param coeffs: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('mortality.calculate_crown_scorch_mortality', locals())

def calculate_scorch_height(
        fireline_intensity_btu_ft_s: Union[float, np.ndarray],
        midflame_wind_mph: Union[float, np.ndarray],
        air_temp_f: Union[float, np.ndarray],
) -> np.ndarray:
    """Dispatch to the standalone native implementation.

    :param fireline_intensity_btu_ft_s: Input matching the reference API.
    :param midflame_wind_mph: Input matching the reference API.
    :param air_temp_f: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('mortality.calculate_scorch_height', locals())
