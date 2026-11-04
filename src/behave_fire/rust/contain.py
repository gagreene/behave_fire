"""Explicit native contain adapters."""
from __future__ import annotations
from typing import Union, Optional
import warnings
import numpy as np
from ._bridge import _call, _native
STATUS_REPORTED = np.int32(1)
STATUS_CONTAINED = np.int32(3)
STATUS_OVERRUN = np.int32(4)
STATUS_EXHAUSTED = np.int32(5)
STATUS_TIME_LIMIT = np.int32(8)

def run_contain_sim_array(
        report_spread_rate: Union[float, np.ndarray],
        lw_ratio: Union[float, np.ndarray],
        report_size: Union[float, np.ndarray],
        production_rate: Union[float, np.ndarray],
        attack_time: Union[float, np.ndarray],
        max_steps: int=500,
        dist_step: float=0.01,
) -> dict:
    """Dispatch to the standalone native implementation.

    :param report_spread_rate: Input matching the reference API.
    :param lw_ratio: Input matching the reference API.
    :param report_size: Input matching the reference API.
    :param production_rate: Input matching the reference API.
    :param attack_time: Input matching the reference API.
    :param max_steps: Input matching the reference API.
    :param dist_step: Input matching the reference API.
    :return: Owned native calculation result.
    """
    return _call('contain.run_contain_sim_array', locals())
