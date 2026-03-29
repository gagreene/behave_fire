"""
conftest.py — pytest path fixture for behave_py tests.

pytest.ini already adds src/behave/ via the `pythonpath` option (runs
before collection).  This file is kept as a belt-and-suspenders guard:
if someone runs pytest from inside src/behave/tests/ directly (without
pytest.ini being picked up), this conftest ensures the path is still set.
"""

import sys
import os

# src/behave/  (two levels up from this file: tests/ -> behave/ -> ...)
_BEHAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BEHAVE_DIR not in sys.path:
    sys.path.insert(0, _BEHAVE_DIR)
