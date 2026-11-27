"""Boundary coverage for the Python fine dead fuel moisture lookup."""

import numpy as np

from behave_fire.components.fine_dead_fuel_moisture_tool import calculate_fine_dead_fuel_moisture

# ---- moisture lookup tests ----

def test_moisture_lookup_routes_exposed_and_shaded_cells():
    moisture = calculate_fine_dead_fuel_moisture(
        dry_bulb_i=np.array([[0], [5]]),
        rh_i=20,
        slope_i=np.array([0, 1, 0, 1, 1]),
        aspect_i=np.array([0, 0, 0, 0, 3]),
        shading_i=np.array([0, 0, 1, 1, 1]),
        month_i=np.array([0, 0, 0, 0, 2]),
        elev_i=np.array([1, 1, 1, 1, 2]),
        time_i=np.array([3, 3, 3, 3, 5]),
    )

    # Pinned fineDeadFuelMoistureTool.cpp reference cells [0, 20] and [5, 20].
    reference = np.array([[0.14], [0.12]])
    # Correction cells [0, 10], [1, 10], [8, 10], [8, 10], and [35, 17].
    # The first two pairs differ only in slope, which affects exposed cells.
    correction = np.array([0.0, 0.01, 0.03, 0.03, 0.06])
    np.testing.assert_array_equal(moisture, reference + correction)
