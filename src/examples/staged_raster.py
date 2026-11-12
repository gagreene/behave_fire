"""Demonstrate owned Rust stages and base-unit field injection on a common grid."""
import numpy as np

from behave_fire.rust import CrownGrid, SurfaceGrid


def main():
    """Compare staged injection with a fused restart on independent native grids.

    :return: None; prints Surface and Crown spread arrays in feet/minute.
    """
    weather = dict(m1h=0.06, m10h=0.07, m100h=0.08, mlh=0.6, mlw=0.9,
                   wind_speed=88., wind_direction=35., slope=12., aspect=15.)
    grid = SurfaceGrid(dict(weather, fuel_model_grid=124, canopy_cover=0.4,
                            canopy_height=30., crown_ratio=0.5), shape=(2, 3), threads=4)
    for stage in ('particles', 'fuelbed', 'reaction'):
        grid.run_stage(stage)
    # BTU/ft²/min; editing the returned copy alone cannot mutate the native session.
    grid.set('reaction_intensity', grid.get('reaction_intensity') * 0.9)
    grid.run_remaining()
    print('Injected Surface spread (ft/min):', grid.get('spread_rate'))
    grid.run_fused()  # Restart discards transient injections; persistent overrides survive.
    print('Original Surface spread (ft/min):', grid.get('spread_rate'))

    crown = CrownGrid(dict(weather, canopy_base_height=3., canopy_height=30., canopy_bulk_density=0.02,
                           moisture_foliar=100., surface_ros=5., surface_fli=1000., surface_hpua=1000.),
                      shape=(2, 3), keep=('fire_type',), threads=4)
    crown.run_stage('crown_fraction')
    crown.set('crown_fraction_burned', np.full((2, 3), 0.5))
    crown.run_remaining()
    print('Injected Crown final spread (ft/min):', crown.get('final_spread_rate'))


if __name__ == '__main__':
    main()
