"""Independent widened Python references and explicit component NoData contracts."""
import numpy as np


def normalize(arguments):
    """Copy supplied real physical values at float64 while retaining control roles."""
    controls = {'S', 'lut', 'is_crown', 'is_surface_fire', 'is_passive_crown', 'is_active_crown',
                'is_defined', 'is_dynamic', 'charge_type', 'max_steps', 'species_master_table'}
    result = {}
    for name, value in arguments.items():
        if name in controls or name.endswith(('units', '_grid', '_i')) or isinstance(value, str) or value is None:
            result[name] = value
        elif isinstance(value, dict):
            result[name] = normalize(value)
        else:
            result[name] = np.asarray(np.ma.asarray(value, dtype=np.float64).filled(np.nan))
    return result


def reference(function, arguments):
    """Evaluate Python on the actual widened inputs, then apply reviewed missing rules."""
    from behave_fire.components import crown

    args = normalize(arguments)
    result = function(**args)
    name = function.__name__
    nan = lambda value: np.isnan(np.asarray(value, dtype=np.float64))
    scalar_conditions = {
        'calculate_eccentricity': 'fire_length_to_width_ratio',
        'calculate_fire_length_to_width_ratio': 'effective_wind_speed',
        'calculate_flame_length': 'fireline_intensity',
        'calculate_critical_crown_fire_spread_rate': 'canopy_bulk_density',
        'calculate_surface_fire_critical_spread_rate': 'surface_hpua',
        'calculate_scorch_height': 'fireline_intensity_btu_ft_s' if 'fireline_intensity_btu_ft_s' in args else 'fireline_intensity',
    }
    if name in scalar_conditions:
        result = np.where(nan(args[scalar_conditions[name]]), np.nan, result)
    if name == 'calculate_crown_fire_ratios':
        for field, denominator in [('crown_fire_transition_ratio', 'critical_surface_fli'), ('crown_fire_active_ratio', 'critical_crown_ros')]:
            result[field] = np.where(nan(args[denominator]), np.nan, result[field])
    if name == 'classify_crown_fire_type':
        missing = nan(args['transition_ratio']) | nan(args['active_ratio'])
        result['fire_type'] = np.where(missing, -1, result['fire_type']).astype(np.int32)
        for key in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
            result[key] = np.where(missing, False, result[key])
    if name == 'calculate_crown_fraction_burned':
        missing = nan(args['crowning_surface_ros'] - args['surface_critical_ros'])
        result = np.where(missing, np.nan, result)
    if name in ('calculate_crown_heat_and_intensity', 'calculate_passive_crown_fire_behavior'):
        intensity, flame = ('crown_fire_line_intensity', 'crown_flame_length') if name == 'calculate_crown_heat_and_intensity' else (
            'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length')
        result[flame] = np.where(nan(result[intensity]), np.nan, result[flame])
    if name == 'run_surface_fire':
        result['flame_length'] = np.where(nan(result['fireline_intensity']), np.nan, result['flame_length'])
        result['direction_of_max_spread'] = np.where(nan(result['spread_rate']), np.nan, result['direction_of_max_spread'])
    if name == 'calculate_crown_surface_fire':
        result['crown_surface']['flame_length'] = np.where(nan(result['crown_surface']['fireline_intensity']), np.nan, result['crown_surface']['flame_length'])
        result['crown_surface']['direction_of_max_spread'] = np.where(nan(result['crown_surface']['spread_rate']), np.nan, result['crown_surface']['direction_of_max_spread'])
    if name == 'assign_final_crown_fire_behavior':
        missing = np.asarray(args['is_surface_fire'], bool) & nan(args['surface_fli'])
        result['final_flame_length'] = np.where(missing, np.nan, result['final_flame_length'])
    if name == 'calculate_crown_scorch_mortality':
        missing = nan(args['tree_height_ft'] * args['crown_ratio'])
        for key in ('crown_length_scorch', 'crown_volume_scorch'):
            result[key] = np.where(missing, np.nan, result[key])
        eq = np.clip(np.atleast_1d(args['equation_number_grid']).astype(int), 0, 120)
        known = np.any(args['coeffs'][eq] != 0., axis=-1)
        result['probability_mortality'] = np.where(missing & known, np.nan, result['probability_mortality'])
    if name == 'run_contain_sim_array' and args.get('max_steps', 500) > 0:
        # A NaN spread or infinity producing NaN geometry makes termination unknown.
        missing = ~np.isfinite(np.asarray(args['report_spread_rate']))
        missing &= np.asarray(args['report_spread_rate']) != -np.inf
        for key in result:
            result[key] = np.where(missing, -1 if key == 'status' else np.nan, result[key])
        result['status'] = result['status'].astype(np.int32)
    return result
