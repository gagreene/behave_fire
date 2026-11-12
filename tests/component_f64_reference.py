"""Independent widened Python references and explicit component NoData contracts."""
import numpy as np


def _surface_missing(outputs, intermediates):
    """Apply reviewed unknown-denominator/ellipse rules to existing Python output."""
    missing = np.isnan(intermediates['heat_sink']) | np.isnan(outputs['no_wind_no_slope_spread_rate'])
    for field in ('no_wind_no_slope_spread_rate', 'spread_rate', 'backing_spread_rate', 'flanking_spread_rate',
                  'fireline_intensity', 'flame_length', 'direction_of_max_spread'):
        outputs[field] = np.where(missing, np.nan, outputs[field])
    from behave_fire.components.surface import calculate_surface_wind_coefficients
    coefficients = calculate_surface_wind_coefficients(intermediates['sigma'])
    r0 = outputs['no_wind_no_slope_spread_rate']
    phi = np.where(np.isnan(r0), np.nan, np.where(r0>1e-7, outputs['spread_rate']/r0-1., 0.))
    unknown = _unknown_all_above([(phi,0.),(coefficients['wind_b'],1e-7),
                                (coefficients['wind_c'],1e-7),(intermediates['relative_packing_ratio'],1e-7)])
    for field in ('effective_wind_speed','fire_length_to_width_ratio','eccentricity','backing_spread_rate','flanking_spread_rate'):
        outputs[field] = np.where(unknown,np.nan,outputs[field])
    return outputs


def _unknown_all_above(values):
    """Return unknown conjunctions, allowing an independent known false condition."""
    unknown, inactive = False, False
    for value, limit in values:
        value = np.asarray(value)
        unknown = unknown | np.isnan(value)
        inactive = inactive | (value <= limit)
    return unknown & ~inactive


def normalize(arguments):
    """Copy supplied real physical values at float64 while retaining control roles."""
    controls = {'S', 'lut', 'is_crown', 'is_surface_fire', 'is_passive_crown', 'is_active_crown',
                'is_defined', 'is_dynamic', 'charge_type', 'species_master_table'}
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
        'calculate_residence_time': 'sigma',
        'calculate_no_wind_no_slope_spread_rate': 'heat_sink',
        'calculate_scorch_height': 'fireline_intensity_btu_ft_s' if 'fireline_intensity_btu_ft_s' in args else 'fireline_intensity',
    }
    if name in scalar_conditions:
        result = np.where(nan(args[scalar_conditions[name]]), np.nan, result)
    if name == 'calculate_surface_wind_coefficients':
        result = {field: np.where(nan(args['sigma']), np.nan, value) for field, value in result.items()}
    if name == 'calculate_wind_phi':
        active = (args['sigma'] > 1e-7) & (args['midflame_wind_speed'] > 1e-7)
        missing = _unknown_all_above([(args['sigma'], 1e-7), (args['midflame_wind_speed'], 1e-7)])
        result = np.where(missing | (active & nan(args['relative_packing_ratio'])), np.nan, result)
    if name == 'calculate_slope_phi':
        from behave_fire.components.behave_units import SlopeUnits

        slope = np.tan(np.radians(SlopeUnits.toBaseUnits(args['slope'], args['slope_units'])))
        result = np.where(_unknown_all_above([(slope,1e-7),(args['packing_ratio'],1e-7)]), np.nan, result)
    if name == 'calculate_effective_wind_speed':
        r0 = args['no_wind_no_slope_spread_rate']
        phi = np.where(nan(r0),np.nan,np.where(r0>1e-7,args['spread_rate']/r0-1.,0.))
        missing = _unknown_all_above([(phi,0.)]+[(args[key],1e-7) for key in ('wind_b','wind_c','relative_packing_ratio')])
        result = np.where(missing, np.nan, result)
    if name in ('calculate_wind_adjustment_factor', 'calculate_surface_wind_adjustment_factor'):
        if str(args.get('waf_method','')).replace('_','').lower() != 'userinput' or args.get('user_waf') is None:
            cover,height,ratio,depth = (args[key] for key in ('canopy_cover','canopy_height','crown_ratio','depth'))
            fraction = cover*ratio/3.
            opened = (cover<1e-7)|(fraction<.05)|(height<6.)
            missing = (~opened & (nan(cover)|nan(fraction)|nan(height))) | (opened & nan(depth)) | (~opened & np.isinf(height))
            result = np.where(missing,np.nan,result)
    if name == 'calculate_safety_zone':
        core = args['number_of_personnel']*args['area_per_person_sqft']+args['number_of_equipment']*args['area_per_equipment_sqft']
        for key in ('radius','area'):
            result[key] = np.where(nan(core),np.nan,result[key])
    if name.startswith('calculate_spotting_from_'):
        wind = args.get('wind_mph', args.get('wind_speed'))
        first = next(args[key] for key in ('flame_height_ft','flame_height','flame_length_ft','flame_length','dbh_in','dbh') if key in args)
        required = [(wind,1e-7),(first,1e-7)]
        if name == 'calculate_spotting_from_torching_trees':
            required.extend((args[key],0.) for key in ('height_ft','height','count','tree_height','number_of_trees') if key in args)
        result = np.where(_unknown_all_above(required), np.nan, result)
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
        result = _surface_missing(result, args['ib'])
        result['flame_length'] = np.where(nan(result['fireline_intensity']), np.nan, result['flame_length'])
        result['direction_of_max_spread'] = np.where(nan(result['spread_rate']), np.nan, result['direction_of_max_spread'])
    if name == 'calculate_crown_surface_fire':
        result['crown_surface'] = _surface_missing(result['crown_surface'], result['ib_crown'])
        result['crown_length_to_width_ratio'] = np.where(nan(result['crown_surface']['effective_wind_speed']),np.nan,result['crown_length_to_width_ratio'])
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
    return result
