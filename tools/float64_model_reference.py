"""Independent Python equations with the reviewed float64/NaN model contract.

This module is a correctness oracle only. Timed Python calls must receive the
original caller data; widening here must never replace that timing baseline.
"""
import numpy as np


def _ratio(numerator, denominator):
    """Apply a three-valued denominator guard without treating NaN as zero.

    :param numerator: Actual numerator values.
    :param denominator: Actual critical values.
    :return: Ratios with the finite threshold rule and unavailable denominators.
    """
    denominator = np.asarray(denominator)
    return np.where(np.isnan(denominator), np.nan,
                    np.where(denominator > 1e-7, numerator / denominator, 0.))


def apply_crown_rules(outputs, supplied, *, density=None):
    """Apply reviewed missing thresholds and selected-branch rules to Python science.

    :param outputs: Complete Python Crown output mapping on widened inputs.
    :param supplied: Actual widened and NaN-filled Surface fields.
    :param density: Optional actual canopy density to distinguish missing from finite zero.
    :return: Independent output arrays using float64 physical values and int32 codes.
    """
    from behave_fire.components import crown

    result = {name: np.array(value, dtype=np.int32 if name == 'fire_type' else np.float64, copy=True)
              for name, value in outputs.items()}
    critical = result['crown_critical_surface_fire_line_intensity']
    active_critical = result['crown_critical_fire_spread_rate']
    if density is not None:
        active_critical = np.where(np.isnan(density), np.nan, active_critical)
    result['crown_critical_fire_spread_rate'] = active_critical
    # An unavailable spatial rate has no defined effective wind or crown shape.
    result['crown_length_to_width_ratio'] = np.where(np.isnan(result['crown_fire_spread_rate']),
        np.nan, result['crown_length_to_width_ratio'])
    transition = _ratio(supplied['fireline_intensity'], critical)
    active = _ratio(result['crown_fire_spread_rate'], active_critical)
    result['crown_fire_transition_ratio'] = transition
    result['crown_fire_active_ratio'] = active
    classified = crown.classify_crown_fire_type(transition, active)
    code = np.where(np.isnan(transition) | np.isnan(active), -1, classified['fire_type']).astype(np.int32)
    result['fire_type'] = code
    surface_critical = crown.calculate_surface_fire_critical_spread_rate(critical, supplied['heat_per_unit_area'])
    surface_critical = np.where(np.isnan(supplied['heat_per_unit_area']), np.nan, surface_critical)
    result['surface_fire_critical_spread_rate'] = surface_critical
    crowning = np.where(np.isnan(active_critical), np.nan, result['crowning_surface_fire_spread_rate'])
    result['crowning_surface_fire_spread_rate'] = crowning
    fraction = crown.calculate_crown_fraction_burned(supplied['spread_rate'], surface_critical, crowning)
    fraction = np.where(np.isnan(crowning - surface_critical), np.nan, fraction)
    result['crown_fraction_burned'] = fraction
    passive = crown.calculate_passive_crown_fire_behavior(supplied['spread_rate'], supplied['heat_per_unit_area'],
        result['crown_fire_spread_rate'], result['canopy_heat_per_unit_area'], fraction)
    passive['passive_crown_fire_flame_length'] = np.where(
        np.isnan(passive['passive_crown_fire_line_intensity']), np.nan, passive['passive_crown_fire_flame_length'])
    result.update(passive)
    result['crown_flame_length'] = np.where(np.isnan(result['crown_fire_line_intensity']),
                                           np.nan, result['crown_flame_length'])
    with np.errstate(invalid='ignore'):
        surface_flame = np.where(np.isnan(supplied['fireline_intensity']), np.nan,
            np.where(supplied['fireline_intensity'] > 0., .2 * supplied['fireline_intensity'] ** (2. / 3.), 0.))
    for name, surface, passive_value, active_value in [
        ('final_spread_rate', supplied['spread_rate'], result['passive_crown_fire_spread_rate'], result['crown_fire_spread_rate']),
        ('final_heat_per_unit_area', supplied['heat_per_unit_area'], result['passive_crown_fire_heat_per_unit_area'], result['crown_fire_heat_per_unit_area']),
        ('final_fireline_intensity', supplied['fireline_intensity'], result['passive_crown_fire_line_intensity'], result['crown_fire_line_intensity']),
        ('final_flame_length', surface_flame, result['passive_crown_fire_flame_length'], result['crown_flame_length']),
    ]:
        result[name] = np.where(code < 0, np.nan, np.where(np.isin(code, [0, 2]), surface,
            np.where(code == 1, passive_value, active_value)))
    return result


def model_reference(function, arguments):
    """Run explicit Python on widened inputs, then apply the reviewed missing policy.

    :param function: Explicit Python model calculation callable.
    :param arguments: Original model keyword operands.
    :return: Complete independent normalized model reference.
    """
    arguments = normalize_inputs(arguments)
    with np.errstate(all='ignore'):
        result = function(**arguments)
        if 'surface_results' in arguments:
            result = apply_crown_rules(result, arguments['surface_results'],
                                       density=arguments.get('canopy_bulk_density', 0.))
    return result


def normalize_inputs(arguments):
    """Widen real physical operands while preserving exact selectors and lookup roles.

    :param arguments: Actual model keyword arguments, including nested supplied outputs.
    :return: Independent normalized arguments for explicit Python correctness calls.
    """
    result = {}
    controls = {'fuel_model_grid', 'validity', 'wind_orientation_mode', 'wind_height_mode', 'waf_method',
                'lut', 'out_units', 'crown_ratio_units'}
    for name, value in arguments.items():
        if name in controls or name.endswith(('units', '_grid', '_i')) or name in ('charge_type', 'max_steps') or value is None:
            result[name] = value
        elif name == 'surface_results':
            result[name] = {key: np.asarray(np.ma.asarray(item, dtype=np.float64).filled(np.nan))
                            for key, item in value.items()}
        elif name.startswith('_'):
            result[name] = value
        else:
            result[name] = np.asarray(np.ma.asarray(value, dtype=np.float64).filled(np.nan))
    return result
