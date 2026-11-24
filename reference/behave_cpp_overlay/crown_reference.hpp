#ifndef BEHAVE_CPP_OVERLAY_CROWN_REFERENCE_HPP
#define BEHAVE_CPP_OVERLAY_CROWN_REFERENCE_HPP

#include "crown.h"
#include "fuelModels.h"
#include "surface.h"

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace crown_reference {

struct Request {
    double wind_fpm;
    double canopy_base_ft;
    double crown_ratio;
    FireType::FireTypeEnum expected_type;
};

namespace detail {

// Fixed inputs shared by the six reviewed Scott and Reinhardt cases.
constexpr int fuel_model = 10;
constexpr std::array<double, 5> moisture = {{6, 7, 8, 60, 90}};
constexpr double foliar = 100;
constexpr double canopy_height = 30;
constexpr double canopy_cover = 50;
constexpr double density = 0.02;
constexpr double direction = 0;
constexpr double slope = 0;
constexpr double aspect = 0;
constexpr double waf = 0.4;

constexpr auto area_heat = HeatPerUnitAreaUnits::BtusPerSquareFoot;
constexpr auto density_units = DensityUnits::PoundsPerCubicFoot;
constexpr auto feet = LengthUnits::Feet;
constexpr auto fpm = SpeedUnits::FeetPerMinute;
constexpr auto fraction = FractionUnits::Fraction;
constexpr auto intensity = FirelineIntensityUnits::BtusPerFootPerSecond;
constexpr auto percent = FractionUnits::Percent;
constexpr auto twenty = WindHeightInputMode::TwentyFoot;
constexpr auto upslope = WindAndSpreadOrientationMode::RelativeToUpslope;
constexpr auto waf_mode = WindAdjustmentFactorCalculationMethod::UserInput;

struct SurfaceReadback {
    double reaction;
    double sink;
    double heat;
    double ros;
    double no_wind_ros;
    double fireline;
    double flame;
    double midflame_wind;
    double length_to_width;
};

struct Inversion {
    double critical_ros;
    double threshold_wind;
};

// Stage dependencies determine the order of these private guard functions.

/// Stop immediately when a reviewed assumption fails.
///
/// :param condition: Whether the guarded condition holds.
/// :param reason: Condition identifier for the protocol error.
/// :return: Nothing. Throws on failure.
inline void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(std::string("crown_guard_") + reason);
}

/// Require an intermediate to be safely inside binary64 arithmetic limits.
///
/// :param value: Positive intermediate for the reviewed cases.
/// :param reason: Condition identifier for the protocol error.
/// :return: The checked value.
inline double positive(double value, const char* reason) {
    check(std::isfinite(value) && value >= 1e-12 && value <= 1e16, reason);
    return value;
}

/// Check the public surface inputs that both upstream models expose.
///
/// :param model: Configured Surface or Crown object.
/// :param request: Reviewed Crown input.
/// :param wind: Expected twenty-foot wind in feet per minute.
/// :return: Nothing. Throws on a mismatch.
template<class Model>
void check_common_inputs(const Model& model, const Request& request, double wind) {
    const std::array<double, 5> actual_moisture = {{
        model.getMoistureOneHour(fraction), model.getMoistureTenHour(fraction),
        model.getMoistureHundredHour(fraction), model.getMoistureLiveHerbaceous(fraction),
        model.getMoistureLiveWoody(fraction),
    }};
    const std::array<double, 5> expected_moisture = {{
        moisture[0] / 100, moisture[1] / 100, moisture[2] / 100, moisture[3] / 100, moisture[4] / 100,
    }};
    check(model.getFuelModelNumber() == fuel_model && actual_moisture == expected_moisture &&
          model.getCanopyHeight(feet) == canopy_height && model.getCanopyCover(fraction) == canopy_cover / 100 &&
          model.getCrownRatio(fraction) == request.crown_ratio && model.getAspect() == aspect &&
          model.getWindDirection() == direction && model.getSlope(SlopeUnits::Degrees) == slope &&
          model.getWindSpeed(fpm, twenty) == wind, "input_readback");
}

/// Confirm the denominators and radicand used by Crown's internal geometry.
///
/// :param wind: Recovered twenty-foot wind in feet per minute.
/// :param active_ros: Derived Crown spread rate in feet per minute.
/// :return: Guarded Crown length-to-width ratio.
inline double check_geometry(double wind, double active_ros) {
    positive(active_ros, "geometry_ros");
    // Crown supplies its ft/min readback with an mph tag. FireSize retains the
    // number, then applies its ft/min-to-mph conversion for length-to-width.
    const bool calm = wind == 0;
    const double length_to_width = 1.0 + 0.125 * (wind * 0.01136363636);
    check(calm ? length_to_width == 1 : length_to_width > 1 && length_to_width < (wind == 2200 ? 4.2 : 3),
          "geometry_length_to_width");
    const double radicand = length_to_width * length_to_width - 1;
    if (calm) check(radicand == 0, "geometry_radicand");
    else positive(radicand, "geometry_radicand");
    const double root = std::sqrt(radicand);
    const double eccentricity = root / length_to_width;
    check(calm ? eccentricity == 0 : eccentricity > 0 && eccentricity < 1, "geometry_eccentricity");
    const double backing = positive(active_ros * (1 - eccentricity) / (1 + eccentricity), "geometry_backing");
    const double axis_b = positive((active_ros + backing) / 2, "geometry_axis_b");
    positive((backing + active_ros) / length_to_width * 0.5, "geometry_flanking");
    positive(axis_b / length_to_width, "geometry_axis_a");
    if (calm) check(axis_b == backing, "geometry_axis_c");
    else positive(axis_b - backing, "geometry_axis_c");
    const double heading_divisor = positive(length_to_width - root, "geometry_heading_divisor");
    positive((length_to_width + root) / heading_divisor, "geometry_heading_ratio");
    return length_to_width;
}

/// Check the reviewed branch before reproducing its crown fraction calculation.
///
/// :param type: Expected upstream fire classification.
/// :param numerator: Actual surface spread minus critical surface spread.
/// :param denominator: Crowning surface spread minus critical surface spread.
/// :return: Guarded crown fraction burned.
inline double checked_fraction(FireType::FireTypeEnum type, double numerator, double denominator) {
    check(std::isfinite(numerator) && std::isfinite(denominator), "cfb_terms");
    if (type == FireType::Torching) {
        check(denominator > 1e-7 && numerator > 0 && numerator < denominator, "torch_cfb_domain");
    } else if (type == FireType::Crowning) {
        check(denominator < 0, "active_cfb_domain");
    }
    const double raw = denominator > 1e-7 ? numerator / denominator : 0;
    check(std::isfinite(raw), "cfb_division");
    if (type == FireType::Torching) check(raw > 0 && raw < 1, "torch_cfb_division");
    return std::fmax(0, std::fmin(raw, 1));
}

/// Guard the FM10 wind inversion before the threshold Surface calculation.
///
/// :param surface: Readbacks from the flat FM10 Surface replay.
/// :return: Checked critical spread rate and threshold wind.
inline Inversion invert(const SurfaceReadback& surface) {
    const double metric_density = positive(16.0185 * positive(density, "bulk_density"), "converted_bulk_density");
    const double critical_ros = positive(3.28084 * (3 / metric_density), "critical_ros");
    const double target_ros = positive(critical_ros / 3.34, "target_fm10_ros");
    // These coefficients are specific to the pinned, reviewed FM10 inversion.
    const double flux = 0.048317062998571636;
    const double ros0 = positive(positive(surface.reaction, "reaction") * flux / positive(surface.sink, "sink"), "ros0");
    const double argument = positive(((target_ros / ros0) - 1) / 0.0016102128596515481, "inversion_argument");
    const double exponent = 1.0 / 1.4308256324729873;
    check(std::isfinite(exponent) && exponent > 0 && exponent <= 1, "inversion_exponent");
    const double midflame = positive(std::pow(argument, exponent), "derived_midflame");
    const double threshold_wind = midflame / waf;
    check(std::isfinite(threshold_wind) && threshold_wind > 0 && threshold_wind <= 40 * 88, "threshold_wind_domain");
    return {critical_ros, threshold_wind};
}

/// Verify the geometry and arithmetic of the expected full Crown branch.
///
/// :param request: Reviewed Crown input.
/// :param surface: External Surface readbacks at the reviewed wind.
/// :param fm10: Flat FM10 Surface readbacks at the reviewed wind.
/// :param inversion: Checked wind inversion.
/// :param threshold_ros: Surface spread at the inversion wind.
/// :return: Guarded Crown length-to-width ratio.
inline double guard_downstream(const Request& request, const SurfaceReadback& surface, const SurfaceReadback& fm10,
                               const Inversion& inversion, double threshold_ros) {
    const double heat = positive(surface.heat, "surface_hpua");
    const double base_height_m = request.canopy_base_ft * 0.3048;
    // Crown floors the canopy base only for the transition intensity.
    const double critical_base_height_m = base_height_m < 0.1 ? 0.1 : base_height_m;
    const double critical_argument =
        positive(0.010 * critical_base_height_m * (460.0 + 25.9 * foliar), "critical_argument");
    const double critical_intensity = positive(std::pow(critical_argument, 1.5) * 0.2886719, "critical_intensity");
    const double critical_ros = positive(60 * critical_intensity / heat, "critical_surface_ros");
    const double active_ros = positive(3.34 * fm10.ros, "active_ros");
    const double canopy_heat = positive(density * (canopy_height - request.canopy_base_ft) * 8000, "canopy_heat");
    const double active_intensity = positive(active_ros / 60 * positive(heat + canopy_heat, "active_heat"), "active_intensity");
    const double active_ratio = positive(active_ros / inversion.critical_ros, "active_ratio");
    const double transition_ratio = positive(surface.fireline / critical_intensity, "transition_ratio");
    const auto type = request.expected_type;
    const bool active_spread = type == FireType::Crowning || type == FireType::ConditionalCrownFire;
    const bool transition = type == FireType::Crowning || type == FireType::Torching;
    check((active_ratio >= 1) == active_spread && (transition_ratio >= 1) == transition, "candidate_branch");

    const double numerator = surface.ros - critical_ros;
    const double denominator = positive(threshold_ros, "threshold_ros") - critical_ros;
    const double cfb = checked_fraction(type, numerator, denominator);
    const double passive_ros = positive(surface.ros + cfb * (active_ros - surface.ros), "passive_ros");
    const double passive_heat = positive(heat + canopy_heat * cfb, "passive_heat");
    const double passive_intensity = positive(passive_heat * passive_ros / 60, "passive_intensity");
    positive(0.2 * std::pow(active_intensity, 2.0 / 3.0), "active_flame");
    positive(0.2 * std::pow(passive_intensity, 2.0 / 3.0), "passive_flame");
    return check_geometry(request.wind_fpm, active_ros);
}

/// Set the fixed modes shared by both upstream models before input updates.
///
/// :param model: Fresh Surface or Crown object.
/// :return: Nothing. Sets the moisture and wind adjustment modes.
template<class Model>
void set_modes(Model& model) {
    model.setMoistureInputMode(MoistureInputMode::BySizeClass);
    model.setUserProvidedWindAdjustmentFactor(waf);
    model.setWindAdjustmentFactorCalculationMethod(waf_mode);
}

/// Run one Surface stage after checking its input readbacks and modes.
///
/// :param surface: Configured Surface object.
/// :param request: Reviewed Crown input.
/// :param wind: Expected twenty-foot wind in feet per minute.
/// :return: Checked Surface readbacks used by subsequent stages.
inline SurfaceReadback run_surface(Surface& surface, const Request& request, double wind) {
    check_common_inputs(surface, request, wind);
    check(surface.getMoistureInputMode() == MoistureInputMode::BySizeClass && surface.getWindHeightInputMode() == twenty &&
          surface.getWindAndSpreadOrientationMode() == upslope &&
          surface.getWindAdjustmentFactorCalculationMethod() == waf_mode &&
          !surface.isUsingTwoFuelModels() && !surface.getIsUsingChaparral() && !surface.getIsUsingWesternAspen() &&
          !surface.getIsUsingPalmettoGallberry(), "surface_modes");
    check(surface.getMidflameWindspeed(fpm) == wind * waf, "midflame_wind");
    surface.doSurfaceRunInDirectionOfMaxSpread();
    using HeatSource = HeatSourceAndReactionIntensityUnits;
    const SurfaceReadback result = {
        positive(surface.getReactionIntensity(HeatSource::BtusPerSquareFootPerMinute), "reaction"),
        positive(surface.getHeatSink(HeatSinkUnits::BtusPerCubicFoot), "sink"),
        positive(surface.getHeatPerUnitArea(area_heat), "surface_hpua"),
        positive(surface.getSpreadRate(fpm), "surface_ros"),
        positive(surface.getNoWindSpreadRate(fpm), "no_wind_ros"),
        positive(surface.getFirelineIntensity(intensity), "surface_fireline"),
        positive(surface.getFlameLength(feet), "surface_flame"),
        surface.getMidflameWindspeed(fpm),
        positive(surface.getFireLengthToWidthRatio(), "surface_length_to_width"),
    };
    check(result.midflame_wind == wind * waf, "midflame_after_run");
    return result;
}

} // namespace detail

/// Configure and run the reviewed Scott and Reinhardt calculation.
///
/// :param crown: Fresh Crown object.
/// :param request: Reviewed Crown input whose preflight passed.
/// :return: Nothing. Calculates upstream Crown outputs.
inline void calculate(Crown& crown, const Request& request) {
    using namespace detail;
    set_modes(crown);
    crown.updateCrownInputs(fuel_model, moisture[0], moisture[1], moisture[2], moisture[3], moisture[4], foliar, percent,
        request.wind_fpm, fpm, twenty, direction, upslope, slope, SlopeUnits::Degrees, aspect, canopy_cover, percent,
        canopy_height, request.canopy_base_ft, feet, request.crown_ratio, fraction, density, density_units);
    check_common_inputs(crown, request, request.wind_fpm);
    check(crown.getMoistureFoliar(fraction) == foliar / 100 &&
          crown.getCanopyBaseHeight(feet) == request.canopy_base_ft &&
          crown.getCanopyBulkDensity(density_units) == density, "crown_only_inputs");
    crown.doCrownRunScottAndReinhardt();
}

/// Check the exact input's staged Surface replay before entering the full Crown calculation.
///
/// :param fuels: Fresh upstream fuel model table.
/// :param request: Reviewed Crown input.
/// :return: Guarded geometry readback expected after the full Crown calculation.
inline double preflight(FuelModels& fuels, const Request& request) {
    using namespace detail;
    check(fuels.isFuelModelDefined(fuel_model) && !fuels.isAllFuelLoadZero(fuel_model), "nonburnable");
    Surface external(fuels);
    set_modes(external);
    external.updateSurfaceInputs(fuel_model, moisture[0], moisture[1], moisture[2], moisture[3], moisture[4],
        percent, request.wind_fpm, fpm, twenty, direction, upslope, slope, SlopeUnits::Degrees, aspect,
        canopy_cover, percent, canopy_height, feet, request.crown_ratio, fraction);
    const double recovered = external.getWindSpeed(fpm, twenty);
    check(recovered == request.wind_fpm, "initial_twenty_foot_wind");
    external.setWindSpeed(recovered, fpm, twenty);
    const SurfaceReadback surface = run_surface(external, request, recovered);

    Surface fm10(fuels);
    fm10 = external;
    fm10.setFuelModelNumber(fuel_model);
    fm10.setWindAdjustmentFactorCalculationMethod(waf_mode);
    fm10.setUserProvidedWindAdjustmentFactor(waf);
    fm10.setSlope(slope, SlopeUnits::Degrees);
    fm10.setWindAndSpreadOrientationMode(upslope);
    fm10.setWindDirection(direction);
    fm10.setWindSpeed(recovered, fpm, twenty);
    const SurfaceReadback flat = run_surface(fm10, request, recovered);
    check(flat.reaction == surface.reaction && flat.sink == surface.sink && flat.heat == surface.heat &&
          flat.ros == surface.ros && flat.no_wind_ros == surface.no_wind_ros && flat.fireline == surface.fireline &&
          flat.flame == surface.flame && flat.midflame_wind == surface.midflame_wind &&
          flat.length_to_width == surface.length_to_width, "independent_flat_fm10_replay");

    const Inversion inversion = invert(flat);
    const Surface snapshot = external;
    external.setWindSpeed(inversion.threshold_wind, fpm, twenty);
    const SurfaceReadback raised = run_surface(external, request, inversion.threshold_wind);
    external = snapshot;
    check_common_inputs(external, request, recovered);
    check(external.getSpreadRate(fpm) == surface.ros && external.getMidflameWindspeed(fpm) == surface.midflame_wind &&
          external.getHeatPerUnitArea(area_heat) == surface.heat &&
          external.getFirelineIntensity(intensity) == surface.fireline && external.getFlameLength(feet) == surface.flame &&
          external.getFireLengthToWidthRatio() == surface.length_to_width, "snapshot_restore");
    return guard_downstream(request, surface, flat, inversion, raised.ros);
}

/// Read the selected output fields after the guarded full Crown calculation.
///
/// :param crown: Crown object after the Scott and Reinhardt calculation.
/// :param request: Reviewed Crown input.
/// :param expected_length_to_width: Guarded length-to-width ratio from preflight.
/// :return: Crown fields for the driver's JSON encoder.
inline std::map<std::string, double> read_result(const Crown& crown, const Request& request, double expected_length_to_width) {
    using namespace detail;
    const auto fire_type = crown.getFireType();
    check(fire_type == request.expected_type, "output_fire_type");
    const double length_to_width = crown.getCrownFireLengthToWidthRatio();
    check(length_to_width == expected_length_to_width, "output_length_to_width");
    return {
        {"crown_fireline_btu_ft_s", crown.getCrownFirelineIntensity(intensity)},
        {"crown_flame_ft", crown.getCrownFlameLength(feet)},
        {"crown_fraction_burned", crown.getCrownFractionBurned()},
        {"crown_length_to_width_ratio", length_to_width},
        {"crown_ros_ft_min", crown.getCrownFireSpreadRate(fpm)},
        {"final_fireline_btu_ft_s", crown.getFinalFirelineIntesity(intensity)},
        {"final_flame_ft", crown.getFinalFlameLength(feet)},
        {"final_hpua_btu_ft2", crown.getFinalHeatPerUnitArea(area_heat)},
        {"final_ros_ft_min", crown.getFinalSpreadRate(fpm)},
        {"fire_type", fire_type},
        {"surface_ros_ft_min", crown.getSurfaceFireSpreadRate(fpm)},
    };
}

} // namespace crown_reference

#endif
