// Phase 1 protocol probe. Scientific calculations are called in pinned upstream sources.
#include "fireSize.h"
#include "fuelModels.h"
#include "ignite.h"
#include "surface.h"
#include <cctype>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
struct Value {
    enum Kind { Number, String, Boolean, Null } kind;
    std::string text;
};

// Constructor placement follows C++ idiom; parser helpers precede its public API.
class Parser {
    const std::string& input_;
    std::size_t offset_ = 0;

    char peek() const { return offset_ < input_.size() ? input_[offset_] : '\0'; }
    void space() { while (peek() == ' ' || peek() == '\t' || peek() == '\r' || peek() == '\n') ++offset_; }
    std::string string() {
        if (peek() != '"') throw std::runtime_error("expected_string");
        ++offset_;
        std::string result;
        while (peek() != '"') {
            unsigned char c = static_cast<unsigned char>(peek());
            if (c < 32 || c > 126) throw std::runtime_error("invalid_or_non_ascii_string");
            ++offset_;
            if (c == '\\') {
                char escaped = peek();
                if (escaped != '"' && escaped != '\\' && escaped != '/')
                    throw std::runtime_error("unsupported_string_escape");
                c = static_cast<unsigned char>(escaped);
                ++offset_;
            }
            result.push_back(static_cast<char>(c));
        }
        ++offset_;
        return result;
    }
    Value value() {
        space();
        if (peek() == '"') return {Value::String, string()};
        std::size_t begin = offset_;
        for (const std::string token : {"true", "false", "null"}) {
            if (input_.compare(offset_, token.size(), token) == 0) {
                offset_ += token.size();
                return {token == "null" ? Value::Null : Value::Boolean, token};
            }
        }
        if (peek() == '-') ++offset_;
        if (peek() == '0') ++offset_;
        else {
            if (peek() < '1' || peek() > '9') throw std::runtime_error("expected_json_number");
            while (peek() >= '0' && peek() <= '9') ++offset_;
        }
        if (peek() == '.') {
            ++offset_;
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid_fraction");
            while (peek() >= '0' && peek() <= '9') ++offset_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++offset_;
            if (peek() == '+' || peek() == '-') ++offset_;
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid_exponent");
            while (peek() >= '0' && peek() <= '9') ++offset_;
        }
        return {Value::Number, input_.substr(begin, offset_ - begin)};
    }
public:
    explicit Parser(const std::string& input) : input_(input) {}
    std::map<std::string, Value> parse() {
        space();
        if (peek() != '{') throw std::runtime_error("expected_object");
        ++offset_;
        space();
        std::map<std::string, Value> result;
        if (peek() != '}') {
            while (true) {
                space();
                std::string key = string();
                space();
                if (peek() != ':') throw std::runtime_error("expected_colon");
                ++offset_;
                Value item = value();
                if (!result.emplace(key, item).second) throw std::runtime_error("duplicate_key");
                space();
                if (peek() != ',') break;
                ++offset_;
            }
        }
        if (peek() != '}') throw std::runtime_error("expected_object_end");
        ++offset_;
        space();
        if (offset_ != input_.size()) throw std::runtime_error("trailing_input");
        return result;
    }
};

std::string encode(const std::string& value) {
    std::string result = "\"";
    for (char c : value) {
        if (c == '"' || c == '\\') result.push_back('\\');
        result.push_back(c);
    }
    return result + '"';
}

/// Encode a finite numeric result with the reference driver's JSON formatting.
///
/// :param result: Output fields and their values.
/// :return: JSON object preserving negative zeros.
std::string encode_result(const std::map<std::string, double>& result) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << '{';
    bool first = true;
    for (const auto& item : result) {
        if (!std::isfinite(item.second)) throw std::runtime_error("nonfinite_upstream_output");
        if (!first) output << ',';
        first = false;
        output << encode(item.first) << ':';
        if (item.second == 0.0 && std::signbit(item.second)) output << "-0.0";
        else output << item.second;
    }
    output << '}';
    return output.str();
}

struct UpstreamTrap {};

/// Register a validated request and check the trap before its upstream call.
///
/// :param id: Validated request ID.
/// :param ids: IDs already seen on this invocation.
/// :param forbid: Whether the upstream entry trap is enabled.
/// :param entered: Entry flag to set once the trap is cleared.
/// :return: Nothing. Records the ID and marks upstream entry on success.
void enter_upstream(const std::string& id, std::set<std::string>& ids, bool forbid, bool& entered) {
    if (!ids.insert(id).second) throw std::runtime_error("duplicate_id");
    if (forbid) throw UpstreamTrap{};
    entered = true;
}

double numeric(const Value& value, const std::string& field) {
    if (value.kind == Value::String) {
        if (value.text == "NaN") return std::numeric_limits<double>::quiet_NaN();
        if (value.text == "Inf") return std::numeric_limits<double>::infinity();
        if (value.text == "-Inf") return -std::numeric_limits<double>::infinity();
    }
    if (value.kind != Value::Number) throw std::runtime_error("numeric_type_" + field);
    char* end = nullptr;
    errno = 0;
    const double result = std::strtod(value.text.c_str(), &end);
    if (errno == ERANGE) throw std::runtime_error("number_range_" + field);
    if (!end || *end) throw std::runtime_error("invalid_number_" + field);
    return result;
}

const std::map<int, double> percent_controls = {
    {110,24},{111,24},{150,20},{151,13},{152,15},{153,15},{154,25},
    {155,25},{156,20},{157,35},{158,24},{159,35},{166,26},{167,30},
    {168,35},{169,30},{170,27},{171,40},{172,32},{190,28},{191,25},
    {192,45},{193,26}
};
struct Request { std::string id; int model; double direction; std::map<std::string, double> physical; std::map<std::string,std::string> modes; double user_waf; std::string session; bool percent_control; double moe_percent; };
Request validate(const std::map<std::string, Value>& input) {
    // Deny known hazardous special-mode inputs before any upstream construction.
    // These checks do not enable those out-of-scope operations.
    if (input.count("curing")) {
        const double curing = numeric(input.at("curing"), "curing");
        if (!std::isfinite(curing)) throw std::runtime_error("nonfinite_curing");
        if (curing < 0 || curing > 1) throw std::runtime_error("curing_range");
    }
    for (const auto& field : {std::string("life_state"), std::string("size_class")}) {
        if (input.count(field)) {
            const double value = numeric(input.at(field), field);
            const double maximum = field == "life_state" ? 1 : 4;
            if (!std::isfinite(value) || std::floor(value) != value || value < 0 || value > maximum)
                throw std::runtime_error("chaparral_index_" + field);
        }
    }
    const std::set<std::string> optional = {"wind_height","wind_orientation","waf_method","user_waf","moisture_mode","session","fuel_mode","moe_percent"};
    const std::set<std::string> fields = {"id", "operation", "fuel_model", "wind_direction", "m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed", "slope", "aspect", "canopy_cover", "canopy_height", "crown_ratio", "wind_height","wind_orientation","waf_method","user_waf","moisture_mode","session","fuel_mode","moe_percent"};
    for (const auto& item : input)
        if (!fields.count(item.first)) throw std::runtime_error("unsupported_field");
    for (const auto& field : fields)
        if (!optional.count(field) && !input.count(field)) throw std::runtime_error("missing_field_" + field);
    if (input.at("id").kind != Value::String || input.at("id").text.empty() || input.at("id").text.size() > 64)
        throw std::runtime_error("invalid_id");
    for (unsigned char c : input.at("id").text)
        if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') throw std::runtime_error("invalid_id");
    if (input.at("operation").kind != Value::String || input.at("operation").text != "surface")
        throw std::runtime_error("unsupported_operation");
    const double model = numeric(input.at("fuel_model"), "fuel_model");
    if (!std::isfinite(model) || std::floor(model) != model || model < 0 || model > 255)
        throw std::runtime_error("fuel_model_storage_bounds");
    const double direction = numeric(input.at("wind_direction"), "wind_direction");
    if (!std::isfinite(direction)) throw std::runtime_error("nonfinite_angle");
    if (direction != 0 && direction != 90 && direction != 180 && direction != 270 && direction != 360)
        throw std::runtime_error("angle_not_in_reviewed_grid");
    for (const auto& name : {std::string("fuel_model"), std::string("wind_direction")}) {
        const Value& value = input.at(name);
        if (value.kind != Value::Number || value.text.find_first_of(".eE") != std::string::npos)
            throw std::runtime_error("integer_token_required_" + name);
    }
    std::map<std::string, double> physical;
    const std::map<std::string, double> maximum = {{"m1h",300},{"m10h",300},{"m100h",300},{"mlh",300},{"mlw",300},{"wind_speed",40},{"slope",60},{"aspect",360},{"canopy_cover",100},{"canopy_height",200},{"crown_ratio",1}};
    for (const auto& bound : maximum) {
        const double value = numeric(input.at(bound.first), bound.first);
        if (!std::isfinite(value) || value < 0 || value > bound.second)
            throw std::runtime_error("physical_domain_" + bound.first);
        if (bound.first == "aspect" && value != 0 && value != 90 && value != 180 && value != 270 && value != 360)
            throw std::runtime_error("aspect_not_in_reviewed_grid");
        physical.emplace(bound.first,value);
    }
    std::map<std::string,std::string> modes = {{"wind_height","TwentyFoot"},{"wind_orientation","North"},{"waf_method","CrownRatio"},{"moisture_mode","BySizeClass"}};
    const std::map<std::string,std::set<std::string>> choices = {
        {"wind_height",{"TwentyFoot","TenMeter"}}, {"wind_orientation",{"North","Upslope"}},
        {"waf_method",{"CrownRatio","UserInput"}}, {"moisture_mode",{"BySizeClass"}}};
    for (const auto& choice : choices) {
        if (input.count(choice.first)) {
            const Value& value = input.at(choice.first);
            if (value.kind != Value::String || !choice.second.count(value.text))
                throw std::runtime_error("unsupported_mode_" + choice.first);
            modes[choice.first] = value.text;
        }
    }
    double user_waf = 0.0;
    if (modes.at("waf_method") == "UserInput") {
        if (!input.count("user_waf")) throw std::runtime_error("missing_user_waf");
        user_waf = numeric(input.at("user_waf"), "user_waf");
        if (!std::isfinite(user_waf) || user_waf < 0 || user_waf > 1) throw std::runtime_error("user_waf_range");
    } else if (input.count("user_waf")) throw std::runtime_error("unused_user_waf");
    std::string session;
    if (input.count("session")) {
        const Value& value = input.at("session");
        if (value.kind != Value::String || value.text.empty() || value.text.size() > 32) throw std::runtime_error("invalid_session");
        for (unsigned char c : value.text) if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') throw std::runtime_error("invalid_session");
        session = value.text;
    }
    bool percent_control = false;
    if (input.count("fuel_mode")) {
        const Value& mode = input.at("fuel_mode");
        if (mode.kind != Value::String || (mode.text != "numbered" && mode.text != "percent_unit_custom"))
            throw std::runtime_error("unsupported_fuel_mode");
        percent_control = mode.text == "percent_unit_custom";
    }
    double moe_percent = 0.0;
    if (percent_control) {
        const auto expected = percent_controls.find(static_cast<int>(model));
        if (expected == percent_controls.end() || !input.count("moe_percent"))
            throw std::runtime_error("unreviewed_percent_control");
        moe_percent = numeric(input.at("moe_percent"), "moe_percent");
        if (!std::isfinite(moe_percent) || moe_percent != expected->second)
            throw std::runtime_error("unreviewed_percent_quantity");
    } else if (input.count("moe_percent")) throw std::runtime_error("unused_custom_moe");
    return {input.at("id").text, static_cast<int>(model), direction, physical, modes, user_waf, session, percent_control, moe_percent};
}

struct FireSizeRequest {
    bool crown;
    double wind_mph;
    double spread_ft_min;
    double elapsed_min;
};

/// Validate a standalone request against the three reviewed FireSize inputs.
///
/// :param input: Parsed request object.
/// :return: The matching reviewed input for a fresh FireSize object.
FireSizeRequest validate_fire_size(const std::map<std::string, Value>& input) {
    const std::set<std::string> fields = {"id", "operation", "mode", "wind_mph", "spread_ft_min", "elapsed_min"};
    for (const auto& item : input)
        if (!fields.count(item.first)) throw std::runtime_error("unsupported_field");
    for (const auto& field : fields)
        if (!input.count(field)) throw std::runtime_error("missing_field_" + field);

    const Value& id = input.at("id");
    if (id.kind != Value::String || id.text.empty()) throw std::runtime_error("invalid_id");
    const Value& mode = input.at("mode");
    if (mode.kind != Value::String || (mode.text != "Crown" && mode.text != "Surface"))
        throw std::runtime_error("unsupported_fire_size_mode");
    for (const auto& field : {"elapsed_min", "spread_ft_min", "wind_mph"})
        if (input.at(field).kind != Value::Number) throw std::runtime_error(std::string("numeric_type_") + field);

    const FireSizeRequest request = {
        mode.text == "Crown",
        numeric(input.at("wind_mph"), "wind_mph"),
        numeric(input.at("spread_ft_min"), "spread_ft_min"),
        numeric(input.at("elapsed_min"), "elapsed_min"),
    };
    const std::map<std::string, FireSizeRequest> reviewed = {
        {"calm-crown", {true, 0.0, 4.0, 2.0}},
        {"surface-5-mph", {false, 5.0, 10.0, 10.0}},
        {"zero-elapsed", {false, 5.0, 10.0, 0.0}},
    };
    const auto match = reviewed.find(id.text);
    if (match == reviewed.end()) throw std::runtime_error("unreviewed_fire_size_request");
    const auto& expected = match->second;
    if (request.crown != expected.crown || request.wind_mph != expected.wind_mph
            || request.spread_ft_min != expected.spread_ft_min || request.elapsed_min != expected.elapsed_min
            || std::signbit(request.wind_mph) || std::signbit(request.elapsed_min))
        throw std::runtime_error("unreviewed_fire_size_request");
    return expected;
}

struct IgniteRequest {
    double air_temperature_f;
    double sun_shade_fraction;
    double moisture_1h;
};

/// Validate a standalone request against the four reviewed firebrand inputs.
///
/// :param input: Parsed request object.
/// :return: The matching reviewed input for a fresh Ignite object.
IgniteRequest validate_ignite(const std::map<std::string, Value>& input) {
    const std::set<std::string> fields = {
        "id", "operation", "air_temperature_f", "sun_shade_fraction", "moisture_1h",
    };
    for (const auto& item : input)
        if (!fields.count(item.first)) throw std::runtime_error("unsupported_field");
    for (const auto& field : fields)
        if (!input.count(field)) throw std::runtime_error("missing_field_" + field);

    const Value& id = input.at("id");
    if (id.kind != Value::String || id.text.empty()) throw std::runtime_error("invalid_id");
    for (const auto& field : {"air_temperature_f", "moisture_1h", "sun_shade_fraction"})
        if (input.at(field).kind != Value::Number) throw std::runtime_error(std::string("numeric_type_") + field);

    const IgniteRequest request = {
        numeric(input.at("air_temperature_f"), "air_temperature_f"),
        numeric(input.at("sun_shade_fraction"), "sun_shade_fraction"),
        numeric(input.at("moisture_1h"), "moisture_1h"),
    };
    const double air_temperature_f = 71.0;
    const double sun_shade_fraction = 0.5;
    const std::map<std::string, double> reviewed_moisture = {
        {"ignite-dry", 0.0},
        {"ignite-moisture-12-5", 0.125},
        {"ignite-moisture-25", 0.25},
        {"ignite-wet", 0.5},
    };
    const auto match = reviewed_moisture.find(id.text);
    if (match == reviewed_moisture.end() || request.air_temperature_f != air_temperature_f
            || request.sun_shade_fraction != sun_shade_fraction || request.moisture_1h != match->second
            || std::signbit(request.moisture_1h))
        throw std::runtime_error("unreviewed_ignite_request");
    return {air_temperature_f, sun_shade_fraction, match->second};
}

struct Unavailable {};
struct GeometryTrap {};
struct Session {
    FuelModels fuels;
    Surface surface;
    Session() : fuels(), surface(fuels) {}
};

// Registration follows validation and precedes execution; no numbered record is edited.
int register_percent_control(FuelModels& fuels, const Request& request) {
    if (!request.percent_control) return request.model;
    const int source = request.model;
    const int custom = 250;
    const auto heat = HeatOfCombustionUnits::BtusPerPound;
    const auto loading = LoadingUnits::PoundsPerSquareFoot;
    const auto savr = SurfaceAreaToVolumeUnits::SquareFeetOverCubicFeet;
    const bool accepted = fuels.setCustomFuelModel(custom, "UNIT_CONTROL", "Explicit Percent-unit control",
        fuels.getFuelbedDepth(source, LengthUnits::Feet), LengthUnits::Feet,
        request.moe_percent, FractionUnits::Percent,
        fuels.getHeatOfCombustionDead(source, heat), fuels.getHeatOfCombustionLive(source, heat), heat,
        fuels.getFuelLoadOneHour(source, loading), fuels.getFuelLoadTenHour(source, loading),
        fuels.getFuelLoadHundredHour(source, loading), fuels.getFuelLoadLiveHerbaceous(source, loading),
        fuels.getFuelLoadLiveWoody(source, loading), loading, fuels.getSavrOneHour(source, savr),
        fuels.getSavrLiveHerbaceous(source, savr), fuels.getSavrLiveWoody(source, savr), savr,
        fuels.getIsDynamic(source));
    if (!accepted) throw std::runtime_error("custom_registration_failed");
    return custom;
}

std::string run(const Request& request, bool forbid_geometry, bool& geometry_entered, std::unique_ptr<Session>& session) {
    FuelModels fuels;
    if (!fuels.isFuelModelDefined(request.model) || fuels.isAllFuelLoadZero(request.model))
        throw Unavailable{};
    if (!session) session.reset(new Session);
    const int model = register_percent_control(session->fuels, request);
    Surface& surface = session->surface;
    surface.updateSurfaceInputs(model, request.physical.at("m1h"), request.physical.at("m10h"), request.physical.at("m100h"), request.physical.at("mlh"), request.physical.at("mlw"),
        FractionUnits::Percent, request.physical.at("wind_speed"), SpeedUnits::MilesPerHour,
        (request.modes.at("wind_height") == "TenMeter" ? WindHeightInputMode::TenMeter : WindHeightInputMode::TwentyFoot), request.direction,
        (request.modes.at("wind_orientation") == "North" ? WindAndSpreadOrientationMode::RelativeToNorth : WindAndSpreadOrientationMode::RelativeToUpslope), request.physical.at("slope"),
        SlopeUnits::Percent, request.physical.at("aspect"), request.physical.at("canopy_cover"), FractionUnits::Percent,
        request.physical.at("canopy_height"), LengthUnits::Feet, request.physical.at("crown_ratio"), FractionUnits::Fraction);
    surface.setMoistureInputMode(MoistureInputMode::BySizeClass);
    surface.setWindAdjustmentFactorCalculationMethod(request.modes.at("waf_method") == "UserInput"
        ? WindAdjustmentFactorCalculationMethod::UserInput : WindAdjustmentFactorCalculationMethod::UseCrownRatio);
    if (request.modes.at("waf_method") == "UserInput") surface.setUserProvidedWindAdjustmentFactor(request.user_waf);
    surface.doSurfaceRunInDirectionOfMaxSpread();
    if (forbid_geometry) throw GeometryTrap{};
    geometry_entered = true;
    const std::map<std::string, double> result = {
        {"fireline_intensity_btu_ft_s", surface.getFirelineIntensity(FirelineIntensityUnits::BtusPerFootPerSecond)},
        {"flame_length_ft", surface.getFlameLength(LengthUnits::Feet)},
        {"heat_per_unit_area_btu_ft2", surface.getHeatPerUnitArea(HeatPerUnitAreaUnits::BtusPerSquareFoot)},
        {"length_to_width_ratio", surface.getFireLengthToWidthRatio()},
        {"spread_chains_per_hour", surface.getSpreadRate(SpeedUnits::ChainsPerHour)}
    };
    return encode_result(result);
}

/// Calculate and read geometry on a fresh upstream FireSize object.
///
/// :param request: Reviewed FireSize input.
/// :param forbid_geometry: Whether the geometry entry trap is enabled.
/// :param geometry_entered: Geometry entry flag to set once the trap is cleared.
/// :return: JSON object with length-to-width ratio, area, and perimeter.
std::string run_fire_size(const FireSizeRequest& request, bool forbid_geometry, bool& geometry_entered) {
    if (forbid_geometry) throw GeometryTrap{};
    geometry_entered = true;
    FireSize size;
    size.calculateFireBasicDimensions(request.crown, request.wind_mph, SpeedUnits::MilesPerHour,
        request.spread_ft_min, SpeedUnits::FeetPerMinute);
    return encode_result({
        {"area_ft2", size.getFireArea(request.crown, AreaUnits::SquareFeet, request.elapsed_min, TimeUnits::Minutes)},
        {"length_to_width_ratio", size.getFireLengthToWidthRatio()},
        {"perimeter_ft", size.getFirePerimeter(request.crown, LengthUnits::Feet, request.elapsed_min, TimeUnits::Minutes)},
    });
}

/// Calculate firebrand ignition probability on a fresh upstream Ignite object.
///
/// :param request: Reviewed Ignite input.
/// :param forbid_output: Whether the output entry trap is enabled.
/// :return: JSON object with ignition probability and fuel temperature.
std::string run_ignite(const IgniteRequest& request, bool forbid_output) {
    if (forbid_output) throw GeometryTrap{};
    Ignite ignite;
    ignite.initializeMembers();
    ignite.setAirTemperature(request.air_temperature_f, TemperatureUnits::Fahrenheit);
    ignite.setSunShade(request.sun_shade_fraction, FractionUnits::Fraction);
    ignite.setMoistureOneHour(request.moisture_1h, FractionUnits::Fraction);
    const double probability = ignite.calculateFirebrandIgnitionProbability(FractionUnits::Fraction);
    return encode_result({
        {"firebrand_probability", probability},
        {"fuel_temperature_f", ignite.getFuelTemperature(TemperatureUnits::Fahrenheit)},
    });
}
}

int main(int argc, char** argv) {
    std::setlocale(LC_ALL, "C");
    if (argc != 3 && argc != 4) return 64;
    const bool forbid = argc == 4 && std::string(argv[3]) == "--forbid-upstream";
    const bool forbid_geometry = argc == 4 && std::string(argv[3]) == "--forbid-geometry";
    if (argc == 4 && !forbid && !forbid_geometry) return 64;
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary);
    if (!input || !output) return 66;
    std::set<std::string> ids;
    std::map<std::string,std::unique_ptr<Session>> sessions;
    std::string line;
    while (std::getline(input, line)) {
        std::string id;
        bool entered = false;
        bool geometry_entered = false;
        try {
            if (line.size() > 65536) throw std::runtime_error("line_too_long");
            const auto object = Parser(line).parse();
            if (object.count("id") && object.at("id").kind == Value::String) id = object.at("id").text;
            const auto operation_field = object.find("operation");
            const std::string operation = operation_field != object.end() && operation_field->second.kind == Value::String
                ? operation_field->second.text : "";
            std::string result;
            if (operation == "fire_size") {
                const FireSizeRequest request = validate_fire_size(object);
                enter_upstream(id, ids, forbid, entered);
                result = run_fire_size(request, forbid_geometry, geometry_entered);
            } else if (operation == "ignite") {
                const IgniteRequest request = validate_ignite(object);
                enter_upstream(id, ids, forbid, entered);
                result = run_ignite(request, forbid_geometry);
            } else {
                const Request request = validate(object);
                enter_upstream(request.id, ids, forbid, entered);
                std::unique_ptr<Session> fresh;
                auto& session = request.session.empty() ? fresh : sessions[request.session];
                result = run(request, forbid_geometry, geometry_entered, session);
            }
            output << "{\"id\":" << encode(id) << ",\"status\":\"ok\",\"upstream_entries\":1,"
                   << (operation == "ignite" ? "\"output_entries\"" : "\"geometry_entries\"")
                   << ":1,\"output\":" << result << "}\n";
        } catch (const UpstreamTrap&) {
            std::cerr << "UPSTREAM_ENTRY_BLOCKED\n";
            return 86;
        } catch (const Unavailable&) {
            output << "{\"id\":" << encode(id) << ",\"status\":\"unavailable\",\"upstream_entries\":1,\"geometry_entries\":0,\"finding\":\"F-013\"}\n";
        } catch (const GeometryTrap&) {
            std::cerr << "GEOMETRY_ENTRY_BLOCKED\n";
            return 87;
        } catch (const std::exception& error) {
            // IDs are schema-limited for accepted rows; encode also escapes rejected strings.
            output << "{\"id\":" << (id.empty() ? "null" : encode(id))
                   << ",\"status\":\"rejected\",\"upstream_entries\":" << (entered ? 1 : 0)
                   << ",\"error\":" << encode(error.what()) << "}\n";
            if (entered) return 70;
        }
    }
    output.flush();
    const bool flush_failed = !output;
    output.close();
    return input.bad() || flush_failed || !output ? 74 : 0;
}
