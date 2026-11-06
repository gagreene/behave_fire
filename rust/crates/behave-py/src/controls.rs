//! Interpret Python's permissive string controls at the binding boundary.
use crate::arrays::Arguments;
use behave_fire::engine::{SurfaceOptions, WafMethod, WindHeight, WindOrientation};

fn contains(value: &str, needle: &str) -> bool {
    value
        .as_bytes()
        .windows(needle.len())
        .any(|part| part.eq_ignore_ascii_case(needle.as_bytes()))
}

pub fn crown(arguments: &Arguments) -> SurfaceOptions {
    SurfaceOptions {
        wind_orientation: orientation(
            arguments
                .s("wind_orientation_mode")
                .unwrap_or("RelativeToUpslope"),
        ),
        ..SurfaceOptions::default()
    }
}

pub fn height(value: &str) -> WindHeight {
    if contains(value, "10") || contains(value, "ten") {
        WindHeight::TenMetre
    } else {
        WindHeight::TwentyFoot
    }
}

pub fn orientation(value: &str) -> WindOrientation {
    if contains(value, "north") {
        WindOrientation::North
    } else {
        WindOrientation::Upslope
    }
}

pub fn surface(arguments: &Arguments) -> Result<SurfaceOptions, String> {
    Ok(SurfaceOptions {
        wind_orientation: orientation(arguments.s("wind_orientation_mode")?),
        wind_height: height(arguments.s("wind_height_mode")?),
        waf_method: waf(arguments.s("waf_method")?),
    })
}

pub fn waf(value: &str) -> WafMethod {
    if value
        .bytes()
        .filter(|byte| *byte != b'_')
        .map(|byte| byte.to_ascii_lowercase())
        .eq(b"userinput".iter().copied())
    {
        WafMethod::UserInput
    } else {
        WafMethod::CrownRatio
    }
}
