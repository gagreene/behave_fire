#include "fuelModels.h"
#include "surface.h"
#include <iostream>

int main()
{
    FuelModels fuels;
    Surface surface(fuels);
    surface.updateSurfaceInputs(124, 6.0, 7.0, 8.0, 60.0, 90.0,
        FractionUnits::Percent, 5.0, SpeedUnits::MilesPerHour,
        WindHeightInputMode::TwentyFoot, 0.0,
        WindAndSpreadOrientationMode::RelativeToNorth, 30.0,
        SlopeUnits::Percent, 0.0, 50.0, FractionUnits::Percent,
        30.0, LengthUnits::Feet, 0.5, FractionUnits::Fraction);
    surface.doSurfaceRunInDirectionOfMaxSpread();
    const double spread = surface.getSpreadRate(SpeedUnits::ChainsPerHour);
    std::cout << spread << '\n';
    return spread > 0.0 ? 0 : 2;
}
