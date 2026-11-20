cmake_minimum_required(VERSION 3.20)
project(behave_phase1_spike LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(INCLUDE_EXRATE "Compile upstream EXRATE implementation files" ON)
set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/../behave_cpp/src/behave")
set(SURFACE_NAMES behaveUnits chaparralFuel fireSize fuelModels moistureScenarios
    palmettoGallberry surface surfaceFireReactionIntensity surfaceFuelbedIntermediates
    surfaceInputs surfaceFire surfaceTwoFuelModels westernAspen windAdjustmentFactor windSpeedUtility)
set(SURFACE_SOURCES)
foreach(NAME IN LISTS SURFACE_NAMES)
    list(APPEND SURFACE_SOURCES "${UPSTREAM}/${NAME}.cpp")
endforeach()
if(INCLUDE_EXRATE)
    list(APPEND SURFACE_SOURCES "${UPSTREAM}/newext.cpp" "${UPSTREAM}/randfuel.cpp" "${UPSTREAM}/randthread.cpp")
endif()
add_library(behave_sources STATIC ${SURFACE_SOURCES})
target_include_directories(behave_sources PUBLIC "${UPSTREAM}")
if(MSVC)
    set(PRECISE_FLAGS /fp:precise /D_USE_MATH_DEFINES= /EHsc)
else()
    set(PRECISE_FLAGS -fno-fast-math -ffp-contract=off)
endif()
target_compile_options(behave_sources PRIVATE ${PRECISE_FLAGS})
add_executable(surface_link_probe probe.cpp)
target_compile_options(surface_link_probe PRIVATE ${PRECISE_FLAGS})
target_link_libraries(surface_link_probe PRIVATE behave_sources)
if(MSVC)
    target_link_options(surface_link_probe PRIVATE "/MAP:${CMAKE_BINARY_DIR}/surface_link_probe.map")
endif()

add_executable(behave_ref_driver driver.cpp
    "${UPSTREAM}/ignite.cpp" "${UPSTREAM}/igniteInputs.cpp"
    "${UPSTREAM}/canopy_coefficient_table.cpp" "${UPSTREAM}/mortality.cpp"
    "${UPSTREAM}/mortality_equation_table.cpp" "${UPSTREAM}/mortality_inputs.cpp"
    "${UPSTREAM}/species_master_table.cpp"
    "${UPSTREAM}/CrownFirebrandProcessor.cpp" "${UPSTREAM}/spot.cpp" "${UPSTREAM}/spotInputs.cpp")
target_compile_options(behave_ref_driver PRIVATE ${PRECISE_FLAGS})
target_link_libraries(behave_ref_driver PRIVATE behave_sources)
if(MSVC)
    target_link_options(behave_ref_driver PRIVATE "/MAP:${CMAKE_BINARY_DIR}/behave_ref_driver.map")
endif()
