cmake_minimum_required(VERSION 4.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_INSTALLATION_ROOT $ENV{VCPKG_ROOT})
    message(STATUS "VCPKG_INSTALLATION_ROOT is ${VCPKG_INSTALLATION_ROOT}")
endif()

if(EXISTS ${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake)
    message(
        STATUS
        "Using Vcpkg toolchain: ${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
    )
    set(CMAKE_TOOLCHAIN_FILE
        "${VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING
        ""
    )
    set(USING_VCPKG TRUE)
endif()

project(ConstrainedMiniball LANGUAGES CXX)
include(CTest)

find_package(Eigen3 CONFIG REQUIRED)
if (Eigen3_FOUND)
	message(STATUS "Found Eigen3 headers in: ${EIGEN3_INCLUDE_DIR}")
endif()

find_package(CGAL CONFIG REQUIRED)
if (DEFINED CGAL_MODULES_DIR AND Eigen3_FOUND)
    include(${CGAL_MODULES_DIR}/CGAL_Eigen3_support.cmake)
    message(STATUS "CGAL has Eigen3 support.")
endif()

add_executable(example example.cpp)
target_link_libraries(example PRIVATE Eigen3::Eigen CGAL::CGAL CGAL::Eigen3_support)

file(GLOB test_files tests/*.cpp)
foreach(file ${test_files})
    get_filename_component(name ${file} NAME_WE)
    add_executable(${name} ${file})
    target_link_libraries(${name} PRIVATE Eigen3::Eigen CGAL::CGAL CGAL::Eigen3_support)
    add_test(NAME "${name} EXACT" COMMAND ${name} "EXACT")
    add_test(NAME "${name} DOUBLE" COMMAND ${name} "DOUBLE")
endforeach()

