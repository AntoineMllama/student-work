cmake_minimum_required(VERSION 3.10)
project(ISIM_72)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/utility)
include_directories(${CMAKE_SOURCE_DIR}/src/texture)

file(GLOB_RECURSE SOURCES "src/*.cc")

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

add_executable(ISIM_72 ${SOURCES})

target_link_libraries(ISIM_72 sfml-graphics sfml-window sfml-system)
