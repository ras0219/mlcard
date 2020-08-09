cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0091 NEW)
set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")
set(ENV{VCPKG_OVERLAY_PORTS} "${CMAKE_CURRENT_SOURCE_DIR}/overlay-ports")

project(mlcard CXX)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(MSVC)
    add_compile_options(/permissive- /FC /arch:AVX2 /fp:fast /Gs /GS-)
endif()
find_package(RapidJSON CONFIG REQUIRED)
find_package(FLTK CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

file(GLOB SHARED_SRC shared/*)

add_library(mlcard_objs INTERFACE)
target_sources(mlcard_objs INTERFACE ${SHARED_SRC})
target_link_libraries(mlcard_objs INTERFACE fmt::fmt)
target_include_directories(mlcard_objs INTERFACE shared ${RAPIDJSON_INCLUDE_DIRS})

file(GLOB LIB_SRC lib/*)
add_library(libmlcard SHARED ${LIB_SRC})
target_link_libraries(libmlcard PRIVATE mlcard_objs)
set_target_properties(libmlcard
    PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    PREFIX ""
)

file(GLOB BIN_SRC bin/*)
add_executable(mlcard ${BIN_SRC})
set_property(TARGET mlcard PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(mlcard PRIVATE fltk mlcard_objs)
target_include_directories(mlcard PRIVATE ${RAPIDJSON_INCLUDE_DIRS})
