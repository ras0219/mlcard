cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0091 NEW)
set(VCPKG_TARGET_TRIPLET x64-windows-static CACHE STRING "" FORCE)
project(mlcard CXX)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/permissive- /FC)
endif()

find_package(FLTK CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(main main.cpp "vec.h")
set_property(TARGET main PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(main PRIVATE fltk fmt::fmt)
