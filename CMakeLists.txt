cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0091 NEW)
set(VCPKG_TARGET_TRIPLET x64-windows-static CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")

project(mlcard CXX)

if(MSVC)
    add_compile_options(/permissive- /FC /arch:AVX2 /fp:fast /Gs /GS-)
 #   add_link_options(/PROFILE)
endif()

find_package(FLTK CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_path(CPPWINRT_INCLUDE_DIRS "winrt/Windows.ApplicationModel.Activation.h")

add_executable(main main.cpp "vec.h" "model.cpp" "game.h" "game.cpp" "worker.h" "worker.cpp" "ai_play.h"  "rjwriter.h" "modeldims.h")
set_property(TARGET main PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(main PRIVATE fltk fmt::fmt)
target_include_directories(main PRIVATE ${CPPWINRT_INCLUDE_DIRS})
