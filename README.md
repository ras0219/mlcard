# MLCard

## How to Compile

Using Visual Studio 2019 CMake Open Folder; requires recent vcpkg (last known good commit: https://github.com/ras0219/vcpkg/commit/be0c13ba694162ddf6e8e96e0f6700f8d336edfc).

A convenience submodule is available at `/vcpkg`.

```
git clone https://github.com/ras0219/mlcard --recursive --depth=1
.\mlcard\vcpkg\bootstrap-vcpkg.bat
```

CMake needs to use the vcpkg toolchain file. An example CMakeSettings.json would be:

```json
{
  "configurations": [
    {
      "name": "x64-Release",
      "generator": "Ninja",
      "configurationType": "RelWithDebInfo",
      "buildRoot": "${projectDir}\\out\\build\\${name}",
      "installRoot": "${projectDir}\\out\\install\\${name}",
      "cmakeCommandArgs": "",
      "buildCommandArgs": "-v",
      "ctestCommandArgs": "",
      "cmakeToolchain": "${projectDir}/vcpkg/scripts/buildsystems/vcpkg.cmake",
      "inheritEnvironments": [ "msvc_x64_x64" ],
      "variables": []
    },
    {
      "name": "x64-Debug",
      "generator": "Ninja",
      "configurationType": "Debug",
      "buildRoot": "${projectDir}\\out\\build\\${name}",
      "installRoot": "${projectDir}\\out\\install\\${name}",
      "cmakeCommandArgs": "",
      "buildCommandArgs": "-v",
      "ctestCommandArgs": "",
      "cmakeToolchain": "${projectDir}/vcpkg/scripts/buildsystems/vcpkg.cmake",
      "inheritEnvironments": [ "msvc_x64_x64" ],
      "variables": []
    }
  ]
}
```
