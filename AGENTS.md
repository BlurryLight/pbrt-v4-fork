# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++17 implementation of pbrt-v4 built with CMake. Core renderer code lives in `src/pbrt/`, with major subsystems split into `base/`, `cpu/`, `gpu/`, `wavefront/`, and `util/`. Command-line tools are under `src/pbrt/cmd/` and third-party dependencies are vendored under `src/ext/`; avoid editing vendored code unless the change is explicitly dependency-related. Unit tests are colocated with implementation files as `*_test.cpp` under `src/pbrt/` and `src/pbrt/util/`. Example scenes and assets live in `scenes/` and `images/`. Generated and local build output belongs in `build/` or `build-release/`.

## Build, Test, and Development Commands

- `git submodule update --init --recursive`: fetch required third-party submodules.
- `cmake --preset debug`: configure a Debug Ninja build in `build/`.
- `cmake --preset release`: configure a Release Ninja build in `build-release/`.
- `cmake --build --preset debug --parallel`: compile the Debug build.
- `cmake --build --preset release --parallel`: compile the Release build.
- `ctest --preset debug`: run Debug tests with failure output enabled.
- `./build/pbrt_test --gtest-filter=Parser.*`: run a focused GTest suite.
- `./build/pbrt scenes/<scene>.pbrt`: render a local scene after building.

Both configure presets generate `compile_commands.json` for editor and tooling integration.

GPU builds depend on CUDA and OptiX configuration. Set `PBRT_OPTIX7_PATH` or pass `-DPBRT_OPTIX7_PATH=...`; set `-DPBRT_GPU_SHADER_MODEL=sm_80` when auto-detection is unreliable.

## Coding Style & Naming Conventions

Follow the surrounding C++ style: 4-space indentation, braces on the same line for functions and control blocks, `UpperCamelCase` for types, `lowerCamelCase` for variables/functions where already used, and uppercase `PBRT_*` for compile definitions. Keep headers in the matching subsystem and include from `pbrt/...`. Add new CMake sources to the appropriate `PBRT_*_SOURCE` or header list in `CMakeLists.txt`.

## Testing Guidelines

Tests use GoogleTest and should be named `*_test.cpp`. Keep tests close to the code they validate, use `TEST(SuiteName, CaseName)`, and prefer deterministic numeric checks. Run `ctest --preset debug` before submitting; for targeted work, also run `./build/pbrt_test --gtest-filter=SuiteName.*`.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects, for example `add cornellbox` and `Fix zlib cache vars for OpenEXR detection`. Keep commits focused and mention affected subsystems when helpful. Pull requests should describe the behavioral change, list build/test commands run, link related issues, and include rendered output or image comparisons when changes affect rendering.

## Agent-Specific Instructions

Do not modify `src/ext/` or generated build directories unless requested. Prefer narrow patches, preserve existing scene assets, and verify renderer-facing changes with both unit tests and a small scene render when practical.
