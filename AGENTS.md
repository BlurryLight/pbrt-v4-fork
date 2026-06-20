# Repository Guidelines

## Project Structure & Module Organization

This repository is a C++17 implementation of pbrt-v4 built with CMake. Core renderer code lives in `src/pbrt/`, with major subsystems split into `base/`, `cpu/`, `gpu/`, `wavefront/`, and `util/`. Command-line tools are under `src/pbrt/cmd/` and third-party dependencies are vendored under `src/ext/`; avoid editing vendored code unless the change is explicitly dependency-related. Unit tests are colocated with implementation files as `*_test.cpp` under `src/pbrt/` and `src/pbrt/util/`. Example scenes and assets live in `scenes/` and `images/`. Generated and local build output belongs in `build/` or `build-release/`. The agent build dir is `build/cmake-build-agent/` (see below).

## Build, Test, and Development Commands

### Agent build (recommended for code agents)

Code agents must use the dedicated agent build script so that their
intermediate build artifacts do not collide with a developer's working
`build/` or `build-release/` tree. The agent build dir is fully isolated
at `build/cmake-build-agent/`.

- `./local_build_test.py`: full pipeline (configure + build + test) in Release.
- `./local_build_test.py --configure`: only configure (Ninja, Release by default).
- `./local_build_test.py --build`: only build an already-configured tree.
- `./local_build_test.py --test`: only run the unit-test binary.
- `./local_build_test.py --render`: also run a 1-spp smoke render of `scenes/cornell_box_v4.pbrt` and verify a non-trivial EXR is produced.
- `./local_build_test.py --clean`: wipe the agent build dir before running.
- `./local_build_test.py --debug`: Debug build instead of Release.
- `./local_build_test.py --jobs N`: cap parallelism.
- `./local_build_test.py --gtest-filter=Parser.*`: forward a filter to `pbrt_test`.
- `./local_build_test.py --extra-cmake -DFOO=bar ...`: append extra cmake flags.

The script auto-initializes git submodules, reuses the pre-generated
`rgbspectrum_*.cpp` tables from `build/` if present (skipping the
`rgb2spec_opt` build step), and exits non-zero on any failure
(1 = build/test failure, 2 = missing prerequisites, 130 = interrupted).
Always run it (or at minimum `./local_build_test.py --configure --build`)
after editing a renderer-facing file and before reporting success.

### User-facing presets (avoid in agent sessions)

These are the presets from `CMakePresets.json` intended for human
developers. They share the `build/` and `build-release/` directories and
must not be used by code agents:

- `git submodule update --init --recursive`: fetch required third-party submodules.
- `cmake --preset debug`: configure a Debug Ninja build in `build/`.
- `cmake --preset release`: configure a Release Ninja build in `build-release/`.
- `cmake --build --preset debug --parallel`: compile the Debug build.
- `cmake --build --preset release --parallel`: compile the Release build.
- `ctest --preset debug`: run Debug tests with failure output enabled.
- `./build/pbrt_test --gtest-filter=Parser.*`: run a focused GTest suite.
- `./build/pbrt scenes/<scene>.pbrt`: render a local scene after building.

Both configure presets generate `compile_commands.json` for editor and tooling integration.

### GPU builds

GPU builds depend on CUDA and OptiX configuration. Set `PBRT_OPTIX7_PATH` or pass `-DPBRT_OPTIX7_PATH=...`; set `-DPBRT_GPU_SHADER_MODEL=sm_80` when auto-detection is unreliable. To opt in for the agent build, pass `--cuda` (default sm_80) or `--extra-cmake -DPBRT_GPU_SHADER_MODEL=sm_75 ...`.

## Coding Style & Naming Conventions

Follow the surrounding C++ style: 4-space indentation, braces on the same line for functions and control blocks, `UpperCamelCase` for types, `lowerCamelCase` for variables/functions where already used, and uppercase `PBRT_*` for compile definitions. Keep headers in the matching subsystem and include from `pbrt/...`. Add new CMake sources to the appropriate `PBRT_*_SOURCE` or header list in `CMakeLists.txt`.

## Testing Guidelines

Tests use GoogleTest and should be named `*_test.cpp`. Keep tests close to the code they validate, use `TEST(SuiteName, CaseName)`, and prefer deterministic numeric checks. Run `ctest --preset debug` before submitting; for targeted work, also run `./build/pbrt_test --gtest-filter=SuiteName.*`.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive subjects, for example `add cornellbox` and `Fix zlib cache vars for OpenEXR detection`. Keep commits focused and mention affected subsystems when helpful. Pull requests should describe the behavioral change, list build/test commands run, link related issues, and include rendered output or image comparisons when changes affect rendering.

## Agent-Specific Instructions

Do not modify `src/ext/` or generated build directories unless requested. Prefer narrow patches, preserve existing scene assets, and verify renderer-facing changes with both unit tests and a small scene render when practical.
