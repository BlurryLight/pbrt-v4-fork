#!/usr/bin/env python3
"""
local_build_test.py - Build & smoke-test pbrt-v4 for code agents.

This script is the canonical way for code agents to verify that pbrt-v4
still compiles and the test suite still passes after a change. It is
deliberately isolated from the normal user-facing build directories
(`build/` and `build-release/` from CMakePresets.json) and writes its
own outputs to `build/cmake-build-agent/`.

Usage:
    ./local_build_test.py                  # full: configure + build + test
    ./local_build_test.py --configure      # only configure
    ./local_build_test.py --build          # only build (after configure)
    ./local_build_test.py --test           # only run unit tests
    ./local_build_test.py --render         # also run a tiny smoke render
    ./local_build_test.py --clean          # wipe build dir before running
    ./local_build_test.py --debug          # Debug build (default: Release)
    ./local_build_test.py --jobs N         # limit parallelism

Exit codes:
    0   success
    1   configuration / dependency / build / test failure
    2   prerequisites missing (cmake, ninja, system libs, submodules)
    130 interrupted (SIGINT)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build" / "cmake-build-agent"
USER_BUILD_DIR = REPO_ROOT / "build"
SCENES_DIR = REPO_ROOT / "scenes"
DEFAULT_SMOKE_SCENE = SCENES_DIR / "cornell_box_v4.pbrt"

PREGENERATED_TABLES = (
    "rgbspectrum_aces.cpp",
    "rgbspectrum_dci_p3.cpp",
    "rgbspectrum_rec2020.cpp",
    "rgbspectrum_srgb.cpp",
)

# Submodule paths (relative to REPO_ROOT) that must be populated for the
# build to succeed. The list mirrors `.gitmodules`.
REQUIRED_SUBMODULES = [
    "src/ext/zlib",
    "src/ext/ptex",
    "src/ext/double-conversion",
    "src/ext/stb",
    "src/ext/openexr",
    "src/ext/filesystem",
    "src/ext/openvdb",
    "src/ext/libdeflate",
    "src/ext/lodepng",
    "src/ext/utf8proc",
    "src/ext/qoi",
    "src/ext/glfw",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class Log:
    """Minimal leveled logger with optional ANSI color."""

    USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    @classmethod
    def _emit(cls, level: str, color: str, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        if cls.USE_COLOR:
            sys.stdout.write(f"\033[{color}m[{ts} {level:>5}]\033[0m {msg}\n")
        else:
            sys.stdout.write(f"[{ts} {level:>5}] {msg}\n")
        sys.stdout.flush()

    @classmethod
    def info(cls, msg: str) -> None:
        cls._emit("INFO", "1;36", msg)

    @classmethod
    def step(cls, msg: str) -> None:
        cls._emit("STEP", "1;35", msg)

    @classmethod
    def ok(cls, msg: str) -> None:
        cls._emit("OK", "1;32", msg)

    @classmethod
    def warn(cls, msg: str) -> None:
        cls._emit("WARN", "1;33", msg)

    @classmethod
    def error(cls, msg: str) -> None:
        cls._emit("ERROR", "1;31", msg)

    @classmethod
    def cmd(cls, msg: str) -> None:
        cls._emit("CMD", "0;37", msg)


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True,
        env: dict | None = None, stream: bool = True) -> int:
    """Run a command, streaming output. Returns the exit code."""
    Log.cmd(" ".join(map(str, cmd)))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if stream else None,
        stderr=subprocess.STDOUT if stream else None,
        text=True,
        bufsize=1,
    )
    try:
        if stream and proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise


def fail(msg: str, code: int = 1) -> "NoReturn":
    Log.error(msg)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# Prerequisite checks
# --------------------------------------------------------------------------- #


def check_tools() -> None:
    """Verify that the external tools we need are on PATH."""
    missing: list[str] = []
    for tool in ("cmake", "ninja"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        fail(
            f"missing required tools: {', '.join(missing)}. "
            f"Install them (e.g. `apt install cmake ninja-build`) and retry.",
            code=2,
        )
    # cmake version sanity-check
    try:
        out = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, check=True
        )
        first = out.stdout.splitlines()[0]
        Log.info(first)
    except Exception as e:
        fail(f"could not determine cmake version: {e}", code=2)


def check_submodules() -> None:
    """Make sure required git submodules are populated."""
    missing: list[str] = []
    for rel in REQUIRED_SUBMODULES:
        p = REPO_ROOT / rel
        if not p.is_dir() or not any(p.iterdir()):
            missing.append(rel)
    if not missing:
        return
    Log.warn(f"uninitialized submodules: {', '.join(missing)}")
    Log.step("running `git submodule update --init --recursive`")
    rc = run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        fail("git submodule update failed", code=2)


def copy_pregenerated_tables() -> bool:
    """If a previous user build produced the pre-generated rgbspectrum .cpp
    files, copy them into the agent build dir so we can skip running
    `rgb2spec_opt` at build time. Returns True if all four files were
    staged successfully.
    """
    staged = BUILD_DIR / "_pregenerated"
    staged.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in PREGENERATED_TABLES:
        src = USER_BUILD_DIR / name
        dst = staged / name
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            ok = False
    if ok:
        Log.info(f"staged pre-generated rgb tables in {staged}")
    else:
        shutil.rmtree(staged, ignore_errors=True)
        Log.info("pre-generated rgb tables unavailable; rgb2spec_opt will run during build")
    return ok


# --------------------------------------------------------------------------- #
# Build steps
# --------------------------------------------------------------------------- #


def configure(args: argparse.Namespace) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    build_type = "Debug" if args.debug else "Release"

    cmake_args: list[str] = [
        "cmake",
        "-S", str(REPO_ROOT),
        "-B", str(BUILD_DIR),
        "-G", "Ninja",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]

    # If we got pre-generated tables, force the cmake flag and copy them
    # into the binary dir before configuring.
    if copy_pregenerated_tables():
        cmake_args.append("-DPBRT_USE_PREGENERATED_RGB_TO_SPECTRUM_TABLES=True")
        # The pre-generated files need to live in the binary dir when
        # the flag is set, so we copy them after configure too. Do it now
        # in case the configure step inspects the binary dir.
        staged = BUILD_DIR / "_pregenerated"
        for name in PREGENERATED_TABLES:
            shutil.copy2(staged / name, BUILD_DIR / name)
    else:
        # Do not pass the flag; the build will run rgb2spec_opt for us.
        pass

    if args.cuda:
        # Caller opted into a GPU build.
        cmake_args.append("-DPBRT_GPU_SHADER_MODEL=sm_80")

    if args.extra_cmake:
        cmake_args.extend(args.extra_cmake)

    Log.step(f"configure ({build_type}, Ninja) -> {BUILD_DIR}")
    rc = run(cmake_args, cwd=REPO_ROOT)
    if rc != 0:
        fail("cmake configure failed", code=1)

    Log.ok("configure succeeded")


def build(args: argparse.Namespace) -> None:
    if not (BUILD_DIR / "CMakeCache.txt").exists():
        fail("build directory is not configured; run with --configure first", code=1)

    Log.step("build (parallel)")
    rc = run(
        [
            "cmake", "--build", str(BUILD_DIR),
            "--config", "Debug" if args.debug else "Release",
            "--parallel", str(args.jobs) if args.jobs else "",
        ],
        cwd=REPO_ROOT,
    )
    if rc != 0:
        fail("build failed", code=1)
    Log.ok("build succeeded")


def test(args: argparse.Namespace) -> None:
    test_bin = BUILD_DIR / "pbrt_test"
    if not test_bin.exists():
        fail("pbrt_test not found; build first", code=1)

    # Honour any --gtest-filter passed by the caller; otherwise run the
    # full suite.
    gtest_args: list[str] = []
    if args.gtest_filter:
        gtest_args = [f"--gtest-filter={args.gtest_filter}"]

    Log.step("running unit tests")
    rc = run([str(test_bin), *gtest_args], cwd=BUILD_DIR)
    if rc != 0:
        fail("tests failed", code=1)
    Log.ok("all tests passed")


def render(args: argparse.Namespace) -> None:
    pbrt_bin = BUILD_DIR / "pbrt"
    if not pbrt_bin.exists():
        fail("pbrt binary not found; build first", code=1)

    scene = Path(args.scene) if args.scene else DEFAULT_SMOKE_SCENE
    if not scene.is_file():
        fail(f"scene not found: {scene}", code=1)

    out_exr = BUILD_DIR / "smoke.exr"
    if out_exr.exists():
        out_exr.unlink()

    Log.step(f"smoke render: {scene.name} -> {out_exr.name}")
    rc = run(
        [
            str(pbrt_bin),
            "--spp", str(args.render_spp),
            "--outfile", str(out_exr),
            str(scene),
        ],
        cwd=BUILD_DIR,
    )
    if rc != 0 or not out_exr.is_file():
        fail("smoke render failed", code=1)

    size = out_exr.stat().st_size
    if size < 1024:
        fail(f"smoke render produced suspiciously small file ({size} bytes)", code=1)
    Log.ok(f"smoke render ok ({size:,} bytes)")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and test pbrt-v4 for code agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--configure", action="store_true", help="run configure step")
    p.add_argument("--build", action="store_true", help="run build step")
    p.add_argument("--test", action="store_true", help="run unit tests")
    p.add_argument("--render", action="store_true", help="run a smoke render")
    p.add_argument("--no-test", action="store_true", help="skip unit tests")
    p.add_argument("--no-render", action="store_true", help="skip smoke render")
    p.add_argument("--clean", action="store_true", help="wipe build dir before running")
    p.add_argument("--debug", action="store_true", help="Debug build (default Release)")
    p.add_argument("--cuda", action="store_true", help="enable CUDA/OptiX build (sm_80)")
    p.add_argument("--jobs", type=int, default=0, help="parallel jobs (0 = use all cores)")
    p.add_argument(
        "--gtest-filter",
        default=None,
        help="forwarded to pbrt_test (e.g. 'Parser.*')",
    )
    p.add_argument("--scene", default=None, help="scene file for the smoke render")
    p.add_argument(
        "--render-spp", type=int, default=1,
        help="samples per pixel for the smoke render",
    )
    p.add_argument(
        "--extra-cmake", nargs=argparse.REMAINDER, default=None,
        help="extra cmake flags appended at the end of the configure command",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Default action: do everything.
    if not (args.configure or args.build or args.test or args.render or args.clean):
        args.configure = args.build = args.test = True

    # --no-test / --no-render override their counterparts.
    if args.no_test:
        args.test = False
    if args.no_render:
        args.render = False

    Log.info(f"repo:   {REPO_ROOT}")
    Log.info(f"build:  {BUILD_DIR}")

    check_tools()
    check_submodules()

    if args.clean and BUILD_DIR.exists():
        Log.step(f"cleaning {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    try:
        if args.configure:
            configure(args)
        if args.build:
            build(args)
        if args.test:
            test(args)
        if args.render:
            render(args)
    except KeyboardInterrupt:
        Log.warn("interrupted")
        sys.exit(130)

    Log.ok("done")


if __name__ == "__main__":
    main()
