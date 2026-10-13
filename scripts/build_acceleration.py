"""Build the optional solver locally; ordinary installation needs no compiler."""

import argparse
import sys
from pathlib import Path
from shutil import copy2

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernels-only", action="store_true", help="Build only C++ inference kernels"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "src",
        help="Package root for compiled modules (defaults to the source checkout)",
    )
    args = parser.parse_args()
    kernels = Extension(
        "alphaboxes._kernels",
        [str(ROOT / "src/alphaboxes/_kernels.pyx"), str(ROOT / "src/alphaboxes/_kernels_core.cpp")],
        include_dirs=[str(ROOT / "src/alphaboxes")],
        language="c++",
        extra_compile_args=["/O2", "/fp:strict", "/std:c++17"]
        if sys.platform == "win32"
        else ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fno-math-errno", "-std=c++17"],
    )
    extensions = (
        []
        if args.kernels_only
        else [
            Extension(f"alphaboxes.{name}", [str(ROOT / f"src/alphaboxes/{name}.pyx")])
            for name in ("_endgame", "_search", "_components", "_rollout")
        ]
    )
    distribution = setup(
        name="alphaboxes-acceleration",
        ext_modules=cythonize(
            [*extensions, kernels],
            build_dir=str(ROOT / "build/cython"),
            compiler_directives={"language_level": 3},
        ),
        package_dir={"": "src"},
        script_args=["build_ext"],
    )
    command = distribution.get_command_obj("build_ext")
    for extension in distribution.ext_modules:
        relative = command.get_ext_filename(extension.name)
        built, target = Path(command.build_lib) / relative, args.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            copy2(built, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
