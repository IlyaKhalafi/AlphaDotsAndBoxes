"""Build the optional solver locally; ordinary installation needs no compiler."""

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    setup(
        name="alphaboxes-acceleration",
        ext_modules=cythonize(
            [Extension("alphaboxes._endgame", [str(ROOT / "src/alphaboxes/_endgame.pyx")])],
            build_dir=str(ROOT / "build/cython"),
            compiler_directives={"language_level": 3},
        ),
        package_dir={"": "src"},
        script_args=["build_ext", "--inplace"],
    )
