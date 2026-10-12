"""Build the optional solver locally; ordinary installation needs no compiler."""

from pathlib import Path
from shutil import copy2

from Cython.Build import cythonize
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    distribution = setup(
        name="alphaboxes-acceleration",
        ext_modules=cythonize(
            [
                Extension(f"alphaboxes.{name}", [str(ROOT / f"src/alphaboxes/{name}.pyx")])
                for name in ("_endgame", "_search", "_components", "_rollout")
            ],
            build_dir=str(ROOT / "build/cython"),
            compiler_directives={"language_level": 3},
        ),
        package_dir={"": "src"},
        script_args=["build_ext"],
    )
    command = distribution.get_command_obj("build_ext")
    for extension in distribution.ext_modules:
        relative = command.get_ext_filename(extension.name)
        built, target = Path(command.build_lib) / relative, ROOT / "src" / relative
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            copy2(built, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
