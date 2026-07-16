"""Build and install the hnsw_healer extension.

Default path: pybind11.setup_helpers (reliable in cibuildwheel/manylinux).
Optional: HNSW_HEALER_USE_CMAKE=1 for the CMake build path.

Usage:
    pip install -e .
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

HERE = Path(__file__).parent.resolve()


def read_version() -> str:
    init_py = HERE / "api" / "__init__.py"
    if init_py.is_file():
        match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            init_py.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return match.group(1)
    return "0.3.2"


VERSION = read_version()
USE_CMAKE = os.environ.get("HNSW_HEALER_USE_CMAKE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# ---------------------------------------------------------------------------
# Path A: pybind11 setup_helpers (default — works well on manylinux/macOS CI)
# ---------------------------------------------------------------------------


def make_pybind11_extension() -> list:
    from pybind11.setup_helpers import Pybind11Extension

    return [
        Pybind11Extension(
            "hnsw_healer",
            ["src/healer.cpp"],
            include_dirs=[str(HERE / "src")],
            cxx_std=17,
            # Becomes #define VERSION_INFO 0.3.2 — stringified in healer.cpp
            define_macros=[("VERSION_INFO", VERSION)],
        )
    ]


# ---------------------------------------------------------------------------
# Path B: CMake (optional local / advanced)
# ---------------------------------------------------------------------------


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir or str(HERE))


class CMakeBuild(_build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        if sys.platform == "darwin":
            deploy = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "12.0")
            cmake_args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deploy}"]

        build_args = ["--config", cfg]
        using_ninja = False
        if "CMAKE_GENERATOR" not in os.environ:
            for candidate in (
                "ninja",
                str(Path(sys.executable).parent / "Scripts" / "ninja.exe"),
                str(Path(sys.executable).parent / "ninja.exe"),
            ):
                try:
                    subprocess.run(
                        [candidate, "--version"],
                        check=True,
                        capture_output=True,
                    )
                    cmake_args += [
                        "-G",
                        "Ninja",
                        f"-DCMAKE_MAKE_PROGRAM={candidate}",
                    ]
                    using_ninja = True
                    break
                except (FileNotFoundError, subprocess.CalledProcessError, OSError):
                    continue

        if sys.platform.startswith("win"):
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            ]
            gen = os.environ.get("CMAKE_GENERATOR", "")
            joined = " ".join(cmake_args)
            vs_generator = (
                "Visual Studio" in gen
                or os.environ.get("CMAKE_GENERATOR_PLATFORM")
            )
            if (
                not using_ninja
                and "Ninja" not in gen
                and "NMake" not in gen
                and "-G" not in joined
                and vs_generator
                and sys.maxsize > 2**32
            ):
                cmake_args += ["-A", "x64"]
            if not using_ninja and "Ninja" not in joined:
                if "Visual Studio" in gen or vs_generator:
                    build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{os.cpu_count() or 2}"]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                arg for arg in os.environ["CMAKE_ARGS"].split(" ") if arg
            ]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args], cwd=build_temp
        )

        if sys.platform.startswith("win"):
            self._copy_mingw_runtime_dlls(extdir)

    @staticmethod
    def _copy_mingw_runtime_dlls(extdir: Path) -> None:
        candidates: list[Path] = []
        for key in ("CXX", "CC", "CMAKE_CXX_COMPILER"):
            val = os.environ.get(key)
            if val:
                candidates.append(Path(val).resolve().parent)
        winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        if winget.is_dir():
            for p in winget.glob("MartinStorsjo.LLVM-MinGW*/**/bin"):
                candidates.append(p)
        for bindir in candidates:
            if not bindir.is_dir():
                continue
            copied = False
            for name in ("libc++.dll", "libunwind.dll", "libssp-0.dll"):
                src = bindir / name
                if src.is_file():
                    try:
                        import shutil

                        shutil.copy2(src, Path(extdir) / name)
                        copied = True
                    except OSError:
                        pass
            if copied:
                break


if USE_CMAKE:
    ext_modules: list = [CMakeExtension("hnsw_healer")]
    cmdclass: dict = {"build_ext": CMakeBuild}
else:
    try:
        from pybind11.setup_helpers import build_ext as pybind11_build_ext

        ext_modules = make_pybind11_extension()
        cmdclass = {"build_ext": pybind11_build_ext}
    except ImportError:
        # First-pass metadata collection without pybind11 installed yet
        ext_modules = []
        cmdclass = {}


setup(
    name="hnsw-healer",
    version=VERSION,
    description=(
        "Hard-delete residual vectors in HNSW indexes: physical wipe, "
        "graph rebuild/compact, residual proofs, and signed erasure receipts."
    ),
    author="HNSW Healer contributors",
    python_requires=">=3.10",
    keywords=[
        "machine-unlearning",
        "hnsw",
        "vector-database",
        "gdpr",
        "privacy",
        "rag",
        "embeddings",
    ],
    packages=find_packages(exclude=("tests", "tests.*", "docs", "docs.*")),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pybind11>=2.12.0",
        "numpy>=1.26.0",
        "cryptography>=42.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "httpx>=0.27.0",
            "matplotlib>=3.8.0",
        ],
        "hnswlib": ["hnswlib>=0.8.0"],
        "faiss": ["faiss-cpu>=1.7.4"],
        "chroma": ["chromadb>=0.4.0", "hnswlib>=0.8.0"],
        "qdrant": ["qdrant-client>=1.7.0"],
        "weaviate": ["weaviate-client>=3.25.0"],
        "redis": ["redis>=5.0.0"],
        "aws": ["boto3>=1.28.0"],
        "gcp": ["google-cloud-kms>=2.0.0"],
        "vault": ["hvac>=1.0.0"],
        "enterprise": [
            "hnswlib>=0.8.0",
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.4",
            "qdrant-client>=1.7.0",
        ],
        "kms": [
            "boto3>=1.28.0",
            "google-cloud-kms>=2.0.0",
            "hvac>=1.0.0",
        ],
    },
    long_description=(
        (HERE / "README.md").read_text(encoding="utf-8")
        if (HERE / "README.md").is_file()
        else ""
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/cbritt0n/unlearning",
    project_urls={
        "Documentation": "https://github.com/cbritt0n/unlearning#readme",
        "Changelog": "https://github.com/cbritt0n/unlearning/blob/main/CHANGELOG.md",
        "Bug Tracker": "https://github.com/cbritt0n/unlearning/issues",
        "Source": "https://github.com/cbritt0n/unlearning",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
