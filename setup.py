"""Build and install the hnsw_healer pybind11 extension.

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
from setuptools.command.build_ext import build_ext

# ---------------------------------------------------------------------------
# CMake-backed extension (production-grade local builds via pip install -e .)
# ---------------------------------------------------------------------------


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            # setuptools expects the extension in extdir — CMakeLists must not override this.
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        # macOS: keep deployment target consistent with cibuildwheel
        if sys.platform == "darwin":
            deploy = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "12.0")
            cmake_args += [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deploy}"]
        build_args = ["--config", cfg]
        using_ninja = False

        # Prefer Ninja when available (Docker, cibuildwheel, pip-installed ninja).
        if "CMAKE_GENERATOR" not in os.environ:
            ninja_exe = None
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
                    ninja_exe = candidate
                    break
                except (FileNotFoundError, subprocess.CalledProcessError, OSError):
                    continue
            if ninja_exe:
                cmake_args += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja_exe}"]
                using_ninja = True

        if sys.platform.startswith("win"):
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            ]
            # -A is only valid for Visual Studio generators. Never pass it for
            # Ninja / NMake (breaks configure with "does not support platform").
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
            ):
                if sys.maxsize > 2**32:
                    cmake_args += ["-A", "x64"]
            if not using_ninja and "Ninja" not in joined:
                # Multi-proc flag is MSBuild-specific.
                if "Visual Studio" in gen or vs_generator:
                    build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{os.cpu_count() or 2}"]

        # Optional: pass through extra CMake args, e.g.
        #   CMAKE_ARGS="-Dpybind11_DIR=..." pip install -e .
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

        # On Windows + MinGW/LLVM toolchains, copy C++ runtime DLLs next to
        # the extension so import works without a global PATH change.
        if sys.platform.startswith("win"):
            self._copy_mingw_runtime_dlls(extdir)

    @staticmethod
    def _copy_mingw_runtime_dlls(extdir: Path) -> None:
        """Best-effort: ship libc++/libunwind beside the .pyd."""
        candidates: list[Path] = []
        for key in ("CXX", "CC", "CMAKE_CXX_COMPILER"):
            val = os.environ.get(key)
            if val:
                candidates.append(Path(val).resolve().parent)
        # Common WinGet LLVM-MinGW layout
        winget = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        if winget.is_dir():
            for p in winget.glob("MartinStorsjo.LLVM-MinGW*/**/bin"):
                candidates.append(p)
        dll_names = ("libc++.dll", "libunwind.dll", "libssp-0.dll")
        for bindir in candidates:
            if not bindir.is_dir():
                continue
            copied = False
            for name in dll_names:
                src = bindir / name
                if src.is_file():
                    dest = Path(extdir) / name
                    try:
                        import shutil

                        shutil.copy2(src, dest)
                        copied = True
                    except OSError:
                        pass
            if copied:
                break


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

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
    return "0.1.0"


setup(
    name="hnsw-healer",
    version=read_version(),
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
    packages=find_packages(
        exclude=("tests", "tests.*", "docs", "docs.*")
    ),
    ext_modules=[CMakeExtension("hnsw_healer")],
    cmdclass={"build_ext": CMakeBuild},
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
    # Update after creating the GitHub repository:
    url="https://github.com/YOUR_ORG/unlearning",
    project_urls={
        "Documentation": "https://github.com/YOUR_ORG/unlearning#readme",
        "Changelog": "https://github.com/YOUR_ORG/unlearning/blob/main/CHANGELOG.md",
        "Bug Tracker": "https://github.com/YOUR_ORG/unlearning/issues",
        "Source": "https://github.com/YOUR_ORG/unlearning",
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
