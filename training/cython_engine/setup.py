"""
Cython 引擎构建脚本
==================
用法:
    cd training/cython_engine
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "game_core",
        ["game_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="xiangqi_cython_engine",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
            "language_level": 3,
        },
    ),
)
