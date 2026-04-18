# juniorstock/quant/metal_ext/setup.py
from setuptools import setup, Extension
import pybind11
import os

# Flags for Apple Clang and MLX headers
cpp_args = ['-std=c++17', '-O3', '-fPIC', '-Wall']

ext_modules = [
    Extension(
        'mlx_ternary_ext',
        ['bindings.cpp'],
        include_dirs=[pybind11.get_include(), '/usr/local/include'], # Assuming MLX headers available
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='mlx_ternary_ext',
    ext_modules=ext_modules,
)
