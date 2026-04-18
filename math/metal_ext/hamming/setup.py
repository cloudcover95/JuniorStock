# juniorstock/math/metal_ext/hamming/setup.py
from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++17', '-O3', '-fPIC', '-Wall']

ext_modules = [
    Extension(
        'mlx_hamming_ext',
        ['bindings_hamming.cpp'],
        include_dirs=[pybind11.get_include(), '/usr/local/include'],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='mlx_hamming_ext',
    ext_modules=ext_modules,
)
