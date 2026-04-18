# juniorstock/quant/spiking/setup.py
from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++17', '-O3', '-fPIC', '-Wall']

ext_modules = [
    Extension(
        'mlx_lif_ext',
        ['bindings_lif.cpp'],
        include_dirs=[pybind11.get_include(), '/usr/local/include'],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='mlx_lif_ext',
    ext_modules=ext_modules,
)
