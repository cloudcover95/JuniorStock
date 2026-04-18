# juniorstock/build/setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Target strictly the proprietary logic kernels for obfuscation and speed
# Excludes I/O and UI boundaries to maintain dynamic routing flexibility
target_modules = [
    "../math/fsd/kinematics.py",
    "../quant/atml_core.py",
    "../blackbox/strategy_container.py",
    "../risk/topo_allocator.py"
]

# Enforce extreme C-level optimizations
# boundscheck=False: Disables IndexError checks (relies on FSD logic gates)
# wraparound=False: Disables negative indexing
# cdivision=True: Disables ZeroDivisionError checks (speed over safety, handled at tensor level)
compiler_directives = {
    'language_level': "3",
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'nonecheck': False
}

setup(
    name="JuniorStock_Sovereign_Kernels",
    ext_modules=cythonize(
        target_modules,
        compiler_directives=compiler_directives,
        annotate=False
    )
)
