#!/bin/zsh
# juniorstock/quant/metal_ext/build_metal.sh

echo "[METAL GATE] Compiling C++ bindings and Metal Shaders..."
cd "$(dirname "$0")"

# Build pybind11 extension
python3 setup.py build_ext --inplace

# Optional: Precompile metal library for direct instantiation
xcrun -sdk macosx metal -c ternary_gemm.metal -o ternary_gemm.air
xcrun -sdk macosx metallib ternary_gemm.air -o ternary_gemm.metallib

rm ternary_gemm.air

echo "[METAL GATE] Ternary SIMD Shaders deployed."
