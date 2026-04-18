#!/bin/zsh
# juniorstock/math/metal_ext/hamming/build_hamming.sh

echo "[METAL GATE] Compiling Hamming SIMD Shaders..."
cd "$(dirname "$0")"

python3 setup.py build_ext --inplace

xcrun -sdk macosx metal -c hamming_dist.metal -o hamming_dist.air
xcrun -sdk macosx metallib hamming_dist.air -o hamming_dist.metallib
rm hamming_dist.air

echo "[METAL GATE] Ternary LSH Shaders deployed."
