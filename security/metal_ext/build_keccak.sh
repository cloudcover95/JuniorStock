#!/bin/zsh
# juniorstock/security/metal_ext/build_keccak.sh

echo "[METAL GATE] Compiling Keccak Sponge Shaders..."
cd "$(dirname "$0")"

python3 setup.py build_ext --inplace

xcrun -sdk macosx metal -c keccak_sponge.metal -o keccak_sponge.air
xcrun -sdk macosx metallib keccak_sponge.air -o keccak_sponge.metallib
rm keccak_sponge.air

echo "[METAL GATE] Keccak Enclave Shaders deployed."
