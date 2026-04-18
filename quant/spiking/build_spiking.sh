#!/bin/zsh
# juniorstock/quant/spiking/build_spiking.sh

echo "[METAL GATE] Compiling Ternary LIF Shaders..."
cd "$(dirname "$0")"

python3 setup.py build_ext --inplace

xcrun -sdk macosx metal -c lif_ternary.metal -o lif_ternary.air
xcrun -sdk macosx metallib lif_ternary.air -o lif_ternary.metallib
rm lif_ternary.air

echo "[METAL GATE] ASTA Shaders deployed."
