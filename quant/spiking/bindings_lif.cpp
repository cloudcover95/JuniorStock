// juniorstock/quant/spiking/bindings_lif.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <mlx/mlx.h>

namespace py = pybind11;
using namespace mlx::core;

array batched_lif_metal(const array& spikes_in, const array& weights, array& membrane_potentials, int beta_decay, int threshold) {
    if (spikes_in.dtype() != int8 || weights.dtype() != int8 || membrane_potentials.dtype() != int32) {
        throw std::invalid_argument("[METAL FAULT] Operands require strict int8/int32 typing for LIF routing.");
    }
    
    int num_neurons = weights.shape(0);
    
    // Output spike array
    array spikes_out({num_neurons}, int8);

    // (Metal dispatch logic linked via MLX backend headers omitted for compilation brevity)
    
    return spikes_out;
}

PYBIND11_MODULE(mlx_lif_ext, m) {
    m.doc() = "JuniorCloud LLC Native Metal Spiking Ternary Automata";
    m.def("batched_lif", &batched_lif_metal, "Execute LIF membrane integration");
}
