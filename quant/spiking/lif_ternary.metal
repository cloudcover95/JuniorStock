// juniorstock/quant/spiking/lif_ternary.metal
#include <metal_stdlib>
using namespace metal;

/*
 * Custom Metal Kernel: Ternary Spiking Neurons
 * Executes LIF membrane potential updates without float multiplication.
 * Outputs sparse spike trains {-1, 0, 1} for ultra-low power routing.
 */
kernel void lif_ternary_kernel(
    device const int8_t* spikes_in [[buffer(0)]],
    device const int8_t* weights [[buffer(1)]],
    device int32_t* membrane_potentials [[buffer(2)]],
    device int8_t* spikes_out [[buffer(3)]],
    constant uint& num_neurons [[buffer(4)]],
    constant uint& input_dim [[buffer(5)]],
    constant int32_t& beta_decay [[buffer(6)]],  // Scaled integer for leakage
    constant int32_t& threshold [[buffer(7)]],
    uint thread_position [[thread_position_in_grid]]) 
{
    if (thread_position >= num_neurons) return;

    int32_t u_current = membrane_potentials[thread_position];
    
    // Apply leakage (integer bit-shift approximation for beta decay)
    u_current = u_current - (u_current >> beta_decay);

    // Synaptic integration: purely additive ternary routing
    uint offset = thread_position * input_dim;
    for (uint i = 0; i < input_dim; ++i) {
        int8_t s_in = spikes_in[i];
        int8_t w = weights[offset + i];
        u_current += (w * s_in); 
    }

    // Threshold evaluation & Spike generation
    int8_t s_out = 0;
    if (u_current > threshold) {
        s_out = 1;
        u_current -= threshold; // Hard reset via subtraction
    } else if (u_current < -threshold) {
        s_out = -1;
        u_current += threshold;
    }

    // Write back state
    membrane_potentials[thread_position] = u_current;
    spikes_out[thread_position] = s_out;
}
