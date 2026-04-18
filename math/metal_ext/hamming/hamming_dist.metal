// juniorstock/math/metal_ext/hamming/hamming_dist.metal
#include <metal_stdlib>
using namespace metal;

/*
 * Custom Metal Kernel: Batched Hamming Distance
 * Executes bitwise XOR and popcount across packed uint8 memory arrays.
 * Bypasses fp16 ALUs entirely. Optimized for M4 SIMD execution.
 */
kernel void batched_hamming_distance(
    device const uint8_t* query_vector [[buffer(0)]],
    device const uint8_t* memory_matrix [[buffer(1)]],
    device uint32_t* distances [[buffer(2)]],
    constant uint& num_records [[buffer(3)]],
    constant uint& vector_dim [[buffer(4)]],
    uint thread_position [[thread_position_in_grid]]) 
{
    if (thread_position >= num_records) return;

    uint32_t current_distance = 0;
    uint offset = thread_position * vector_dim;

    // Loop unrolling for 8-bit popcount accumulation
    for (uint i = 0; i < vector_dim; ++i) {
        uint8_t q_val = query_vector[i];
        uint8_t m_val = memory_matrix[offset + i];
        
        // ^ is bitwise XOR, popcount computes the number of set bits
        current_distance += popcount(q_val ^ m_val);
    }

    distances[thread_position] = current_distance;
}
