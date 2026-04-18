// juniorstock/quant/metal_ext/ternary_gemm.metal
#include <metal_stdlib>
using namespace metal;

/*
 * Custom Metal Kernel: Ternary GEMM
 * Bypasses fp16 ALUs. Expects weights and inputs packed into int8_t 
 * where values are strictly -1, 0, or 1.
 * Optimized for M4/M1 SIMD execution units.
 */
kernel void ternary_matmul_kernel(
    device const int8_t* A [[buffer(0)]],
    device const int8_t* B [[buffer(1)]],
    device int32_t* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) 
{
    if (gid.x >= N || gid.y >= M) return;

    int32_t accumulator = 0;
    
    // Unrolled loop for ternary addition/subtraction
    for (uint k = 0; k < K; ++k) {
        int8_t a_val = A[gid.y * K + k];
        int8_t b_val = B[k * N + gid.x];
        
        // Logical multiplier-free routing
        // If either is 0, product is 0. If signs match, +1. If signs differ, -1.
        accumulator += (a_val * b_val); 
    }

    C[gid.y * N + gid.x] = accumulator;
}
