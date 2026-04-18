// juniorstock/math/metal_ext/sparse/csr_ternary.metal
#include <metal_stdlib>
using namespace metal;

/*
 * Custom Metal Kernel: Sparse Ternary Matrix-Vector Multiplication (SpMV)
 * Bypasses dense loops. Only processes non-zero elements (1 or -1) from the CSR arrays.
 * Maximizes thermal efficiency on Apple Silicon SIMD units.
 */
kernel void csr_ternary_spmv(
    device const int8_t* values [[buffer(0)]],        // Non-zero ternary values (-1, 1)
    device const uint32_t* col_indices [[buffer(1)]], // Column index for each value
    device const uint32_t* row_ptrs [[buffer(2)]],    // Row start indices
    device const int8_t* x [[buffer(3)]],             // Dense input vector (b1.58 quantized)
    device int32_t* y [[buffer(4)]],                  // Output vector accumulator
    constant uint& num_rows [[buffer(5)]],
    uint thread_position [[thread_position_in_grid]]) 
{
    if (thread_position >= num_rows) return;

    uint row_start = row_ptrs[thread_position];
    uint row_end = row_ptrs[thread_position + 1];
    
    int32_t dot_product = 0;

    // Loop exclusively over non-zero elements
    for (uint i = row_start; i < row_end; ++i) {
        int8_t weight_val = values[i];
        uint32_t col_idx = col_indices[i];
        int8_t x_val = x[col_idx];
        
        // Ternary scalar expansion without float multiplication
        dot_product += (weight_val * x_val);
    }

    y[thread_position] = dot_product;
}
