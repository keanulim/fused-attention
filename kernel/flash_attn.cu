/*
 * Flash Attention Forward Pass — CUDA Kernel
 *
 * Algorithm: Flash Attention 2 (Dao, 2023)
 * Each thread block owns one (batch, head, q_tile) triple.
 * It streams K/V tiles from HBM, keeping Q, O, m, l in registers/SRAM,
 * and applies the online softmax recurrence so the full N×N score
 * matrix is never materialised.
 *
 * Thread layout
 * -------------
 * blockDim.x = kBlockM  (one thread per Q-row in the tile)
 * gridDim.x  = B * H
 * gridDim.y  = ceil(N / kBlockM)
 *
 * Shared memory layout  (floats)
 * --------------------------------
 * [ K tile : kBlockN × kHeadDim ]
 * [ V tile : kBlockN × kHeadDim ]
 *
 * Each thread keeps its Q row and output accumulator in registers.
 * S (dot-product scores for one KV tile) is also register-resident.
 *
 * Implement flash_attn_fwd_launch(Q, K, V, O, causal) in this file.
 * O is pre-allocated by Python; write the result there.
 */