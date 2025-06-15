use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Original nested implementation - O(V²) complexity
/// Checks each vertex to see if it's a universal sink
pub fn find_universal_sink_nested(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    
    for candidate in 0..n {
        let mut is_sink = true;
        
        // Check if candidate has no outgoing edges
        for j in 0..n {
            if matrix[candidate][j] == 1 {
                is_sink = false;
                break;
            }
        }
        
        if is_sink {
            // Check if all other vertices point to candidate
            let mut all_point_to_candidate = true;
            for i in 0..n {
                if i != candidate && matrix[i][candidate] == 0 {
                    all_point_to_candidate = false;
                    break;
                }
            }
            
            if all_point_to_candidate {
                return Some(candidate);
            }
        }
    }
    
    None
}

/// Sequential O(V) implementation - the clever algorithm
pub fn find_universal_sink_sequential(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    
    // Step 1: Find a candidate in O(V)
    // Key insight: If u is a universal sink, then:
    // - matrix[u][v] = 0 for all v (u has no outgoing edges)
    // - matrix[v][u] = 1 for all v ≠ u (all point to u)
    
    let mut candidate = 0;
    
    // Smart traversal of the matrix
    // If matrix[i][j] = 1, then i cannot be a sink (has outgoing edge)
    // If matrix[i][j] = 0, then j cannot be a sink (not all point to j)
    for j in 1..n {
        if matrix[candidate][j] == 1 {
            // candidate has an outgoing edge, so cannot be a sink
            candidate = j;
        }
        // If matrix[candidate][j] == 0, keep candidate as j cannot be a sink
    }
    
    // Step 2: Verify if the candidate is really a universal sink in O(V)
    // Check candidate's row (must be all 0s)
    for j in 0..n {
        if matrix[candidate][j] == 1 {
            return None; // Candidate has an outgoing edge
        }
    }
    
    // Check candidate's column (must be all 1s except diagonal)
    for i in 0..n {
        if i != candidate && matrix[i][candidate] == 0 {
            return None; // A vertex doesn't point to candidate
        }
    }
    
    Some(candidate)
}

/// Bitwise optimized version for small graphs (≤64 vertices)
/// Uses bit operations to check conditions faster
pub fn find_universal_sink_bitwise(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    
    if n > 64 {
        // Fall back to sequential for larger graphs
        return find_universal_sink_sequential(matrix);
    }
    
    // Convert matrix to bit representation
    let mut rows: Vec<u64> = vec![0; n];
    let mut cols: Vec<u64> = vec![0; n];
    
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j] == 1 {
                rows[i] |= 1u64 << j;  // Set bit j in row i
                cols[j] |= 1u64 << i;  // Set bit i in column j
            }
        }
    }
    
    // Find candidate using bit operations
    let mut candidate = 0;
    for j in 1..n {
        if (rows[candidate] & (1u64 << j)) != 0 {
            candidate = j;
        }
    }
    
    // Verify candidate using bit operations
    // Check row: should be 0 (no outgoing edges)
    if rows[candidate] != 0 {
        return None;
    }
    
    // Check column: should have all bits set except position 'candidate'
    let expected_col = (1u64 << n) - 1 - (1u64 << candidate);
    if cols[candidate] != expected_col {
        return None;
    }
    
    Some(candidate)
}

/// Cache-optimized version using column-major traversal
/// Better cache locality when checking columns
pub fn find_universal_sink_cache_optimized(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    
    // Transpose matrix for better cache locality when checking columns
    let mut transposed = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            transposed[j][i] = matrix[i][j];
        }
    }
    
    // Find candidate
    let mut candidate = 0;
    for j in 1..n {
        if matrix[candidate][j] == 1 {
            candidate = j;
        }
    }
    
    // Verify candidate with better cache usage
    // Check row (original matrix)
    for j in 0..n {
        if matrix[candidate][j] == 1 {
            return None;
        }
    }
    
    // Check column (transposed matrix - sequential memory access)
    for i in 0..n {
        if i != candidate && transposed[candidate][i] == 0 {
            return None;
        }
    }
    
    Some(candidate)
}

/// Parallel version that checks multiple candidates simultaneously
/// Note: This is mainly for demonstration - the O(V) algorithm is hard to parallelize effectively
pub fn find_universal_sink_parallel(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    if n == 0 {
        return None;
    }
    
    // For small matrices, use sequential version
    if n < 1000 {
        return find_universal_sink_sequential(matrix);
    }
    
    let matrix_arc = Arc::new(matrix);
    let found_sink = Arc::new(AtomicBool::new(false));
    let sink_vertex = Arc::new(AtomicUsize::new(0));
    
    // Find candidate (sequential - hard to parallelize effectively)
    let mut candidate = 0;
    for j in 1..n {
        if matrix[candidate][j] == 1 {
            candidate = j;
        }
    }
    
    // Verify candidate in parallel
    let matrix_ref = matrix_arc.clone();
    let found_ref = found_sink.clone();
    
    // Check row in parallel
    let row_valid = (0..n).into_par_iter()
        .chunks(64) // Process in chunks for better efficiency
        .all(|chunk| {
            if found_ref.load(Ordering::Relaxed) {
                return false; // Early exit if already found invalid
            }
            chunk.iter().all(|&j| matrix_ref[candidate][j] == 0)
        });
    
    if !row_valid {
        return None;
    }
    
    // Check column in parallel
    let col_valid = (0..n).into_par_iter()
        .chunks(64)
        .all(|chunk| {
            chunk.iter().all(|&i| {
                i == candidate || matrix_ref[i][candidate] == 1
            })
        });
    
    if col_valid {
        Some(candidate)
    } else {
        None
    }
}

/// SIMD-optimized version for x86_64
/// Uses AVX2 to check multiple matrix elements simultaneously
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn find_universal_sink_simd(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    
    let n = matrix.len();
    if n == 0 || n > 256 {
        return find_universal_sink_sequential(matrix);
    }
    
    unsafe {
        find_universal_sink_simd_impl(matrix)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn find_universal_sink_simd_impl(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    
    let n = matrix.len();
    let words_per_row = (n + 63) / 64;
    
    // Convert to bit representation aligned for SIMD
    let mut rows: Vec<Vec<u64>> = vec![vec![0u64; words_per_row]; n];
    let mut cols: Vec<Vec<u64>> = vec![vec![0u64; words_per_row]; n];
    
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j] == 1 {
                rows[i][j / 64] |= 1u64 << (j % 64);
                cols[j][i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    
    // Find candidate
    let mut candidate = 0;
    for j in 1..n {
        if matrix[candidate][j] == 1 {
            candidate = j;
        }
    }
    
    // Verify candidate using SIMD
    // Check row - should be all zeros
    let row_valid = if words_per_row >= 4 {
        let zero_vec = _mm256_setzero_si256();
        let mut all_zero = true;
        
        for w in (0..words_per_row).step_by(4) {
            if w + 3 < words_per_row {
                let row_vec = _mm256_loadu_si256(rows[candidate][w..].as_ptr() as *const __m256i);
                let cmp = _mm256_cmpeq_epi64(row_vec, zero_vec);
                if _mm256_movemask_epi8(cmp) != -1 {
                    all_zero = false;
                    break;
                }
            } else {
                // Handle remaining words
                for j in w..words_per_row {
                    if rows[candidate][j] != 0 {
                        all_zero = false;
                        break;
                    }
                }
            }
        }
        all_zero
    } else {
        rows[candidate].iter().all(|&w| w == 0)
    };
    
    if !row_valid {
        return None;
    }
    
    // Check column - complex with SIMD, fall back to regular check
    for i in 0..n {
        if i != candidate && matrix[i][candidate] == 0 {
            return None;
        }
    }
    
    Some(candidate)
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn find_universal_sink_simd(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    find_universal_sink_sequential(matrix)
}

/// Ultra-optimized version combining all techniques
/// Automatically selects the best algorithm based on matrix size
pub fn find_universal_sink_ultra_optimized(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    let n = matrix.len();
    
    if n == 0 {
        return None;
    } else if n == 1 {
        return Some(0); // Single vertex is always a universal sink
    } else if n <= 64 {
        // Small graphs: use bitwise operations
        find_universal_sink_bitwise(matrix)
    } else if n <= 256 {
        // Medium graphs: use SIMD if available
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return find_universal_sink_simd(matrix);
            }
        }
        find_universal_sink_cache_optimized(matrix)
    } else {
        // Large graphs: use cache-optimized sequential
        // (Parallel version has too much overhead for this algorithm)
        find_universal_sink_cache_optimized(matrix)
    }
}

/// Auto-selecting version for production use
pub fn find_universal_sink_auto(matrix: &Vec<Vec<usize>>) -> Option<usize> {
    find_universal_sink_ultra_optimized(matrix)
}

#[cfg(test)]
mod tests {
    use crate::matrix::{generate_matrix_with_sink, generate_matrix_without_sink};

    use super::*;
    
    #[test]
    fn test_all_algorithms_with_sink() {
        let matrix = vec![
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 0],  // Vertex 2 is the sink
            vec![0, 0, 1, 0],
        ];
        
        assert_eq!(find_universal_sink_nested(&matrix), Some(2));
        assert_eq!(find_universal_sink_sequential(&matrix), Some(2));
        assert_eq!(find_universal_sink_bitwise(&matrix), Some(2));
        assert_eq!(find_universal_sink_cache_optimized(&matrix), Some(2));
        assert_eq!(find_universal_sink_parallel(&matrix), Some(2));
        assert_eq!(find_universal_sink_simd(&matrix), Some(2));
        assert_eq!(find_universal_sink_ultra_optimized(&matrix), Some(2));
    }
    
    #[test]
    fn test_all_algorithms_without_sink() {
        let matrix = vec![
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 0],
        ];
        
        assert_eq!(find_universal_sink_nested(&matrix), None);
        assert_eq!(find_universal_sink_sequential(&matrix), None);
        assert_eq!(find_universal_sink_bitwise(&matrix), None);
        assert_eq!(find_universal_sink_cache_optimized(&matrix), None);
        assert_eq!(find_universal_sink_parallel(&matrix), None);
        assert_eq!(find_universal_sink_simd(&matrix), None);
        assert_eq!(find_universal_sink_ultra_optimized(&matrix), None);
    }
    
    #[test]
    fn test_edge_cases() {
        // Empty matrix
        let empty: Vec<Vec<usize>> = vec![];
        assert_eq!(find_universal_sink_sequential(&empty), None);
        assert_eq!(find_universal_sink_ultra_optimized(&empty), None);
        
        // Single vertex
        let single = vec![vec![0]];
        assert_eq!(find_universal_sink_sequential(&single), Some(0));
        assert_eq!(find_universal_sink_ultra_optimized(&single), Some(0));
        
        // Two vertices with sink
        let two_with_sink = vec![
            vec![0, 1],
            vec![0, 0],
        ];
        assert_eq!(find_universal_sink_sequential(&two_with_sink), Some(1));
        assert_eq!(find_universal_sink_ultra_optimized(&two_with_sink), Some(1));
    }
    
    #[test]
    fn test_large_matrices() {
        // Test with various sizes
        for size in [10, 50, 100, 200] {
            // Test with sink
            let sink_pos = size / 2;
            let matrix_with_sink = generate_matrix_with_sink(size, sink_pos);
            
            let result_seq = find_universal_sink_sequential(&matrix_with_sink);
            let result_ultra = find_universal_sink_ultra_optimized(&matrix_with_sink);
            
            assert_eq!(result_seq, Some(sink_pos));
            assert_eq!(result_ultra, Some(sink_pos));
            
            // Test without sink
            let matrix_without_sink = generate_matrix_without_sink(size);
            
            let result_seq = find_universal_sink_sequential(&matrix_without_sink);
            let result_ultra = find_universal_sink_ultra_optimized(&matrix_without_sink);
            
            assert_eq!(result_seq, None);
            assert_eq!(result_ultra, None);
        }
    }
}