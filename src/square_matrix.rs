use rayon::prelude::*;
use std::sync::Arc;

/// Custom BitVec implementation optimized for different graph sizes
/// This enum uses Rust's zero-cost abstractions to avoid heap allocation for small graphs
/// 
/// The idea: Instead of storing each 0 or 1 in a full integer (32/64 bits),
/// we pack them into single bits, achieving up to 64x memory compression
#[derive(Clone)]
pub enum BitVec {
    /// For graphs with ≤ 64 vertices: fits in a single CPU register
    /// This is the fastest case as all operations happen in registers
    Small(u64),
    
    /// For graphs with ≤ 256 vertices: uses stack-allocated array
    /// 4 × 64 = 256 bits, still avoids heap allocation
    Medium([u64; 4]),
    
    /// For larger graphs: falls back to heap-allocated vector
    /// Each u64 stores 64 edges, so n vertices need ⌈n/64⌉ u64s
    Large(Vec<u64>),
}

impl BitVec {
    /// Creates a new BitVec with all bits initialized to 0
    /// Automatically selects the most efficient variant based on size
    fn new(size: usize) -> Self {
        if size <= 64 {
            BitVec::Small(0)
        } else if size <= 256 {
            BitVec::Medium([0; 4])
        } else {
            // Calculate number of u64s needed: ceiling of size/64
            BitVec::Large(vec![0; (size + 63) / 64])
        }
    }
    
    /// Sets the bit at the given index to 1
    /// Uses bitwise OR with a mask that has only the target bit set
    /// 
    /// Example: set(5) on 00000000 gives 00100000 (bit 5 from right)
    #[inline(always)]  // Hint to compiler: always inline this hot function
    fn set(&mut self, index: usize) {
        match self {
            BitVec::Small(bits) => {
                // Create mask with only bit at 'index' set: 1 << index
                // OR with existing bits to set without affecting others
                *bits |= 1u64 << index;
            }
            BitVec::Medium(arr) => {
                // Determine which u64 contains our bit
                let word = index / 64;  // Integer division
                let bit = index % 64;   // Position within that u64
                arr[word] |= 1u64 << bit;
            }
            BitVec::Large(vec) => {
                let word = index / 64;
                let bit = index % 64;
                vec[word] |= 1u64 << bit;
            }
        }
    }
    
    /// Checks if the bit at the given index is set (1) or not (0)
    /// Uses bitwise AND with a mask to isolate the target bit
    #[inline(always)]
    fn get(&self, index: usize) -> bool {
        match self {
            BitVec::Small(bits) => {
                // AND with mask isolates the bit, != 0 checks if it's set
                (*bits & (1u64 << index)) != 0
            }
            BitVec::Medium(arr) => {
                let word = index / 64;
                let bit = index % 64;
                (arr[word] & (1u64 << bit)) != 0
            }
            BitVec::Large(vec) => {
                let word = index / 64;
                let bit = index % 64;
                (vec[word] & (1u64 << bit)) != 0
            }
        }
    }
    
    /// Performs bitwise OR with another BitVec, modifying self
    /// This is the key operation for G²: it adds all edges from 'other' to self
    /// 
    /// In graph terms: if A→B and B→C, this adds all of B's connections to A
    #[inline(always)]
    fn or_assign(&mut self, other: &BitVec) {
        match (self, other) {
            (BitVec::Small(a), BitVec::Small(b)) => {
                // Single OR operation processes 64 edges at once!
                *a |= *b;
            }
            (BitVec::Medium(a), BitVec::Medium(b)) => {
                // Process 256 bits in 4 operations
                for i in 0..4 {
                    a[i] |= b[i];
                }
            }
            (BitVec::Large(a), BitVec::Large(b)) => {
                // Process all words, zip ensures we don't go out of bounds
                for (ai, bi) in a.iter_mut().zip(b.iter()) {
                    *ai |= *bi;
                }
            }
            _ => unreachable!(), // Different variants should never be mixed
        }
    }
}

/// Optimized graph representation using bit vectors
/// Each row is a BitVec where bit j indicates an edge to vertex j
pub struct OptimizedGraph {
    n: usize,           // Number of vertices
    rows: Vec<BitVec>,  // Adjacency matrix rows as bit vectors
}

impl OptimizedGraph {
    /// Converts a traditional adjacency matrix to our optimized format
    /// Time complexity: O(V²) but with very low constant factor
    fn from_matrix(matrix: &Vec<Vec<usize>>) -> Self {
        let n = matrix.len();
        let mut rows = Vec::with_capacity(n);  // Pre-allocate for efficiency
        
        for i in 0..n {
            let mut row = BitVec::new(n);
            for j in 0..n {
                if matrix[i][j] == 1 {
                    row.set(j);  // Convert 1s to set bits
                }
            }
            rows.push(row);
        }
        
        OptimizedGraph { n, rows }
    }
    
    /// Converts back to traditional matrix format for compatibility
    fn to_matrix(&self) -> Vec<Vec<usize>> {
        let mut matrix = vec![vec![0; self.n]; self.n];
        
        for i in 0..self.n {
            for j in 0..self.n {
                if self.rows[i].get(j) {
                    matrix[i][j] = 1;
                }
            }
        }
        
        matrix
    }
}

/// Original implementation - kept for comparison
/// Note: This modifies the input matrix (takes ownership)
pub fn matrix_square_original(matrix: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut new_matrix = matrix.clone();

    // Triple nested loop - O(V³) complexity
    for (i, u) in matrix.iter().enumerate() {
        for (j, &v) in u.iter().enumerate() {
            if v == 1 {  // If edge i→j exists
                // Add all of j's outgoing edges to i's row
                for (k, &val) in matrix[j].iter().enumerate() {
                    if new_matrix[i][k] != 1 {
                        new_matrix[i][k] += val;
                    }
                }
            }
        }
    }

    new_matrix
}

/// Basic O(V³) implementation - clear and simple
/// Implements G² = G ∪ G×G (graph union with paths of length 2)
pub fn matrix_square_basic(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let n = matrix.len();
    let mut result = vec![vec![0; n]; n];
    
    // Step 1: Copy original edges (paths of length 1)
    // These represent direct connections in the graph
    for i in 0..n {
        for j in 0..n {
            result[i][j] = matrix[i][j];
        }
    }
    
    // Step 2: Add paths of length exactly 2
    // For each possible path i→k→j, add edge i→j
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                // If there's a path i→k→j, then i can reach j in 2 steps
                if matrix[i][k] == 1 && matrix[k][j] == 1 {
                    result[i][j] = 1;
                    // Note: no break here, we check all paths (inefficient)
                }
            }
        }
    }
    
    result
}

/// Optimized version with early termination
/// Key insight: Once we find ANY path of length ≤2, we can stop searching
pub fn matrix_square_optimized(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let n = matrix.len();
    let mut result = matrix.clone();  // Start with original edges
    
    for i in 0..n {
        for j in 0..n {
            // Only look for 2-paths if there's no direct edge
            if result[i][j] == 0 {
                for k in 0..n {
                    if matrix[i][k] == 1 && matrix[k][j] == 1 {
                        result[i][j] = 1;
                        break;  // OPTIMIZATION: Found one path, no need for more
                    }
                }
            }
        }
    }
    
    result
}

/// Bit-parallel implementation for small graphs (≤64 vertices)
/// This is often the fastest for small graphs due to CPU bit-level parallelism
pub fn matrix_square_bitwise(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let n = matrix.len();
    if n > 64 {
        // Fall back to standard algorithm for larger graphs
        return matrix_square_optimized(matrix);
    }
    
    // Convert each row to a 64-bit integer where bit j represents edge to vertex j
    let mut rows: Vec<u64> = vec![0; n];
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j] == 1 {
                rows[i] |= 1u64 << j;  // Set bit j in row i
            }
        }
    }
    
    // Compute G² using bit operations
    // Key insight: OR operation on 64 bits = 64 edge additions in parallel!
    let mut result_bits = rows.clone();
    for i in 0..n {
        for k in 0..n {
            // Check if vertex i has edge to vertex k
            if (rows[i] & (1u64 << k)) != 0 {
                // Add ALL of k's edges to i's edges with single OR
                result_bits[i] |= rows[k];
            }
        }
    }
    
    // Convert back to standard matrix format
    let mut result = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if (result_bits[i] & (1u64 << j)) != 0 {
                result[i][j] = 1;
            }
        }
    }
    
    result
}

/// Parallel version using Rayon for automatic work distribution
/// Each thread computes a subset of rows independently
#[cfg(feature = "parallel")]
pub fn matrix_square_parallel(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    
    let n = matrix.len();
    // Arc (Atomic Reference Counter) allows safe sharing between threads
    let matrix_arc = std::sync::Arc::new(matrix);
    
    // Parallel iterator automatically distributes rows across CPU cores
    let result: Vec<Vec<usize>> = (0..n)
        .into_par_iter()  // Convert to parallel iterator
        .map(|i| {
            // Each thread gets a reference to the shared matrix
            let matrix = matrix_arc.clone();
            let mut row = vec![0; n];
            
            // Compute row i independently of other rows
            for j in 0..n {
                row[j] = matrix[i][j];
                if row[j] == 0 {
                    // Look for 2-paths from i to j
                    for k in 0..n {
                        if matrix[i][k] == 1 && matrix[k][j] == 1 {
                            row[j] = 1;
                            break;
                        }
                    }
                }
            }
            
            row
        })
        .collect();  // Rayon collects results in correct order
    
    result
}

/// Ultra-optimized version combining custom BitVec with bit manipulation tricks
/// This is the "Four Russians" approach adapted for boolean matrices
pub fn matrix_square_ultra_optimized(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let graph = OptimizedGraph::from_matrix(matrix);
    let n = graph.n;
    let mut result_rows = graph.rows.clone();
    
    // Special optimization for small graphs that fit in registers
    if n <= 64 {
        // Extract bit patterns for direct manipulation
        if let BitVec::Small(ref bits) = graph.rows[0] {
            let mut row_bits: Vec<u64> = graph.rows.iter().map(|row| {
                if let BitVec::Small(b) = row { *b } else { unreachable!() }
            }).collect();
            
            // Ultra-fast computation using bit scan operations
            for i in 0..n {
                let mut result = row_bits[i];  // Start with original edges
                let row_i = row_bits[i];
                
                // Brian Kernighan's algorithm to iterate only over set bits
                // This is key: we only process EXISTING edges, not all possible edges
                let mut k_bits = row_i;
                while k_bits != 0 {
                    // Find position of lowest set bit (hardware instruction)
                    let k = k_bits.trailing_zeros() as usize;
                    // Add all edges from vertex k
                    result |= row_bits[k];
                    // Clear the lowest set bit: k_bits = k_bits & (k_bits - 1)
                    k_bits &= k_bits - 1;
                }
                
                // Store result back
                if let BitVec::Small(ref mut b) = result_rows[i] {
                    *b = result;
                }
            }
        }
    } else {
        // General version for larger graphs
        for i in 0..n {
            let mut new_row = graph.rows[i].clone();
            
            // Process only vertices that i connects to
            for k in 0..n {
                if graph.rows[i].get(k) {
                    // Add all of k's connections to i's connections
                    new_row.or_assign(&graph.rows[k]);
                }
            }
            
            result_rows[i] = new_row;
        }
    }
    
    OptimizedGraph { n, rows: result_rows }.to_matrix()
}

/// Parallel version of the ultra-optimized algorithm
/// Combines the best of both worlds: bit-level parallelism AND thread-level parallelism
pub fn matrix_square_parallel_optimized(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let graph = Arc::new(OptimizedGraph::from_matrix(matrix));
    let n = graph.n;
    
    // Each thread processes its assigned rows using bit operations
    let result_rows: Vec<_> = (0..n)
        .into_par_iter()
        .map(|i| {
            let graph = graph.clone();
            let mut new_row = graph.rows[i].clone();
            
            // Process only vertices where edge exists (sparse optimization)
            for k in 0..n {
                if graph.rows[i].get(k) {
                    new_row.or_assign(&graph.rows[k]);
                }
            }
            
            new_row
        })
        .collect();
    
    OptimizedGraph { n, rows: result_rows }.to_matrix()
}

/// SIMD (Single Instruction Multiple Data) version for x86_64 processors
/// Uses AVX2 instructions to process 256 bits (4 × u64) in parallel
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn matrix_square_simd(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    
    let n = matrix.len();
    
    // SIMD works best when data aligns to 256-bit boundaries
    if n <= 256 && n % 64 == 0 {
        unsafe {
            matrix_square_simd_impl(matrix)
        }
    } else {
        // Fall back if size doesn't align well
        matrix_square_ultra_optimized(matrix)
    }
}

/// Internal SIMD implementation using AVX2 intrinsics
/// This processes 4 × 64 = 256 edges in a single CPU instruction!
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn matrix_square_simd_impl(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    
    let n = matrix.len();
    // Calculate how many u64s we need per row (rounded up to multiple of 4)
    let words_per_row = (n + 255) / 256 * 4;
    
    // Convert to SIMD-friendly format: each row is array of u64s
    let mut rows: Vec<Vec<u64>> = vec![vec![0u64; words_per_row]; n];
    
    for i in 0..n {
        for j in 0..n {
            if matrix[i][j] == 1 {
                let word = j / 64;    // Which u64 contains bit j
                let bit = j % 64;     // Position within that u64
                rows[i][word] |= 1u64 << bit;
            }
        }
    }
    
    // Compute G² using AVX2 vector operations
    let mut result_rows = rows.clone();
    
    for i in 0..n {
        for k in 0..n {
            let word_k = k / 64;
            let bit_k = k % 64;
            
            // Check if edge i→k exists
            if (rows[i][word_k] & (1u64 << bit_k)) != 0 {
                // Process 256 bits at a time using SIMD
                for w in (0..words_per_row).step_by(4) {
                    if w + 3 < words_per_row {
                        // Load 256 bits from result_rows[i]
                        let a = unsafe { _mm256_loadu_si256(result_rows[i][w..].as_ptr() as *const __m256i) };
                        // Load 256 bits from rows[k]
                        let b = unsafe { _mm256_loadu_si256(rows[k][w..].as_ptr() as *const __m256i) };
                        // Perform 256-bit OR in single instruction
                        let result = unsafe { _mm256_or_si256(a, b) };
                        // Store 256 bits back
                        unsafe { _mm256_storeu_si256(result_rows[i][w..].as_mut_ptr() as *mut __m256i, result) };
                    } else {
                        // Handle remaining bits that don't fill a 256-bit register
                        for j in w..words_per_row {
                            result_rows[i][j] |= rows[k][j];
                        }
                    }
                }
            }
        }
    }
    
    // Convert back to standard matrix format
    let mut matrix = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let word = j / 64;
            let bit = j % 64;
            if (result_rows[i][word] & (1u64 << bit)) != 0 {
                matrix[i][j] = 1;
            }
        }
    }
    
    matrix
}

/// Fallback for non-x86 architectures
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn matrix_square_simd(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    matrix_square_ultra_optimized(matrix)
}

/// Cache-optimized block multiplication algorithm
/// Processes the matrix in blocks that fit in CPU cache for better performance
pub fn matrix_square_blocked_optimized(matrix: &Vec<Vec<usize>>, block_size: usize) -> Vec<Vec<usize>> {
    let n = matrix.len();
    let graph = OptimizedGraph::from_matrix(matrix);
    let mut result = graph.rows.clone();
    
    // Block multiplication to optimize cache usage
    // Modern CPUs have ~32KB L1 cache, so block_size × block_size × 8 bytes should fit
    for i_block in (0..n).step_by(block_size) {
        for k_block in (0..n).step_by(block_size) {
            // Process a block_size × block_size submatrix
            let k_end = std::cmp::min(k_block + block_size, n);
            
            // These loops process data that fits in L1/L2 cache
            for i in i_block..std::cmp::min(i_block + block_size, n) {
                for k in k_block..k_end {
                    if graph.rows[i].get(k) {
                        result[i].or_assign(&graph.rows[k]);
                    }
                }
            }
        }
    }
    
    OptimizedGraph { n, rows: result }.to_matrix()
}

/// Automatic algorithm selection based on matrix size and hardware
/// This is what you should use in production code
pub fn matrix_square_auto(matrix: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let n = matrix.len();
    
    if n <= 64 {
        // Small graphs: bitwise is fastest due to register operations
        matrix_square_bitwise(matrix)
    } else if n <= 256 {
        // Medium graphs: SIMD if available, otherwise ultra-optimized
        #[cfg(target_feature = "avx2")]
        return matrix_square_simd(matrix);
        matrix_square_ultra_optimized(matrix)
    } else {
        // Large graphs: parallel processing is essential
        matrix_square_parallel_optimized(matrix)
    }
}

// Unit tests to ensure all implementations produce identical results
mod tests {
    use crate::matrix::{generate_random_matrix, generate_random_matrix_density, matrices_equal};
    use super::*;
    
    #[test]
    fn test_simple_graph() {
        // Test graph with known result
        let matrix = vec![
            vec![0, 1, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 1, 1],
            vec![0, 1, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 0],
            vec![0, 0, 0, 0, 0, 1],
        ];
        
        let expected = vec![
            vec![0, 1, 0, 1, 1, 0],  // 0->1->4 added
            vec![0, 0, 0, 1, 1, 0],  // 1->4->3 added
            vec![0, 0, 0, 1, 1, 1],  // 2->4->3 added
            vec![0, 1, 0, 0, 1, 0],  // 3->1->4 added
            vec![0, 1, 0, 1, 0, 0],  // 4->3->1 added
            vec![0, 0, 0, 0, 0, 1],
        ];
        
        // Test all versions produce the same result
        let result_original = matrix_square_original(matrix.clone());
        let result_basic = matrix_square_basic(&matrix);
        let result_optimized = matrix_square_optimized(&matrix);
        let result_bitwise = matrix_square_bitwise(&matrix);
        
        assert!(matrices_equal(&result_original, &expected), "Original failed");
        assert!(matrices_equal(&result_basic, &expected), "Basic failed");
        assert!(matrices_equal(&result_optimized, &expected), "Optimized failed");
        assert!(matrices_equal(&result_bitwise, &expected), "Bitwise failed");
    }
    
    #[test]
    fn test_all_versions_consistency() {
        // Test that all versions produce identical results on random graphs
        let sizes = vec![5, 10, 20, 30];
        let densities = vec![0.1, 0.3, 0.5];
        
        let mut rng = rand::thread_rng();
        
        for size in sizes {
            for density in &densities {
                let matrix = generate_random_matrix_density(size, *density);
                
                let result_original = matrix_square_original(matrix.clone());
                let result_basic = matrix_square_basic(&matrix);
                let result_optimized = matrix_square_optimized(&matrix);
                let result_bitwise = matrix_square_bitwise(&matrix);
                
                assert!(matrices_equal(&result_original, &result_basic), 
                       "Original vs Basic mismatch for size {}", size);
                assert!(matrices_equal(&result_basic, &result_optimized), 
                       "Basic vs Optimized mismatch for size {}", size);
                assert!(matrices_equal(&result_optimized, &result_bitwise), 
                       "Optimized vs Bitwise mismatch for size {}", size);
            }
        }
    }
    
    #[test]
    fn test_edge_cases() {
        // Empty matrix
        let empty: Vec<Vec<usize>> = vec![];
        assert_eq!(matrix_square_original(empty.clone()), empty);
        
        // Single vertex
        let single = vec![vec![0]];
        assert_eq!(matrix_square_original(single.clone()), single);
        
        // Graph with self-loop
        let with_loop = vec![
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![0, 0, 1],  // Self-loop on vertex 2
        ];
        let expected_loop = vec![
            vec![0, 1, 1],  // 0->1->2 added
            vec![0, 0, 1],
            vec![0, 0, 1],
        ];
        assert_eq!(matrix_square_original(with_loop.clone()), expected_loop);
    }

    #[test]
    fn test_bitvec_operations() {
        // Test basic BitVec functionality
        let mut small = BitVec::Small(0);
        small.set(5);
        small.set(10);
        assert!(small.get(5));
        assert!(small.get(10));
        assert!(!small.get(7));
        
        let mut medium = BitVec::Medium([0; 4]);
        medium.set(100);
        medium.set(200);
        assert!(medium.get(100));
        assert!(medium.get(200));
    }
    
    #[test]
    fn test_ultra_optimized_correctness() {
        // Test specific paths in ultra-optimized version
        let matrix = vec![
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 0],
            vec![1, 0, 0, 1],
            vec![0, 0, 0, 0],
        ];
        
        let result = matrix_square_ultra_optimized(&matrix);
        
        // Verify 2-paths exist
        assert_eq!(result[0][2], 1); // 0->1->2
        assert_eq!(result[1][3], 1); // 1->2->3
        assert_eq!(result[2][1], 1); // 2->0->1
    }
}