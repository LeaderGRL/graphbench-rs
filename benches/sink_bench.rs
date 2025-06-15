use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use graph::{matrix::{generate_matrix_with_sink, generate_matrix_without_sink}, sink::{find_universal_sink_auto, find_universal_sink_bitwise, find_universal_sink_cache_optimized, find_universal_sink_nested, find_universal_sink_parallel, find_universal_sink_sequential, find_universal_sink_ultra_optimized}};

/// Benchmark all algorithms on matrices WITH universal sinks
fn bench_with_sink(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_sink_with_sink");
    
    // Test different matrix sizes
    for size in [10, 20, 50, 64, 100, 128, 256, 500, 1000].iter() {
        // Place sink at different positions to test worst-case scenarios
        let sink_positions = match *size {
            s if s <= 64 => vec![0, s/2, s-1], // Beginning, middle, end
            _ => vec![size/2], // Just middle for large matrices
        };
        
        for sink_pos in sink_positions {
            let matrix = generate_matrix_with_sink(*size, sink_pos);
            group.throughput(Throughput::Elements((*size * *size) as u64));
            
            // Original nested (only for small sizes due to O(V²))
            group.bench_with_input(
                BenchmarkId::new(format!("nestedl/sink_{}", sink_pos), size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_nested(black_box(m)))
            );
            
            // Sequential O(V) - always test
            group.bench_with_input(
                BenchmarkId::new(format!("sequential/sink_{}", sink_pos), size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_sequential(black_box(m)))
            );
            
            // Bitwise (only for n ≤ 64)
            if *size <= 64 {
                group.bench_with_input(
                    BenchmarkId::new(format!("bitwise/sink_{}", sink_pos), size),
                    &matrix,
                    |b, m| b.iter(|| find_universal_sink_bitwise(black_box(m)))
                );
            }
            
            // Cache-optimized
            group.bench_with_input(
                BenchmarkId::new(format!("cache_optimized/sink_{}", sink_pos), size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_cache_optimized(black_box(m)))
            );
            
            // Parallel (only for larger matrices where it might help)
            if *size >= 100 {
                group.bench_with_input(
                    BenchmarkId::new(format!("parallel/sink_{}", sink_pos), size),
                    &matrix,
                    |b, m| b.iter(|| find_universal_sink_parallel(black_box(m)))
                );
            }
            
            // SIMD
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if *size <= 256 {
                use graph::sink::find_universal_sink_simd;

                group.bench_with_input(
                    BenchmarkId::new(format!("simd/sink_{}", sink_pos), size),
                    &matrix,
                    |b, m| b.iter(|| find_universal_sink_simd(black_box(m)))
                );
            }
            
            // Ultra-optimized
            group.bench_with_input(
                BenchmarkId::new(format!("ultra_optimized/sink_{}", sink_pos), size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_ultra_optimized(black_box(m)))
            );
            
            // Auto
            group.bench_with_input(
                BenchmarkId::new(format!("auto/sink_{}", sink_pos), size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_auto(black_box(m)))
            );
        }
    }
    
    group.finish();
}

/// Benchmark all algorithms on matrices WITHOUT universal sinks
fn bench_without_sink(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_sink_without_sink");
    
    for size in [10, 20, 50, 64, 100, 256, 500, 1000].iter() {
        let matrix = generate_matrix_without_sink(*size);
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // Original nested (only for small sizes)
        group.bench_with_input(
            BenchmarkId::new("nested", size),
            &matrix,
            |b, m| b.iter(|| find_universal_sink_nested(black_box(m)))
        );
        
        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &matrix,
            |b, m| b.iter(|| find_universal_sink_sequential(black_box(m)))
        );
        
        // Bitwise
        if *size <= 64 {
            group.bench_with_input(
                BenchmarkId::new("bitwise", size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_bitwise(black_box(m)))
            );
        }
        
        // Cache-optimized
        group.bench_with_input(
            BenchmarkId::new("cache_optimized", size),
            &matrix,
            |b, m| b.iter(|| find_universal_sink_cache_optimized(black_box(m)))
        );
        
        // Parallel
        if *size >= 100 {
            group.bench_with_input(
                BenchmarkId::new("parallel", size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_parallel(black_box(m)))
            );
        }
        
        // SIMD
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if *size <= 256 {
            use graph::sink::find_universal_sink_simd;

            group.bench_with_input(
                BenchmarkId::new("simd", size),
                &matrix,
                |b, m| b.iter(|| find_universal_sink_simd(black_box(m)))
            );
        }
        
        // Ultra-optimized
        group.bench_with_input(
            BenchmarkId::new("ultra_optimized", size),
            &matrix,
            |b, m| b.iter(|| find_universal_sink_ultra_optimized(black_box(m)))
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_with_sink,
    bench_without_sink,
);
criterion_main!(benches);