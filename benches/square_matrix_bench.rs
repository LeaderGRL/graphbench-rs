use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use graph::{matrix::{generate_random_matrix, generate_random_matrix_density}, square_matrix::{matrix_square_auto, matrix_square_basic, matrix_square_bitwise, matrix_square_blocked_optimized, matrix_square_optimized, matrix_square_original, matrix_square_parallel_optimized, matrix_square_ultra_optimized}};

fn benchmark_all_versions(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_squared_all");
    
    // Test sur différentes tailles
    for size in [16, 32, 64, 128, 256].iter() {
        let matrix = generate_random_matrix_density(*size, 0.1);
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // Version originale
        group.bench_with_input(
            BenchmarkId::new("original", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_original(black_box(m.clone())))
        );
        
        // Version basique
        group.bench_with_input(
            BenchmarkId::new("basic", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_basic(black_box(m)))
        );
        
        // Version optimisée avec early break
        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_optimized(black_box(m)))
        );
        
        // Version bitwise simple
        if *size <= 64 {
            group.bench_with_input(
                BenchmarkId::new("bitwise_simple", size),
                &matrix,
                |b, m| b.iter(|| matrix_square_bitwise(black_box(m)))
            );
        }
        
        // Version ultra-optimisée avec BitVec
        group.bench_with_input(
            BenchmarkId::new("ultra_optimized", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_ultra_optimized(black_box(m)))
        );
        
        // Version SIMD
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if *size <= 256 {
            use graph::square_matrix::matrix_square_simd;

            group.bench_with_input(
                BenchmarkId::new("simd", size),
                &matrix,
                |b, m| b.iter(|| matrix_square_simd(black_box(m)))
            );
        }
        
        // Version parallèle
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_parallel_optimized(black_box(m)))
        );
        
        // Version auto (choisit la meilleure)
        group.bench_with_input(
            BenchmarkId::new("auto", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_auto(black_box(m)))
        );
    }
    
    group.finish();
}

fn benchmark_large_graphs(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_squared_large");
    group.sample_size(10); // Moins d'échantillons pour les grands graphes
    
    for size in [500, 1000, 2000].iter() {
        let matrix = generate_random_matrix_density(*size, 0.05); // Densité plus faible
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // Seulement les versions qui peuvent gérer de grands graphes efficacement
        
        group.bench_with_input(
            BenchmarkId::new("optimized", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_optimized(black_box(m)))
        );
        
        group.bench_with_input(
            BenchmarkId::new("ultra_optimized", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_ultra_optimized(black_box(m)))
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_parallel_optimized(black_box(m)))
        );
        
        group.bench_with_input(
            BenchmarkId::new("blocked", size),
            &matrix,
            |b, m| b.iter(|| matrix_square_blocked_optimized(black_box(m), 64))
        );
    }
    
    group.finish();
}

fn benchmark_density_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_scaling");
    let size = 256;
    
    for density in [0.01, 0.05, 0.1, 0.2, 0.5].iter() {
        let matrix = generate_random_matrix_density(size, *density);
        
        group.bench_with_input(
            BenchmarkId::new("ultra_optimized", format!("{}%", (*density * 100.0) as u32)),
            &matrix,
            |b, m| b.iter(|| matrix_square_ultra_optimized(black_box(m)))
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}%", (*density * 100.0) as u32)),
            &matrix,
            |b, m| b.iter(|| matrix_square_parallel_optimized(black_box(m)))
        );
    }
    
    group.finish();
}

fn benchmark_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");
    
    // Tester différentes tailles de blocs
    let size = 1024;
    let matrix = generate_random_matrix_density(size, 0.05);
    
    for block_size in [32, 64, 128, 256].iter() {
        group.bench_with_input(
            BenchmarkId::new("blocked", format!("block_{}", block_size)),
            &matrix,
            |b, m| b.iter(|| matrix_square_blocked_optimized(black_box(m), *block_size))
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_all_versions,
    benchmark_large_graphs,
    benchmark_density_scaling,
    benchmark_cache_effects
);
criterion_main!(benches);