# GraphBench-rs
[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/github/workflow/status/LeaderGRL/GraphBench-rs/CI)](https://github.com/LeaderGRL/GraphBench-rs/actions)

**GraphBench-rs** is a highly optimized, Rust-based benchmarking suite specifically designed to evaluate and compare algorithms for graph-related computations, such as detecting universal sinks and performing matrix operations. Developed within the scope of research towards building an efficient Entity Component System (ECS) for a high-performance 3D rendering engine, GraphBench-rs provides a comprehensive set of optimized implementations, each tuned for different graph sizes and hardware architectures.

---

## 📖 Overview

Efficient graph algorithms are critical for numerous applications, particularly in the context of entity management and spatial queries in 3D game engines and simulations. **GraphBench-rs** helps you:

- Evaluate and benchmark different algorithmic strategies.
- Understand the performance implications of various optimization techniques.
- Select the optimal graph-handling implementations tailored to your ECS-based 3D engine requirements.

---

## 🚀 Features

GraphBench-rs comes with a range of optimized implementations, leveraging advanced techniques such as:

- **Sequential Optimization**: Clever linear-time algorithms for universal sink detection.
- **Bitwise Operations**: Ultra-fast bit-parallel implementations ideal for small graphs (≤64 vertices).
- **Cache Optimization**: Transpose-based strategies to leverage CPU cache efficiency.
- **Parallel Processing**: Leveraging Rust’s Rayon library for concurrent computation.
- **SIMD (AVX2)**: Hardware-accelerated implementations for medium-sized graphs (≤256 vertices).
- **Adaptive Algorithms**: Auto-selects the best implementation based on the graph size and available CPU features.

---

## 📊 Algorithms

### 1. Graph Squared (G²)

Computes the transitive closure up to path length 2, essential for:
- Finding indirect dependencies between systems
- Identifying parallelization opportunities
- Detecting dependency chains

**Implementations:**
- `matrix_square_original`: Basic O(V³) implementation
- `matrix_square_bitwise`: Bit-parallel for graphs ≤64 vertices (up to 26x faster)
- `matrix_square_simd`: AVX2 vectorized for medium graphs (up to 43x faster)
- `matrix_square_parallel`: Multi-threaded for large graphs (up to 46x faster)
- `matrix_square_ultra_optimized`: Combines bit manipulation with sparse optimization

### 2. Universal Sink Detection

Identifies systems that depend on all others but have no dependents - typically the final render system.

**Implementations:**
- `find_universal_sink_nested`: O(V²) naive approach
- `find_universal_sink_sequential`: Clever O(V) algorithm
- `find_universal_sink_bitwise`: Bit manipulation variant
- `find_universal_sink_cache_optimized`: Improved memory access patterns

## 📈 Performance Results

### Graph Squared (G²) Performance

| Graph Size | Naive | Optimized | Speedup | Best Algorithm |
|------------|-------|-----------|---------|----------------|
| 64 vertices | 55.7 µs | 8.58 µs | **6.5x** | Bitwise |
| 256 vertices | 3.28 ms | 304 µs | **10.8x** | SIMD |
| 1000 vertices | 174 ms | 4.18 ms | **41.6x** | Parallel |
| 2000 vertices | 932 ms | 20.1 ms | **46.3x** | Parallel |

### Universal Sink Detection Performance

| Graph Size | O(V²) | O(V) | Speedup |
|------------|-------|------|---------|
| 100 vertices | 100 µs | 3 µs | **33x** |
| 1000 vertices | 10 ms | 30 µs | **333x** |
| 10000 vertices | 1000 ms | 300 µs | **3333x** |

## 🚀 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
graph-algorithms-ecs = { git = "https://github.com/yourusername/graph-algorithms-ecs" }

# Optional features
[features]
parallel = ["rayon"]
simd = []  # Requires nightly Rust or specific CPU features
```

## 📊 Benchmarks

Run the complete benchmark suite:

```bash
# All benchmarks
cargo bench

# Specific algorithm
cargo bench graph_squared

# With HTML report
cargo bench -- --save-baseline baseline
open target/criterion/report/index.html

# Quick performance test
cargo test performance --release -- --nocapture
```

### Benchmark Groups

- `graph_squared_all`: Tests all G² implementations
- `universal_sink_*`: Various universal sink scenarios
- `complexity_comparison`: O(V) vs O(V²) vs O(V³) scaling
- `cache_effects`: Impact of different memory access patterns
## 🙏 Acknowledgments

- Based on algorithms from "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein
- Inspired by the Bevy ECS scheduling system
- Thanks to the Rust community for excellent libraries (Rayon, Criterion)
