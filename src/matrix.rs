/// Génère une matrice aléatoire avec possibilité d'avoir un puits universel
pub fn generate_random_matrix(n: usize, has_sink: bool) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; n]; n];
    
    if has_sink {
        // Choisir un sommet aléatoire comme puits
        let sink = n / 2; // Utiliser le sommet du milieu
        
        // Tous les autres sommets pointent vers le puits
        for i in 0..n {
            if i != sink {
                matrix[i][sink] = 1;
            }
        }
        
        // Le puits n'a aucune arête sortante (déjà initialisé à 0)
        
        // Ajouter quelques arêtes aléatoires entre les autres sommets
        for i in 0..n {
            for j in 0..n {
                if i != j && i != sink && j != sink && rand() % 4 == 0 {
                    matrix[i][j] = 1;
                }
            }
        }
    } else {
        // Graphe aléatoire sans puits universel
        for i in 0..n {
            for j in 0..n {
                if i != j && rand() % 3 == 0 {
                    matrix[i][j] = 1;
                }
            }
        }
    }
    
    matrix
}

/// Simple générateur de nombres aléatoires
fn rand() -> usize {
    static mut SEED: usize = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1664525).wrapping_add(1013904223);
        SEED
    }
}

// Vérifie si deux matrices sont égales
pub fn matrices_equal(a: &Vec<Vec<usize>>, b: &Vec<Vec<usize>>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            return false;
        }
        for j in 0..a[i].len() {
            if a[i][j] != b[i][j] {
                return false;
            }
        }
    }
    
    true
}

/// Génère une matrice aléatoire
pub fn generate_random_matrix_density(n: usize, density: f32) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; n]; n];
    let mut seed = 42u64;
    
    for i in 0..n {
        for j in 0..n {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let random = (seed >> 32) as f32 / u32::MAX as f32;
            if i != j && random < density {
                matrix[i][j] = 1;
            }
        }
    }
    
    matrix
}

/// Generate a matrix with a universal sink at given position
pub fn generate_matrix_with_sink(n: usize, sink_pos: usize) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; n]; n];
    
    // All vertices point to sink except sink itself
    for i in 0..n {
        if i != sink_pos {
            matrix[i][sink_pos] = 1;
        }
    }
    
    // Add some random edges between non-sink vertices
    for i in 0..n {
        for j in 0..n {
            if i != j && i != sink_pos && j != sink_pos && rand::random::<f32>() < 0.3 {
                matrix[i][j] = 1;
            }
        }
    }
    
    matrix
}

/// Generate a matrix without a universal sink
pub fn generate_matrix_without_sink(n: usize) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; n]; n];
    
    // Create a cycle to ensure no universal sink
    for i in 0..n {
        matrix[i][(i + 1) % n] = 1;
    }
    
    // Add random edges
    for i in 0..n {
        for j in 0..n {
            if i != j && matrix[i][j] == 0 && rand::random::<f32>() < 0.2 {
                matrix[i][j] = 1;
            }
        }
    }
    
    matrix
}