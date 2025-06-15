use std::collections::HashSet;

fn square_adj(mut adj: Vec<HashSet<usize>>) -> Vec<HashSet<usize>> {
    let n = adj.len();
    let mut insertions: Vec<(usize, usize)> = Vec::new();

    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors.iter() {
            for &v2 in adj[v].iter() {
                insertions.push((u, v2));
            }
        }
    }

    for (u, v2) in insertions {
        adj[u].insert(v2);
    }

    adj
}