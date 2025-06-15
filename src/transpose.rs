fn transpose(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    
    let mut adj_t: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut last_seen = vec![usize::MAX; n];
    adj_t.extend((0..n).map(|_| Vec::new()));

    
    for (u, neighbors) in adj.iter().enumerate() {
        for &v in neighbors {
            if u == v { continue; }
            
            if last_seen[u] != v {
                last_seen[u] = v;
                adj_t[v].push(u);
            }
            
        }
    }

    adj_t
}