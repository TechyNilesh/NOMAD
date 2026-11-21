/// Streaming Isolation Forest (SIF) in Rust
///
/// Based on: Liu et al., "Streaming Isolation Forest", PAKDD 2025.
/// https://doi.org/10.1007/978-981-96-8170-9_8
///
/// Each tree uses reservoir sampling to maintain a fixed window of `window_size`
/// instances. New instances are inserted and the tree is regrown locally where
/// needed. Anomaly score is the normalised average path length (2^{-E[h]/c(n)}).

use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ─── Harmonic-number approximation ──────────────────────────────────────────

fn harmonic(i: usize) -> f64 {
    (i as f64).ln() + 0.5772156649
}

/// Average path length of unsuccessful search in a BST of n nodes.
fn c(n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    2.0 * harmonic(n - 1) - 2.0 * (n - 1) as f64 / n as f64
}

// ─── Tree node ───────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Node {
    Leaf {
        instances: Vec<Vec<f64>>,
    },
    Branch {
        feature: usize,
        split_value: f64,
        feature_min: f64,
        feature_max: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    fn mass(&self) -> usize {
        match self {
            Node::Leaf { instances } => instances.len(),
            Node::Branch { left, right, .. } => left.mass() + right.mass(),
        }
    }

    /// Path length for x (number of edges traversed + leaf correction).
    fn path_length(&self, x: &[f64], depth: usize) -> f64 {
        match self {
            Node::Leaf { instances } => {
                let m = instances.len();
                if m <= 1 { depth as f64 } else { depth as f64 + c(m) }
            }
            Node::Branch { feature, split_value, left, right, .. } => {
                if x[*feature] < *split_value {
                    left.path_length(x, depth + 1)
                } else {
                    right.path_length(x, depth + 1)
                }
            }
        }
    }

    fn instances(&self) -> Vec<Vec<f64>> {
        match self {
            Node::Leaf { instances } => instances.clone(),
            Node::Branch { left, right, .. } => {
                let mut v = left.instances();
                v.extend(right.instances());
                v
            }
        }
    }
}

// ─── Build a subtree from a set of instances ────────────────────────────────

fn build_tree(instances: &[Vec<f64>], height: usize, rng: &mut StdRng) -> Node {
    if height == 0 || instances.len() <= 1 {
        return Node::Leaf { instances: instances.to_vec() };
    }

    let n_features = instances[0].len();
    let mut feature_order: Vec<usize> = (0..n_features).collect();
    // shuffle to try features in random order
    for i in (1..feature_order.len()).rev() {
        let j = rng.gen_range(0..=i);
        feature_order.swap(i, j);
    }

    for &feat in &feature_order {
        let min_val = instances.iter().map(|x| x[feat]).fold(f64::INFINITY, f64::min);
        let max_val = instances.iter().map(|x| x[feat]).fold(f64::NEG_INFINITY, f64::max);
        if (max_val - min_val).abs() < 1e-10 {
            continue;
        }
        let split = rng.gen_range(min_val..max_val);
        let left_pts: Vec<Vec<f64>> = instances.iter().filter(|x| x[feat] < split).cloned().collect();
        let right_pts: Vec<Vec<f64>> = instances.iter().filter(|x| x[feat] >= split).cloned().collect();

        let left = Box::new(build_tree(&left_pts, height - 1, rng));
        let right = Box::new(build_tree(&right_pts, height - 1, rng));

        return Node::Branch {
            feature: feat,
            split_value: split,
            feature_min: min_val,
            feature_max: max_val,
            left,
            right,
        };
    }

    // All features constant — make a leaf
    Node::Leaf { instances: instances.to_vec() }
}

// ─── Single streaming isolation tree ────────────────────────────────────────

struct SiTree {
    window_size: usize,
    height_limit: usize,
    instances: Vec<Vec<f64>>,  // reservoir
    n_seen: usize,
    root: Option<Node>,
    rng: StdRng,
}

impl SiTree {
    fn new(window_size: usize, height_limit: usize, seed: u64) -> Self {
        Self {
            window_size,
            height_limit,
            instances: Vec::with_capacity(window_size),
            n_seen: 0,
            root: None,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reservoir-sample x into the window; rebuild tree when window is full.
    fn train(&mut self, x: &[f64]) {
        if self.n_seen < self.window_size {
            // Fill window
            self.instances.push(x.to_vec());
            if self.instances.len() == self.window_size {
                self.root = Some(build_tree(&self.instances, self.height_limit, &mut self.rng));
            }
        } else {
            // Reservoir sampling: replace random slot with probability window_size/n_seen
            let replace_idx = self.rng.gen_range(0..self.n_seen);
            if replace_idx < self.window_size {
                self.instances[replace_idx] = x.to_vec();
                // Rebuild tree after update
                self.root = Some(build_tree(&self.instances, self.height_limit, &mut self.rng));
            }
        }
        self.n_seen += 1;
    }

    fn score(&self, x: &[f64]) -> f64 {
        match &self.root {
            None => 0.5,
            Some(root) => root.path_length(x, 0),
        }
    }
}

// ─── Forest (ensemble of SiTrees) ────────────────────────────────────────────

struct SiForest {
    trees: Vec<SiTree>,
    window_size: usize,
}

impl SiForest {
    fn new(n_trees: usize, window_size: usize, height_limit: usize, seed: u64) -> Self {
        let trees = (0..n_trees)
            .map(|i| SiTree::new(window_size, height_limit, seed.wrapping_add(i as u64 * 6364136223846793005)))
            .collect();
        Self { trees, window_size }
    }

    fn fit_partial(&mut self, x: &[f64]) {
        for tree in &mut self.trees {
            tree.train(x);
        }
    }

    fn score_partial(&self, x: &[f64]) -> f64 {
        let ready: Vec<&SiTree> = self.trees.iter().filter(|t| t.root.is_some()).collect();
        if ready.is_empty() {
            return 0.5;
        }
        let avg_path: f64 = ready.iter().map(|t| t.score(x)).sum::<f64>() / ready.len() as f64;
        let norm = avg_path / c(self.window_size);
        2.0_f64.powf(-norm)
    }
}

// ─── PyO3 wrapper ────────────────────────────────────────────────────────────

#[pyclass]
pub struct StreamingIsolationForest {
    forest: SiForest,
}

#[pymethods]
impl StreamingIsolationForest {
    /// StreamingIsolationForest(n_trees=100, window_size=256, height=None, seed=None)
    #[new]
    #[pyo3(signature = (n_trees=100, window_size=256, height=None, seed=None))]
    fn new(n_trees: usize, window_size: usize, height: Option<usize>, seed: Option<u64>) -> Self {
        let height_limit = height.unwrap_or_else(|| (window_size as f64).log2().ceil() as usize);
        let seed = seed.unwrap_or(42);
        Self {
            forest: SiForest::new(n_trees, window_size, height_limit, seed),
        }
    }

    pub fn fit_partial(&mut self, x: Vec<f64>) {
        self.forest.fit_partial(&x);
    }

    pub fn score_partial(&self, x: Vec<f64>) -> f64 {
        self.forest.score_partial(&x)
    }

    pub fn fit_score_partial(&mut self, x: Vec<f64>) -> f64 {
        self.forest.fit_partial(&x);
        self.forest.score_partial(&x)
    }
}
