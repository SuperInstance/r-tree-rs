//! Range query and k-NN search for the R-tree.

use crate::node::{BBox, Entry, Node};

/// Find all entries whose bounding boxes overlap with the query bbox.
pub fn range_query(node: &Node, query: &BBox) -> Vec<usize> {
    let mut results = Vec::new();
    range_query_rec(node, query, &mut results);
    results
}

fn range_query_rec(node: &Node, query: &BBox, results: &mut Vec<usize>) {
    for entry in &node.entries {
        if !entry.bbox().overlaps(query) {
            continue;
        }
        match entry {
            Entry::Leaf { id, .. } => {
                results.push(*id);
            }
            Entry::Internal { child, .. } => {
                range_query_rec(child, query, results);
            }
        }
    }
}

/// Find all entries whose bounding boxes contain the given point.
pub fn point_query(node: &Node, px: f64, py: f64) -> Vec<usize> {
    let mut results = Vec::new();
    point_query_rec(node, px, py, &mut results);
    results
}

fn point_query_rec(node: &Node, px: f64, py: f64, results: &mut Vec<usize>) {
    for entry in &node.entries {
        let bbox = entry.bbox();
        if !bbox.contains_point(px, py) {
            continue;
        }
        match entry {
            Entry::Leaf { id, .. } => {
                results.push(*id);
            }
            Entry::Internal { child, .. } => {
                point_query_rec(child, px, py, results);
            }
        }
    }
}

/// k-nearest neighbor search using a priority-queue approach.
///
/// Returns up to k entries sorted by distance (nearest first).
/// Distance is measured from the query point to the bbox center.
pub fn knn_query(node: &Node, point: (f64, f64), k: usize) -> Vec<usize> {
    let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
    knn_rec(node, point, k, &mut heap);
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    heap.into_iter().map(|(_, id)| id).collect()
}

fn knn_rec(node: &Node, point: (f64, f64), k: usize, heap: &mut Vec<(f64, usize)>) {
    // Collect entries with distances and sort by distance
    let mut entries: Vec<(f64, &Entry)> = node.entries.iter().map(|e| {
        (e.bbox().dist_sq_to_center(point), e)
    }).collect();
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (dist, entry) in entries {
        // Prune if this entry is farther than the k-th nearest
        if heap.len() >= k && dist >= heap[0].0 {
            continue;
        }

        match entry {
            Entry::Leaf { id, .. } => {
                if heap.len() < k {
                    heap.push((dist, *id));
                    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                } else if dist < heap[0].0 {
                    heap[0] = (dist, *id);
                    heap.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            }
            Entry::Internal { child, .. } => {
                knn_rec(child, point, k, heap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert::insert as insert_entry;

    fn build_test_tree() -> Node {
        let mut root = Node::new(true);
        for i in 0..10 {
            let x = (i % 5) as f64 * 2.0;
            let y = (i / 5) as f64 * 2.0;
            insert_entry(&mut root, BBox::new(x, y, x + 1.0, y + 1.0), i, 4);
        }
        root
    }

    #[test]
    fn test_range_query_all() {
        let tree = build_test_tree();
        let results = range_query(&tree, &BBox::new(-10.0, -10.0, 20.0, 20.0));
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_range_query_none() {
        let tree = build_test_tree();
        let results = range_query(&tree, &BBox::new(100.0, 100.0, 110.0, 110.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_query_partial() {
        let tree = build_test_tree();
        let results = range_query(&tree, &BBox::new(0.0, 0.0, 4.0, 2.0));
        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[test]
    fn test_point_query() {
        let tree = build_test_tree();
        let results = point_query(&tree, 0.5, 0.5);
        assert!(results.contains(&0));
    }

    #[test]
    fn test_point_query_miss() {
        let tree = build_test_tree();
        let results = point_query(&tree, 100.0, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_knn_basic() {
        let tree = build_test_tree();
        let results = knn_query(&tree, (0.5, 0.5), 3);
        assert!(results.len() <= 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_knn_returns_nearest() {
        let mut root = Node::new(true);
        for i in 0..5 {
            insert_entry(&mut root, BBox::new(i as f64 * 10.0, 0.0, i as f64 * 10.0 + 1.0, 1.0), i, 4);
        }
        let results = knn_query(&root, (1.0, 0.5), 1);
        assert!(results.contains(&0));
    }

    #[test]
    fn test_knn_empty_tree() {
        let tree = Node::new(true);
        let results = knn_query(&tree, (0.0, 0.0), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_range_empty_tree() {
        let tree = Node::new(true);
        let results = range_query(&tree, &BBox::new(0.0, 0.0, 10.0, 10.0));
        assert!(results.is_empty());
    }
}
