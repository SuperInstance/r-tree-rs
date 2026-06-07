//! R-tree main data structure.

use crate::node::{BBox, Entry, Node};
use crate::insert;
use crate::query;

/// An R-tree spatial index.
pub struct RTree {
    /// Root node.
    root: Node,
    /// Maximum entries per node before split.
    max_entries: usize,
    /// Number of entries in the tree.
    len: usize,
}

impl RTree {
    /// Create a new empty R-tree with the given maximum entries per node.
    ///
    /// Typical values are 4-10. Higher values use more memory but may give
    /// better query performance.
    pub fn new(max_entries: usize) -> Self {
        Self {
            root: Node::new(true),
            max_entries: max_entries.max(2),
            len: 0,
        }
    }

    /// Insert a bounding box with an associated ID.
    pub fn insert(&mut self, bbox: (f64, f64, f64, f64), id: usize) {
        let bb = BBox::new(bbox.0, bbox.1, bbox.2, bbox.3);
        let split = insert::insert(&mut self.root, bb, id, self.max_entries);

        if let Some((n1, n2)) = split {
            // Root was split — create a new root
            let b1 = n1.bbox().unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));
            let b2 = n2.bbox().unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));
            let mut new_root = Node::new(false);
            new_root.entries.push(Entry::Internal { bbox: b1, child: Box::new(n1) });
            new_root.entries.push(Entry::Internal { bbox: b2, child: Box::new(n2) });
            self.root = new_root;
        }

        self.len += 1;
    }

    /// Delete an entry by ID.
    ///
    /// Returns true if an entry was found and removed.
    pub fn delete(&mut self, id: usize) -> bool {
        let removed = delete_rec(&mut self.root, id);
        if removed {
            self.len -= 1;
        }
        // If root has only one internal child, shorten the tree
        if !self.root.is_leaf && self.root.entries.len() == 1 {
            if let Entry::Internal { child, .. } = self.root.entries.pop().unwrap() {
                self.root = *child;
            }
        }
        removed
    }

    /// Find all entries whose bounding boxes overlap with the query.
    pub fn range_query(&self, query: (f64, f64, f64, f64)) -> Vec<usize> {
        let qb = BBox::new(query.0, query.1, query.2, query.3);
        query::range_query(&self.root, &qb)
    }

    /// Find the k nearest entries to a point.
    pub fn knn(&self, point: (f64, f64), k: usize) -> Vec<usize> {
        query::knn_query(&self.root, point, k)
    }

    /// Number of entries in the tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the root bounding box.
    pub fn bbox(&self) -> Option<BBox> {
        self.root.bbox()
    }
}

fn delete_rec(node: &mut Node, id: usize) -> bool {
    if node.is_leaf {
        let idx = node.entries.iter().position(|e| match e {
            Entry::Leaf { id: eid, .. } => *eid == id,
            _ => false,
        });
        if let Some(idx) = idx {
            node.entries.remove(idx);
            return true;
        }
        return false;
    }

    // Search children
    for i in 0..node.entries.len() {
        let found = match &mut node.entries[i] {
            Entry::Internal { child, bbox } => {
                let f = delete_rec(child, id);
                if f {
                    // Update parent bbox
                    if let Some(new_bb) = child.bbox() {
                        *bbox = new_bb;
                    }
                    // Remove empty child
                    if child.is_empty() {
                        // Mark for removal
                        return true; // handled below
                    }
                }
                f
            }
            _ => false,
        };
        if found {
            // Clean up empty internal nodes
            node.entries.retain(|e| match e {
                Entry::Internal { child, .. } => !child.is_empty(),
                _ => true,
            });
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let tree = RTree::new(4);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_insert_and_count() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 1.0, 1.0), 0);
        tree.insert((2.0, 2.0, 3.0, 3.0), 1);
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_range_query_basic() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 2.0, 2.0), 0);
        tree.insert((1.0, 1.0, 3.0, 3.0), 1);
        tree.insert((10.0, 10.0, 12.0, 12.0), 2);

        let results = tree.range_query((0.5, 0.5, 1.5, 1.5));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_range_query_none() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 1.0, 1.0), 0);
        let results = tree.range_query((5.0, 5.0, 6.0, 6.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_delete() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 1.0, 1.0), 0);
        tree.insert((2.0, 2.0, 3.0, 3.0), 1);
        assert!(tree.delete(0));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 1.0, 1.0), 0);
        assert!(!tree.delete(99));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_knn() {
        let mut tree = RTree::new(4);
        for i in 0..10 {
            tree.insert((i as f64, 0.0, i as f64 + 1.0, 1.0), i);
        }
        let results = tree.knn((5.5, 0.5), 3);
        assert!(results.len() <= 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_many_inserts() {
        let mut tree = RTree::new(4);
        for i in 0..100 {
            let x = (i % 10) as f64;
            let y = (i / 10) as f64;
            tree.insert((x, y, x + 0.5, y + 0.5), i);
        }
        assert_eq!(tree.len(), 100);
        let results = tree.range_query((0.0, 0.0, 10.0, 10.0));
        assert_eq!(results.len(), 100);
    }

    #[test]
    fn test_bbox() {
        let mut tree = RTree::new(4);
        tree.insert((0.0, 0.0, 1.0, 1.0), 0);
        tree.insert((5.0, 5.0, 6.0, 6.0), 1);
        let bb = tree.bbox().unwrap();
        assert!((bb.min_x - 0.0).abs() < 1e-10);
        assert!((bb.max_x - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_split_balance() {
        let mut tree = RTree::new(3);
        // Insert enough entries to cause splits
        for i in 0..20 {
            let x = (i % 5) as f64 * 3.0;
            let y = (i / 5) as f64 * 3.0;
            tree.insert((x, y, x + 1.0, y + 1.0), i);
        }
        assert_eq!(tree.len(), 20);
        // All should be findable
        let results = tree.range_query((-1.0, -1.0, 20.0, 20.0));
        assert_eq!(results.len(), 20);
    }
}
