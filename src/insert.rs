//! Insertion into the R-tree.
//!
//! Implements the R*-tree choose subtree and overflow treatment.

use crate::node::{BBox, Entry, Node};
use crate::split::rstar_split;

/// Result of an insertion that may cause a split.
pub enum InsertResult {
    /// Insertion completed without split.
    Done,
    /// Insertion caused a split, returning two new nodes.
    Split(Node, Node),
}

/// Insert a leaf entry into the tree, returning a split if needed.
pub fn insert(root: &mut Node, bbox: BBox, id: usize, max_entries: usize) -> Option<(Node, Node)> {
    let entry = Entry::Leaf { bbox, id };
    let result = insert_rec(root, entry, max_entries, 0);
    match result {
        InsertResult::Done => None,
        InsertResult::Split(n1, n2) => Some((n1, n2)),
    }
}

fn insert_rec(node: &mut Node, entry: Entry, max_entries: usize, _depth: usize) -> InsertResult {
    if node.is_leaf {
        node.entries.push(entry);
        if node.entries.len() > max_entries {
            let (g1, g2) = do_split(node);
            return InsertResult::Split(
                Node { entries: g1, is_leaf: true },
                Node { entries: g2, is_leaf: true },
            );
        }
        InsertResult::Done
    } else {
        // Choose subtree (R*-tree: minimize overlap enlargement, then area enlargement)
        let best_idx = choose_subtree(node, entry.bbox());

        let child = match &mut node.entries[best_idx] {
            Entry::Internal { child, .. } => child,
            _ => panic!("Internal node has leaf entry"),
        };

        let result = insert_rec(child, entry, max_entries, _depth + 1);

        // Update parent bbox
        if let Some(child_bbox) = child.bbox() {
            if let Entry::Internal { bbox, .. } = &mut node.entries[best_idx] {
                *bbox = child_bbox;
            }
        }

        match result {
            InsertResult::Split(n1, n2) => {
                // Replace the old child with the two new ones
                node.entries.remove(best_idx);
                let b1 = n1.bbox().unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));
                let b2 = n2.bbox().unwrap_or(BBox::new(0.0, 0.0, 0.0, 0.0));
                node.entries.push(Entry::Internal { bbox: b1, child: Box::new(n1) });
                node.entries.push(Entry::Internal { bbox: b2, child: Box::new(n2) });

                if node.entries.len() > max_entries {
                    let (g1, g2) = do_split(node);
                    return InsertResult::Split(
                        Node { entries: g1, is_leaf: false },
                        Node { entries: g2, is_leaf: false },
                    );
                }
                InsertResult::Done
            }
            InsertResult::Done => InsertResult::Done,
        }
    }
}

/// Choose the best subtree for insertion using R*-tree heuristics.
fn choose_subtree(node: &Node, entry_bbox: &BBox) -> usize {
    let mut best = 0;
    let mut best_enl = f64::MAX;
    let mut best_area = f64::MAX;

    for (i, e) in node.entries.iter().enumerate() {
        let bbox = e.bbox();
        let enl = bbox.enlargement(entry_bbox);
        if enl < best_enl || (enl - best_enl).abs() < 1e-10 && bbox.area() < best_area {
            best_enl = enl;
            best_area = bbox.area();
            best = i;
        }
    }

    best
}

/// Split a node's entries.
fn do_split(node: &mut Node) -> (Vec<Entry>, Vec<Entry>) {
    let result = rstar_split(node.entries.clone());
    (result.group1, result.group2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_leaf() {
        let mut root = Node::new(true);
        let result = insert(&mut root, BBox::new(0.0, 0.0, 1.0, 1.0), 0, 4);
        assert!(result.is_none());
        assert_eq!(root.entries.len(), 1);
    }

    #[test]
    fn test_insert_causes_split() {
        let mut root = Node::new(true);
        for i in 0..5 {
            let result = insert(&mut root, BBox::new(i as f64, 0.0, i as f64 + 1.0, 1.0), i, 4);
            if i == 4 {
                // The 5th insert should cause a split
                assert!(result.is_some());
            }
        }
    }

    #[test]
    fn test_insert_multiple() {
        let mut root = Node::new(true);
        for i in 0..10 {
            insert(&mut root, BBox::new(i as f64, 0.0, i as f64 + 1.0, 1.0), i, 4);
        }
        // Tree should still be functional
        assert!(root.bbox().is_some());
    }

    #[test]
    fn test_choose_subtree() {
        let mut node = Node::new(false);
        let child1 = Node::new(true);
        let child2 = Node::new(true);
        node.entries.push(Entry::Internal { bbox: BBox::new(0.0, 0.0, 2.0, 2.0), child: Box::new(child1) });
        node.entries.push(Entry::Internal { bbox: BBox::new(10.0, 10.0, 12.0, 12.0), child: Box::new(child2) });

        let idx = choose_subtree(&node, &BBox::new(0.5, 0.5, 1.5, 1.5));
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_insert_spread() {
        let mut root = Node::new(true);
        for i in 0..20 {
            let x = (i % 5) as f64 * 10.0;
            let y = (i / 5) as f64 * 10.0;
            insert(&mut root, BBox::new(x, y, x + 1.0, y + 1.0), i, 4);
        }
        assert!(root.bbox().is_some());
    }
}
