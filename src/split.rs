//! R*-tree split algorithm.
//!
//! Implements the R*-tree split heuristic: minimize overlap between
//! the two resulting groups.

use crate::node::{BBox, Entry, Node};

/// Result of splitting a node: two groups of entries.
pub struct SplitResult {
    /// First group.
    pub group1: Vec<Entry>,
    /// Second group.
    pub group2: Vec<Entry>,
    /// Bounding box of group 1.
    pub bbox1: BBox,
    /// Bounding box of group 2.
    pub bbox2: BBox,
}

/// Split a node's entries using the R*-tree axis-based split heuristic.
///
/// Tries both x and y axes, picks the one with the best split quality
/// (minimum overlap or minimum total area).
pub fn rstar_split(entries: Vec<Entry>) -> SplitResult {
    let m = entries.len();
    let min_entries = (m as f64 * 0.4).ceil() as usize;

    let mut best_split: Option<SplitResult> = None;
    let mut best_goodness = f64::MAX;

    // Try splitting along each axis
    for dim in 0..2 {
        let mut sorted = entries.clone();
        sorted.sort_by(|a, b| {
            let (ca, cb) = (a.bbox().center(), b.bbox().center());
            if dim == 0 { ca.0.partial_cmp(&cb.0).unwrap() }
            else { ca.1.partial_cmp(&cb.1).unwrap() }
        });

        // Try all valid split positions
        for k in min_entries..=(m - min_entries) {
            let g1: Vec<Entry> = sorted[..k].to_vec();
            let g2: Vec<Entry> = sorted[k..].to_vec();

            let b1 = group_bbox(&g1);
            let b2 = group_bbox(&g2);

            let overlap = overlap_area(&b1, &b2);
            let total_area = b1.area() + b2.area();

            let goodness = overlap + total_area * 0.001;
            if goodness < best_goodness {
                best_goodness = goodness;
                best_split = Some(SplitResult { group1: g1, group2: g2, bbox1: b1, bbox2: b2 });
            }
        }
    }

    best_split.unwrap_or_else(|| {
        // Fallback: simple half split
        let mid = entries.len() / 2;
        let g1 = entries[..mid].to_vec();
        let g2 = entries[mid..].to_vec();
        let b1 = group_bbox(&g1);
        let b2 = group_bbox(&g2);
        SplitResult { group1: g1, group2: g2, bbox1: b1, bbox2: b2 }
    })
}

/// Compute the bounding box of a group of entries.
pub fn group_bbox(entries: &[Entry]) -> BBox {
    entries.iter().fold(
        BBox::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN),
        |acc, e| acc.merge(e.bbox()),
    )
}

/// Compute the overlap area between two bounding boxes.
pub fn overlap_area(a: &BBox, b: &BBox) -> f64 {
    if !a.overlaps(b) {
        return 0.0;
    }
    let ox = a.max_x.min(b.max_x) - a.min_x.max(b.min_x);
    let oy = a.max_y.min(b.max_y) - a.min_y.max(b.min_y);
    ox * oy
}

/// Create two new nodes from a split result.
pub fn split_node(entries: Vec<Entry>, is_leaf: bool) -> (Node, Node, BBox, BBox) {
    let result = rstar_split(entries);
    let n1 = Node { entries: result.group1, is_leaf };
    let n2 = Node { entries: result.group2, is_leaf };
    (n1, n2, result.bbox1, result.bbox2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_leaf(x1: f64, y1: f64, x2: f64, y2: f64, id: usize) -> Entry {
        Entry::Leaf { bbox: BBox::new(x1, y1, x2, y2), id }
    }

    #[test]
    fn test_split_even() {
        let entries: Vec<Entry> = (0..6).map(|i| {
            make_leaf(i as f64, 0.0, i as f64 + 1.0, 1.0, i)
        }).collect();
        let result = rstar_split(entries);
        assert!(!result.group1.is_empty());
        assert!(!result.group2.is_empty());
        assert_eq!(result.group1.len() + result.group2.len(), 6);
    }

    #[test]
    fn test_split_preserves_entries() {
        let entries: Vec<Entry> = (0..8).map(|i| {
            make_leaf(i as f64 * 2.0, i as f64 * 2.0, i as f64 * 2.0 + 1.0, i as f64 * 2.0 + 1.0, i)
        }).collect();
        let result = rstar_split(entries);
        assert_eq!(result.group1.len() + result.group2.len(), 8);
    }

    #[test]
    fn test_overlap_area() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 1.0, 3.0, 3.0);
        assert!((overlap_area(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap_area() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(2.0, 2.0, 3.0, 3.0);
        assert!((overlap_area(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn test_split_node() {
        let entries: Vec<Entry> = (0..6).map(|i| {
            make_leaf(i as f64, 0.0, i as f64 + 1.0, 1.0, i)
        }).collect();
        let (n1, n2, b1, b2) = split_node(entries, true);
        assert!(n1.is_leaf);
        assert!(n2.is_leaf);
        assert!(b1.area() >= 0.0);
        assert!(b2.area() >= 0.0);
    }

    #[test]
    fn test_group_bbox() {
        let entries = vec![
            make_leaf(0.0, 0.0, 1.0, 1.0, 0),
            make_leaf(5.0, 5.0, 6.0, 6.0, 1),
        ];
        let bb = group_bbox(&entries);
        assert!((bb.min_x - 0.0).abs() < 1e-10);
        assert!((bb.max_x - 6.0).abs() < 1e-10);
    }
}
