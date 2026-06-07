//! Node types for the R-tree.

/// An axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Minimum x.
    pub min_x: f64,
    /// Minimum y.
    pub min_y: f64,
    /// Maximum x.
    pub max_x: f64,
    /// Maximum y.
    pub max_y: f64,
}

impl BBox {
    /// Create a new bounding box from (min_x, min_y, max_x, max_y).
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
    }

    /// Area of the bounding box.
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// Margin (perimeter) of the bounding box.
    pub fn margin(&self) -> f64 {
        2.0 * (self.max_x - self.min_x + self.max_y - self.min_y)
    }

    /// Check if this bbox overlaps with another.
    pub fn overlaps(&self, other: &BBox) -> bool {
        self.min_x < other.max_x
            && self.max_x > other.min_x
            && self.min_y < other.max_y
            && self.max_y > other.min_y
    }

    /// Check if this bbox contains another.
    pub fn contains(&self, other: &BBox) -> bool {
        self.min_x <= other.min_x
            && self.max_x >= other.max_x
            && self.min_y <= other.min_y
            && self.max_y >= other.max_y
    }

    /// Merge this bbox with another, returning the union.
    pub fn merge(&self, other: &BBox) -> BBox {
        BBox::new(
            self.min_x.min(other.min_x),
            self.min_y.min(other.min_y),
            self.max_x.max(other.max_x),
            self.max_y.max(other.max_y),
        )
    }

    /// Enlarge this bbox to include another (in-place).
    pub fn enlarge(&mut self, other: &BBox) {
        self.min_x = self.min_x.min(other.min_x);
        self.min_y = self.min_y.min(other.min_y);
        self.max_x = self.max_x.max(other.max_x);
        self.max_y = self.max_y.max(other.max_y);
    }

    /// Area enlargement needed to include another bbox.
    pub fn enlargement(&self, other: &BBox) -> f64 {
        self.merge(other).area() - self.area()
    }

    /// Center point of the bbox.
    pub fn center(&self) -> (f64, f64) {
        ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
    }

    /// Squared distance from a point to the bbox center.
    pub fn dist_sq_to_center(&self, point: (f64, f64)) -> f64 {
        let (cx, cy) = self.center();
        let dx = cx - point.0;
        let dy = cy - point.1;
        dx * dx + dy * dy
    }

    /// Check if a point is inside the bbox.
    pub fn contains_point(&self, px: f64, py: f64) -> bool {
        px >= self.min_x && px <= self.max_x && py >= self.min_y && py <= self.max_y
    }
}

/// An entry in an R-tree node.
#[derive(Debug, Clone)]
pub enum Entry {
    /// A leaf entry containing a bounding box and a data ID.
    Leaf { bbox: BBox, id: usize },
    /// An internal node entry containing a child node.
    Internal { bbox: BBox, child: Box<Node> },
}

impl Entry {
    /// Get the bounding box of this entry.
    pub fn bbox(&self) -> &BBox {
        match self {
            Entry::Leaf { bbox, .. } => bbox,
            Entry::Internal { bbox, .. } => bbox,
        }
    }

    /// Get a mutable reference to the bounding box.
    pub fn bbox_mut(&mut self) -> &mut BBox {
        match self {
            Entry::Leaf { bbox, .. } => bbox,
            Entry::Internal { bbox, .. } => bbox,
        }
    }
}

/// A node in the R-tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// Entries in this node.
    pub entries: Vec<Entry>,
    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl Node {
    /// Create a new empty node.
    pub fn new(is_leaf: bool) -> Self {
        Self { entries: Vec::new(), is_leaf }
    }

    /// Bounding box encompassing all entries.
    pub fn bbox(&self) -> Option<BBox> {
        self.entries.iter().fold(None, |acc, e| {
            Some(match acc {
                None => *e.bbox(),
                Some(b) => b.merge(e.bbox()),
            })
        })
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the node is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbox_area() {
        let b = BBox::new(0.0, 0.0, 4.0, 3.0);
        assert!((b.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_margin() {
        let b = BBox::new(0.0, 0.0, 4.0, 3.0);
        assert!((b.margin() - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_overlaps() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 1.0, 3.0, 3.0);
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_bbox_no_overlap() {
        let a = BBox::new(0.0, 0.0, 1.0, 1.0);
        let b = BBox::new(2.0, 2.0, 3.0, 3.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_bbox_contains() {
        let outer = BBox::new(0.0, 0.0, 5.0, 5.0);
        let inner = BBox::new(1.0, 1.0, 3.0, 3.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_bbox_merge() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 1.0, 3.0, 3.0);
        let merged = a.merge(&b);
        assert!((merged.min_x - 0.0).abs() < 1e-10);
        assert!((merged.max_x - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_enlargement() {
        let a = BBox::new(0.0, 0.0, 2.0, 2.0);
        let b = BBox::new(1.0, 1.0, 3.0, 3.0);
        let enl = a.enlargement(&b);
        assert!(enl > 0.0);
    }

    #[test]
    fn test_node_new() {
        let n = Node::new(true);
        assert!(n.is_leaf);
        assert!(n.is_empty());
    }

    #[test]
    fn test_node_bbox() {
        let mut n = Node::new(true);
        n.entries.push(Entry::Leaf { bbox: BBox::new(0.0, 0.0, 1.0, 1.0), id: 0 });
        n.entries.push(Entry::Leaf { bbox: BBox::new(2.0, 2.0, 3.0, 3.0), id: 1 });
        let bb = n.bbox().unwrap();
        assert!((bb.min_x - 0.0).abs() < 1e-10);
        assert!((bb.max_x - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbox_contains_point() {
        let b = BBox::new(0.0, 0.0, 5.0, 5.0);
        assert!(b.contains_point(2.5, 2.5));
        assert!(!b.contains_point(6.0, 2.5));
    }
}
