//! # r-tree-rs
//!
//! R-tree spatial indexing with R*-tree split heuristics.
//!
//! # Example
//!
//! ```
//! use r_tree_rs::RTree;
//!
//! let mut tree = RTree::new(3);
//! tree.insert((0.0, 0.0, 2.0, 2.0), 1);  // (min_x, min_y, max_x, max_y), id
//! tree.insert((1.0, 1.0, 3.0, 3.0), 2);
//!
//! let results = tree.range_query((0.5, 0.5, 1.5, 1.5));
//! assert_eq!(results.len(), 2);
//! ```

pub mod node;
pub mod insert;
pub mod split;
pub mod query;
mod tree;

pub use tree::RTree;
