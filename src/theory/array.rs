//! The theory of arrays: `select` and `store`, lowered to bitvectors plus
//! read-over-write lemmas.
//!
//! An array is a total function from an index word to an element word. Two
//! operations matter:
//!
//! - `select(a, i)` is the element stored at `i`;
//! - `store(a, i, v)` is `a` with position `i` overwritten by `v`.
//!
//! Their axiom is *read-over-write*, and it is the whole theory:
//!
//! ```text
//! select(store(a, i, v), j) = (j == i) ? v : select(a, j)
//! ```
//!
//! Together with congruence for the base array - equal indices select equal
//! elements - that is enough to reason about keyed maps and keyed storage
//! without ever deciding a formula about the entire index space.
//!
//! ## Encoding
//!
//! Every `select` term is a fresh `Bv`; the theory records which array and
//! which index term it was taken at. This module's job at model-check time is
//! to instantiate the axiom instance for the one branch the model chose:
//!
//! - when the model makes the stored index and the selected index **equal**,
//!   the select must equal the stored value;
//! - when it makes them **different**, the select must equal the parent array's
//!   select at the same index (which was created when the term was built, so a
//!   chain of stores becomes a chain of already-existing terms).
//!
//! A base (symbolic) array gets the congruence instance instead: two selects at
//! model-equal indices must agree.
//!
//! This is the standard lazy array encoding, and it fits the crate's soundness
//! law exactly: lemmas are monotone, so `Unsat` is a fact the moment it is
//! seen; a `Sat` is returned only after the model survives every instantiation;
//! and the round budget running out is `Unknown`.
//!
//! ## What this replaces
//!
//! A symbolic executor that wants to know "does this store land on cell C?"
//! otherwise keeps a fixed-size list of guarded writes and folds them with a
//! chain of `ite`s - an array theory with its arms cut to a constant. That is
//! sound but runs out after a handful of stores; here the number of stores is
//! bounded by the query, not by the encoding.
//!
//! ## Cost
//!
//! A select through a chain of `k` stores creates `k` terms (one per level) and
//! at validation each level is one comparison plus one implication. The model
//! check is linear in the number of select terms plus, for base arrays, the
//! pairwise congruence scan. Nothing is quadratic in the array's *index space*:
//! the index width may be 128 and the encoding never enumerates it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::Solver;
use crate::bv::Bv;
use crate::theory::{ModelView, Theory};

/// One array term: a base array or a store over another array term.
#[derive(Clone)]
enum Node {
    /// A symbolic array, or a constant array (every index selects `constant`).
    Base {
        constant: Option<Bv>,
        index_width: u32,
        elem_width: u32,
    },
    /// `parent` with `index` overwritten by `value`.
    Store { parent: usize, index: Bv, value: Bv },
}

/// A distinct `select` term: the array it reads, the index it reads at, the
/// fresh result, and - for a store - the parent array's select at that same
/// index (the read-over-write fallback).
struct Select {
    array: usize,
    index: Bv,
    result: Bv,
    parent: Option<Bv>,
}

struct Inner {
    nodes: RefCell<Vec<Node>>,
    /// `(array id, index node address) -> select record`, so rebuilding the same
    /// term returns the same result instead of a second independent cell.
    by_key: RefCell<HashMap<(usize, usize), usize>>,
    /// `(parent id, index address, value address) -> store id`, so two identical
    /// stores are one array term.
    store_key: RefCell<HashMap<(usize, usize, usize), usize>>,
    selects: RefCell<Vec<Select>>,
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayTheory")
            .field("arrays", &self.nodes.borrow().len())
            .field("selects", &self.selects.borrow().len())
            .finish()
    }
}

impl Inner {
    fn base(&self, constant: Option<Bv>, index_width: u32, elem_width: u32) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        nodes.push(Node::Base {
            constant,
            index_width,
            elem_width,
        });
        nodes.len() - 1
    }

    fn push_select(&self, array: usize, index: Bv, result: Bv, parent: Option<Bv>) -> usize {
        let key = (array, index.ptr_key());
        let i = {
            let mut selects = self.selects.borrow_mut();
            selects.push(Select {
                array,
                index,
                result,
                parent,
            });
            selects.len() - 1
        };
        self.by_key.borrow_mut().insert(key, i);
        i
    }
}

/// A handle on the array theory's term log. Clones share it, so the handle used
/// to build terms and the one registered with the solver are the same theory.
///
/// One theory serves every array in a solver; each array declares its own index
/// and element widths, so arrays of different shapes coexist.
///
/// ```
/// use smtlite::{ArrayTheory, Bv, Solution, Solver};
///
/// let mut s = Solver::new();
/// let arrays = ArrayTheory::new();
/// let a = arrays.var(&mut s, 8, 8);
/// let i = s.var("i", 8);
/// let v = s.var("v", 8);
/// let b = a.store(&mut s, &i, &v);
/// // Reading back the cell just written is the written value, by construction.
/// let read = b.select(&mut s, &i);
/// s.assert(read.ne(&v));
/// s.register_theory(arrays.clone());
/// assert!(matches!(s.check_theories(200_000, None), Solution::Unsat));
/// ```
#[derive(Clone, Debug)]
pub struct ArrayTheory {
    inner: Rc<Inner>,
}

impl Default for ArrayTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayTheory {
    /// A fresh array theory with no arrays.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Rc::new(Inner {
                nodes: RefCell::new(Vec::new()),
                by_key: RefCell::new(HashMap::new()),
                store_key: RefCell::new(HashMap::new()),
                selects: RefCell::new(Vec::new()),
            }),
        }
    }

    /// A symbolic array `index_width -> elem_width`.
    ///
    /// # Panics
    ///
    /// Panics unless both widths are 1..=128.
    #[must_use]
    pub fn var(&self, _solver: &mut Solver, index_width: u32, elem_width: u32) -> Array {
        assert!(
            (1..=128).contains(&index_width),
            "array index width must be 1..=128, got {index_width}"
        );
        assert!(
            (1..=128).contains(&elem_width),
            "array element width must be 1..=128, got {elem_width}"
        );
        let id = self.inner.base(None, index_width, elem_width);
        Array {
            inner: Rc::clone(&self.inner),
            id,
        }
    }

    /// The constant array: every index selects `element`.
    ///
    /// `index_width` only fixes the width a later [`store`](Array::store) must
    /// use; the elements never depend on it.
    #[must_use]
    pub fn constant(&self, element: &Bv, index_width: u32) -> Array {
        assert!(
            (1..=128).contains(&index_width),
            "array index width must be 1..=128, got {index_width}"
        );
        let id = self
            .inner
            .base(Some(element.clone()), index_width, element.width());
        Array {
            inner: Rc::clone(&self.inner),
            id,
        }
    }

    /// The number of array terms created (bases plus stores).
    #[must_use]
    pub fn arrays(&self) -> usize {
        self.inner.nodes.borrow().len()
    }

    /// The number of distinct `select` terms created.
    #[must_use]
    pub fn selects(&self) -> usize {
        self.inner.selects.borrow().len()
    }
}

/// An array term of a fixed shape. Created by [`ArrayTheory`].
#[derive(Clone, Debug)]
pub struct Array {
    inner: Rc<Inner>,
    id: usize,
}

impl Array {
    /// The index and element widths, `(index, element)`.
    ///
    /// # Panics
    ///
    /// Panics if the internal id does not name an array - unreachable, because
    /// ids only come from this module.
    #[must_use]
    pub fn shape(&self) -> (u32, u32) {
        match &self.inner.nodes.borrow()[self.id] {
            Node::Base {
                index_width,
                elem_width,
                ..
            } => (*index_width, *elem_width),
            Node::Store { parent, .. } => {
                let shape = &self.inner.nodes.borrow()[*parent];
                match shape {
                    Node::Base {
                        index_width,
                        elem_width,
                        ..
                    } => (*index_width, *elem_width),
                    // Stores always sit on an array of some shape; walk down
                    // rather than trusting the immediate parent's variant.
                    Node::Store { .. } => Array {
                        inner: Rc::clone(&self.inner),
                        id: *parent,
                    }
                    .shape(),
                }
            }
        }
    }

    /// The element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index`'s width differs from the array's index width.
    pub fn select(&self, solver: &mut Solver, index: &Bv) -> Bv {
        let (iw, _) = self.shape();
        assert_eq!(
            index.width(),
            iw,
            "select index must be {iw} bits, got {}",
            index.width()
        );
        self.select_at(solver, self.id, index)
    }

    fn select_at(&self, solver: &mut Solver, array: usize, index: &Bv) -> Bv {
        // Same array term and same index term: one cell, one result.
        let key = (array, index.ptr_key());
        if let Some(&i) = self.inner.by_key.borrow().get(&key) {
            return self.inner.selects.borrow()[i].result.clone();
        }
        match self.inner.nodes.borrow()[array].clone() {
            Node::Base {
                constant: Some(c), ..
            } => c,
            Node::Base {
                constant: None,
                elem_width,
                ..
            } => {
                let n = self.inner.selects.borrow().len();
                let result = solver.var(&format!("arr#{array}[{n}]"), elem_width);
                self.inner
                    .push_select(array, index.clone(), result.clone(), None);
                result
            }
            Node::Store {
                parent,
                index: stored,
                value,
            } => {
                // The syntactic fast path: the same term is the same cell.
                if stored.ptr_eq(index) {
                    return value;
                }
                // Build the fallback term now, so the lemma instantiated
                // later names a term the solver already knows.
                let parent_sel = self.select_at(solver, parent, index);
                let n = self.inner.selects.borrow().len();
                let result = solver.var(&format!("arr#{array}[{n}]"), value.width());
                self.inner
                    .push_select(array, index.clone(), result.clone(), Some(parent_sel));
                result
            }
        }
    }

    /// `self` with `index` overwritten by `value`.
    ///
    /// # Panics
    ///
    /// Panics if either width differs from the array's shape.
    pub fn store(&self, _solver: &mut Solver, index: &Bv, value: &Bv) -> Array {
        let (iw, ew) = self.shape();
        assert_eq!(
            index.width(),
            iw,
            "store index must be {iw} bits, got {}",
            index.width()
        );
        assert_eq!(
            value.width(),
            ew,
            "stored value must be {ew} bits, got {}",
            value.width()
        );
        let key = (self.id, index.ptr_key(), value.ptr_key());
        if let Some(&id) = self.inner.store_key.borrow().get(&key) {
            return Array {
                inner: Rc::clone(&self.inner),
                id,
            };
        }
        let id = {
            let mut nodes = self.inner.nodes.borrow_mut();
            nodes.push(Node::Store {
                parent: self.id,
                index: index.clone(),
                value: value.clone(),
            });
            nodes.len() - 1
        };
        self.inner.store_key.borrow_mut().insert(key, id);
        Array {
            inner: Rc::clone(&self.inner),
            id,
        }
    }
}

impl Theory for ArrayTheory {
    fn name(&self) -> &'static str {
        "arrays"
    }

    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
        let nodes = self.inner.nodes.borrow();
        let selects = self.inner.selects.borrow();
        let mut out = Vec::new();

        // Read-over-write: instantiate the branch the model chose.
        for s in selects.iter() {
            let Node::Store {
                index: stored,
                value,
                ..
            } = &nodes[s.array]
            else {
                continue;
            };
            let (Some(sv), Some(iv)) = (model.eval128(stored), model.eval128(&s.index)) else {
                continue;
            };
            if sv == iv {
                if model.eval128(&s.result) != model.eval128(value) {
                    out.push(stored.eq(&s.index).implies(&s.result.eq(value)));
                }
            } else if let Some(parent) = &s.parent {
                if model.eval128(&s.result) != model.eval128(parent) {
                    out.push(stored.ne(&s.index).implies(&s.result.eq(parent)));
                }
            }
        }

        // Congruence at a base array: equal indices select equal elements.
        for i in 0..selects.len() {
            if !matches!(nodes[selects[i].array], Node::Base { constant: None, .. }) {
                continue;
            }
            for j in (i + 1)..selects.len() {
                if selects[j].array != selects[i].array {
                    continue;
                }
                let (Some(a), Some(b)) = (
                    model.eval128(&selects[i].index),
                    model.eval128(&selects[j].index),
                ) else {
                    continue;
                };
                if a != b {
                    continue;
                }
                let results_differ = match (
                    model.eval128(&selects[i].result),
                    model.eval128(&selects[j].result),
                ) {
                    (Some(x), Some(y)) => x != y,
                    _ => false,
                };
                if results_differ {
                    out.push(
                        selects[i]
                            .index
                            .eq(&selects[j].index)
                            .implies(&selects[i].result.eq(&selects[j].result)),
                    );
                }
            }
        }
        out
    }

    fn eager_lemmas(&self) -> Vec<Bv> {
        let nodes = self.inner.nodes.borrow();
        let selects = self.inner.selects.borrow();
        let mut out = Vec::new();
        for s in selects.iter() {
            match &nodes[s.array] {
                Node::Store {
                    index: stored,
                    value,
                    ..
                } => {
                    // Both branches of read-over-write, always: the disjunction
                    // is exhaustive, so emitting both is the standard eager
                    // encoding and cannot exclude a real model.
                    out.push(stored.eq(&s.index).implies(&s.result.eq(value)));
                    if let Some(parent) = &s.parent {
                        out.push(stored.ne(&s.index).implies(&s.result.eq(parent)));
                    }
                }
                Node::Base { .. } => {}
            }
        }
        for i in 0..selects.len() {
            if !matches!(nodes[selects[i].array], Node::Base { constant: None, .. }) {
                continue;
            }
            for j in (i + 1)..selects.len() {
                if selects[j].array != selects[i].array {
                    continue;
                }
                out.push(
                    selects[i]
                        .index
                        .eq(&selects[j].index)
                        .implies(&selects[i].result.eq(&selects[j].result)),
                );
            }
        }
        out
    }
}
