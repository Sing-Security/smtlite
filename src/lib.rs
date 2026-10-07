//! smtlite — a small, pure-Rust QF_BV SMT solver for the RE suite's symbolic executor.
//!
//! Build bitvector expressions with [`Solver::var`] and the [`Bv`] combinators, assert 1-bit
//! constraints, and [`Solver::check`]. Under the hood the expression DAG is bit-blasted to
//! CNF (`blast`) and solved by a self-contained DPLL core (`sat`); a satisfying
//! assignment is read back as concrete values — the PoC seeds the executor needs to turn a
//! static *candidate* into a *confirmed* finding, or an `Unsat` that kills it.
//!
//! ```
//! use smtlite::{Solver, Bv, Solution};
//! let mut s = Solver::new();
//! let x = s.var("x", 64);
//! s.assert(x.add(&Bv::val(3, 64)).eq(&Bv::val(10, 64))); // x + 3 == 10
//! match s.check() {
//!     Solution::Sat(m) => assert_eq!(m.get("x"), Some(7)),
//!     other => panic!("expected SAT, got {other:?}"),
//! }
//! ```

mod blast;
mod bv;
mod sat;

use std::collections::HashMap;

pub use bv::{Bv, mask};
pub use sat::SatResult;

use blast::Blaster;
use bv::Node;
use sat::SatResult::{Sat, Unknown, Unsat};

/// Default decision budget before a query degrades to [`Solution::Unknown`].
const DEFAULT_BUDGET: u64 = 4_000_000;

/// Default formula-size cap: a blasted formula larger than this is [`Solution::Unknown`] without
/// being solved at all.
///
/// `propagate` rescans every clause per fixpoint, so a huge circuit (deep path constraints over
/// 64-bit multiplies) can burn minutes at a low *decision* count. Every operation at 32 bits fits
/// under 40,000 clauses; a **64-bit division needs roughly 110,000**, so a caller that genuinely
/// needs one raises the cap deliberately via [`Solver::with_max_clauses`].
const DEFAULT_MAX_CLAUSES: usize = 40_000;

/// A satisfiability result with a readable model on success.
#[derive(Debug)]
pub enum Solution {
    Sat(Model),
    Unsat,
    Unknown,
}

/// The SMT context: named bitvector variables and asserted constraints.
#[derive(Debug)]
pub struct Solver {
    vars: Vec<(String, u32)>,
    asserts: Vec<Bv>,
    max_clauses: usize,
}

impl Default for Solver {
    fn default() -> Self {
        Self { vars: Vec::new(), asserts: Vec::new(), max_clauses: DEFAULT_MAX_CLAUSES }
    }
}

impl Solver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise (or lower) the formula-size cap above which a query becomes [`Solution::Unknown`]
    /// without being solved — see `DEFAULT_MAX_CLAUSES`. The default fits every operation at 32
    /// bits; a 64-bit division needs roughly 110,000 clauses, and such a query should be paired
    /// with a wall-clock deadline via [`check_all_within`](Self::check_all_within).
    #[must_use]
    pub fn with_max_clauses(mut self, n: usize) -> Self {
        self.max_clauses = n;
        self
    }

    /// A fresh `width`-bit symbolic variable.
    pub fn var(&mut self, name: &str, width: u32) -> Bv {
        let id = self.vars.len();
        self.vars.push((name.to_string(), width));
        Bv::wrap(Node::Var(width, id))
    }

    /// Does `bv` reference any variable whose name starts with `prefix`? Lets a caller tell a
    /// value derived from a loaded field (`mem…`) apart from one that is only an untouched
    /// initial register (`init_…`).
    #[must_use]
    pub fn depends_on(&self, bv: &Bv, prefix: &str) -> bool {
        bv.var_ids().into_iter().any(|id| self.vars.get(id).is_some_and(|(n, _)| n.starts_with(prefix)))
    }

    /// Assert a 1-bit constraint must hold.
    pub fn assert(&mut self, constraint: Bv) {
        debug_assert_eq!(constraint.width(), 1, "assert expects a 1-bit (boolean) value");
        self.asserts.push(constraint);
    }

    /// Solve with the default budget.
    #[must_use]
    pub fn check(&self) -> Solution {
        self.check_with_budget(DEFAULT_BUDGET)
    }

    /// Solve, spending at most `budget` decisions.
    #[must_use]
    pub fn check_with_budget(&self, budget: u64) -> Solution {
        self.solve_constraints(&self.asserts, budget, None)
    }

    /// Check an explicit set of constraints WITHOUT storing them — for a symbolic executor
    /// that carries per-path constraints and asks many independent questions against one
    /// shared variable namespace.
    #[must_use]
    pub fn check_all(&self, constraints: &[Bv]) -> Solution {
        self.solve_constraints(constraints, DEFAULT_BUDGET, None)
    }

    /// [`check_all`](Self::check_all) with an explicit decision budget — a caller running many
    /// queries under a wall-clock deadline (e.g. a corpus sweep) uses a small budget so a single
    /// hard instance degrades to [`Solution::Unknown`] fast instead of stalling the batch.
    #[must_use]
    pub fn check_all_with_budget(&self, constraints: &[Bv], budget: u64) -> Solution {
        self.solve_constraints(constraints, budget, None)
    }

    /// [`check_all_with_budget`](Self::check_all_with_budget) plus a wall-clock `deadline` — the
    /// hard bound a corpus sweep needs so one propagation-heavy query can't stall the batch.
    #[must_use]
    pub fn check_all_within(&self, constraints: &[Bv], budget: u64, deadline: std::time::Instant) -> Solution {
        self.solve_constraints(constraints, budget, Some(deadline))
    }

    fn solve_constraints(&self, constraints: &[Bv], budget: u64, deadline: Option<std::time::Instant>) -> Solution {
        // One bump arena per solve holds every CNF clause; it drops (freeing the lot) when this
        // function returns. The `Model` we hand back copies its bits out, so it never borrows the arena.
        let bump = bumpalo::Bump::new();
        let mut b = Blaster::new(&bump);
        for c in constraints {
            b.assert_true(c);
        }
        if b.cnf.clauses.len() > self.max_clauses {
            return Solution::Unknown;
        }
        match b.cnf.solve_within(budget, deadline) {
            Sat(assignment) => {
                let name_to_id = self.vars.iter().enumerate().map(|(id, (n, _))| (n.clone(), id)).collect();
                Solution::Sat(Model { assignment, var_bits: b.var_bits, name_to_id })
            }
            Unsat => Solution::Unsat,
            Unknown => Solution::Unknown,
        }
    }
}

/// A satisfying assignment, queryable by variable name.
#[derive(Debug)]
pub struct Model {
    assignment: Vec<bool>,
    var_bits: HashMap<usize, Vec<i32>>,
    name_to_id: HashMap<String, usize>,
}

impl Model {
    /// The concrete value of a variable, or `None` if it never appeared in a constraint.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<u64> {
        let id = *self.name_to_id.get(name)?;
        let bits = self.var_bits.get(&id)?;
        let mut v = 0u64;
        for (i, &lit) in bits.iter().enumerate() {
            let base = self.assignment[(lit.unsigned_abs() - 1) as usize];
            if if lit > 0 { base } else { !base } {
                v |= 1u64 << i;
            }
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sat(s: &Solver) -> Model {
        match s.check() {
            Solution::Sat(m) => m,
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn test_linear_add() {
        let mut s = Solver::new();
        let x = s.var("x", 64);
        s.assert(x.add(&Bv::val(3, 64)).eq(&Bv::val(10, 64)));
        assert_eq!(sat(&s).get("x"), Some(7));
    }

    #[test]
    fn test_multiply_by_const() {
        // 2*x == 10 (mod 2^32) has two solutions (5 and 0x80000005) — check the property,
        // not a specific root.
        let mut s = Solver::new();
        let x = s.var("x", 32);
        s.assert(x.mul(&Bv::val(2, 32)).eq(&Bv::val(10, 32)));
        let v = sat(&s).get("x").unwrap();
        assert_eq!(v.wrapping_mul(2) & 0xffff_ffff, 10, "got x={v}");
    }

    #[test]
    fn test_contradiction_is_unsat() {
        let mut s = Solver::new();
        let x = s.var("x", 16);
        s.assert(x.eq(&Bv::val(5, 16)));
        s.assert(x.eq(&Bv::val(6, 16)));
        assert!(matches!(s.check(), Solution::Unsat));
    }

    #[test]
    fn test_range_model() {
        // 5 <u x <u 10  ⇒  a value strictly between.
        let mut s = Solver::new();
        let x = s.var("x", 32);
        s.assert(x.ugt(&Bv::val(5, 32)));
        s.assert(x.ult(&Bv::val(10, 32)));
        let v = sat(&s).get("x").unwrap();
        assert!((6..=9).contains(&v), "got {v}");
    }

    #[test]
    fn test_unsigned_add_overflow() {
        // x + 1 <u x is only true on wraparound: x must be all-ones (255 at width 8).
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.assert(x.add(&Bv::val(1, 8)).ult(&x));
        assert_eq!(sat(&s).get("x"), Some(255));
    }

    #[test]
    fn test_signed_vs_unsigned_compare() {
        // -1 <s 0 holds; 0xff <u 0 does not.
        let mut s = Solver::new();
        s.assert(Bv::val(0xff, 8).slt(&Bv::val(0, 8)));
        assert!(matches!(s.check(), Solution::Sat(_)));

        let mut u = Solver::new();
        u.assert(Bv::val(0xff, 8).ult(&Bv::val(0, 8)));
        assert!(matches!(u.check(), Solution::Unsat));
    }

    #[test]
    fn test_overflow_in_size_math() {
        // The classic: count * elem overflows the 32-bit size, so the product is < elem —
        // exactly the "integer overflow in an allocation size" the executor will ask about.
        let mut s = Solver::new();
        let count = s.var("count", 32);
        let elem = Bv::val(0x10, 32);
        let size = count.mul(&elem);
        s.assert(size.ult(&elem)); // wrapped
        s.assert(count.ugt(&Bv::val(0, 32)));
        let c = sat(&s).get("count").unwrap();
        assert!(c.wrapping_mul(0x10) & 0xffff_ffff < 0x10, "count={c} should overflow");
    }
}
