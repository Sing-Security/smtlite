//! smtlite - a small, pure-Rust QF_BV SMT solver.
//!
//! Quantifier-free bitvector formulas only: build expressions with [`Solver::var`] and the
//! [`Bv`] combinators, [`Solver::assert`] the 1-bit constraints, and call [`Solver::check`].
//! A **satisfying** answer comes back as a [`Model`] you can read concrete values out of; the
//! alternative answers are `Unsat` and a budget-limited `Unknown`, and the three are kept
//! distinct so a caller can tell a proof from a timeout ([`Solution`]).
//!
//! It is scoped to *directed* queries - "is there an input that makes this operation
//! overflow?", "can this length reach that copy?" - asked many times over, rather than to
//! general theorem proving. See the README for what that costs and what it rules out.
//!
//! Under the hood the expression DAG is bit-blasted to CNF and solved by a self-contained
//! DPLL core. There is no external solver, no C, and no `unsafe`; `bumpalo` is the only
//! dependency, and it is an allocator.
//!
//! ```
//! use smtlite::{Bv, Solution, Solver};
//!
//! let mut s = Solver::new();
//! let x = s.var("x", 64);
//! s.assert(x.add(&Bv::val(3, 64)).eq(&Bv::val(10, 64))); // x + 3 == 10
//!
//! match s.check() {
//!     Solution::Sat(m) => assert_eq!(m.get("x"), Some(7)),
//!     other => panic!("expected SAT, got {other:?}"),
//! }
//! ```
//!
//! # Bounded by construction
//!
//! Three bounds, any of which yields [`Solution::Unknown`] rather than a wrong verdict:
//! [`Solver::check_with_budget`] caps decisions, [`Solver::check_all_within`] adds a
//! wall-clock deadline, and [`Solver::with_max_clauses`] caps the formula size.
//!
//! # Width and threading limits
//!
//! Widths are 1..=64 bits, because [`Model::get`] reads a variable back as a `u64`. An
//! operation whose result or target width would fall outside 1..=64 **panics**, as does one
//! given two operands of different widths; each such method says so under `# Panics`.
//!
//! A [`Bv`] is an `Rc`-shared node, so neither [`Bv`] nor [`Solver`] is `Send` or `Sync` - one
//! solve runs on one thread.
#![cfg_attr(doctest, doc = include_str!("../README.md"))]

mod blast;
mod bv;
mod sat;

use std::cell::RefCell;
use std::collections::HashMap;

pub use bv::{Bv, mask};

use blast::Blaster;
use bv::Node;
use sat::SatResult::{Sat, Unknown, Unsat};

/// Default decision budget before a query degrades to [`Solution::Unknown`].
const DEFAULT_BUDGET: u64 = 4_000_000;

/// Default formula-size cap: a blasted formula larger than this is [`Solution::Unknown`] without
/// being solved at all.
///
/// `propagate` rescans every clause per fixpoint, so a huge circuit can burn minutes at a low
/// *decision* count. Every operation at 32 bits fits; a **64-bit division needs roughly
/// 110,000 clauses**, so a caller that needs one raises the cap via
/// [`Solver::with_max_clauses`].
const DEFAULT_MAX_CLAUSES: usize = 40_000;

/// The answer to a satisfiability query, three-valued.
///
/// The distinction that matters is [`Unknown`](Solution::Unknown) against the other two: `Sat`
/// and `Unsat` are verdicts about the formula, while `Unknown` says only that a configured
/// bound was reached first. Nothing may treat `Unknown` as "no solution exists" - it is the
/// absence of an answer, not a negative one.
#[derive(Debug)]
pub enum Solution {
    /// Satisfiable. The [`Model`] holds a concrete assignment under which every asserted
    /// constraint holds simultaneously.
    Sat(Model),
    /// Unsatisfiable - no assignment satisfies the constraints, proven within the bounds.
    Unsat,
    /// No verdict: the decision budget, the wall-clock deadline, or the clause cap was reached
    /// before the search finished.
    ///
    /// There is no partial model to salvage. Raise the bound that was hit
    /// ([`Solver::check_with_budget`], [`Solver::check_all_within`],
    /// [`Solver::with_max_clauses`]) and ask again, or treat the query as unresolved.
    Unknown,
}

/// The SMT context: named bitvector variables and asserted constraints.
#[derive(Debug)]
pub struct Solver {
    vars: Vec<(String, u32)>,
    asserts: Vec<Bv>,
    max_clauses: usize,
    cache: RefCell<Option<Cache>>,
}

/// The blasted background of the stored constraints, kept so the per-path entry points can reuse
/// it instead of re-blasting the shared part for every query. Owned clause vectors, so it outlives
/// the arena it was blasted from.
#[derive(Debug)]
struct Cache {
    clauses: Vec<Vec<i32>>,
    nvars: usize,
    var_bits: HashMap<usize, Vec<i32>>,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            vars: Vec::new(),
            asserts: Vec::new(),
            max_clauses: DEFAULT_MAX_CLAUSES,
            cache: RefCell::new(None),
        }
    }
}

impl Solver {
    /// An empty solver: no variables, no constraints.
    ///
    /// ```
    /// use smtlite::Solver;
    ///
    /// let mut s = Solver::new();
    /// let x = s.var("x", 32);
    /// assert_eq!(x.width(), 32);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Raise (or lower) the formula-size cap above which a query becomes [`Solution::Unknown`]
    /// without being solved - see `DEFAULT_MAX_CLAUSES`. The default fits every operation at 32
    /// bits; a 64-bit division needs roughly 110,000 clauses, and such a query should be paired
    /// with a wall-clock deadline via [`check_all_within`](Self::check_all_within).
    #[must_use]
    pub fn with_max_clauses(mut self, n: usize) -> Self {
        self.max_clauses = n;
        self
    }

    /// A fresh `width`-bit symbolic variable.
    ///
    /// The name is a label for [`Model::get`] readback and
    /// [`depends_on`](Self::depends_on) prefix matching, not an identity: reusing a name
    /// creates an independent variable that merely reads back under the same key.
    ///
    /// # Panics
    ///
    /// Panics unless `width` is 1..=64. A model is read back through a `u64`, so every width
    /// in the crate is bounded by that; a wider variable would be unreadable and could not
    /// hold a constant.
    pub fn var(&mut self, name: &str, width: u32) -> Bv {
        assert!(
            (1..=64).contains(&width),
            "variable width must be 1..=64, got {width}"
        );
        let id = self.vars.len();
        self.vars.push((name.to_string(), width));
        self.cache.replace(None);
        Bv::wrap(Node::Var(width, id))
    }

    /// Does `bv` reference any variable whose name starts with `prefix`?
    ///
    /// This is how a caller tells where a value came from without threading provenance
    /// through every operation: name the variables by origin - one prefix for data read from
    /// outside, another for values the caller started with - and ask which of them a
    /// subexpression actually depends on. Use [`ptr_eq`](Bv::ptr_eq) to ask whether two
    /// handles are the *same* value.
    ///
    /// ```
    /// use smtlite::{Bv, Solver};
    ///
    /// let mut s = Solver::new();
    /// let len = s.var("hdr.len", 16);
    /// let base = s.var("init.base", 16);
    ///
    /// assert!(s.depends_on(&len.add(&Bv::val(1, 16)), "hdr."));
    /// assert!(!s.depends_on(&base, "hdr."));
    /// ```
    #[must_use]
    pub fn depends_on(&self, bv: &Bv, prefix: &str) -> bool {
        bv.var_ids().into_iter().any(|id| {
            self.vars
                .get(id)
                .is_some_and(|(n, _)| n.starts_with(prefix))
        })
    }

    /// Assert that a 1-bit constraint must hold.
    ///
    /// Constraints accumulate; the next solve has to satisfy all of them at once. A
    /// contradiction between two of them is a legitimate way to get [`Solution::Unsat`], not
    /// an error.
    ///
    /// ```
    /// use smtlite::{Bv, Solution, Solver};
    ///
    /// let mut s = Solver::new();
    /// let x = s.var("x", 8);
    /// s.assert(x.eq(&Bv::val(0, 8)));
    /// s.assert(x.eq(&Bv::val(1, 8))); // contradicts the line above
    /// assert!(matches!(s.check(), Solution::Unsat));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `constraint` is not 1 bit wide. A multi-bit value is not a proposition;
    /// blasting one bit of it would answer a question the caller did not ask.
    pub fn assert(&mut self, constraint: Bv) {
        assert_eq!(
            constraint.width(),
            1,
            "a constraint must be 1 bit, got {}",
            constraint.width()
        );
        self.asserts.push(constraint);
        self.cache.replace(None);
    }

    /// Solve the accumulated constraints with the default budget.
    ///
    /// ```
    /// use smtlite::{Bv, Solution, Solver};
    ///
    /// let mut s = Solver::new();
    /// let n = s.var("n", 8);
    /// s.assert(n.mul(&Bv::val(4, 8)).ult(&n)); // n*4 wrapped: an overflow exists
    ///
    /// assert!(matches!(s.check(), Solution::Sat(_)));
    /// ```
    #[must_use]
    pub fn check(&self) -> Solution {
        self.check_with_budget(DEFAULT_BUDGET)
    }

    /// Solve, spending at most `budget` decisions.
    ///
    /// The stored constraints are blasted once and cached, so repeated checks against an
    /// unchanged background only pay for the blast on the first one.
    #[must_use]
    pub fn check_with_budget(&self, budget: u64) -> Solution {
        self.solve_assumptions(&[], budget, None)
    }

    /// Solve the stored constraints together with an extra, per-call set of 1-bit constraints,
    /// with the default budget.
    ///
    /// Each element of `assumptions` is asserted just for this call and dropped after, so the
    /// next solve sees only the stored constraints. This is the "background once, many paths"
    /// pattern: a caller asserts the shared part once, then asks one question per path with that
    /// path's branch conditions on top.
    ///
    /// The background is blasted once and reused; each call only blasts its assumptions. See
    /// [`check_all`](Self::check_all) for the other mode, where the passed constraints are the
    /// whole formula and nothing stored takes part.
    ///
    /// ```
    /// use smtlite::{Bv, Solution, Solver};
    ///
    /// let mut s = Solver::new();
    /// let x = s.var("x", 8);
    /// s.assert(x.ult(&Bv::val(10, 8))); // background: x < 10
    ///
    /// assert!(matches!(s.check_assumptions(&[x.ugt(&Bv::val(5, 8))]), Solution::Sat(_)));
    /// // The assumption did not leak: x < 10 and x < 5 still have a solution.
    /// assert!(matches!(s.check_assumptions(&[x.ult(&Bv::val(5, 8))]), Solution::Sat(_)));
    /// // x < 10 and x > 20 is unsat.
    /// assert!(matches!(s.check_assumptions(&[x.ugt(&Bv::val(20, 8))]), Solution::Unsat));
    /// ```
    #[must_use]
    pub fn check_assumptions(&self, assumptions: &[Bv]) -> Solution {
        self.check_assumptions_with_budget(assumptions, DEFAULT_BUDGET)
    }

    /// [`check_assumptions`](Self::check_assumptions) with an explicit decision budget.
    #[must_use]
    pub fn check_assumptions_with_budget(&self, assumptions: &[Bv], budget: u64) -> Solution {
        self.solve_assumptions(assumptions, budget, None)
    }

    /// [`check_assumptions_with_budget`](Self::check_assumptions_with_budget) plus a wall-clock
    /// `deadline`.
    #[must_use]
    pub fn check_assumptions_within(
        &self,
        assumptions: &[Bv],
        budget: u64,
        deadline: std::time::Instant,
    ) -> Solution {
        self.solve_assumptions(assumptions, budget, Some(deadline))
    }

    /// Solve an explicit set of constraints *without* storing them, against the variables
    /// already declared.
    ///
    /// The constraints passed here are the whole formula: nothing previously
    /// [`assert`](Self::assert)ed takes part. That makes this the entry point for asking many
    /// independent questions against one shared variable namespace - where each question
    /// carries its own set of constraints (one path's assumptions, say) and none of them
    /// should leak into the next.
    #[must_use]
    pub fn check_all(&self, constraints: &[Bv]) -> Solution {
        self.solve_constraints(constraints, DEFAULT_BUDGET, None)
    }

    /// [`check_all`](Self::check_all) with an explicit decision budget.
    ///
    /// A sweep over many queries uses a small budget, so one hard instance degrades to
    /// [`Solution::Unknown`] instead of dominating the batch's wall-clock.
    #[must_use]
    pub fn check_all_with_budget(&self, constraints: &[Bv], budget: u64) -> Solution {
        self.solve_constraints(constraints, budget, None)
    }

    /// [`check_all_with_budget`](Self::check_all_with_budget) plus a wall-clock `deadline`.
    ///
    /// The budget bounds decisions, not time. A batch that has to finish inside a wall-clock
    /// bound passes a deadline here, so the slowest query degrades rather than stalling the
    /// rest.
    #[must_use]
    pub fn check_all_within(
        &self,
        constraints: &[Bv],
        budget: u64,
        deadline: std::time::Instant,
    ) -> Solution {
        self.solve_constraints(constraints, budget, Some(deadline))
    }

    fn solve_constraints(
        &self,
        constraints: &[Bv],
        budget: u64,
        deadline: Option<std::time::Instant>,
    ) -> Solution {
        // One arena per solve holds every clause and drops when this function returns. The
        // `Model` we hand back copies its bits out, so it never borrows the arena.
        let bump = bumpalo::Bump::new();
        let mut b = Blaster::new(&bump);
        // Every declared variable gets its SAT bits up front, so a model is total over the
        // declaration: `Model::get` reads back a variable no constraint happened to mention.
        for (id, (_, width)) in self.vars.iter().enumerate() {
            b.declare_var(id, *width);
        }
        for c in constraints {
            b.assert_true(c);
        }
        if b.cnf.clauses.len() > self.max_clauses {
            return Solution::Unknown;
        }
        match b.cnf.solve_within(budget, deadline) {
            Sat(assignment) => Solution::Sat(Model {
                assignment,
                var_bits: b.var_bits,
                name_to_id: self.name_to_id(),
            }),
            Unsat => Solution::Unsat,
            Unknown => Solution::Unknown,
        }
    }

    fn solve_assumptions(
        &self,
        assumptions: &[Bv],
        budget: u64,
        deadline: Option<std::time::Instant>,
    ) -> Solution {
        self.ensure_cache();
        let cache = self.cache.borrow();
        let cache = cache.as_ref().expect("the cache was just built");

        // A fresh arena and blaster for this call's clauses. Seed it with the background's
        // literals for every declared variable and continue the variable counter after them, so
        // the assumption gates never collide with the cached bits.
        let bump = bumpalo::Bump::new();
        let mut b = Blaster::new(&bump);
        b.var_bits = cache.var_bits.clone();
        b.cnf.nvars = cache.nvars;
        for c in assumptions {
            b.assert_true(c);
        }

        // Background clauses first, then the assumption clauses that borrow this call's arena.
        let mut clauses: Vec<&[i32]> = cache.clauses.iter().map(|c| c.as_slice()).collect();
        clauses.extend(b.cnf.clauses.iter().copied());

        if clauses.len() > self.max_clauses {
            return Solution::Unknown;
        }
        match sat::solve(&clauses, b.cnf.nvars, budget, deadline) {
            Sat(assignment) => Solution::Sat(Model {
                assignment,
                var_bits: b.var_bits,
                name_to_id: self.name_to_id(),
            }),
            Unsat => Solution::Unsat,
            Unknown => Solution::Unknown,
        }
    }

    /// Build the cached blast of the stored constraints if it is not already there.
    fn ensure_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(self.blast_background());
        }
    }

    /// Blast the stored constraints once, copying the clauses out of the arena so they outlive it.
    fn blast_background(&self) -> Cache {
        let bump = bumpalo::Bump::new();
        let mut b = Blaster::new(&bump);
        for (id, (_, width)) in self.vars.iter().enumerate() {
            b.declare_var(id, *width);
        }
        for c in &self.asserts {
            b.assert_true(c);
        }
        Cache {
            clauses: b.cnf.clauses.iter().map(|c| c.to_vec()).collect(),
            nvars: b.cnf.nvars,
            var_bits: b.var_bits,
        }
    }

    fn name_to_id(&self) -> HashMap<String, usize> {
        self.vars
            .iter()
            .enumerate()
            .map(|(id, (n, _))| (n.clone(), id))
            .collect()
    }
}

/// A satisfying assignment, queryable by variable name.
///
/// Only [`Solution::Sat`] carries one. The assignment is total over the variables declared on
/// the [`Solver`]: [`get`](Model::get) reads back any declared variable, whether or not a
/// constraint mentioned it, so a query that constrains only some of the declared variables
/// still hands back a value for every one.
#[derive(Debug)]
pub struct Model {
    assignment: Vec<bool>,
    var_bits: HashMap<usize, Vec<i32>>,
    name_to_id: HashMap<String, usize>,
}

impl Model {
    /// The concrete value of a variable, or `None` if the model has no such variable.
    ///
    /// `None` means the name was never declared on the [`Solver`] that produced this model -
    /// not that the value is unknown. A declared variable always reads back, including one no
    /// constraint mentioned (the solver still assigns it a value).
    ///
    /// ```
    /// use smtlite::{Bv, Solution, Solver};
    ///
    /// let mut s = Solver::new();
    /// let x = s.var("x", 64);
    /// s.assert(x.and(&Bv::val(0xf, 64)).eq(&Bv::val(0xc, 64)));
    ///
    /// let Solution::Sat(m) = s.check() else { unreachable!() };
    /// assert_eq!(m.get("x").map(|v| v & 0xf), Some(0xc));
    /// assert_eq!(m.get("never_declared"), None);
    /// ```
    ///
    /// A model reads back as a `u64`, so a value wider than that has no representation and
    /// reads back as `None`. Widths are capped at 64 throughout ([`Solver::var`]), so this is
    /// unreachable through the public API; it is checked rather than assumed so the readback
    /// can never shift a bit off the end of the word.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<u64> {
        let id = *self.name_to_id.get(name)?;
        let bits = self.var_bits.get(&id)?;
        if bits.len() > 64 {
            return None;
        }
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
        // 2*x == 10 (mod 2^32) has two solutions (5 and 0x80000005) - check the property,
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
        // 5 <u x <u 10  ==>  a value strictly between.
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
        // `count * elem` overflows a 32-bit size, so the product wraps below `elem`.
        let mut s = Solver::new();
        let count = s.var("count", 32);
        let elem = Bv::val(0x10, 32);
        let size = count.mul(&elem);
        s.assert(size.ult(&elem)); // wrapped
        s.assert(count.ugt(&Bv::val(0, 32)));
        let c = sat(&s).get("count").unwrap();
        assert!(
            c.wrapping_mul(0x10) & 0xffff_ffff < 0x10,
            "count={c} should overflow"
        );
    }

    #[test]
    fn test_unused_variable_still_reads_back() {
        // The model is total over the declaration: a variable no constraint mentions is still
        // assigned, so `get` returns a value rather than the `None` that means "never declared".
        let mut s = Solver::new();
        let x = s.var("x", 8);
        let _unused = s.var("unused", 16);
        s.assert(x.eq(&Bv::val(5, 8)));

        let m = sat(&s);
        assert_eq!(m.get("x"), Some(5));
        let u = m.get("unused").expect("a declared variable reads back");
        assert!(u <= 0xffff, "unused value {u} exceeds its 16-bit width");
    }

    #[test]
    fn test_variables_read_back_with_no_constraints() {
        // Even an empty constraint set yields a total model over the declared variables.
        let mut s = Solver::new();
        let _a = s.var("a", 1);
        let _b = s.var("b", 64);

        let m = sat(&s);
        assert!(m.get("a").is_some());
        assert!(m.get("b").is_some());
    }
}
