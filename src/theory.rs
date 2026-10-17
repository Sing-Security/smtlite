//! The theory layer: lemmas over the blasted bitvector core.
//!
//! The core is QF_BV and stays that way. A theory here is not a new decision
//! procedure bolted beside DPLL - it is a *source of lemmas*, each a 1-bit
//! [`Bv`] term over the terms the core already knows, instantiated lazily
//! against a model and re-solved. The blaster, the SAT core, the clause cap
//! and the deadlines are untouched; a theory only ever adds constraints.
//!
//! The soundness law, stated once:
//!
//! > **SAT is a claim until validated; UNSAT is a fact.**
//!
//! - **(a) UNSAT is immediate.** Lemma instantiation is *monotone* - adding a
//!   lemma only removes models - so an `Unsat` under any subset of the theory's
//!   lemmas is already an `Unsat` of the full theory formula. Nothing is owed.
//! - **(b) SAT must be validated.** A model is handed to every registered
//!   theory's [`Theory::violated_lemmas`]; a non-empty answer means the model
//!   breaks an axiom, the lemmas are added, and the query is re-solved. Only a
//!   model no theory objects to is returned as `Sat`. `Unknown` stays
//!   first-class throughout: the round budget running out is `Unknown`, never
//!   a guess.
//!
//! That is a model-based DPLL(T) loop implemented entirely on the existing
//! core - no new SAT, no C, no `unsafe`, no new dependency.
//!
//! A theory must be *pure*: its lemmas are built when the theory's terms were
//! built, so [`Theory::violated_lemmas`] takes only a [`ModelView`] and never
//! needs the solver. That is what keeps the round loop free of re-entrancy and
//! what lets [`Solver::model_consistent`] re-run the validation afterwards,
//! from outside.
//!
//! A theory's lemmas are instantiated per *query*, not asserted into the
//! solver's stored constraints: they are derived under that query's model, and
//! the next query re-derives its own (cheap next to re-blasting a background).

pub mod array;
pub mod uf;

pub use array::{Array, ArrayTheory};
pub use uf::Uninterpreted;

use std::collections::HashMap;
use std::rc::Rc;

use crate::bv::{Bv, Cmp, eval_bin, mask128, rot_, shl_, shr_, signed};
use crate::{Model, Node, Solution, Solver};

/// The value each of a model's variables took, and the ability to evaluate an
/// arbitrary term under it.
///
/// This is the only thing a [`Theory`] sees of a solve: enough to answer "what
/// does the model say this term is". Evaluation is exact - it walks the same
/// expression DAG the model was built for, reading variable values out of the
/// assignment - so a theory never has to reconstruct what the blaster did.
///
/// [`eval`](ModelView::eval) answers `None` for a term wider than 64 bits (a
/// model reads back through a `u64`; see the crate-level note) or one mentioning
/// a variable no longer declared - neither can arise for a term built against
/// the solver that produced the model. [`eval128`](ModelView::eval128) is the
/// same walk without the `u64` ceiling.
#[derive(Debug)]
pub struct ModelView<'a> {
    model: &'a Model,
}

impl<'a> ModelView<'a> {
    /// A view over a model. Public so a caller can re-check a `Sat` it carried
    /// out of the solver - the independent half of law (b) - rather than only
    /// through the traits that produced it.
    #[must_use]
    pub fn new(model: &'a Model) -> Self {
        Self { model }
    }

    /// The concrete value of a named variable, as [`Model::get`].
    #[must_use]
    pub fn get(&self, name: &str) -> Option<u64> {
        self.model.get(name)
    }

    /// The value of `bv` under the model at a `u64`, or `None` when the term is
    /// wider than 64 bits (see the type note).
    #[must_use]
    pub fn eval(&self, bv: &Bv) -> Option<u64> {
        if bv.width() > 64 {
            return None;
        }
        self.eval128(bv).and_then(|v| u64::try_from(v).ok())
    }

    /// The value of `bv` under the model at the crate's full 128-bit width.
    #[must_use]
    pub fn eval128(&self, bv: &Bv) -> Option<u128> {
        // Post-order walk with the values memoised against node identity: the
        // DAG is walked once, a shared subgraph evaluated once, and no
        // recursion depth is spent (an expression chain can be arbitrarily
        // deep - the crate's other walks are iterative for the same reason).
        let mut memo: HashMap<usize, u128> = HashMap::new();
        let key = |b: &Bv| Rc::as_ptr(&b.0) as usize;
        let mut stack: Vec<(Bv, bool)> = vec![(bv.clone(), false)];
        while let Some((cur, ready)) = stack.pop() {
            if memo.contains_key(&key(&cur)) {
                continue;
            }
            if !ready {
                stack.push((cur.clone(), true));
                for child in children_of(&cur.0) {
                    if !memo.contains_key(&key(&child)) {
                        stack.push((child, false));
                    }
                }
                continue;
            }
            let val = eval_node(&cur.0, &memo, self.model)?;
            memo.insert(key(&cur), val);
        }
        memo.get(&key(bv)).copied()
    }
}

/// The operands of a node, in any order - the post-order walk's children.
fn children_of(n: &Node) -> Vec<Bv> {
    match n {
        Node::Const(_, _) | Node::Var(_, _) => Vec::new(),
        Node::Not(a)
        | Node::Neg(a)
        | Node::ShlC(a, _)
        | Node::ShrC(a, _, _)
        | Node::RotC(a, _, _)
        | Node::Zext(a, _)
        | Node::Sext(a, _)
        | Node::Extract(a, _, _) => vec![a.clone()],
        Node::Bin(_, a, b)
        | Node::Compare(_, a, b)
        | Node::Concat(a, b)
        | Node::ShlV(a, b)
        | Node::ShrV(a, b, _)
        | Node::RotV(a, b, _) => vec![a.clone(), b.clone()],
        Node::Ite(a, b, c) => vec![a.clone(), b.clone(), c.clone()],
    }
}

/// Evaluate one node whose children are already in `memo`, using the same
/// semantics the folder in `bv` uses for constants.
fn eval_node(n: &Node, memo: &HashMap<usize, u128>, model: &Model) -> Option<u128> {
    let v = |b: &Bv| -> Option<u128> { memo.get(&(Rc::as_ptr(&b.0) as usize)).copied() };
    Some(match n {
        Node::Const(w, value) => mask128(*value, *w),
        Node::Var(w, id) => mask128(model.value_of128(*id)?, *w),
        Node::Not(a) => mask128(!v(a)?, a.width()),
        Node::Neg(a) => mask128(0u128.wrapping_sub(v(a)?), a.width()),
        Node::Bin(op, a, b) => eval_bin(*op, v(a)?, v(b)?, a.width()),
        Node::ShlC(a, k) => shl_(v(a)?, u128::from(*k), a.width()),
        Node::ShrC(a, k, arith) => shr_(v(a)?, u128::from(*k), a.width(), *arith),
        Node::ShlV(a, k) => shl_(v(a)?, v(k)?, a.width()),
        Node::ShrV(a, k, arith) => shr_(v(a)?, v(k)?, a.width(), *arith),
        Node::RotC(a, k, left) => rot_(v(a)?, u128::from(*k), a.width(), *left),
        Node::RotV(a, k, left) => rot_(v(a)?, v(k)?, a.width(), *left),
        Node::Compare(c, a, b) => {
            let (x, y, w) = (v(a)?, v(b)?, a.width());
            let bit = match c {
                Cmp::Eq => x == y,
                Cmp::Ne => x != y,
                Cmp::Ult => x < y,
                Cmp::Ule => x <= y,
                Cmp::Slt => signed(x, w) < signed(y, w),
                Cmp::Sle => signed(x, w) <= signed(y, w),
            };
            u128::from(bit)
        }
        Node::Zext(a, _) => v(a)?,
        Node::Sext(a, w) => mask128(signed(v(a)?, a.width()) as u128, *w),
        Node::Extract(a, hi, lo) => mask128(v(a)? >> *lo, *hi - *lo + 1),
        Node::Concat(hi, lo) => mask128((v(hi)? << lo.width()) | v(lo)?, hi.width() + lo.width()),
        Node::Ite(c, t, e) => {
            if v(c)? != 0 {
                v(t)?
            } else {
                v(e)?
            }
        }
    })
}

/// A theory over the core: a term layer that lowers to [`Bv`] and carries
/// axioms, instantiated lazily against a model.
///
/// Implementations own their terms; the solver owns only a boxed handle. A
/// lemma must be a 1-bit term (the loop asserts it), and must be an *instance
/// of an axiom the theory's own lowering is sound for* - the loop trusts them
/// exactly as far as it trusts an `assert`.
pub trait Theory {
    /// A short name for reports.
    fn name(&self) -> &'static str;

    /// The axiom instances `model` violates, as 1-bit lemmas over terms that
    /// already exist. Empty means the model is consistent with the theory.
    ///
    /// Called with a model every time the core answers `Sat`; the lemmas are
    /// added and the query re-solved. Must be a pure function of the model and
    /// the theory's terms - no solver access, no fresh terms.
    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv>;

    /// Every axiom instance the theory can name, regardless of a model - the
    /// eager encoding.
    ///
    /// Not used by the solve loop, which instantiates lazily; it exists so a
    /// caller can compare the two encodings, and so the soundness tests can
    /// check law (a): an `Unsat` under a strict lemma subset must still be
    /// `Unsat` under the full, eager set. The default is empty - a theory with
    /// an infinite instantiation space owes only the lazy path.
    fn eager_lemmas(&self) -> Vec<Bv> {
        Vec::new()
    }
}

/// A closed theory: `0 == 1`. The scaffolding's own falsifier - registering it
/// must make every query `Unsat`, which is the cheapest end-to-end check that
/// the loop actually consults its theories.
#[derive(Debug, Default)]
pub struct Contradiction;

impl Theory for Contradiction {
    fn name(&self) -> &'static str {
        "contradiction"
    }
    fn violated_lemmas(&self, _model: &ModelView<'_>) -> Vec<Bv> {
        vec![Bv::val(0, 1).eq(&Bv::val(1, 1))]
    }
    fn eager_lemmas(&self) -> Vec<Bv> {
        vec![Bv::val(0, 1).eq(&Bv::val(1, 1))]
    }
}

/// A boxed theory is a theory, so a caller that builds theories dynamically
/// (and holds them behind `Box<dyn Theory>`) can register them directly.
impl<T: Theory + ?Sized> Theory for Box<T> {
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
        (**self).violated_lemmas(model)
    }
    fn eager_lemmas(&self) -> Vec<Bv> {
        (**self).eager_lemmas()
    }
}

impl Solver {
    /// Register a theory for the `check_*_theories` entry points.
    ///
    /// Registration is additive and does not touch the stored constraints, so
    /// a solver without registered theories behaves exactly as before at every
    /// existing entry point.
    ///
    /// # Panics
    ///
    /// Panics if a theory with the same [`Theory::name`] is already registered:
    /// a duplicated name makes a report ambiguous, and a doubled theory would
    /// silently double every lemma it emits.
    pub fn register_theory(&mut self, theory: impl Theory + 'static) {
        assert!(
            !self.theories.iter().any(|t| t.name() == theory.name()),
            "a theory named `{}` is already registered",
            theory.name()
        );
        self.theories.push(Box::new(theory));
    }

    /// How many model-check rounds [`check_theories`](Self::check_theories) and
    /// friends may run before answering [`Solution::Unknown`].
    ///
    /// Each round is one solve plus one validation of its model; theories that
    /// need many rounds converge over several, and a query that would need
    /// unboundedly many (a theory whose lemmas never converge) degrades to
    /// `Unknown` instead of looping. Zero disables the loop entirely: the query
    /// is solved once and answered without model validation - sound only for a
    /// solver with no registered theories, and provided as the escape hatch
    /// that makes that statement testable.
    #[must_use]
    pub fn with_theory_rounds(mut self, rounds: usize) -> Self {
        self.theory_rounds = rounds;
        self
    }

    /// Solve the stored constraints with every registered theory's axioms.
    ///
    /// The `Sat` that comes back has been validated against every theory
    /// (law (b)); `Unsat` is a fact the moment it is seen (law (a)); `Unknown`
    /// covers both the core's own bounds and the round budget. See the
    /// [module docs](crate::theory).
    #[must_use]
    pub fn check_theories(&self, budget: u64, deadline: Option<std::time::Instant>) -> Solution {
        self.solve_theory_rounds(&[], true, budget, deadline)
    }

    /// [`check_theories`](Self::check_theories) with an extra, per-call set of
    /// constraints - the per-path query a symbolic walker makes.
    #[must_use]
    pub fn check_assumptions_theories(
        &self,
        assumptions: &[Bv],
        budget: u64,
        deadline: Option<std::time::Instant>,
    ) -> Solution {
        self.solve_theory_rounds(assumptions, true, budget, deadline)
    }

    /// [`check_theories`](Self::check_theories) against an explicit constraint
    /// set instead of the stored ones - nothing asserted into the solver takes
    /// part, exactly as in [`check_all`](Self::check_all).
    #[must_use]
    pub fn check_all_theories(
        &self,
        constraints: &[Bv],
        budget: u64,
        deadline: Option<std::time::Instant>,
    ) -> Solution {
        self.solve_theory_rounds(constraints, false, budget, deadline)
    }

    /// Is `model` consistent with every registered theory? The validation step
    /// of law (b), exposed so a caller can re-check a `Sat` (or a model it
    /// carried somewhere else) after the fact.
    #[must_use]
    pub fn model_consistent(&self, model: &Model) -> bool {
        let view = ModelView::new(model);
        self.theories
            .iter()
            .all(|t| t.violated_lemmas(&view).is_empty())
    }

    /// Every lemma every registered theory can name eagerly - the full
    /// instantiation, used by the soundness tests to check law (a) against the
    /// lazy loop's answer. Assumes the constraints as [`assert`](Self::assert)
    /// would; see [`Theory::eager_lemmas`].
    #[must_use]
    pub fn eager_theory_lemmas(&self) -> Vec<Bv> {
        self.theories
            .iter()
            .flat_map(|t| t.eager_lemmas())
            .collect()
    }

    /// The shared round loop: solve, validate the model, add what it violates,
    /// repeat. `assumptions` selects the entry-point shape: `true` solves the
    /// stored constraints plus the given terms, `false` treats the terms as the
    /// whole formula.
    ///
    /// The first round keeps the incremental shape, so a large background is
    /// blasted once. From the first instantiated lemma on, the rounds switch to
    /// a single blast over `stored ++ terms ++ lemmas`, for one reason: a
    /// lemma's antecedent is almost always a term the background already
    /// constrains - the equality a path condition pinned - and the blaster's
    /// structural interning only shares a circuit between terms blasted by the
    /// *same* blaster. Blasting the background separately makes the core prove
    /// the two copies equivalent, which is exactly the query a learning-free
    /// DPLL is worst at (measured: seconds to `Unknown` instead of an
    /// immediate `Unsat`). Re-blasting a background is linear; the duplicate
    /// circuit is exponential, so the trade is deliberately taken here and
    /// nowhere else.
    fn solve_theory_rounds(
        &self,
        base: &[Bv],
        assumptions: bool,
        budget: u64,
        deadline: Option<std::time::Instant>,
    ) -> Solution {
        let solve = |lemmas: &[Bv]| {
            let mut all: Vec<Bv> = base.to_vec();
            all.extend(lemmas.iter().cloned());
            if !assumptions {
                return self.solve_constraints(&all, budget, deadline);
            }
            if lemmas.is_empty() {
                return self.solve_assumptions(&all, budget, deadline);
            }
            let mut together: Vec<Bv> = self.asserts.clone();
            together.extend(all);
            self.solve_constraints(&together, budget, deadline)
        };
        if self.theories.is_empty() || self.theory_rounds == 0 {
            return solve(&[]);
        }
        let mut lemmas: Vec<Bv> = Vec::new();
        for _round in 0..self.theory_rounds {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Solution::Unknown;
            }
            match solve(&lemmas) {
                Solution::Sat(model) => {
                    let mut fresh: Vec<Bv> = Vec::new();
                    {
                        let view = ModelView::new(&model);
                        for t in &self.theories {
                            fresh.extend(t.violated_lemmas(&view));
                        }
                    }
                    for l in &fresh {
                        assert_eq!(
                            l.width(),
                            1,
                            "theory lemma must be 1 bit, got {}",
                            l.width()
                        );
                    }
                    if fresh.is_empty() {
                        return Solution::Sat(model);
                    }
                    lemmas.extend(fresh);
                }
                // Law (a): Unsat under any lemma subset is Unsat for the full
                // theory formula. Unknown is the absence of an answer and is
                // passed through untouched.
                other => return other,
            }
        }
        Solution::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Solution, Solver};

    /// x is even: the toy theory the scaffolding's own tests exercise.
    struct Even {
        x: Bv,
    }

    impl Theory for Even {
        fn name(&self) -> &'static str {
            "even"
        }
        fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
            match model.eval(&self.x) {
                Some(v) if v % 2 == 1 => vec![self.x.and(&Bv::val(1, 8)).eq(&Bv::val(0, 8))],
                _ => Vec::new(),
            }
        }
    }

    #[test]
    fn eval_reads_the_model_the_way_the_folder_does() {
        // ModelView::eval must agree with the folding semantics for an
        // expression with a good sample of node shapes in it.
        let mut s = Solver::new();
        let x = s.var("x", 8);
        let y = s.var("y", 8);
        s.assert(x.eq(&Bv::val(0x37, 8)));
        s.assert(y.eq(&Bv::val(0x9a, 8)));
        let cond = x.add(&y).xor(&x.rotl(3)).ult(&Bv::val(0x20, 8));
        let e = Bv::ite(&cond, &x.extract(3, 0), &y.extract(3, 0));
        let Solution::Sat(m) = s.check() else {
            panic!("expected SAT")
        };
        let view = ModelView::new(&m);
        // The same expression, evaluated independently in u8 arithmetic: the
        // rotation is an 8-bit rotate, which is what `u8::rotate_left` is.
        let (xv, yv) = (0x37u8, 0x9au8);
        let rot = xv.rotate_left(3);
        let expect = if (xv.wrapping_add(yv) ^ rot) < 0x20 {
            xv & 0xf
        } else {
            yv & 0xf
        };
        assert_eq!(view.eval(&e), Some(u64::from(expect)));
        assert_eq!(view.get("x"), Some(0x37));
        // A term mentioning an undeclared id is the only `None`: ids are read
        // out of the model, and everything here was declared.
        assert_eq!(view.eval(&x), Some(0x37));
    }

    #[test]
    fn a_model_is_validated_and_re_solved_until_consistent() {
        // x + 1 == 4 is Sat with x = 3 and only x = 3. The plain entry point
        // hands back a model the theory rejects; the theory query must see the
        // rejection (law (b)) and answer Unsat, because an even x cannot make
        // the constraint hold.
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.assert(x.add(&Bv::val(1, 8)).eq(&Bv::val(4, 8)));
        s.register_theory(Even { x });
        let Solution::Sat(m) = s.check() else {
            panic!("expected SAT")
        };
        assert_eq!(m.get("x"), Some(3));
        assert!(
            !s.model_consistent(&m),
            "plain check() ignores theories, and the violation is visible"
        );
        assert!(matches!(s.check_theories(100_000, None), Solution::Unsat));

        // With a model that can be repaired, the loop repairs it: x + y == 4
        // has even solutions, and the Sat that comes out is validated.
        let mut s2 = Solver::new();
        let (x2, y2) = (s2.var("x", 8), s2.var("y", 8));
        s2.assert(x2.add(&y2).eq(&Bv::val(4, 8)));
        s2.register_theory(Even { x: x2 });
        let Solution::Sat(m2) = s2.check_theories(100_000, None) else {
            panic!("expected SAT")
        };
        assert_eq!(m2.get("x").map(|v| v % 2), Some(0));
        assert!(s2.model_consistent(&m2));
    }

    #[test]
    fn a_contradictory_theory_is_unsat_immediately() {
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.assert(x.eq(&Bv::val(1, 8)));
        s.register_theory(Contradiction);
        assert!(matches!(s.check_theories(100_000, None), Solution::Unsat));
        // The stored constraints are untouched: the same solver without the
        // theory loop answers as it always did.
        assert!(matches!(s.check(), Solution::Sat(_)));
    }

    #[test]
    fn round_exhaustion_is_unknown_never_a_wrong_verdict() {
        // A theory that always objects and can always be obeyed: each round
        // rules out the model just seen, so the loop keeps finding a new one
        // and must run out of rounds - `Unknown`, never the unvalidated model.
        struct NotThisOne {
            x: Bv,
        }
        impl Theory for NotThisOne {
            fn name(&self) -> &'static str {
                "not-this-one"
            }
            fn violated_lemmas(&self, m: &ModelView<'_>) -> Vec<Bv> {
                let v = m.eval(&self.x).expect("8-bit term");
                vec![self.x.ne(&Bv::val(v, 8))]
            }
        }
        let mut s = Solver::new().with_theory_rounds(3);
        let x = s.var("x", 8);
        s.register_theory(NotThisOne { x });
        assert!(matches!(s.check_theories(100_000, None), Solution::Unknown));
    }

    #[test]
    fn no_theories_means_no_change() {
        // The regression bar for every existing caller: an unregistered
        // solver behaves identically at the theory entry points.
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.assert(x.ult(&Bv::val(10, 8)));
        assert!(matches!(
            s.check_assumptions_theories(&[x.ugt(&Bv::val(5, 8))], 100_000, None),
            Solution::Sat(_)
        ));
        assert!(matches!(
            s.check_all_theories(
                &[x.ugt(&Bv::val(30, 8)), x.ult(&Bv::val(20, 8))],
                100_000,
                None
            ),
            Solution::Unsat
        ));
    }

    #[test]
    fn duplicate_registration_panics() {
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.register_theory(Even { x: x.clone() });
        let dup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.register_theory(Even { x: x.clone() });
        }));
        assert!(dup.is_err(), "a duplicated theory name must be refused");
    }
}
