//! Uninterpreted functions: the congruence theory.
//!
//! An uninterpreted function `f` is a function with a signature and nothing
//! else. The solver may pick any value it likes for an application; the one law
//! it must obey is *congruence*:
//!
//! > equal arguments imply equal results — `a = b  =>  f(a) = f(b)`.
//!
//! That is exactly the property a symbolic executor needs from a call it cannot
//! or will not model (a syscall's status, a CPI's return value, a hash): the
//! program's own checks may compare two derivations, and the solver is entitled
//! to make them differ only when the inputs really differ. It is *not* a claim
//! that the function does anything in particular - see the crate's honest-limit
//! note in [`super`].
//!
//! ## How it is encoded
//!
//! Every application gets a fresh `Bv` of the declared result width; the theory
//! records the pair `(arguments, result)`. When the core answers `Sat`, every
//! recorded pair whose arguments the model makes equal but whose results it
//! makes different is a congruence violation, and the theory emits that one
//! axiom instance:
//!
//! ```text
//! (∧ₖ argᵢ[k] = argⱼ[k]) => resultᵢ = resultⱼ
//! ```
//!
//! Lemmas are monotone, so the module-level law holds unchanged: `Unsat` is
//! immediate, `Sat` is returned only after a model survives validation, and the
//! round budget running out is `Unknown`. A pair is instantiated at most once -
//! the emitted set is kept, because the lemma that was asserted cannot be
//! violated by the *next* model, and re-emitting it would only grow the
//! formula.
//!
//! ## Determinism of applications
//!
//! Applying the same function to the *same terms* twice returns the same fresh
//! result rather than two results plus a lemma. Terms are compared by node
//! identity, which is how a built expression is shared in this crate anyway; two
//! separately-built but semantically equal argument tuples are still covered,
//! by congruence, against whatever values the model gives them.
//!
//! ## Cost
//!
//! Lazy instantiation examines every pair of applications per round, so a round
//! is quadratic in the application count; the number of rounds is the growth of
//! the congruence closure. That is deliberate: the alternative is eagerly
//! emitting the same quadratic set into every query, including the thousands a
//! symbolic sweep fires at paths that never touch the function.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::Solver;
use crate::bv::Bv;
use crate::theory::{ModelView, Theory};

/// One application: the argument terms and the fresh result term.
#[derive(Clone, Debug)]
struct App {
    args: Vec<Bv>,
    result: Bv,
}

struct Inner {
    name: &'static str,
    /// The width of each argument, in order.
    arg_widths: Vec<u32>,
    /// The width of the result.
    result_width: u32,
    /// Every application made, in creation order.
    apps: RefCell<Vec<App>>,
    /// Applications by argument-node identity, so a repeat returns the same
    /// result instead of a second independent one.
    by_args: RefCell<HashMap<Vec<usize>, usize>>,
}

impl fmt::Debug for Inner {
    /// Hand-written because `RefCell`s of `Bv`-bearing applications are not
    /// interesting to print beyond their shape, and the `DAG` debug form of a
    /// deep term is enormous.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Uninterpreted")
            .field("name", &self.name)
            .field("arg_widths", &self.arg_widths)
            .field("result_width", &self.result_width)
            .field("applications", &self.apps.borrow().len())
            .field("distinct_argument_tuples", &self.by_args.borrow().len())
            .finish()
    }
}

/// An uninterpreted function: apply it to build terms, register a clone with the
/// solver as a [`Theory`], and let congruence do the rest.
///
/// Clones share their application log, so the handle used to build terms and
/// the handle registered with the solver are the *same* function.
///
/// ```
/// use smtlite::{Bv, Solution, Solver, Uninterpreted};
///
/// let mut s = Solver::new();
/// let a = s.var("a", 8);
/// let b = s.var("b", 8);
/// let f = Uninterpreted::new("f", &[8], 8);
/// let (fa, fb) = (f.apply(&mut s, &[a.clone()]), f.apply(&mut s, &[b.clone()]));
/// s.assert(a.eq(&b));
/// s.assert(fa.ne(&fb));
/// s.register_theory(f);
/// // Equal arguments, unequal results: no model can satisfy congruence.
/// assert!(matches!(s.check_theories(100_000, None), Solution::Unsat));
/// ```
#[derive(Clone, Debug)]
pub struct Uninterpreted {
    inner: Rc<Inner>,
}

impl Uninterpreted {
    /// Declare a function `name` from `arg_widths` to `result_width` bits.
    ///
    /// # Panics
    ///
    /// Panics if an argument or the result width is outside `1..=64` (the
    /// crate's term ceiling), or if the name is empty - an empty name makes a
    /// report ambiguous.
    #[must_use]
    pub fn new(name: &'static str, arg_widths: &[u32], result_width: u32) -> Self {
        assert!(!name.is_empty(), "an uninterpreted function needs a name");
        for &w in arg_widths {
            assert!(
                (1..=64).contains(&w),
                "argument width must be 1..=64, got {w}"
            );
        }
        assert!(
            (1..=64).contains(&result_width),
            "result width must be 1..=64, got {result_width}"
        );
        Self {
            inner: Rc::new(Inner {
                name,
                arg_widths: arg_widths.to_vec(),
                result_width,
                apps: RefCell::new(Vec::new()),
                by_args: RefCell::new(HashMap::new()),
            }),
        }
    }

    /// The function's declared name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.inner.name
    }

    /// Apply the function to `args`, returning the fresh result term.
    ///
    /// The same argument tuple (by term identity) returns the same result term,
    /// so a caller that rebuilds a query does not accumulate duplicate
    /// applications. The result is a variable the solver's model carries, so it
    /// can be constrained, compared, and read back like any other term.
    ///
    /// # Panics
    ///
    /// Panics if the arity or any argument width differs from the declaration.
    pub fn apply(&self, solver: &mut Solver, args: &[Bv]) -> Bv {
        assert_eq!(
            args.len(),
            self.inner.arg_widths.len(),
            "`{}` takes {} argument(s), got {}",
            self.inner.name,
            self.inner.arg_widths.len(),
            args.len()
        );
        for (k, (arg, &want)) in args.iter().zip(&self.inner.arg_widths).enumerate() {
            assert_eq!(
                arg.width(),
                want,
                "`{}` argument {k} must be {want} bits, got {}",
                self.inner.name,
                arg.width()
            );
        }
        let key: Vec<usize> = args.iter().map(Bv::ptr_key).collect();
        if let Some(&i) = self.inner.by_args.borrow().get(&key) {
            return self.inner.apps.borrow()[i].result.clone();
        }
        let result = solver.var(
            &format!("uf:{}#{}", self.inner.name, self.inner.apps.borrow().len()),
            self.inner.result_width,
        );
        let i = {
            let mut apps = self.inner.apps.borrow_mut();
            apps.push(App {
                args: args.to_vec(),
                result: result.clone(),
            });
            apps.len() - 1
        };
        self.inner.by_args.borrow_mut().insert(key, i);
        result
    }

    /// The number of applications made so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.apps.borrow().len()
    }

    /// Whether no application has been made yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The congruence lemma for the pair `(i, j)`: the conjunction of the
    /// arguments' equalities implies the results' equality.
    fn congruence(&self, apps: &[App], i: usize, j: usize) -> Bv {
        let (a, b) = (&apps[i], &apps[j]);
        debug_assert_eq!(a.args.len(), b.args.len());
        let mut antecedent: Option<Bv> = None;
        for (x, y) in a.args.iter().zip(&b.args) {
            let eq = x.eq(y);
            antecedent = Some(match antecedent {
                None => eq,
                Some(acc) => acc.land(&eq),
            });
        }
        let consequent = a.result.eq(&b.result);
        match antecedent {
            // 0-arity: congruence says every application of a constant function
            // is the same constant, unconditionally.
            None => consequent,
            Some(a) => a.implies(&consequent),
        }
    }
}

impl Theory for Uninterpreted {
    fn name(&self) -> &'static str {
        self.inner.name
    }

    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
        // No instantiated-pair bookkeeping here on purpose: `violated_lemmas`
        // is also the public validation step (`Solver::model_consistent`), and
        // a validation that recorded what it had seen would suppress the lemma
        // in the *next* query's round loop. Re-emitting within one query is
        // bounded - once a pair's lemma is asserted every later model satisfies
        // it, so each pair is emitted at most once per solve naturally.
        let apps = self.inner.apps.borrow();
        let mut out = Vec::new();
        for i in 0..apps.len() {
            for j in (i + 1)..apps.len() {
                // Both directions have to be readable: an argument no model
                // can value (a wider term, a lost variable) is not a violation
                // - the query is about this model, and the model cannot speak
                // to it.
                let args_equal = apps[i].args.iter().zip(&apps[j].args).all(|(x, y)| {
                    match (model.eval(x), model.eval(y)) {
                        (Some(xv), Some(yv)) => xv == yv,
                        _ => false,
                    }
                });
                if !args_equal {
                    continue;
                }
                let results_differ =
                    match (model.eval(&apps[i].result), model.eval(&apps[j].result)) {
                        (Some(xv), Some(yv)) => xv != yv,
                        _ => false,
                    };
                if results_differ {
                    out.push(self.congruence(&apps, i, j));
                }
            }
        }
        out
    }

    fn eager_lemmas(&self) -> Vec<Bv> {
        let apps = self.inner.apps.borrow();
        let mut out = Vec::new();
        for i in 0..apps.len() {
            for j in (i + 1)..apps.len() {
                out.push(self.congruence(&apps, i, j));
            }
        }
        out
    }
}
