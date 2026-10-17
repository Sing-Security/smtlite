//! The theory layer's soundness laws, checked from outside the crate.
//!
//! Every test here is one of two laws - they are the whole contract, and each
//! theory that lands inherits them by adding a case:
//!
//! - **(a) UNSAT is a fact.** Lemma instantiation is monotone, so an `Unsat`
//!   the lazy loop reaches under a *strict* lemma subset stays `Unsat` when the
//!   full (eager) instantiation is asserted.
//! - **(b) SAT is a claim until validated.** A `Sat` the loop returns has an
//!   empty `violated_lemmas` in every registered theory, and - the independent
//!   half - the model actually satisfies the theory when read back through the
//!   theory's own semantics rather than through its own trait method.
//!
//! Plus the exhaustive check the layered stack relies on everywhere: at small
//! widths, the loop's verdict must agree with brute force over the whole
//! domain.

use smtlite::{Bv, ModelView, Solution, Solver, Theory, Uninterpreted};

/// Finitely many values, so the loop's answer can be brute-forced. `x & 1 == 0`.
struct Even {
    x: Bv,
}

impl Even {
    fn holds_concretely(v: u64) -> bool {
        v % 2 == 0
    }
}

impl Theory for Even {
    fn name(&self) -> &'static str {
        "even"
    }

    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
        let v = model.eval(&self.x).expect("x is 8-bit");
        if Self::holds_concretely(v) {
            Vec::new()
        } else {
            vec![self.x.and(&Bv::val(1, 8)).eq(&Bv::val(0, 8))]
        }
    }

    fn eager_lemmas(&self) -> Vec<Bv> {
        vec![self.x.and(&Bv::val(1, 8)).eq(&Bv::val(0, 8))]
    }
}

/// `x` is a multiple of 3, with the lemma phrased so it cannot be satisfied by
/// accident: `x mod 3 == 0`, expressed through the core's `urem`.
struct MultipleOfThree {
    x: Bv,
}

impl Theory for MultipleOfThree {
    fn name(&self) -> &'static str {
        "multiple-of-three"
    }

    fn violated_lemmas(&self, model: &ModelView<'_>) -> Vec<Bv> {
        match model.eval(&self.x) {
            Some(v) if v % 3 == 0 => Vec::new(),
            _ => vec![self.x.urem(&Bv::val(3, 8)).eq(&Bv::val(0, 8))],
        }
    }

    fn eager_lemmas(&self) -> Vec<Bv> {
        vec![self.x.urem(&Bv::val(3, 8)).eq(&Bv::val(0, 8))]
    }
}

/// Brute force the whole 8-bit domain: is there an `x` satisfying `constraints`
/// *and* the predicate? The oracle the loop's verdict is compared against.
fn brute_force(constraints: &[Bv], predicate: impl Fn(u64) -> bool) -> bool {
    (0u64..=255).any(|v| {
        if !predicate(v) {
            return false;
        }
        let mut s = Solver::new();
        let x = s.var("x", 8);
        for c in constraints {
            s.assert(c.clone());
        }
        s.assert(x.eq(&Bv::val(v, 8)));
        matches!(s.check(), Solution::Sat(_))
    })
}

/// Law (b), the independent half: every `Sat` the loop returns reads back as a
/// value the theory's concrete predicate accepts.
#[test]
fn every_sat_model_satisfies_the_theory_concretely() {
    for k in 0u64..40 {
        let mut s = Solver::new();
        let x = s.var("x", 8);
        s.assert(x.ult(&Bv::val(k + 1, 8)));
        s.assert(x.uge(&Bv::val(k % 7, 8)));
        s.register_theory(Even { x });
        if let Solution::Sat(m) = s.check_theories(50_000, None) {
            let v = m.get("x").expect("x is 8-bit");
            assert!(
                Even::holds_concretely(v),
                "k={k}: loop returned x={v}, which the theory rejects"
            );
            assert!(s.model_consistent(&m), "k={k}: model_consistent disagrees");
        }
    }
}

/// Law (a): whatever the lazy loop proves `Unsat`, the eager instantiation
/// proves `Unsat` too - and by brute force it is `Unsat`, so the loop never
/// claimed a fact that was not one.
#[test]
fn unsat_is_stable_under_full_instantiation() {
    let mut unsat = 0;
    let mut sat = 0;
    for k in 0u64..64 {
        let mut s = Solver::new();
        let x = s.var("x", 8);
        // x == k: satisfiable under the theory exactly for even k, so the
        // sweep exercises both answers - odd k is the Unsat case.
        s.assert(x.eq(&Bv::val(k, 8)));
        s.register_theory(Even { x });
        match s.check_theories(50_000, None) {
            Solution::Unsat => {
                unsat += 1;
                // The same formula with the full eager lemma set, solved with
                // the plain entry point: lemmas are monotone, so this must be
                // Unsat too.
                let mut e = Solver::new();
                let ex = e.var("x", 8);
                e.assert(ex.eq(&Bv::val(k, 8)));
                for l in s.eager_theory_lemmas() {
                    e.assert(l);
                }
                assert!(
                    matches!(e.check(), Solution::Unsat),
                    "k={k}: lazy Unsat but eager instantiation is satisfiable"
                );
                // And brute force agrees there is no witness at all.
                assert!(
                    !brute_force(&[ex.eq(&Bv::val(k, 8))], Even::holds_concretely),
                    "k={k}: loop said Unsat, brute force found a witness"
                );
            }
            Solution::Sat(m) => {
                sat += 1;
                assert_eq!(k % 2, 0, "k={k}: Sat for an odd value");
                assert!(s.model_consistent(&m));
            }
            Solution::Unknown => panic!("k={k}: 8-bit equality is a decision procedure"),
        }
    }
    assert_eq!((sat, unsat), (32, 32), "the sweep must hit both answers");
}

/// The exhaustive-agreement test: over a set of small formulas, the loop's
/// verdict equals brute force over the whole 8-bit domain, for every theory.
#[test]
fn verdict_agrees_with_brute_force_at_small_width() {
    type Case = (&'static str, fn(&Bv) -> Bv);
    let cases: &[Case] = &[
        ("x == 3", |x| x.eq(&Bv::val(3, 8))),
        ("x == 4", |x| x.eq(&Bv::val(4, 8))),
        ("x < 2", |x| x.ult(&Bv::val(2, 8))),
        ("x in [9, 10]", |x| {
            x.uge(&Bv::val(9, 8)).land(&x.ule(&Bv::val(10, 8)))
        }),
        ("x >> 1 == 5", |x| x.lshr(1).eq(&Bv::val(5, 8))),
        ("x * 3 == 6", |x| x.mul(&Bv::val(3, 8)).eq(&Bv::val(6, 8))),
    ];
    /// One theory under test: how to build it over a handle, and its concrete
    /// predicate - the oracle, written independently of the lemmas.
    type TheoryCase = (&'static str, fn(&Bv) -> Box<dyn Theory>, fn(u64) -> bool);
    let theories: &[TheoryCase] = &[
        (
            "even",
            |x| Box::new(Even { x: x.clone() }),
            |v| Even::holds_concretely(v),
        ),
        (
            "multiple-of-three",
            |x| Box::new(MultipleOfThree { x: x.clone() }),
            |v| v % 3 == 0,
        ),
    ];
    for (label, f) in cases {
        for (tlabel, theory_of, holds) in theories {
            let mut s = Solver::new();
            let x = s.var("x", 8);
            let c = f(&x);
            s.assert(c.clone());
            s.register_theory(theory_of(&x));
            let verdict = s.check_theories(50_000, None);
            let expect = brute_force(&[c], holds);
            assert_eq!(
                matches!(verdict, Solution::Sat(_)),
                expect,
                "{tlabel} / {label}: loop said {verdict:?}, brute force said {expect}"
            );
        }
    }
}

/// Round exhaustion is a first-class answer: a theory that always objects
/// yields `Unknown`, never the model it rejected, and never a verdict.
#[test]
fn exhaustion_never_reports_a_verdict() {
    struct AlwaysObjects {
        x: Bv,
    }
    impl Theory for AlwaysObjects {
        fn name(&self) -> &'static str {
            "always-objects"
        }
        fn violated_lemmas(&self, m: &ModelView<'_>) -> Vec<Bv> {
            let v = m.eval(&self.x).expect("x is 8-bit");
            vec![self.x.ne(&Bv::val(v, 8))]
        }
    }
    let mut s = Solver::new().with_theory_rounds(2);
    let x = s.var("x", 8);
    s.register_theory(AlwaysObjects { x });
    assert!(matches!(s.check_theories(50_000, None), Solution::Unknown));
}

/// The escape hatch is explicit: with the round count at zero the loop does not
/// validate at all, and the answer is exactly the plain entry point's.
#[test]
fn zero_rounds_is_the_plain_entry_point() {
    let mut s = Solver::new().with_theory_rounds(0);
    let x = s.var("x", 8);
    s.assert(x.eq(&Bv::val(3, 8)));
    s.register_theory(Even { x });
    let Solution::Sat(m) = s.check_theories(50_000, None) else {
        panic!("expected SAT");
    };
    assert_eq!(m.get("x"), Some(3));
    assert!(!s.model_consistent(&m), "and the model is unvalidated");
}

// region:    --- Uninterpreted functions

/// The function tables over a 2-bit domain: 4^4 = 256 functions.
fn all_tables() -> impl Iterator<Item = [u8; 4]> {
    (0u16..256).map(|t| {
        [
            (t & 3) as u8,
            ((t >> 2) & 3) as u8,
            ((t >> 4) & 3) as u8,
            ((t >> 6) & 3) as u8,
        ]
    })
}

/// Exhaustive search for a (x, y, f) that makes `holds` true, where `f` ranges
/// over every function on the 2-bit domain. The oracle the congruence theory's
/// verdict is checked against.
fn uf_brute_force(holds: impl Fn(u8, u8, &[u8; 4]) -> bool) -> bool {
    (0u8..4).any(|x| (0u8..4).any(|y| all_tables().any(|f| holds(x, y, &f))))
}

/// One query, and the concrete predicate it denotes. Builds `f(x)`, `f(y)` over
/// fresh 2-bit variables, asserts `constraint`, and runs the theory loop.
fn uf_query(
    constraint: fn(&Bv, &Bv, &Bv, &Bv) -> Bv,
    holds: impl Fn(u8, u8, &[u8; 4]) -> bool,
) -> (bool, bool) {
    let mut s = Solver::new();
    let x = s.var("x", 2);
    let y = s.var("y", 2);
    let f = Uninterpreted::new("f", &[2], 2);
    let fx = f.apply(&mut s, std::slice::from_ref(&x));
    let fy = f.apply(&mut s, std::slice::from_ref(&y));
    s.assert(constraint(&x, &y, &fx, &fy));
    s.register_theory(f);
    let verdict = matches!(s.check_theories(200_000, None), Solution::Sat(_));
    (verdict, uf_brute_force(holds))
}

/// Congruence is the whole law, and at a domain small enough to enumerate it
/// must agree with brute force over every function table. Each case is an
/// independently-written concrete predicate; none of them consults the lemma.
#[test]
fn congruence_verdicts_agree_with_exhaustive_function_search() {
    type Case = (
        &'static str,
        fn(&Bv, &Bv, &Bv, &Bv) -> Bv,
        fn(u8, u8, &[u8; 4]) -> bool,
    );
    let cases: &[Case] = &[
        (
            "x == y and f(x) != f(y) is impossible",
            |x, y, fx, fy| x.eq(y).land(&fx.ne(fy)),
            |x, y, f| x == y && f[x as usize] != f[y as usize],
        ),
        (
            "x != y and f(x) != f(y) is possible",
            |x, y, fx, fy| x.ne(y).land(&fx.ne(fy)),
            |x, y, f| x != y && f[x as usize] != f[y as usize],
        ),
        (
            "x == y and f(x) == f(y) is possible",
            |x, y, fx, fy| x.eq(y).land(&fx.eq(fy)),
            |x, y, f| x == y && f[x as usize] == f[y as usize],
        ),
        (
            "f(x) == y and f(y) != x is possible",
            |x, y, fx, fy| fx.eq(y).land(&fy.ne(x)),
            |x, y, f| f[x as usize] == y && f[y as usize] != x,
        ),
    ];
    for (label, constraint, holds) in cases {
        let (verdict, expect) = uf_query(*constraint, holds);
        assert_eq!(
            verdict, expect,
            "{label}: theory said {verdict}, brute force {expect}"
        );
    }
}

/// Nested applications: `f(f(x)) != f(x)` reaches a value through two
/// applications, so congruence has to relate `f(f(x))` to the value of `f` at
/// the *result* of the inner call. Brute force over the same tables.
#[test]
fn nested_applications_obey_congruence() {
    let mut s = Solver::new();
    let x = s.var("x", 2);
    let f = Uninterpreted::new("f", &[2], 2);
    let fx = f.apply(&mut s, std::slice::from_ref(&x));
    let ffx = f.apply(&mut s, std::slice::from_ref(&fx));
    s.assert(ffx.ne(&fx));
    s.register_theory(f);
    let verdict = matches!(s.check_theories(200_000, None), Solution::Sat(_));
    // Exists a table t and an x with t[t[x]] != t[x].
    let expect = (0u8..4).any(|x| all_tables().any(|t| t[t[x as usize] as usize] != t[x as usize]));
    assert!(expect, "the oracle case must be satisfiable");
    assert_eq!(verdict, expect);
}

/// Deduplication: applying `f` to the *same terms* returns the same result
/// term, not a second independent application. That is identity of the shared
/// node, checked with the crate's own cheap test - no solve involved.
#[test]
fn repeated_application_reuses_its_result_term() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let f = Uninterpreted::new("f", &[8], 8);
    let a = f.apply(&mut s, std::slice::from_ref(&x));
    let b = f.apply(&mut s, std::slice::from_ref(&x));
    assert!(
        a.ptr_eq(&b),
        "same arguments must return the same result term"
    );
    assert_eq!(f.len(), 1);

    // A separately-built but *equal* argument is a second application; its
    // result is tied to the first only by congruence, which is exactly the
    // distinction the dedup must not erase.
    // x + 1 - 1 is a genuinely separate node: constant folding does not
    // simplify it back to `x` (had it folded, this test would be vacuous).
    let x2 = x.add(&Bv::val(1, 8)).sub(&Bv::val(1, 8));
    let c = f.apply(&mut s, &[x2]);
    assert!(!a.ptr_eq(&c));
    assert_eq!(f.len(), 2);
}

/// The theory detects the violation on the plain entry point too, so the loop
/// is not hiding a model it should reject: `check()` returns an unvalidated
/// `Sat`, and `model_consistent` names it.
#[test]
fn an_unvalidated_model_is_visible_and_then_rejected() {
    let mut s = Solver::new();
    let (x, y) = (s.var("x", 4), s.var("y", 4));
    let f = Uninterpreted::new("f", &[4], 4);
    let (fx, fy) = (
        f.apply(&mut s, std::slice::from_ref(&x)),
        f.apply(&mut s, std::slice::from_ref(&y)),
    );
    s.assert(x.eq(&y));
    s.assert(fx.ne(&fy));
    s.register_theory(f);
    // The plain entry point knows nothing about congruence and happily returns
    // a model that violates it. That model is visible, and named as invalid.
    let Solution::Sat(plain) = s.check() else {
        panic!("expected SAT")
    };
    assert!(
        !s.model_consistent(&plain),
        "plain check does not consult theories"
    );
    // The loop does, and the only remaining model is the contradictory one.
    assert!(matches!(s.check_theories(200_000, None), Solution::Unsat));
}

/// When congruence alone does not decide the query, the loop repairs the model
/// instead of rejecting it: with equal arguments the results must agree, but
/// their common value is still free.
#[test]
fn a_repairable_model_comes_back_validated() {
    let mut s = Solver::new();
    let (x, y) = (s.var("x", 4), s.var("y", 4));
    let f = Uninterpreted::new("f", &[4], 4);
    let (fx, fy) = (
        f.apply(&mut s, std::slice::from_ref(&x)),
        f.apply(&mut s, std::slice::from_ref(&y)),
    );
    s.assert(x.eq(&y));
    s.assert(fx.ult(&Bv::val(9, 4)));
    s.assert(fy.ult(&Bv::val(9, 4)));
    s.register_theory(f);
    let Solution::Sat(m) = s.check_theories(200_000, None) else {
        panic!("expected SAT")
    };
    assert!(s.model_consistent(&m));
    let (a, b) = (
        m.get("uf:f#0").expect("4-bit"),
        m.get("uf:f#1").expect("4-bit"),
    );
    assert_eq!(a, b, "equal arguments must force equal results");
    assert!(a < 9, "and the query's own bound still holds");
}

/// Law (a) under congruence: whatever the lazy loop proves `Unsat`, the eager
/// instantiation proves `Unsat` too. A theory that emitted a lemma the eager
/// set omits would break monotonicity, and this is the test that would see it.
#[test]
fn congruence_unsat_is_stable_under_full_instantiation() {
    let mut s = Solver::new();
    let (x, y) = (s.var("x", 8), s.var("y", 8));
    let f = Uninterpreted::new("f", &[8], 8);
    let (fx, fy) = (
        f.apply(&mut s, std::slice::from_ref(&x)),
        f.apply(&mut s, std::slice::from_ref(&y)),
    );
    s.assert(x.eq(&y));
    s.assert(fx.ne(&fy));
    s.register_theory(f.clone());
    assert!(matches!(s.check_theories(200_000, None), Solution::Unsat));

    let mut eager = Solver::new();
    let (ex, ey) = (eager.var("x", 8), eager.var("y", 8));
    let g = Uninterpreted::new("f", &[8], 8);
    let (gx, gy) = (
        g.apply(&mut eager, std::slice::from_ref(&ex)),
        g.apply(&mut eager, std::slice::from_ref(&ey)),
    );
    eager.assert(ex.eq(&ey));
    eager.assert(gx.ne(&gy));
    for l in f.eager_lemmas() {
        eager.assert(l);
    }
    assert!(matches!(eager.check(), Solution::Unsat));
}
