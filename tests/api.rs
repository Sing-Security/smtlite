//! Tests for the public surface that is not an arithmetic encoding: model readback, provenance
//! queries, the solver's budgeting knobs, and the small value helpers.

use smtlite::{Bv, Solution, Solver, mask};

fn sat(s: &Solver) -> smtlite::Model {
    match s.check() {
        Solution::Sat(m) => m,
        other => panic!("expected SAT, got {other:?}"),
    }
}

// ---- Model readback -----------------------------------------------------------------------

#[test]
fn a_declared_variable_no_constraint_touches_still_reads_back() {
    // The model is total over the declaration, not just the variables a constraint encoded.
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
fn an_empty_constraint_set_yields_a_total_model() {
    let mut s = Solver::new();
    let _a = s.var("a", 1);
    let _b = s.var("b", 64);

    let m = sat(&s);
    assert!(m.get("a").is_some());
    assert!(m.get("b").is_some());
    assert!(m.get("never_declared").is_none());
}

#[test]
fn every_declared_width_reads_back_exactly() {
    // Walk widths 1..=64, pin each variable, and read it back verbatim.
    for w in 1..=64u32 {
        let v = mask(0xDEAD_BEEF_CAFE_BABE, w);
        let mut s = Solver::new();
        let x = s.var("x", w);
        s.assert(x.eq(&Bv::val(v, w)));
        let m = sat(&s);
        assert_eq!(m.get("x"), Some(v), "width {w} did not read back");
    }
}

// ---- Solver budgeting ---------------------------------------------------------------------

#[test]
fn zero_clause_cap_yields_unknown_without_solving() {
    // The blaster always allocates the pinned `true` literal, so even an empty formula exceeds a
    // zero cap and comes back Unknown rather than a verdict.
    let s = Solver::new().with_max_clauses(0);
    assert!(matches!(s.check(), Solution::Unknown));
}

#[test]
fn zero_budget_refutes_by_propagation_but_cannot_branch() {
    // Unsatisfiable by propagation alone: no decision is spent, so a zero budget still refutes it.
    let mut s = Solver::new();
    let x = s.var("x", 8);
    s.assert(x.eq(&Bv::val(5, 8)));
    s.assert(x.eq(&Bv::val(6, 8)));
    assert!(matches!(s.check_with_budget(0), Solution::Unsat));

    // Satisfiable, but only after branching. `y | ~y` is a tautology that unit propagation cannot
    // see through, so the solver has to decide `y` - and the first decision exceeds a zero budget.
    let mut t = Solver::new();
    let y = t.var("y", 1);
    assert!(matches!(
        t.check_all_with_budget(&[y.or(&y.not())], 0),
        Solution::Unknown
    ));
}

#[test]
fn an_unconstrained_solver_answers_without_spending_a_decision() {
    // A variable no clause mentions is free: it needs no decision, so even a zero budget settles
    // the query, and the model still carries a value for it.
    let mut s = Solver::new();
    let _y = s.var("y", 8);
    match s.check_with_budget(0) {
        Solution::Sat(m) => assert!(m.get("y").is_some(), "the model is total"),
        other => panic!("expected SAT, got {other:?}"),
    }
}

#[test]
fn an_elapsed_deadline_yields_unknown_not_unsat() {
    // `x != x` is unsatisfiable, but an already-elapsed deadline must answer Unknown - "no answer",
    // never a (here, coincidentally correct) "no solution".
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let past = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap_or_else(std::time::Instant::now);
    assert!(matches!(
        s.check_all_within(&[x.ne(&x)], 1_000_000, past),
        Solution::Unknown
    ));
}

#[test]
fn check_all_ignores_stored_asserts_and_does_not_mutate_them() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    s.assert(x.eq(&Bv::val(5, 8)));

    // The explicit set is the whole formula: the stored `x == 5` must not leak in to contradict it.
    assert!(matches!(
        s.check_all(&[x.eq(&Bv::val(9, 8))]),
        Solution::Sat(_)
    ));

    // The stored asserts survive the call and still hold.
    assert_eq!(sat(&s).get("x"), Some(5));
}

// ---- Provenance ---------------------------------------------------------------------------

#[test]
fn depends_on_matches_by_name_prefix() {
    let mut s = Solver::new();
    let hdr = s.var("hdr.len", 16);
    let init = s.var("init.base", 16);

    let from_hdr = hdr.add(&Bv::val(1, 16));
    assert!(s.depends_on(&from_hdr, "hdr."));
    assert!(!s.depends_on(&from_hdr, "init"));
    assert!(s.depends_on(&hdr, "hdr"));
    assert!(!s.depends_on(&Bv::val(1, 16), "hdr"));

    let both = hdr.add(&init);
    assert!(s.depends_on(&both, "hdr."));
    assert!(s.depends_on(&both, "init."));
}

#[test]
fn var_ids_deduplicates_shared_subgraphs() {
    let mut s = Solver::new();
    let a = s.var("a", 8);
    let b = s.var("b", 8);
    // `a` used twice is one node, visited once.
    assert_eq!(a.add(&a).var_ids().len(), 1);
    let mut ids = a.add(&b).var_ids();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1]);
}

#[test]
fn var_ids_walks_a_deep_chain_without_overflowing_the_stack() {
    let mut s = Solver::new();
    let v = s.var("v", 8);
    let mut deep = v;
    // `shl` rather than `not`: the simplifier collapses `~~x` to `x`, so a `not` chain would
    // never grow. A constant shift is left as built and costs no width assert, so the chain stays
    // deep and each step is O(1).
    for _ in 0..100_000 {
        deep = deep.shl(1);
    }
    // The walk is iterative, so a chain this deep must not blow the stack.
    assert_eq!(deep.var_ids(), vec![0]);
    assert!(s.depends_on(&deep, "v"));
    // `Rc` drop, unlike the walk, *is* recursive - leak rather than overflow in a test.
    std::mem::forget(deep);
}

// ---- Value helpers ------------------------------------------------------------------------

#[test]
fn mask_truncates_to_the_low_bits() {
    assert_eq!(mask(0xabcd, 8), 0xcd);
    assert_eq!(mask(0xabcd, 16), 0xabcd);
    assert_eq!(mask(0xff, 4), 0xf);
    assert_eq!(mask(u64::MAX, 64), u64::MAX);
}

#[test]
fn ptr_eq_is_identity_not_structure() {
    let a = Bv::val(1, 8);
    assert!(a.ptr_eq(&a));
    // Structurally equal but separately built: not the same node.
    let b = Bv::val(1, 8);
    assert!(!a.ptr_eq(&b));
    let c = a.add(&b);
    assert!(c.ptr_eq(&c));
}

#[test]
fn as_const_folds_constant_expressions() {
    assert_eq!(Bv::val(7, 8).as_const(), Some(7));
    assert_eq!(Bv::val(0x1ff, 8).as_const(), Some(0xff)); // masked to width on construction
    // A constant expression folds on construction, so it reads back as a constant.
    assert_eq!(Bv::val(1, 8).add(&Bv::val(1, 8)).as_const(), Some(2));
    // Folding follows SMT-LIB zero-divisor semantics, not Rust's divide-by-zero panic.
    assert_eq!(Bv::val(9, 8).udiv(&Bv::val(0, 8)).as_const(), Some(0xff));
    // A rewrite that keeps a variable keeps returning `None`, as a variable is not a constant.
    let mut s = Solver::new();
    let x = s.var("x", 8);
    assert_eq!(x.add(&Bv::val(0, 8)).as_const(), None);
}

#[test]
fn display_and_debug_are_stable() {
    assert_eq!(Bv::val(5, 8).to_string(), "#5:8");
    assert_eq!(format!("{:?}", Bv::val(5, 8)), "Bv(#5:8)");
}

#[test]
fn widths_compose_as_documented() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    assert_eq!(x.width(), 8);
    assert_eq!(x.extract(3, 0).width(), 4);
    assert_eq!(x.zext(16).width(), 16);
    assert_eq!(x.sext(16).width(), 16);
    assert_eq!(x.concat(&x).width(), 16);
    assert_eq!(x.eq(&x).width(), 1);
    assert_eq!(Bv::ite(&x.eq(&x), &x, &x).width(), 8);
}
