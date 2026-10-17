//! The theory of arrays, checked from outside: read-over-write against small
//! concrete tables, the same two soundness laws every theory inherits, and the
//! term-sharing guarantees the engines will lean on.
//!
//! The oracle here is the array's *definition*, written down independently of
//! the lemma encoding: a finite table plus `select(store(t, i, v), j) = if i==j
//! { v } else { t[j] }`. The exhaustive test runs every table at widths small
//! enough to enumerate, pins the array with one select per index, and demands
//! the solver's verdict equal that definition's verdict.

use smtlite::{Array, ArrayTheory, Bv, ModelView, Solution, Solver};

fn check(s: &Solver) -> Solution {
    s.check_theories(200_000, None)
}

/// The concrete array semantics, as a test-local function of a whole table.
fn select_store(table: &[u8], stored: usize, value: u8, at: usize) -> u8 {
    if at == stored { value } else { table[at] }
}

#[test]
fn reading_back_the_cell_just_written_is_free() {
    // A syntactic identity: selecting the very term that was stored returns
    // the stored value, with no theory round at all.
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    let read = b.select(&mut s, &i);
    s.assert(read.ne(&v));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn a_store_through_a_different_index_reads_the_parent() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let j = s.var("j", 4);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    let read = b.select(&mut s, &j);
    let parent = a.select(&mut s, &j);
    s.assert(i.ne(&j));
    s.assert(read.ne(&parent));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn an_equal_index_does_not_read_the_parent() {
    // With i == j the stored value wins, so the select may differ from the
    // parent - but not from the stored value.
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let j = s.var("j", 4);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    let read = b.select(&mut s, &j);
    let parent = a.select(&mut s, &j);
    s.assert(i.eq(&j));
    s.assert(read.ne(&parent));
    s.register_theory(arrays);

    let Solution::Sat(model) = check(&s) else {
        panic!("the stored value may differ from the old cell");
    };
    assert!(s.model_consistent(&model));
    let view = ModelView::new(&model);
    assert_eq!(
        view.eval128(&read),
        view.eval128(&v),
        "at an equal index the store's result must be the stored value"
    );
    assert_eq!(model.get("i"), model.get("j"));
}

#[test]
fn equal_indices_select_equal_elements_at_a_base_array() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let j = s.var("j", 4);
    let x = a.select(&mut s, &i);
    let y = a.select(&mut s, &j);
    s.assert(i.eq(&j));
    s.assert(x.ne(&y));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn a_constant_array_selects_its_element_everywhere() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let c = arrays.constant(&Bv::val(42, 8), 4);
    let k = s.var("k", 4);
    let read = c.select(&mut s, &k);
    s.assert(read.ne(&Bv::val(42, 8)));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn a_chain_of_stores_resolves_each_level_lazily() {
    // Two stores deep, and the read index is *equal* to the lower store's index
    // but a different term: the model must walk two read-over-write levels.
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let j = s.var("j", 4);
    let k = s.var("k", 4);
    let v1 = s.var("v1", 8);
    let v2 = s.var("v2", 8);
    let b1 = a.store(&mut s, &i, &v1);
    let b2 = b1.store(&mut s, &j, &v2);
    let read = b2.select(&mut s, &k);
    s.assert(k.eq(&i));
    s.assert(k.ne(&j));
    s.assert(read.ne(&v1));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn select_and_store_terms_are_shared() {
    // Identical terms are one term: two stores of the same cell are one array,
    // and selecting the same array at the same index is one result.
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let v = s.var("v", 8);
    assert_eq!(arrays.arrays(), 1);
    let b1 = a.store(&mut s, &i, &v);
    let b2 = a.store(&mut s, &i, &v);
    assert_eq!(arrays.arrays(), 2, "the second identical store is shared");
    let j = s.var("j", 4);
    let r1 = b1.select(&mut s, &j);
    let r2 = b2.select(&mut s, &j);
    assert!(r1.ptr_eq(&r2), "the same cell must be the same term");
    // Two terms exist: the store's select and the parent read it falls back to.
    assert_eq!(arrays.selects(), 2);
    // Re-selecting the same array at the same index is also shared.
    let r3 = b1.select(&mut s, &j);
    assert!(r1.ptr_eq(&r3));
    assert_eq!(arrays.selects(), 2);
}

#[test]
fn arrays_of_different_shapes_coexist() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let narrow = arrays.var(&mut s, 4, 8);
    let wide = arrays.var(&mut s, 8, 128);
    assert_eq!(narrow.shape(), (4, 8));
    assert_eq!(wide.shape(), (8, 128));
    let i = s.var("i", 8);
    let v = s.var("v", 128);
    let b = wide.store(&mut s, &i, &v);
    let read = b.select(&mut s, &i);
    s.assert(read.ne(&v));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}

#[test]
fn a_128_bit_element_round_trips_through_a_store() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 5, 128);
    let i = s.var("i", 5);
    let wide = (1u128 << 100) | 7;
    let b = a.store(&mut s, &i, &Bv::val128(wide, 128));
    let read = b.select(&mut s, &i);
    s.assert(read.eq(&Bv::val128(wide, 128)));
    s.register_theory(arrays);
    let Solution::Sat(model) = check(&s) else {
        panic!("the store must be able to hold a 128-bit element");
    };
    assert_eq!(ModelView::new(&model).eval128(&read), Some(wide));
}

/// The exhaustive-agreement test. At index width 1 and element width 2 the whole
/// domain is 4 tables x 2 x 2 x 4 = 64 assignments; for each, the array is
/// pinned cell by cell (one select per index), the query is asserted, and the
/// solver's verdict must equal the concrete definition's.
///
/// Each case's `expect` is the definition rewritten by hand - not the solver's
/// answer - which is what makes this a cross-check rather than a tautology.
#[test]
fn verdicts_agree_with_the_concrete_table_semantics() {
    type Query = fn(&mut Solver, &Array, &Bv, &Bv, &Bv) -> Bv;
    type Expect = fn(&[u8; 2], usize, usize, u8) -> bool;
    // Index width 1: cells {0, 1}. Element width 2: values {0, 1, 2, 3}.
    let cases: &[(&str, Query, Expect)] = &[
        (
            "select(store(a,i,v), j) != select(a,j)",
            |s, a, i, j, v| {
                let b = a.store(s, i, v);
                b.select(s, j).ne(&a.select(s, j))
            },
            |t, i, j, v| select_store(t, i, v, j) != t[j],
        ),
        (
            "select(store(a,i,v), j) != v",
            |s, a, i, j, v| {
                let b = a.store(s, i, v);
                b.select(s, j).ne(v)
            },
            |t, i, j, v| select_store(t, i, v, j) != v,
        ),
        (
            "i == j && select(store(a,i,v), j) != a[i]",
            |s, a, i, j, v| {
                let b = a.store(s, i, v);
                i.eq(j).land(&b.select(s, j).ne(&a.select(s, i)))
            },
            |t, i, j, v| i == j && select_store(t, i, v, j) != t[i],
        ),
        (
            "i == j && select(a,i) != select(a,j)",
            |s, a, i, j, _| i.eq(j).land(&a.select(s, i).ne(&a.select(s, j))),
            // An array is a function: equal indices are the same cell.
            |_, _, _, _| false,
        ),
    ];

    for &(label, query, expect) in cases {
        for table in [[0u8, 0], [0, 1], [1, 0], [3, 2]] {
            for i in 0usize..2 {
                for j in 0usize..2 {
                    for v in 0u8..4 {
                        let mut s = Solver::new();
                        let arrays = ArrayTheory::new();
                        let a = arrays.var(&mut s, 1, 2);
                        // Pinning every cell makes the finite table the whole
                        // array: width 1 has exactly two cells.
                        for (k, cell) in table.iter().enumerate() {
                            let idx = Bv::val(k as u64, 1);
                            let pinned = a.select(&mut s, &idx);
                            s.assert(pinned.eq(&Bv::val(u64::from(*cell), 2)));
                        }
                        let iv = s.var("i", 1);
                        let jv = s.var("j", 1);
                        let vv = s.var("v", 2);
                        s.assert(iv.eq(&Bv::val(i as u64, 1)));
                        s.assert(jv.eq(&Bv::val(j as u64, 1)));
                        s.assert(vv.eq(&Bv::val(u64::from(v), 2)));
                        let goal = query(&mut s, &a, &iv, &jv, &vv);
                        s.assert(goal.clone());
                        s.register_theory(arrays);

                        let wanted = expect(&table, i, j, v);
                        let got = check(&s);
                        match got {
                            Solution::Sat(model) => {
                                assert!(
                                    wanted,
                                    "{label}: table={table:?} i={i} j={j} v={v}: \
                                     solver says Sat, the definition says impossible"
                                );
                                assert!(
                                    s.model_consistent(&model),
                                    "{label}: the returned model fails validation"
                                );
                            }
                            Solution::Unsat => assert!(
                                !wanted,
                                "{label}: table={table:?} i={i} j={j} v={v}: \
                                 the definition has a witness but the solver is Unsat"
                            ),
                            Solution::Unknown => panic!("{label}: this fragment is decidable"),
                        }
                    }
                }
            }
        }
    }
}

/// Law (a) for arrays: a lazy `Unsat` stays `Unsat` under full instantiation.
#[test]
fn lazy_unsat_is_stable_under_eager_array_lemmas() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let j = s.var("j", 4);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    let read = b.select(&mut s, &j);
    let parent = a.select(&mut s, &j);
    let constraints = vec![i.ne(&j), read.ne(&parent)];
    for c in &constraints {
        s.assert(c.clone());
    }
    s.register_theory(arrays.clone());
    assert!(matches!(check(&s), Solution::Unsat));

    // The same formula with every axiom instance asserted eagerly, on the
    // plain QF_BV entry point: still Unsat.
    let mut e = Solver::new();
    for c in &constraints {
        e.assert(c.clone());
    }
    for l in s.eager_theory_lemmas() {
        e.assert(l);
    }
    assert!(matches!(e.check(), Solution::Unsat));

    // And the eager set is non-empty, so the test is about the lemmas and not
    // about the query being contradictory on its own.
    assert!(
        !s.eager_theory_lemmas().is_empty(),
        "read-over-write must contribute instances"
    );
    assert_eq!(arrays.selects(), 2, "one read per array level");
}

/// Law (b) for arrays, the independent half: on a satisfiable read-over-write
/// query the model's own values obey the concrete semantics.
#[test]
fn a_sat_model_obeys_read_over_write_concretely() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 3, 8);
    let i = s.var("i", 3);
    let j = s.var("j", 3);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    let read = b.select(&mut s, &j);
    let parent = a.select(&mut s, &j);
    s.assert(read.ne(&parent));
    s.register_theory(arrays);
    let Solution::Sat(model) = check(&s) else {
        panic!("a store can always change some cell");
    };
    assert!(s.model_consistent(&model));

    let view = ModelView::new(&model);
    let iv = view.eval128(&i).expect("i is 3-bit");
    let jv = view.eval128(&j).expect("j is 3-bit");
    assert_eq!(
        iv, jv,
        "the only way the read differs is at the written cell"
    );
    assert_eq!(
        view.eval128(&read),
        view.eval128(&v),
        "the written cell selects the written value"
    );
}

/// The tractability law the round loop has to obey: a lemma's antecedent is
/// almost always a term the *stored* background already constrains - here the
/// path condition `j == i` and the read-over-write lemma's `i == j` are the
/// same circuit, only built twice.
///
/// The blaster's structural interning shares a circuit only between terms
/// blasted by the *same* [`Blaster`], so a round loop that blasts the background
/// separately makes the core prove the two copies equivalent - which is the
/// query a learning-free DPLL is worst at. This test is the measured case: at
/// 64 bits it answered `Unknown` after 9 s before the rounds blasted
/// `stored ++ lemmas` together, and sub-millisecond `Unsat` after.
#[test]
fn a_lemma_shares_the_background_circuit_it_names() {
    for width in [64, 128] {
        let mut s = Solver::new();
        let arrays = ArrayTheory::new();
        let a = arrays.var(&mut s, width, width);
        let i = s.var("i", width);
        let j = s.var("j", width);
        let one = Bv::val(1, width);
        let b = a.store(&mut s, &i, &one);
        let read = b.select(&mut s, &j);
        // Stored, not passed per call: this is the "background once" shape the
        // symbolic walkers use.
        s.assert(j.eq(&i));
        s.assert(read.ne(&one));
        s.register_theory(arrays);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        assert!(
            matches!(s.check_theories(200_000, Some(deadline)), Solution::Unsat),
            "width {width}: the lemma names the background's own comparator"
        );
    }
}

/// A theory is not a solver feature toggle: the same formula without the theory
/// may be satisfiable, which is why registering it changes the answer.
#[test]
fn the_theory_is_doing_work() {
    let mut s = Solver::new();
    let arrays = ArrayTheory::new();
    let a = arrays.var(&mut s, 4, 8);
    let i = s.var("i", 4);
    let k = s.var("k", 4);
    let v = s.var("v", 8);
    let b = a.store(&mut s, &i, &v);
    // A different term for the same index, so the query is not resolved
    // syntactically: the equality is the solver's to discover.
    let read = b.select(&mut s, &k);
    s.assert(k.eq(&i));
    s.assert(read.ne(&v));
    // Without the theory registered, nothing connects the fresh select var to
    // the store: the formula is satisfiable on the core alone.
    assert!(matches!(s.check(), Solution::Sat(_)));
    s.register_theory(arrays);
    assert!(matches!(check(&s), Solution::Unsat));
}
