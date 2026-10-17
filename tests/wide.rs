//! The 128-bit ceiling: constants, concatenation, arithmetic, and readback.
//!
//! Everything here exists because the suite's target arithmetic is `u128` - gmsol's
//! `checked_mul_div` widens two `u64`s into a `u128` before dividing - so a term the engine
//! cannot build is a value flow it cannot ask about.
//!
//! The division cases are checked by *folding* (both operands constant), which is where the
//! crate's arithmetic semantics are implemented. That is deliberate: a symbolic 128-bit
//! divider is O(w^2) - about four times the 64-bit one, and far over the default clause cap -
//! so a query that needs one must raise the cap deliberately, exactly as a 64-bit division
//! does. Folding has no such cost and pins the same semantics.

use smtlite::{Bv, ModelView, Solution, Solver};
use std::time::Instant;

fn sat(s: &Solver) -> smtlite::Model {
    match s.check() {
        Solution::Sat(m) => m,
        other => panic!("expected SAT, got {other:?}"),
    }
}

#[test]
fn a_128_bit_constant_round_trips_through_the_model() {
    let mut s = Solver::new();
    let x = s.var("x", 128);
    let value = (1u128 << 127) | 0xdead_beef_f00d_1234_5678_9abc_def0_4321u128;
    s.assert(x.eq(&Bv::val128(value, 128)));
    let m = sat(&s);
    assert_eq!(m.get128("x"), Some(value));
    // A value that does not fit a u64 is not silently truncated by `get`.
    assert_eq!(m.get("x"), None);
}

#[test]
fn concatenation_builds_a_wide_value_from_two_words() {
    let mut s = Solver::new();
    let hi = s.var("hi", 64);
    let lo = s.var("lo", 64);
    s.assert(hi.eq(&Bv::val(0x0123_4567_89ab_cdef, 64)));
    s.assert(lo.eq(&Bv::val(0xfedc_ba98_7654_3210, 64)));
    let wide = hi.concat(&lo);
    assert_eq!(wide.width(), 128);
    let want = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
    let m = sat(&s);
    let view = ModelView::new(&m);
    assert_eq!(view.eval128(&wide), Some(want));
    assert_eq!(m.get128("hi"), Some(0x0123_4567_89ab_cdef));
}

#[test]
fn wide_add_sub_mul_wrap_at_128_bits() {
    let max = u128::MAX;
    assert_eq!(
        Bv::val128(max, 128).add(&Bv::val128(1, 128)).as_const128(),
        Some(0),
        "2^128 wraps to zero"
    );
    assert_eq!(
        Bv::val128(0, 128).sub(&Bv::val128(1, 128)).as_const128(),
        Some(max),
        "0 - 1 wraps to all ones"
    );
    assert_eq!(
        Bv::val128(1u128 << 64, 128)
            .mul(&Bv::val128(1u128 << 64, 128))
            .as_const128(),
        Some(0),
        "2^64 * 2^64 is 2^128, which is zero at this width"
    );
    let (a, b) = ((1u128 << 64) + 5, (1u128 << 64) + 7);
    assert_eq!(
        Bv::val128(a, 128).mul(&Bv::val128(b, 128)).as_const128(),
        Some(a.wrapping_mul(b)),
        "the full 128-bit product is kept"
    );
}

#[test]
fn wide_division_follows_the_smt_lib_semantics() {
    // Non-trivial operands, including a quotient that needs more than 64 bits.
    let a = u128::MAX;
    let b = 3u128;
    assert_eq!(
        Bv::val128(a, 128).udiv(&Bv::val128(b, 128)).as_const128(),
        Some(a / b)
    );
    assert_eq!(
        Bv::val128(a, 128).urem(&Bv::val128(b, 128)).as_const128(),
        Some(a % b)
    );

    // Division by zero: all ones for unsigned, `1` for a negative signed dividend.
    assert_eq!(
        Bv::val128(9, 128).udiv(&Bv::val128(0, 128)).as_const128(),
        Some(u128::MAX)
    );
    assert_eq!(
        Bv::val128(9, 128).sdiv(&Bv::val128(0, 128)).as_const128(),
        Some(u128::MAX),
        "a non-negative signed dividend divided by zero is all ones"
    );
    assert_eq!(
        Bv::val128((-9i128) as u128, 128)
            .sdiv(&Bv::val128(0, 128))
            .as_const128(),
        Some(1),
        "a negative signed dividend divided by zero is 1"
    );
    assert_eq!(
        Bv::val128(9, 128).urem(&Bv::val128(0, 128)).as_const128(),
        Some(9)
    );

    // Signed division at the extreme: -2^127 / -1 wraps back to -2^127, and the
    // corresponding remainder is zero. These are the two cases a naive `i128`
    // division would panic on.
    let min = 1u128 << 127;
    let minus_one = u128::MAX;
    assert_eq!(
        Bv::val128(min, 128)
            .sdiv(&Bv::val128(minus_one, 128))
            .as_const128(),
        Some(min)
    );
    assert_eq!(
        Bv::val128(min, 128)
            .srem(&Bv::val128(minus_one, 128))
            .as_const128(),
        Some(0)
    );

    // Ordinary signed division takes the sign of the dividend.
    let neg_ten = (-10i128) as u128;
    let three = 3u128;
    assert_eq!(
        Bv::val128(neg_ten, 128)
            .sdiv(&Bv::val128(three, 128))
            .as_const128(),
        Some((-3i128) as u128)
    );
    assert_eq!(
        Bv::val128(neg_ten, 128)
            .srem(&Bv::val128(three, 128))
            .as_const128(),
        Some((-1i128) as u128)
    );
}

#[test]
fn wide_shifts_extends_and_rotates_fold_exactly() {
    let x = (1u128 << 127) | 1;
    assert_eq!(Bv::val128(x, 128).shl(1).as_const128(), Some(x << 1));
    assert_eq!(Bv::val128(x, 128).shl(128).as_const128(), Some(0));
    assert_eq!(Bv::val128(x, 128).lshr(127).as_const128(), Some(1));
    assert_eq!(
        Bv::val128(x, 128).ashr(127).as_const128(),
        Some(u128::MAX),
        "a negative 128-bit value arithmetic-shifts to all ones"
    );
    assert_eq!(
        Bv::val128(1, 128).rotl(127).as_const128(),
        Some(1u128 << 127)
    );
    assert_eq!(
        Bv::val128(1u128 << 127, 128).rotr(127).as_const128(),
        Some(1)
    );

    // Sign extension from a 64-bit word into 128 bits, and truncation back.
    let word = 0xffff_ffff_ffff_ffffu64;
    assert_eq!(Bv::val(word, 64).sext(128).as_const128(), Some(u128::MAX));
    assert_eq!(
        Bv::val(0x8000_0000_0000_0000, 64).sext(128).as_const128(),
        Some(0xffff_ffff_ffff_ffff_8000_0000_0000_0000)
    );
    assert_eq!(
        Bv::val128(u128::MAX, 128).extract(63, 0).as_const128(),
        Some(u64::MAX as u128)
    );
}

#[test]
fn a_wide_variable_can_be_constrained_and_read_back() {
    // A genuinely symbolic 128-bit variable under a two-sided bound: the model
    // must land inside it. (No division, so the default clause cap holds.)
    let mut s = Solver::new();
    let x = s.var("x", 128);
    let lo = Bv::val128(1u128 << 64, 128);
    let hi = Bv::val128((1u128 << 64) + 1000, 128);
    s.assert(x.uge(&lo));
    s.assert(x.ule(&hi));
    let m = sat(&s);
    let v = m.get128("x").expect("128-bit readback");
    assert!(
        ((1u128 << 64)..=(1u128 << 64) + 1000).contains(&v),
        "x = {v:#x}"
    );
}

#[test]
fn wide_comparisons_are_unsigned_or_signed_as_asked() {
    let min = 1u128 << 127;
    // 0 <u 2^127, but 0 >s 2^127 (the latter is the most negative signed value).
    assert_eq!(
        Bv::val128(0, 128).ult(&Bv::val128(min, 128)).as_const128(),
        Some(1)
    );
    assert_eq!(
        Bv::val128(0, 128).slt(&Bv::val128(min, 128)).as_const128(),
        Some(0)
    );
    assert_eq!(
        Bv::val128(min, 128).slt(&Bv::val128(0, 128)).as_const128(),
        Some(1)
    );
}

#[test]
fn a_128_bit_query_with_a_deadline_is_bounded() {
    // The bounds still apply at the wide end: a deadline that has already
    // passed answers Unknown instead of blasting.
    let mut s = Solver::new();
    let x = s.var("x", 128);
    let past = Instant::now() - std::time::Duration::from_millis(1);
    assert!(matches!(
        s.check_all_within(&[x.eq(&Bv::val128(7, 128))], 1000, past),
        Solution::Unknown
    ));
}
