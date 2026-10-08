//! Contract tests: every documented `# Panics` really panics, at and just past the boundary.
//!
//! The crate's promise is that a width or arity mistake stops the caller rather than quietly
//! blasting the wrong bits - in release builds too. Each test below pins one such boundary to the
//! message the method documents, so a message that drifts (or a check that is dropped under
//! optimisation) is a test failure, not a silent weakening.

use smtlite::{Bv, Solver};

#[test]
#[should_panic(expected = "variable width must be 1..=64")]
fn var_width_zero_panics() {
    let mut s = Solver::new();
    let _ = s.var("x", 0);
}

#[test]
#[should_panic(expected = "variable width must be 1..=64")]
fn var_width_65_panics() {
    let mut s = Solver::new();
    let _ = s.var("x", 65);
}

#[test]
#[should_panic(expected = "constant width must be 1..=64")]
fn const_width_zero_panics() {
    let _ = Bv::val(0, 0);
}

#[test]
#[should_panic(expected = "constant width must be 1..=64")]
fn const_width_65_panics() {
    let _ = Bv::val(0, 65);
}

#[test]
#[should_panic(expected = "bitvector op needs equal widths")]
fn binary_op_width_mismatch_panics() {
    let _ = Bv::val(1, 8).add(&Bv::val(1, 16));
}

#[test]
#[should_panic(expected = "comparison needs equal widths")]
fn comparison_width_mismatch_panics() {
    let _ = Bv::val(1, 8).eq(&Bv::val(1, 16));
}

#[test]
#[should_panic(expected = "a constraint must be 1 bit")]
fn assert_of_multi_bit_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    s.assert(x);
}

#[test]
#[should_panic(expected = "a constraint must be 1 bit")]
fn check_all_of_multi_bit_panics() {
    // `check_all` bypasses `Solver::assert`'s own check and reaches the blaster's - both must hold
    // the line.
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = s.check_all(&[x]);
}

#[test]
#[should_panic(expected = "ite condition must be 1 bit")]
fn ite_non_proposition_condition_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = Bv::ite(&x, &x, &x);
}

#[test]
#[should_panic(expected = "ite branches must match")]
fn ite_width_mismatch_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let y = s.var("y", 16);
    let c = x.eq(&x);
    let _ = Bv::ite(&c, &x, &y);
}

#[test]
#[should_panic(expected = "zext must widen")]
fn zext_narrowing_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 16);
    let _ = x.zext(8);
}

#[test]
#[should_panic(expected = "width must be 1..=64")]
fn zext_past_64_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = x.zext(65);
}

#[test]
#[should_panic(expected = "sext must widen")]
fn sext_narrowing_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 16);
    let _ = x.sext(8);
}

#[test]
#[should_panic(expected = "width must be 1..=64")]
fn sext_past_64_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = x.sext(65);
}

#[test]
#[should_panic(expected = "out of range for width")]
fn extract_hi_below_lo_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = x.extract(2, 5);
}

#[test]
#[should_panic(expected = "out of range for width")]
fn extract_hi_at_width_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = x.extract(8, 0);
}

#[test]
#[should_panic(expected = "over the 64-bit limit")]
fn concat_over_64_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 40);
    let _ = x.concat(&x);
}

#[test]
#[should_panic(expected = "land operand must be 1 bit")]
fn land_on_multi_bit_panics() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let _ = x.land(&x);
}

/// The boundary *just inside* the limit must not panic: width 1 and width 64 are both valid.
#[test]
fn boundary_widths_are_valid() {
    let mut s = Solver::new();
    let one = s.var("one", 1);
    let full = s.var("full", 64);
    assert_eq!(one.width(), 1);
    assert_eq!(full.width(), 64);

    // A 64-bit value can be read back, so it can hold and constrain all ones.
    s.assert(full.eq(&Bv::val(u64::MAX, 64)));
    assert!(matches!(s.check(), smtlite::Solution::Sat(m) if m.get("full") == Some(u64::MAX)));
}
