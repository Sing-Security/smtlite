//! Conformance tests for the arithmetic, shift and rotate operations.
//!
//! Rust's native operators are the oracle. Each expression is built from constants standing in for
//! the operands, and the solver is asked to *prove* the result equals the value Rust computes: the
//! `==` direction must be SAT and the `!=` direction must be UNSAT. A circuit that is merely
//! "probably right" passes the first and fails the second, so the UNSAT half is the real check.
//!
//! Integers here are width-`w` two's-complement bitvectors. `u64` values must be pre-masked to `w`
//! bits; a test that feeds an unmasked value is testing the test.

// The oracle writes the zero-divisor case out as `if b == 0 { … } else { a / b }` rather than
// calling `checked_div`, which would hide the semantics under test: `bvudiv` by zero is all-ones
// and `bvurem` by zero is the dividend.
#![allow(clippy::manual_checked_ops)]

use smtlite::{Bv, Solution, Solver};

/// Formula-size cap for the 64-bit division cases — a 64-bit divider blasts to ~110,000 clauses,
/// well over the crate's deliberate 40,000 default.
const BIG: usize = 200_000;

/// Prove `expr` is exactly the constant `expected`, at the crate's default formula cap.
fn prove(expr: &Bv, expected: u64) {
    prove_capped(expr, expected, 40_000);
}

fn prove_capped(expr: &Bv, expected: u64, cap: usize) {
    let w = expr.width();
    let want = Bv::val(expected, w);

    let mut sat = Solver::new().with_max_clauses(cap);
    sat.assert(expr.eq(&want));
    assert!(
        matches!(sat.check(), Solution::Sat(_)),
        "`{expr}` should be able to equal #{expected:x} ({w} bits)"
    );

    // The UNSAT half: no assignment in the circuit differs from the oracle value.
    let mut unsat = Solver::new().with_max_clauses(cap);
    unsat.assert(expr.ne(&want));
    assert!(
        matches!(unsat.check(), Solution::Unsat),
        "`{expr}` differs from #{expected:x} ({w} bits)"
    );
}

fn is_sat(s: &Solver) -> bool {
    matches!(s.check(), Solution::Sat(_))
}

/// A 4-bit vector read as two's-complement.
fn s4(v: u64) -> i64 {
    if v & 0x8 != 0 {
        v as i64 - 16
    } else {
        v as i64
    }
}

fn maskw(v: u64, w: u32) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}

/// Rotate a `w`-bit value left by `k`, independent of the crate (w < 64).
fn rotl_bits(v: u64, k: u32, w: u32) -> u64 {
    let k = k % w;
    let hi = maskw(v << k, w);
    let lo = if k == 0 { 0 } else { maskw(v, w) >> (w - k) };
    maskw(hi | lo, w)
}

fn rotr_bits(v: u64, k: u32, w: u32) -> u64 {
    let k = k % w;
    let lo = maskw(v, w) >> k;
    let hi = if k == 0 { 0 } else { maskw(v << (w - k), w) };
    maskw(lo | hi, w)
}

// ---- Unsigned division and remainder -------------------------------------------------

#[test]
fn udiv_urem_exhaustive_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            // SMT-LIB: a zero divisor yields all-ones for `/` and the dividend for `%`.
            let (q, r) = if b == 0 { (0xf, a) } else { (a / b, a % b) };
            prove(&x.udiv(&y), q);
            prove(&x.urem(&y), r);
        }
    }
}

#[test]
fn udiv_urem_sampled_width8() {
    const VALS: [u64; 12] = [0, 1, 2, 3, 7, 8, 15, 16, 127, 128, 200, 255];
    for &a in &VALS {
        for &b in &VALS {
            let (x, y) = (Bv::val(a, 8), Bv::val(b, 8));
            let (q, r) = if b == 0 { (0xff, a) } else { (a / b, a % b) };
            prove(&x.udiv(&y), q);
            prove(&x.urem(&y), r);
        }
    }
}

#[test]
fn udiv_urem_edges_width32() {
    const PAIRS: [(u64, u64); 9] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0xffff_ffff, 1),
        (0xffff_ffff, 2),
        (0xffff_ffff, 0xffff_ffff),
        (1, 0xffff_ffff),
        (0xdead_beef, 0x1234),
    ];
    for (a, b) in PAIRS {
        let (x, y) = (Bv::val(a, 32), Bv::val(b, 32));
        let (q, r) = if b == 0 {
            (0xffff_ffff, a)
        } else {
            (a / b, a % b)
        };
        prove(&x.udiv(&y), q);
        prove(&x.urem(&y), r);
    }
}

#[test]
fn udiv_urem_edges_width64() {
    // A 64-bit divider is the one case that genuinely needs a raised cap.
    const PAIRS: [(u64, u64); 6] = [
        (0, 0),
        (1, 0),
        (u64::MAX, 1),
        (u64::MAX, u64::MAX),
        (0x8000_0000_0000_0000, 3),
        (0x0123_4567_89ab_cdef, 0x1_0000_0001),
    ];
    for (a, b) in PAIRS {
        let (x, y) = (Bv::val(a, 64), Bv::val(b, 64));
        let (q, r) = if b == 0 {
            (u64::MAX, a)
        } else {
            (a / b, a % b)
        };
        prove_capped(&x.udiv(&y), q, BIG);
        prove_capped(&x.urem(&y), r, BIG);
    }
}

// ---- Signed division and remainder ---------------------------------------------------

#[test]
fn sdiv_srem_exhaustive_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (sa, sb) = (s4(a), s4(b));
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            // A zero divisor: all-ones when the dividend is non-negative, 1 otherwise.
            let (q, r) = if sb == 0 {
                (if sa >= 0 { 0xf } else { 1 }, a)
            } else {
                (maskw((sa / sb) as u64, 4), maskw((sa % sb) as u64, 4))
            };
            prove(&x.sdiv(&y), q);
            prove(&x.srem(&y), r);
        }
    }
}

#[test]
fn sdiv_srem_edges_width8() {
    // (a, b) as 8-bit two's-complement values: -128, -127, -1, 0, 1, 127.
    const PAIRS: [(u64, u64); 12] = [
        (0x80, 0xff), // MIN / -1  -> wraps back to MIN; remainder 0
        (0x80, 0x01),
        (0xff, 0x80),
        (0xff, 0x02),
        (0x7f, 0xff),
        (0x7f, 0x7f),
        (0x05, 0xfe), // 5 / -2 -> -2, remainder 1 (sign of the DIVIDEND — C's %)
        (0xfb, 0x02), // -5 / 2 -> -2, remainder -1
        (0x80, 0x00), // zero divisor, negative dividend -> 1 (and remainder = dividend)
        (0x7f, 0x00), // zero divisor, positive dividend -> -1
        (0x00, 0x00),
        (0xc9, 0x07), // -55 / 7
    ];
    for (a, b) in PAIRS {
        let (sa, sb) = (a as i8 as i64, b as i8 as i64);
        let (x, y) = (Bv::val(a, 8), Bv::val(b, 8));
        let (q, r) = if sb == 0 {
            (if sa >= 0 { 0xff } else { 1 }, a)
        } else {
            (maskw((sa / sb) as u64, 8), maskw((sa % sb) as u64, 8))
        };
        prove(&x.sdiv(&y), q);
        prove(&x.srem(&y), r);
    }
}

#[test]
fn srem_takes_the_sign_of_the_dividend_not_the_divisor() {
    // The distinction from `bvsmod`: -7 % 3 is -1 (C), where bvsmod would give 2.
    prove(&Bv::val(0xf9, 8).srem(&Bv::val(3, 8)), 0xff); // 0xf9 = -7
    prove(&Bv::val(7, 8).srem(&Bv::val(0xfd, 8)), 1); // 0xfd = -3
}

#[test]
fn sdiv_min_by_negative_one_wraps() {
    // The one signed division that overflows: MIN / -1 is unrepresentable and wraps to MIN.
    prove_capped(
        &Bv::val(0x8000_0000, 32).sdiv(&Bv::val(0xffff_ffff, 32)),
        0x8000_0000,
        BIG,
    );
    prove(&Bv::val(0x8000_0000, 32).srem(&Bv::val(0xffff_ffff, 32)), 0);
}

#[test]
fn signed_division_edges_width32() {
    const PAIRS: [(u64, u64); 6] = [
        (0xffff_fffb, 3),           // -5 / 3
        (5, 0xffff_fffd),           // 5 / -3
        (0x8000_0000, 0xffff_fffe), // MIN / -2
        (0xffff_fffb, 0xffff_fffd), // -5 / -3
        (0x7fff_ffff, 0xffff_ffff), // MAX / -1
        (0xffff_fffe, 0),           // zero divisor, negative dividend
    ];
    for (a, b) in PAIRS {
        let (sa, sb) = (a as u32 as i32 as i64, b as u32 as i32 as i64);
        let (x, y) = (Bv::val(a, 32), Bv::val(b, 32));
        let (q, r) = if sb == 0 {
            (if sa >= 0 { 0xffff_ffff } else { 1 }, a)
        } else {
            (maskw((sa / sb) as u64, 32), maskw((sa % sb) as u64, 32))
        };
        prove(&x.sdiv(&y), q);
        prove(&x.srem(&y), r);
    }
}

// ---- Division with a symbolic operand ------------------------------------------------

#[test]
fn udiv_urem_with_variables() {
    // The constant tests above cover the circuit; this covers the variable plumbing around it.
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let y = s.var("y", 8);
    s.assert(x.eq(&Bv::val(100, 8)));
    s.assert(y.eq(&Bv::val(7, 8)));
    s.assert(x.udiv(&y).eq(&Bv::val(14, 8)));
    s.assert(x.urem(&y).eq(&Bv::val(2, 8)));
    assert!(is_sat(&s), "100 / 7 == 14 remainder 2");
}

#[test]
fn division_is_unsat_when_the_quotient_is_wrong() {
    let mut s = Solver::new();
    let x = s.var("x", 16);
    s.assert(x.eq(&Bv::val(1000, 16)));
    s.assert(x.udiv(&Bv::val(7, 16)).eq(&Bv::val(143, 16))); // 1000 / 7 = 142
    assert!(matches!(s.check(), Solution::Unsat));
}

#[test]
fn a_division_recovers_a_symbolic_divisor() {
    // Run the operation backwards: given the quotient and remainder, what was the divisor?
    let mut s = Solver::new();
    let y = s.var("y", 8);
    s.assert(Bv::val(100, 8).udiv(&y).eq(&Bv::val(9, 8)));
    s.assert(Bv::val(100, 8).urem(&y).eq(&Bv::val(1, 8)));
    // 100 = 9*y + 1 with r < y leaves y = 11.
    match s.check() {
        Solution::Sat(m) => assert_eq!(m.get("y"), Some(11)),
        other => panic!("expected SAT, got {other:?}"),
    }
}

// ---- Symbolic shift amounts ----------------------------------------------------------

#[test]
fn shl_var_exhaustive_amounts_width8() {
    let a = 0b1011_0011u64;
    for k in 0..24u64 {
        // A shift at or beyond the width empties the value — it does not wrap.
        let expected = if k >= 8 { 0 } else { maskw(a << k, 8) };
        prove(&Bv::val(a, 8).shl_var(&Bv::val(k, 8)), expected);
    }
}

#[test]
fn lshr_var_exhaustive_amounts_width8() {
    let a = 0b1011_0011u64;
    for k in 0..24u64 {
        let expected = if k >= 8 { 0 } else { a >> k };
        prove(&Bv::val(a, 8).lshr_var(&Bv::val(k, 8)), expected);
    }
}

#[test]
fn ashr_var_exhaustive_amounts_width8() {
    let a = 0b1011_0011u64; // negative
    let pos = 0b0011_0011u64;
    for k in 0..24u64 {
        let expected = if k >= 8 {
            0xff
        } else {
            ((a as i8) >> k) as u8 as u64
        };
        prove(&Bv::val(a, 8).ashr_var(&Bv::val(k, 8)), expected);
        // A non-negative value must NOT sign-fill.
        let exp_pos = if k >= 8 { 0 } else { pos >> k };
        prove(&Bv::val(pos, 8).ashr_var(&Bv::val(k, 8)), exp_pos);
    }
}

#[test]
fn shift_amount_narrower_than_the_vector() {
    // A 4-bit amount can only reach 15, so the high stages of the barrel are never selected —
    // but every amount it *can* reach must still be right.
    let a = 0xabcu64;
    for k in 0..16u64 {
        prove(&Bv::val(a, 16).shl_var(&Bv::val(k, 4)), maskw(a << k, 16));
        prove(&Bv::val(a, 16).lshr_var(&Bv::val(k, 4)), a >> k);
    }
}

#[test]
fn shift_amount_wider_than_the_vector() {
    // A 64-bit amount against an 8-bit value: every amount >= 8 must empty it.
    let a = 0xd3u64;
    for k in [0u64, 1, 7, 8, 9, 255, 1 << 40, u64::MAX] {
        let expected = if k >= 8 { 0 } else { a << k };
        prove(&Bv::val(a, 8).shl_var(&Bv::val(k, 64)), expected);
    }
}

#[test]
fn symbolic_shift_amount_from_a_variable() {
    let mut s = Solver::new();
    let x = s.var("x", 8);
    let k = s.var("k", 8);
    s.assert(k.eq(&Bv::val(3, 8)));
    s.assert(x.eq(&Bv::val(0x0f, 8)));
    s.assert(x.shl_var(&k).eq(&Bv::val(0x78, 8)));
    assert!(is_sat(&s), "0x0f << 3 == 0x78");
}

#[test]
fn a_shift_recovers_a_symbolic_amount() {
    // The question a symbolic executor actually asks: what shift turns this into that?
    let mut s = Solver::new();
    let k = s.var("k", 8);
    s.assert(Bv::val(0x0f, 8).shl_var(&k).eq(&Bv::val(0x78, 8)));
    match s.check() {
        Solution::Sat(m) => assert_eq!(m.get("k"), Some(3)),
        other => panic!("expected SAT, got {other:?}"),
    }
}

// ---- Rotates -------------------------------------------------------------------------

#[test]
fn rot_const_exhaustive_width8() {
    let a = 0b1001_0110u64;
    for k in 0..=16u32 {
        prove(&Bv::val(a, 8).rotl(k), (a as u8).rotate_left(k) as u64);
        prove(&Bv::val(a, 8).rotr(k), (a as u8).rotate_right(k) as u64);
    }
}

#[test]
fn rot_const_is_a_permutation_not_a_shift() {
    // Rotating away every bit must return the value, where shifting would empty it.
    prove(&Bv::val(0x5a, 8).rotl(8), 0x5a);
    prove(&Bv::val(0x5a, 8).rotr(8), 0x5a);
    prove(&Bv::val(0x5a, 8).rotl(0), 0x5a);
    prove(&Bv::val(0x5a, 8).shl(8), 0);
}

#[test]
fn rot_const_non_power_of_two_widths() {
    for &(a, w) in &[
        (0b1_0101u64, 5u32),
        (0b101u64, 3),
        (0b101_1010u64, 7),
        (0b1_1u64, 6),
    ] {
        for k in 0..=14u32 {
            prove(&Bv::val(a, w).rotl(k), rotl_bits(a, k, w));
            prove(&Bv::val(a, w).rotr(k), rotr_bits(a, k, w));
        }
    }
}

#[test]
fn rot_var_exhaustive_amounts_width8() {
    // Power-of-two width: the low three bits of the amount are already it modulo 8.
    let a = 0b1101_0011u64;
    for k in 0..=40u64 {
        prove(
            &Bv::val(a, 8).rotl_var(&Bv::val(k, 8)),
            (a as u8).rotate_left(k as u32) as u64,
        );
        prove(
            &Bv::val(a, 8).rotr_var(&Bv::val(k, 8)),
            (a as u8).rotate_right(k as u32) as u64,
        );
    }
}

#[test]
fn rot_var_non_power_of_two_widths() {
    for &(a, w) in &[(0b1_0011u64, 5u32), (0b101u64, 3), (0b101_1010u64, 7)] {
        // The amount register is `w` bits, so amounts only run to `2^w - 1` — a larger `k` would
        // be masked on the way in and the oracle would be comparing against a different shift.
        let span = (1u64 << w).min(41);
        for k in 0..span {
            prove(
                &Bv::val(a, w).rotl_var(&Bv::val(k, w)),
                rotl_bits(a, k as u32, w),
            );
            prove(
                &Bv::val(a, w).rotr_var(&Bv::val(k, w)),
                rotr_bits(a, k as u32, w),
            );
        }
    }
}

#[test]
fn rot_var_amount_wider_than_width() {
    // Regression: the amount must reduce modulo the width using every bit of `k`. Truncating `k`
    // to `w` bits first computes `(k mod 2^w) mod w`, which is wrong whenever `w` does not divide
    // `2^w` — for w=5 and k=32 it gives 0 where the answer is 2.
    let a = 0b1_0011u64;
    for k in [0u64, 1, 4, 5, 6, 31, 32, 33, 100, 1023, 0xffff] {
        prove(
            &Bv::val(a, 5).rotl_var(&Bv::val(k, 16)),
            rotl_bits(a, (k % 5) as u32, 5),
        );
        prove(
            &Bv::val(a, 5).rotr_var(&Bv::val(k, 16)),
            rotr_bits(a, (k % 5) as u32, 5),
        );
    }
}

#[test]
fn rot_var_amount_narrower_than_the_vector() {
    // A 3-bit amount cannot reach the width, so the modulo reduction is never exercised.
    let a = 0b1011_0011u64;
    for k in 0..8u64 {
        prove(
            &Bv::val(a, 8).rotl_var(&Bv::val(k, 3)),
            (a as u8).rotate_left(k as u32) as u64,
        );
    }
}

#[test]
fn symbolic_rotate_amount_from_a_variable() {
    // 0b1000_0001 rotated left by 3 is 0b0000_1100. The amount is 3 bits, which is what makes the
    // answer unique: a rotate is periodic in the width, so an 8-bit amount would also admit 251.
    let mut s = Solver::new();
    let k = s.var("k", 3);
    s.assert(
        Bv::val(0b1000_0001, 8)
            .rotl_var(&k)
            .eq(&Bv::val(0b0000_1100, 8)),
    );
    match s.check() {
        Solution::Sat(m) => assert_eq!(m.get("k"), Some(3)),
        other => panic!("expected SAT, got {other:?}"),
    }
}

#[test]
fn symbolic_rotate_amount_is_only_unique_modulo_the_width() {
    // The same rotation, with an amount register as wide as the vector: every solution is
    // congruent to 3 modulo 8, so `k mod 8 == 3` must hold and the raw value must not be pinned.
    let mut s = Solver::new();
    let k = s.var("k", 8);
    s.assert(
        Bv::val(0b1000_0001, 8)
            .rotl_var(&k)
            .eq(&Bv::val(0b0000_1100, 8)),
    );
    match s.check() {
        Solution::Sat(m) => assert_eq!(m.get("k").unwrap() % 8, 3),
        other => panic!("expected SAT, got {other:?}"),
    }
}

// ---- Signed comparisons --------------------------------------------------------------

#[test]
fn sgt_sge_exhaustive_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (sa, sb) = (s4(a), s4(b));
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            prove(&x.sgt(&y), u64::from(sa > sb));
            prove(&x.sge(&y), u64::from(sa >= sb));
        }
    }
}

#[test]
fn signed_and_unsigned_comparisons_disagree_on_the_negative_half() {
    // -1 >s -128, but 0xff >u 0x80 also holds — so compare where they *must* differ: -1 vs 0.
    let (neg1, zero) = (Bv::val(0xff, 8), Bv::val(0, 8));
    prove(&neg1.sgt(&zero), 0);
    prove(&neg1.sge(&zero), 0);
    prove(&neg1.slt(&zero), 1);
    prove(&neg1.ugt(&zero), 1); // 255 > 0 unsigned

    let (min, max) = (Bv::val(0x80, 8), Bv::val(0x7f, 8));
    prove(&min.sgt(&max), 0); // -128 > 127 is false…
    prove(&min.ugt(&max), 1); // …but 128 > 127 unsigned is true
    prove(&max.sgt(&min), 1);
}

#[test]
fn sgt_sge_are_the_mirror_of_slt_sle() {
    for a in 0..8u64 {
        for b in 0..8u64 {
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            // a > b  <=>  b < a, and the two must never both hold or both fail.
            let mut s = Solver::new();
            s.assert(x.sgt(&y));
            s.assert(y.slt(&x).not());
            assert!(
                matches!(s.check(), Solution::Unsat),
                "sgt and slt disagree for {a},{b}"
            );
        }
    }
}
