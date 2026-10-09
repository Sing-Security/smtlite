//! Conformance tests for the arithmetic, shift and rotate operations.
//!
//! Rust's native operators are the oracle, and the suite keeps two layers deliberately apart:
//!
//! - **Circuit conformance** (the `prove*` helpers): each operand is pinned into a *fresh
//!   variable* by an equality constraint, so the expression tree keeps its `Var` nodes and
//!   cannot fold at construction. The solver must then bit-blast the operation's real circuit
//!   and prove the result equals the value Rust computes: the `==` direction is SAT and the
//!   `!=` direction is UNSAT. A circuit that is merely "probably right" passes the first and
//!   fails the second, so the UNSAT half is the real check. Pinning the inputs makes each query
//!   propagation-led once the circuit exists - but the circuit still has to be blasted first,
//!   and blasting it is the work under test. Every proof asserts the expression it sends to the
//!   solver is still symbolic, so a regression back to folded constants fails loudly instead of
//!   passing vacuously. "Coverage guards" at the bottom check the size of the formula the wide
//!   cases really produce.
//!
//! - **Folding conformance** (the tests under "Constant folding"): operands that are both
//!   constants fold on construction through the crate's own `eval_bin`/`fold_cmp` and never
//!   reach the blaster. Those tests check that folding layer against Rust directly via
//!   [`Bv::as_const`], with no solver involved. They are worth having - a folding bug would
//!   poison every caller that relies on eager simplification - but they say nothing about the
//!   circuits, which is why they are labeled and kept separate.
//!
//! Integers here are width-`w` two's-complement bitvectors. `u64` values must be pre-masked to
//! `w` bits; a test that feeds an unmasked value is testing the test.

// The oracle writes the zero-divisor case out as `if b == 0 { ... } else { a / b }` rather than
// calling `checked_div`, which would hide the semantics under test: `bvudiv` by zero is all-ones
// and `bvurem` by zero is the dividend.
#![allow(clippy::manual_checked_ops)]

use smtlite::{Bv, Solution, Solver};

/// Formula-size cap for the wide division cases - a 64-bit divider blasts to ~110,000 clauses,
/// well over the crate's deliberate 40,000 default. The edge tests below really do produce
/// formulas this large now that their operands are pinned variables rather than constants.
const BIG: usize = 200_000;

// ---------------------------------------------------------------------------------------------
// Pinning and the proof helpers.
// ---------------------------------------------------------------------------------------------

/// Pin `v` as the value of a fresh variable at width `w`, returning the variable.
///
/// The expression tree keeps a `Var` node, so nothing built on the return value folds at
/// construction: the solver sees the operation's circuit, not a constant. The equality holds
/// only for this solver, and each call needs a name unique to it.
fn pin(s: &mut Solver, name: &str, v: u64, w: u32) -> Bv {
    let var = s.var(name, w);
    s.assert(var.eq(&Bv::val(v, w)));
    var
}

/// Refuse to prove a folded constant: a constant "proof" exercises `eval_bin`, not the circuit,
/// and the caller asked for the circuit. This is the guard that keeps every test below honest.
fn still_symbolic(expr: &Bv, op: &str) {
    assert!(
        expr.as_const().is_none(),
        "{op}: the expression folded to a constant, so this proof would not exercise the circuit"
    );
}

/// Prove an expression equals `want` exactly, and that no assignment makes it differ.
///
/// `build` constructs the expression on the solver it is handed - pinning whatever operands
/// the caller wants pinned - and runs once per direction on a fresh solver, so the asserted
/// constraint of one direction cannot leak into the other. This is the core the `prove*`
/// wrappers share; the wrappers decide what is pinned and how wide `want` is.
fn prove_with(cap: usize, want: Bv, op: &str, build: impl Fn(&mut Solver) -> Bv) {
    let mut sat = Solver::new().with_max_clauses(cap);
    let expr = build(&mut sat);
    still_symbolic(&expr, op);
    sat.assert(expr.eq(&want));
    assert!(
        matches!(sat.check(), Solution::Sat(_)),
        "{op}: should be able to equal {want} (an Unknown here means the clause cap tripped)"
    );

    // The UNSAT half: no assignment in the circuit differs from the oracle value.
    let mut unsat = Solver::new().with_max_clauses(cap);
    let expr = build(&mut unsat);
    still_symbolic(&expr, op);
    unsat.assert(expr.ne(&want));
    assert!(
        matches!(unsat.check(), Solution::Unsat),
        "{op}: the circuit differs from {want}"
    );
}

/// Prove a same-width binary operation equals `expected`, with both operands pinned at `w`.
fn prove2(w: u32, a: u64, b: u64, expected: u64, op: &str, f: impl Fn(&Bv, &Bv) -> Bv) {
    prove2_capped(w, a, b, expected, 40_000, op, f);
}

/// [`prove2`] against an explicitly raised formula cap, for circuits wider than the default.
fn prove2_capped(
    w: u32,
    a: u64,
    b: u64,
    expected: u64,
    cap: usize,
    op: &str,
    f: impl Fn(&Bv, &Bv) -> Bv,
) {
    prove_with(cap, Bv::val(expected, w), op, |s| {
        let (x, y) = (pin(s, "x", a, w), pin(s, "y", b, w));
        f(&x, &y)
    });
}

/// Prove a comparison: a 1-bit result over two `w`-bit pinned operands.
fn prove_cmp(w: u32, a: u64, b: u64, bit: u64, op: &str, f: impl Fn(&Bv, &Bv) -> Bv) {
    prove_with(40_000, Bv::val(bit, 1), op, |s| {
        let (x, y) = (pin(s, "x", a, w), pin(s, "y", b, w));
        f(&x, &y)
    });
}

/// Prove an operation on a value and a separately-widthed symbolic amount (a variable shift or
/// rotate). Both operands are pinned; the amount at `kw` bits, the value at `w`.
fn prove_shift(
    w: u32,
    kw: u32,
    a: u64,
    k: u64,
    expected: u64,
    op: &str,
    f: impl Fn(&Bv, &Bv) -> Bv,
) {
    prove_with(40_000, Bv::val(expected, w), op, |s| {
        let (x, amount) = (pin(s, "x", a, w), pin(s, "k", k, kw));
        f(&x, &amount)
    });
}

/// Prove a unary operation (a constant-amount rotate or shift) on a pinned value.
fn prove1(w: u32, a: u64, expected: u64, op: &str, f: impl Fn(&Bv) -> Bv) {
    prove_with(40_000, Bv::val(expected, w), op, |s| f(&pin(s, "x", a, w)));
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

// ---- Constant folding (eval_bin), NOT circuit conformance -----------------------------------
//
// Constant operands fold on construction and never reach the blaster. These tests pin down the
// folding semantics against Rust exhaustively at width 4; the circuit proofs live everywhere
// else in this file. No solver is involved.

#[test]
fn folding_matches_rust_for_every_arithmetic_op_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            assert_eq!(x.add(&y).as_const(), Some(maskw(a + b, 4)));
            assert_eq!(x.sub(&y).as_const(), Some(maskw(a.wrapping_sub(b), 4)));
            assert_eq!(x.mul(&y).as_const(), Some(maskw(a * b, 4)));
            assert_eq!(x.and(&y).as_const(), Some(a & b));
            assert_eq!(x.or(&y).as_const(), Some(a | b));
            assert_eq!(x.xor(&y).as_const(), Some(a ^ b));
            assert_eq!(x.neg().as_const(), Some(maskw(a.wrapping_neg(), 4)));
            assert_eq!(x.not().as_const(), Some(maskw(!a, 4)));
            // SMT-LIB: a zero divisor yields all-ones for `/` and the dividend for `%`.
            assert_eq!(
                x.udiv(&y).as_const(),
                Some(if b == 0 { 0xf } else { a / b })
            );
            assert_eq!(x.urem(&y).as_const(), Some(if b == 0 { a } else { a % b }));
            let (sa, sb) = (s4(a), s4(b));
            let (q, r) = if sb == 0 {
                // Zero signed divisor: all-ones for a non-negative dividend, 1 otherwise.
                (if sa >= 0 { 0xf } else { 1 }, a)
            } else {
                (maskw((sa / sb) as u64, 4), maskw((sa % sb) as u64, 4))
            };
            assert_eq!(x.sdiv(&y).as_const(), Some(q));
            assert_eq!(x.srem(&y).as_const(), Some(r));
        }
    }
}

#[test]
fn folding_matches_rust_for_every_comparison_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (x, y) = (Bv::val(a, 4), Bv::val(b, 4));
            let (sa, sb) = (s4(a), s4(b));
            assert_eq!(x.eq(&y).as_const(), Some(u64::from(a == b)));
            assert_eq!(x.ne(&y).as_const(), Some(u64::from(a != b)));
            assert_eq!(x.ult(&y).as_const(), Some(u64::from(a < b)));
            assert_eq!(x.ule(&y).as_const(), Some(u64::from(a <= b)));
            assert_eq!(x.ugt(&y).as_const(), Some(u64::from(a > b)));
            assert_eq!(x.uge(&y).as_const(), Some(u64::from(a >= b)));
            assert_eq!(x.slt(&y).as_const(), Some(u64::from(sa < sb)));
            assert_eq!(x.sle(&y).as_const(), Some(u64::from(sa <= sb)));
            assert_eq!(x.sgt(&y).as_const(), Some(u64::from(sa > sb)));
            assert_eq!(x.sge(&y).as_const(), Some(u64::from(sa >= sb)));
        }
    }
}

#[test]
fn folding_matches_rust_for_shifts_and_rotates() {
    for &w in &[4u32, 7] {
        let a = maskw(0b1011_0011, w);
        let sv = if a >> (w - 1) & 1 == 1 {
            a as i64 - (1i64 << w)
        } else {
            a as i64
        };
        for k in 0..=20u64 {
            let (x, amount) = (Bv::val(a, w), Bv::val(k, 8));
            // A shift at or beyond the width empties (or sign-fills); a rotate wraps.
            let shl = if k >= u64::from(w) {
                0
            } else {
                maskw(a << k, w)
            };
            let lshr = if k >= u64::from(w) { 0 } else { a >> k };
            let ashr = if k >= u64::from(w) {
                if sv < 0 { maskw(u64::MAX, w) } else { 0 }
            } else {
                maskw((sv >> k) as u64, w)
            };
            assert_eq!(x.shl(k as u32).as_const(), Some(shl));
            assert_eq!(x.lshr(k as u32).as_const(), Some(lshr));
            assert_eq!(x.ashr(k as u32).as_const(), Some(ashr));
            assert_eq!(x.rotl(k as u32).as_const(), Some(rotl_bits(a, k as u32, w)));
            assert_eq!(x.rotr(k as u32).as_const(), Some(rotr_bits(a, k as u32, w)));
            // A constant *operand* folds the variable-amount forms too - which is exactly the
            // trap the circuit suite once fell into: `shl_var` on two constants is a constant.
            assert_eq!(x.shl_var(&amount).as_const(), Some(shl));
            assert_eq!(x.lshr_var(&amount).as_const(), Some(lshr));
            assert_eq!(x.ashr_var(&amount).as_const(), Some(ashr));
            assert_eq!(
                x.rotl_var(&amount).as_const(),
                Some(rotl_bits(a, k as u32, w))
            );
            assert_eq!(
                x.rotr_var(&amount).as_const(),
                Some(rotr_bits(a, k as u32, w))
            );
        }
    }
}

// ---- Unsigned division and remainder -------------------------------------------------

#[test]
fn udiv_urem_exhaustive_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            // SMT-LIB: a zero divisor yields all-ones for `/` and the dividend for `%`.
            let (q, r) = if b == 0 { (0xf, a) } else { (a / b, a % b) };
            prove2(4, a, b, q, "udiv", |x, y| x.udiv(y));
            prove2(4, a, b, r, "urem", |x, y| x.urem(y));
        }
    }
}

#[test]
fn udiv_urem_sampled_width8() {
    const VALS: [u64; 12] = [0, 1, 2, 3, 7, 8, 15, 16, 127, 128, 200, 255];
    for &a in &VALS {
        for &b in &VALS {
            let (q, r) = if b == 0 { (0xff, a) } else { (a / b, a % b) };
            prove2(8, a, b, q, "udiv", |x, y| x.udiv(y));
            prove2(8, a, b, r, "urem", |x, y| x.urem(y));
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
        let (q, r) = if b == 0 {
            (0xffff_ffff, a)
        } else {
            (a / b, a % b)
        };
        prove2(32, a, b, q, "udiv", |x, y| x.udiv(y));
        prove2(32, a, b, r, "urem", |x, y| x.urem(y));
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
        let (q, r) = if b == 0 {
            (u64::MAX, a)
        } else {
            (a / b, a % b)
        };
        prove2_capped(64, a, b, q, BIG, "udiv", |x, y| x.udiv(y));
        prove2_capped(64, a, b, r, BIG, "urem", |x, y| x.urem(y));
    }
}

// ---- Signed division and remainder ---------------------------------------------------

#[test]
fn sdiv_srem_exhaustive_width4() {
    for a in 0..16u64 {
        for b in 0..16u64 {
            let (sa, sb) = (s4(a), s4(b));
            // A zero divisor: all-ones when the dividend is non-negative, 1 otherwise.
            let (q, r) = if sb == 0 {
                (if sa >= 0 { 0xf } else { 1 }, a)
            } else {
                (maskw((sa / sb) as u64, 4), maskw((sa % sb) as u64, 4))
            };
            prove2(4, a, b, q, "sdiv", |x, y| x.sdiv(y));
            prove2(4, a, b, r, "srem", |x, y| x.srem(y));
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
        (0x05, 0xfe), // 5 / -2 -> -2, remainder 1 (sign of the DIVIDEND - C's %)
        (0xfb, 0x02), // -5 / 2 -> -2, remainder -1
        (0x80, 0x00), // zero divisor, negative dividend -> 1 (and remainder = dividend)
        (0x7f, 0x00), // zero divisor, positive dividend -> -1
        (0x00, 0x00),
        (0xc9, 0x07), // -55 / 7
    ];
    for (a, b) in PAIRS {
        let (sa, sb) = (a as i8 as i64, b as i8 as i64);
        let (q, r) = if sb == 0 {
            (if sa >= 0 { 0xff } else { 1 }, a)
        } else {
            (maskw((sa / sb) as u64, 8), maskw((sa % sb) as u64, 8))
        };
        prove2(8, a, b, q, "sdiv", |x, y| x.sdiv(y));
        prove2(8, a, b, r, "srem", |x, y| x.srem(y));
    }
}

#[test]
fn sdiv_srem_edges_width64() {
    const PAIRS: [(u64, u64); 8] = [
        (0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff), // MIN / -1 wraps to MIN
        (0x8000_0000_0000_0000, 2),                     // MIN / 2
        (0x8000_0000_0000_0000, 0),                     // zero divisor, negative dividend
        (0x7fff_ffff_ffff_ffff, 0),                     // zero divisor, positive dividend
        (u64::MAX, 3),                                  // -1 / 3
        (5, 0xffff_ffff_ffff_fffd),                     // 5 / -3
        (0xffff_ffff_ffff_fffb, 0xffff_ffff_ffff_fffd), // -5 / -3
        (0x0123_4567_89ab_cdef, 0x8000_0000_0000_0000),
    ];
    for (a, b) in PAIRS {
        let (sa, sb) = (a as i64, b as i64);
        let (q, r) = if sb == 0 {
            (if sa >= 0 { u64::MAX } else { 1 }, a)
        } else if sa == i64::MIN && sb == -1 {
            // Rust itself would panic on MIN / -1; the semantics wrap to MIN with remainder 0.
            (a, 0)
        } else {
            ((sa / sb) as u64, (sa % sb) as u64)
        };
        prove2_capped(64, a, b, q, BIG, "sdiv", |x, y| x.sdiv(y));
        prove2_capped(64, a, b, r, BIG, "srem", |x, y| x.srem(y));
    }
}

#[test]
fn srem_takes_the_sign_of_the_dividend_not_the_divisor() {
    // The distinction from `bvsmod`: -7 % 3 is -1 (C), where bvsmod would give 2.
    prove2(8, 0xf9, 3, 0xff, "srem", |x, y| x.srem(y)); // 0xf9 = -7
    prove2(8, 7, 0xfd, 1, "srem", |x, y| x.srem(y)); // 0xfd = -3
}

#[test]
fn sdiv_min_by_negative_one_wraps() {
    // The one signed division that overflows: MIN / -1 is unrepresentable and wraps to MIN.
    prove2(32, 0x8000_0000, 0xffff_ffff, 0x8000_0000, "sdiv", |x, y| {
        x.sdiv(y)
    });
    prove2(32, 0x8000_0000, 0xffff_ffff, 0, "srem", |x, y| x.srem(y));
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
        let (q, r) = if sb == 0 {
            (if sa >= 0 { 0xffff_ffff } else { 1 }, a)
        } else {
            (maskw((sa / sb) as u64, 32), maskw((sa % sb) as u64, 32))
        };
        // A 32-bit signed division blasts to ~31,000 clauses including the pins, so the
        // default cap holds - this doubles as a check of the documented "every operation at
        // 32 bits fits" claim.
        prove2(32, a, b, q, "sdiv", |x, y| x.sdiv(y));
        prove2(32, a, b, r, "srem", |x, y| x.srem(y));
    }
}

// ---- Division with a symbolic operand ------------------------------------------------

#[test]
fn udiv_urem_with_variables() {
    // The pinned-operand proofs above cover the circuit; this covers querying it the other
    // way round - outputs constrained, inputs read back out of the model.
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
        // A shift at or beyond the width empties the value - it does not wrap.
        let expected = if k >= 8 { 0 } else { maskw(a << k, 8) };
        prove_shift(8, 8, a, k, expected, "shl_var", |x, k| x.shl_var(k));
    }
}

#[test]
fn lshr_var_exhaustive_amounts_width8() {
    let a = 0b1011_0011u64;
    for k in 0..24u64 {
        let expected = if k >= 8 { 0 } else { a >> k };
        prove_shift(8, 8, a, k, expected, "lshr_var", |x, k| x.lshr_var(k));
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
        prove_shift(8, 8, a, k, expected, "ashr_var", |x, k| x.ashr_var(k));
        // A non-negative value must NOT sign-fill.
        let exp_pos = if k >= 8 { 0 } else { pos >> k };
        prove_shift(8, 8, pos, k, exp_pos, "ashr_var", |x, k| x.ashr_var(k));
    }
}

#[test]
fn shifts_var_edges_width64() {
    // The 64-bit barrel shifter: six stages, plus the out-of-range fill.
    let a = 0xdead_beef_cafe_f0edu64;
    for k in [0u64, 1, 63, 64, 65, 1 << 40, u64::MAX] {
        let shl = if k >= 64 { 0 } else { a << k };
        let lshr = if k >= 64 { 0 } else { a >> k };
        let ashr = if k >= 64 {
            if a >> 63 == 1 { u64::MAX } else { 0 }
        } else {
            ((a as i64) >> k) as u64
        };
        prove_shift(64, 64, a, k, shl, "shl_var", |x, k| x.shl_var(k));
        prove_shift(64, 64, a, k, lshr, "lshr_var", |x, k| x.lshr_var(k));
        prove_shift(64, 64, a, k, ashr, "ashr_var", |x, k| x.ashr_var(k));
    }
}

#[test]
fn shift_amount_narrower_than_the_vector() {
    // A 4-bit amount can only reach 15, so the high stages of the barrel are never selected -
    // but every amount it *can* reach must still be right.
    let a = 0xabcu64;
    for k in 0..16u64 {
        prove_shift(16, 4, a, k, maskw(a << k, 16), "shl_var", |x, k| {
            x.shl_var(k)
        });
        prove_shift(16, 4, a, k, a >> k, "lshr_var", |x, k| x.lshr_var(k));
    }
}

#[test]
fn shift_amount_wider_than_the_vector() {
    // A 64-bit amount against an 8-bit value: every amount >= 8 must empty it.
    let a = 0xd3u64;
    for k in [0u64, 1, 7, 8, 9, 255, 1 << 40, u64::MAX] {
        let expected = if k >= 8 { 0 } else { a << k };
        prove_shift(8, 64, a, k, expected, "shl_var", |x, k| x.shl_var(k));
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
        prove1(8, a, (a as u8).rotate_left(k) as u64, "rotl", |x| x.rotl(k));
        prove1(8, a, (a as u8).rotate_right(k) as u64, "rotr", |x| {
            x.rotr(k)
        });
    }
}

#[test]
fn rot_const_is_a_permutation_not_a_shift() {
    // Rotating away every bit must return the value, where shifting would empty it.
    prove1(8, 0x5a, 0x5a, "rotl", |x| x.rotl(8));
    prove1(8, 0x5a, 0x5a, "rotr", |x| x.rotr(8));
    prove1(8, 0x5a, 0x5a, "rotl", |x| x.rotl(0));
    prove1(8, 0x5a, 0, "shl", |x| x.shl(8));
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
            prove1(w, a, rotl_bits(a, k, w), "rotl", |x| x.rotl(k));
            prove1(w, a, rotr_bits(a, k, w), "rotr", |x| x.rotr(k));
        }
    }
}

#[test]
fn rot_var_exhaustive_amounts_width8() {
    // Power-of-two width: the low three bits of the amount are already it modulo 8.
    let a = 0b1101_0011u64;
    for k in 0..=40u64 {
        let l = (a as u8).rotate_left(k as u32) as u64;
        let r = (a as u8).rotate_right(k as u32) as u64;
        prove_shift(8, 8, a, k, l, "rotl_var", |x, k| x.rotl_var(k));
        prove_shift(8, 8, a, k, r, "rotr_var", |x, k| x.rotr_var(k));
    }
}

#[test]
fn rot_var_edges_width64() {
    // The 64-bit barrel rotator: power-of-two width, so the low six bits of the amount are
    // the rotation.
    let a = 0x0123_4567_89ab_cdefu64;
    for k in [0u64, 1, 63, 64, 65, 127, 1 << 40, u64::MAX] {
        let l = a.rotate_left((k % 64) as u32);
        let r = a.rotate_right((k % 64) as u32);
        prove_shift(64, 64, a, k, l, "rotl_var", |x, k| x.rotl_var(k));
        prove_shift(64, 64, a, k, r, "rotr_var", |x, k| x.rotr_var(k));
    }
}

#[test]
fn rot_var_non_power_of_two_widths() {
    for &(a, w) in &[(0b1_0011u64, 5u32), (0b101u64, 3), (0b101_1010u64, 7)] {
        // The amount register is `w` bits, so amounts only run to `2^w - 1` - a larger `k`
        // would be masked on the way in and the oracle would be comparing against a
        // different shift.
        let span = (1u64 << w).min(41);
        for k in 0..span {
            let l = rotl_bits(a, k as u32, w);
            let r = rotr_bits(a, k as u32, w);
            prove_shift(w, w, a, k, l, "rotl_var", |x, k| x.rotl_var(k));
            prove_shift(w, w, a, k, r, "rotr_var", |x, k| x.rotr_var(k));
        }
    }
}

#[test]
fn rot_var_amount_wider_than_width() {
    // Regression: the amount must reduce modulo the width using every bit of `k`. Truncating
    // `k` to `w` bits first computes `(k mod 2^w) mod w`, which is wrong whenever `w` does not
    // divide `2^w` - for w=5 and k=32 it gives 0 where the answer is 2.
    let a = 0b1_0011u64;
    for k in [0u64, 1, 4, 5, 6, 31, 32, 33, 100, 1023, 0xffff] {
        let l = rotl_bits(a, (k % 5) as u32, 5);
        let r = rotr_bits(a, (k % 5) as u32, 5);
        prove_shift(5, 16, a, k, l, "rotl_var", |x, k| x.rotl_var(k));
        prove_shift(5, 16, a, k, r, "rotr_var", |x, k| x.rotr_var(k));
    }
}

#[test]
fn rot_var_amount_narrower_than_the_vector() {
    // A 3-bit amount cannot reach the width, so the modulo reduction is never exercised.
    let a = 0b1011_0011u64;
    for k in 0..8u64 {
        let l = (a as u8).rotate_left(k as u32) as u64;
        prove_shift(8, 3, a, k, l, "rotl_var", |x, k| x.rotl_var(k));
    }
}

#[test]
fn symbolic_rotate_amount_from_a_variable() {
    // 0b1000_0001 rotated left by 3 is 0b0000_1100. The amount is 3 bits, which is what makes
    // the answer unique: a rotate is periodic in the width, so an 8-bit amount would also
    // admit 251.
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
    // congruent to 3 modulo 8, so `k mod 8 == 3` must hold and the raw value must not be
    // pinned.
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
            prove_cmp(4, a, b, u64::from(sa > sb), "sgt", |x, y| x.sgt(y));
            prove_cmp(4, a, b, u64::from(sa >= sb), "sge", |x, y| x.sge(y));
        }
    }
}

#[test]
fn signed_comparisons_edges_width64() {
    // The flip of the sign bit must flip the verdict: the same bit patterns compare the other
    // way round unsigned.
    prove_cmp(64, u64::MAX, 0, 0, "sgt", |x, y| x.sgt(y)); // -1 >s 0 is false
    prove_cmp(64, u64::MAX, 0, 1, "ugt", |x, y| x.ugt(y)); // ...but 2^64-1 >u 0 is true
    prove_cmp(
        64,
        0x8000_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0,
        "sgt",
        |x, y| x.sgt(y),
    ); // MIN >s MAX is false...
    prove_cmp(
        64,
        0x8000_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        1,
        "ugt",
        |x, y| x.ugt(y),
    ); // ...but true unsigned
    prove_cmp(
        64,
        0x8000_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        1,
        "slt",
        |x, y| x.slt(y),
    );
    prove_cmp(64, u64::MAX, 0x8000_0000_0000_0000, 1, "sge", |x, y| {
        x.sge(y)
    });
}

#[test]
fn signed_and_unsigned_comparisons_disagree_on_the_negative_half() {
    // -1 >s -128, but 0xff >u 0x80 also holds - so compare where they *must* differ: -1 vs 0.
    prove_cmp(8, 0xff, 0, 0, "sgt", |x, y| x.sgt(y));
    prove_cmp(8, 0xff, 0, 0, "sge", |x, y| x.sge(y));
    prove_cmp(8, 0xff, 0, 1, "slt", |x, y| x.slt(y));
    prove_cmp(8, 0xff, 0, 1, "ugt", |x, y| x.ugt(y)); // 255 > 0 unsigned

    prove_cmp(8, 0x80, 0x7f, 0, "sgt", |x, y| x.sgt(y)); // -128 > 127 is false...
    prove_cmp(8, 0x80, 0x7f, 1, "ugt", |x, y| x.ugt(y)); // ...but 128 > 127 unsigned is true
    prove_cmp(8, 0x7f, 0x80, 1, "sgt", |x, y| x.sgt(y));
}

#[test]
fn sgt_sge_are_the_mirror_of_slt_sle() {
    for a in 0..8u64 {
        for b in 0..8u64 {
            let mut s = Solver::new();
            let (x, y) = (pin(&mut s, "x", a, 4), pin(&mut s, "y", b, 4));
            // a > b  <=>  b < a, and the two must never both hold or both fail.
            s.assert(x.sgt(&y));
            s.assert(y.slt(&x).not());
            assert!(
                matches!(s.check(), Solution::Unsat),
                "sgt and slt disagree for {a},{b}"
            );
        }
    }
}

// ---- Coverage guards -----------------------------------------------------------------
//
// The suite's own mechanism, checked end to end: pinning must force the wide circuits to
// actually exist. `src/blast.rs` measures the same thing per-circuit; this measures it through
// the public API, where a formula over the clause cap can only answer `Unknown`.

#[test]
fn a_pinned_wide_division_reaches_the_blaster() {
    // A pinned 64-bit division blasts to ~110,000 clauses - over the 40,000 default cap - so
    // the default-cap query must degrade to Unknown. If pinning ever stops forcing the circuit
    // (the folded-constant failure this suite once had), the formula is trivial and this
    // answers Sat, failing the test.
    let q = u64::MAX / 7; // Rust as the oracle

    let mut capped = Solver::new();
    let (x, y) = (
        pin(&mut capped, "x", u64::MAX, 64),
        pin(&mut capped, "y", 7, 64),
    );
    capped.assert(x.udiv(&y).eq(&Bv::val(q, 64)));
    assert!(
        matches!(capped.check(), Solution::Unknown),
        "a pinned 64-bit division should exceed the default clause cap - were the operands \
         folded instead of pinned?"
    );

    // The same query under the raised cap gets a real verdict.
    let mut big = Solver::new().with_max_clauses(BIG);
    let (x, y) = (pin(&mut big, "x", u64::MAX, 64), pin(&mut big, "y", 7, 64));
    big.assert(x.udiv(&y).eq(&Bv::val(q, 64)));
    assert!(
        matches!(big.check(), Solution::Sat(_)),
        "the raised cap should leave room for the whole divider"
    );
}
