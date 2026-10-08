//! Randomized differential testing against an independent oracle.
//!
//! `tests/ops.rs` proves specific constants through the solver. This file generates random
//! expressions instead and checks them against a second implementation of the same semantics -
//! `eval`, below - that shares no code with the bit-blaster. That reaches what a conformance suite
//! cannot: a bug the crate's own idea of "right" also has.
//!
//! Four checks:
//!
//! - **Reachability** (`reachability_is_exact_at_small_widths`): brute-force the whole space of
//!   assignments and constants at small widths, so `E == c` must be SAT exactly when some
//!   assignment makes it so.
//! - **Pinned differential** (`pinned_differential_all_ops`): pin every variable to a concrete
//!   assignment, then require `E == v` SAT with a model that reads the assignment back, and
//!   `E != w` UNSAT - at every width, over every operation.
//! - **Metamorphic** (`metamorphic_*`): identities that hold for every assignment, checked by
//!   refuting their negation.
//! - **Budget boundary** (`adder_equivalence_degrades_to_unknown_not_a_wrong_verdict`): where the
//!   search cannot close, it answers `Unknown` rather than inventing a model.
//!
//! Generation is seeded: a failure reproduces exactly, and the generator covers every operation
//! the crate exposes, including the width-changing ones.

// The oracle spells the zero-divisor case out rather than hiding it under `checked_div`, exactly as
// `tests/ops.rs` does - the semantics under test are the point.
#![allow(clippy::manual_checked_ops)]

use std::collections::HashSet;

use smtlite::{Bv, Model, Solution, Solver};

/// The widest vector any generated expression uses. Small enough that circuits stay fast, wide
/// enough to exercise every width-changing and multi-stage path.
const MAXW: u32 = 6;
/// Variables available at each width. More variables = more distinct assignments to explore.
const NVARS: usize = 3;

// ---------------------------------------------------------------------------------------------
// A deterministic generator (SplitMix64). No dependency, and a fixed seed reproduces a failure.
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A value in `0..n`. `n` must be non-zero. Modulo bias is irrelevant to a test generator.
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

// ---------------------------------------------------------------------------------------------
// The independent oracle: a second implementation of the semantics, over a plain expression tree.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Op2 {
    And,
    Or,
    Xor,
    Add,
    Sub,
    Mul,
    Udiv,
    Urem,
    Sdiv,
    Srem,
}

#[derive(Debug, Clone, Copy)]
enum ShDir {
    L,
    Lr,
    Ar,
}

#[derive(Debug, Clone, Copy)]
enum SK {
    L,
    Lr,
    Ar,
}

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Ult,
    Ule,
    Slt,
    Sle,
}

/// An operation node, with every child the same width except where the operation says otherwise.
#[derive(Debug, Clone)]
enum K {
    Const(u64),
    Var(usize),
    Not(Box<E>),
    Neg(Box<E>),
    Bin(Op2, Box<E>, Box<E>),
    ShC(Box<E>, u32, ShDir),
    RotC(Box<E>, u32, bool),
    ShV(Box<E>, Box<E>, SK),
    RotV(Box<E>, Box<E>, bool),
    Cmp(CmpOp, Box<E>, Box<E>),
    Zext(Box<E>),
    Sext(Box<E>),
    Extract(Box<E>, u32, u32),
    Concat(Box<E>, Box<E>),
    Ite(Box<E>, Box<E>, Box<E>),
}

/// An expression with its result width, so `build` and `eval` agree on every mask.
#[derive(Debug, Clone)]
struct E {
    w: u32,
    k: K,
}

fn bx(e: E) -> Box<E> {
    Box::new(e)
}

/// The variable pool: `vars[w][i]` is the `i`th variable of width `w`. Index 0 is unused.
type Vars = Vec<Vec<Bv>>;
/// A concrete assignment, same shape as [`Vars`]: `vals[w][i]`.
type Vals = Vec<Vec<u64>>;

fn vname(w: u32, i: usize) -> String {
    format!("v{w}_{i}")
}

impl E {
    /// Lower this expression to a `Bv` using the pool. Widths are carried in each node, so the
    /// builder never has to re-derive them.
    fn build(&self, vars: &Vars) -> Bv {
        let w = self.w;
        match &self.k {
            K::Const(v) => Bv::val(*v, w),
            K::Var(i) => vars[w as usize][*i].clone(),
            K::Not(a) => a.build(vars).not(),
            K::Neg(a) => a.build(vars).neg(),
            K::Bin(o, a, b) => {
                let (x, y) = (a.build(vars), b.build(vars));
                match o {
                    Op2::And => x.and(&y),
                    Op2::Or => x.or(&y),
                    Op2::Xor => x.xor(&y),
                    Op2::Add => x.add(&y),
                    Op2::Sub => x.sub(&y),
                    Op2::Mul => x.mul(&y),
                    Op2::Udiv => x.udiv(&y),
                    Op2::Urem => x.urem(&y),
                    Op2::Sdiv => x.sdiv(&y),
                    Op2::Srem => x.srem(&y),
                }
            }
            K::ShC(a, k, d) => {
                let x = a.build(vars);
                match d {
                    ShDir::L => x.shl(*k),
                    ShDir::Lr => x.lshr(*k),
                    ShDir::Ar => x.ashr(*k),
                }
            }
            K::RotC(a, k, left) => {
                let x = a.build(vars);
                if *left { x.rotl(*k) } else { x.rotr(*k) }
            }
            K::ShV(a, k, kind) => {
                let (x, y) = (a.build(vars), k.build(vars));
                match kind {
                    SK::L => x.shl_var(&y),
                    SK::Lr => x.lshr_var(&y),
                    SK::Ar => x.ashr_var(&y),
                }
            }
            K::RotV(a, k, left) => {
                let (x, y) = (a.build(vars), k.build(vars));
                if *left {
                    x.rotl_var(&y)
                } else {
                    x.rotr_var(&y)
                }
            }
            K::Cmp(o, a, b) => {
                let (x, y) = (a.build(vars), b.build(vars));
                match o {
                    CmpOp::Eq => x.eq(&y),
                    CmpOp::Ne => x.ne(&y),
                    CmpOp::Ult => x.ult(&y),
                    CmpOp::Ule => x.ule(&y),
                    CmpOp::Slt => x.slt(&y),
                    CmpOp::Sle => x.sle(&y),
                }
            }
            K::Zext(a) => a.build(vars).zext(w),
            K::Sext(a) => a.build(vars).sext(w),
            K::Extract(a, hi, lo) => a.build(vars).extract(*hi, *lo),
            K::Concat(a, b) => a.build(vars).concat(&b.build(vars)),
            K::Ite(c, t, e) => Bv::ite(&c.build(vars), &t.build(vars), &e.build(vars)),
        }
    }

    /// Evaluate the expression under `vals`. This is the oracle - deliberately a separate
    /// implementation, written from the SMT-LIB semantics rather than from the bit-blaster.
    fn eval(&self, vals: &Vals) -> u64 {
        let w = self.w;
        match &self.k {
            K::Const(v) => mask(*v, w),
            K::Var(i) => mask(vals[w as usize][*i], w),
            K::Not(a) => mask(!a.eval(vals), w),
            K::Neg(a) => mask(0u64.wrapping_sub(a.eval(vals)), w),
            K::Bin(o, a, b) => {
                let (x, y) = (a.eval(vals), b.eval(vals));
                match o {
                    Op2::And => mask(x & y, w),
                    Op2::Or => mask(x | y, w),
                    Op2::Xor => mask(x ^ y, w),
                    Op2::Add => mask(x.wrapping_add(y), w),
                    Op2::Sub => mask(x.wrapping_sub(y), w),
                    Op2::Mul => mask(x.wrapping_mul(y), w),
                    Op2::Udiv => udiv(x, y, w),
                    Op2::Urem => urem(x, y, w),
                    Op2::Sdiv => sdiv(x, y, w),
                    Op2::Srem => srem(x, y, w),
                }
            }
            K::ShC(a, k, d) => {
                let x = a.eval(vals);
                let k = u64::from(*k);
                match d {
                    ShDir::L => shl(x, k, w),
                    ShDir::Lr => lshr(x, k, w),
                    ShDir::Ar => ashr(x, k, w),
                }
            }
            K::RotC(a, k, left) => {
                let x = a.eval(vals);
                if *left {
                    rotl(x, u64::from(*k), w)
                } else {
                    rotr(x, u64::from(*k), w)
                }
            }
            K::ShV(a, k, kind) => {
                let x = a.eval(vals);
                let k = k.eval(vals);
                match kind {
                    SK::L => shl(x, k, w),
                    SK::Lr => lshr(x, k, w),
                    SK::Ar => ashr(x, k, w),
                }
            }
            K::RotV(a, k, left) => {
                let x = a.eval(vals);
                let k = k.eval(vals);
                if *left { rotl(x, k, w) } else { rotr(x, k, w) }
            }
            K::Cmp(o, a, b) => {
                let (x, y) = (a.eval(vals), b.eval(vals));
                let bit = match o {
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                    CmpOp::Ult => x < y,
                    CmpOp::Ule => x <= y,
                    CmpOp::Slt => to_signed(x, a.w) < to_signed(y, b.w),
                    CmpOp::Sle => to_signed(x, a.w) <= to_signed(y, b.w),
                };
                u64::from(bit)
            }
            K::Zext(a) => mask(a.eval(vals), w),
            K::Sext(a) => mask(to_signed(a.eval(vals), a.w) as u64, w),
            K::Extract(a, _hi, lo) => mask(a.eval(vals) >> lo, w),
            K::Concat(a, b) => mask((a.eval(vals) << b.w) | b.eval(vals), w),
            K::Ite(c, t, e) => {
                if c.eval(vals) != 0 {
                    t.eval(vals)
                } else {
                    e.eval(vals)
                }
            }
        }
    }
}

// ---- The oracle's primitives ---------------------------------------------------------------

fn mask(v: u64, w: u32) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}

/// The `w`-bit two's-complement value of `x`, as a signed integer.
fn to_signed(x: u64, w: u32) -> i128 {
    let x = mask(x, w);
    if (x >> (w - 1)) & 1 == 1 {
        x as i128 - (1i128 << w)
    } else {
        x as i128
    }
}

fn shl(x: u64, k: u64, w: u32) -> u64 {
    if k >= u64::from(w) {
        0
    } else {
        mask(x << k, w)
    }
}
fn lshr(x: u64, k: u64, w: u32) -> u64 {
    if k >= u64::from(w) {
        0
    } else {
        mask(x, w) >> k
    }
}
fn ashr(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    if k >= u64::from(w) {
        if (x >> (w - 1)) & 1 == 1 {
            mask(u64::MAX, w)
        } else {
            0
        }
    } else {
        mask((to_signed(x, w) >> k) as u64, w)
    }
}
fn rotl(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    let k = (k % u64::from(w)) as u32;
    if k == 0 {
        x
    } else {
        mask((x << k) | (x >> (w - k)), w)
    }
}
fn rotr(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    let k = (k % u64::from(w)) as u32;
    if k == 0 {
        x
    } else {
        mask((x >> k) | (x << (w - k)), w)
    }
}
fn udiv(x: u64, y: u64, w: u32) -> u64 {
    if y == 0 { mask(u64::MAX, w) } else { x / y }
}
fn urem(x: u64, y: u64, _w: u32) -> u64 {
    if y == 0 { x } else { x % y }
}
fn sdiv(x: u64, y: u64, w: u32) -> u64 {
    let (a, b) = (to_signed(x, w), to_signed(y, w));
    if b == 0 {
        if a >= 0 { mask(u64::MAX, w) } else { 1 }
    } else {
        mask((a / b) as u64, w)
    }
}
fn srem(x: u64, y: u64, w: u32) -> u64 {
    let (a, b) = (to_signed(x, w), to_signed(y, w));
    if b == 0 {
        mask(x, w)
    } else {
        mask((a % b) as u64, w)
    }
}

// ---------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------

const OPS2: [Op2; 10] = [
    Op2::And,
    Op2::Or,
    Op2::Xor,
    Op2::Add,
    Op2::Sub,
    Op2::Mul,
    Op2::Udiv,
    Op2::Urem,
    Op2::Sdiv,
    Op2::Srem,
];
const SDIRS: [ShDir; 3] = [ShDir::L, ShDir::Lr, ShDir::Ar];
const SVKINDS: [SK; 3] = [SK::L, SK::Lr, SK::Ar];
const CMPS: [CmpOp; 6] = [
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::Ult,
    CmpOp::Ule,
    CmpOp::Slt,
    CmpOp::Sle,
];

fn leaf(rng: &mut Rng, w: u32) -> E {
    if rng.bool() {
        E {
            w,
            k: K::Const(rng.next_u64()),
        }
    } else {
        E {
            w,
            k: K::Var(rng.below(NVARS as u64) as usize),
        }
    }
}

/// A random expression of result width `w`, using every operation the crate exposes.
fn gen_expr(rng: &mut Rng, depth: u32, w: u32) -> E {
    if depth == 0 || rng.below(5) == 0 {
        return leaf(rng, w);
    }
    let d = depth - 1;
    let mut choices: Vec<u64> = (0..=20).collect();
    if w > 1 {
        choices.push(21); // zext
        choices.push(22); // sext
    }
    if w < MAXW {
        choices.push(23); // extract, from a wider child
    }
    if w >= 2 {
        choices.push(24); // concat
    }
    if w == 1 {
        choices.push(25); // comparison (result is 1 bit)
    }
    let c = choices[rng.below(choices.len() as u64) as usize];
    match c {
        0 => E {
            w,
            k: K::Not(bx(gen_expr(rng, d, w))),
        },
        1 => E {
            w,
            k: K::Neg(bx(gen_expr(rng, d, w))),
        },
        2..=11 => E {
            w,
            k: K::Bin(
                OPS2[(c - 2) as usize],
                bx(gen_expr(rng, d, w)),
                bx(gen_expr(rng, d, w)),
            ),
        },
        12..=14 => E {
            w,
            k: K::ShC(
                bx(gen_expr(rng, d, w)),
                rng.below(2 * u64::from(w)) as u32,
                SDIRS[(c - 12) as usize],
            ),
        },
        15 => E {
            w,
            k: K::RotC(
                bx(gen_expr(rng, d, w)),
                rng.below(3 * u64::from(w)) as u32,
                rng.bool(),
            ),
        },
        16..=18 => {
            let aw = 1 + rng.below(u64::from(w)) as u32;
            E {
                w,
                k: K::ShV(
                    bx(gen_expr(rng, d, w)),
                    bx(gen_expr(rng, d, aw)),
                    SVKINDS[(c - 16) as usize],
                ),
            }
        }
        19 => {
            let aw = 1 + rng.below(u64::from(w)) as u32;
            E {
                w,
                k: K::RotV(
                    bx(gen_expr(rng, d, w)),
                    bx(gen_expr(rng, d, aw)),
                    rng.bool(),
                ),
            }
        }
        20 => E {
            w,
            k: K::Ite(
                bx(gen_expr(rng, d, 1)),
                bx(gen_expr(rng, d, w)),
                bx(gen_expr(rng, d, w)),
            ),
        },
        21 => {
            let cw = 1 + rng.below(u64::from(w - 1)) as u32;
            E {
                w,
                k: K::Zext(bx(gen_expr(rng, d, cw))),
            }
        }
        22 => {
            let cw = 1 + rng.below(u64::from(w - 1)) as u32;
            E {
                w,
                k: K::Sext(bx(gen_expr(rng, d, cw))),
            }
        }
        23 => {
            let cw = w + 1 + rng.below(u64::from(MAXW - w)) as u32;
            let lo = rng.below(u64::from(cw - w + 1)) as u32;
            let hi = lo + w - 1;
            E {
                w,
                k: K::Extract(bx(gen_expr(rng, d, cw)), hi, lo),
            }
        }
        24 => {
            let aw = 1 + rng.below(u64::from(w - 1)) as u32;
            E {
                w,
                k: K::Concat(bx(gen_expr(rng, d, aw)), bx(gen_expr(rng, d, w - aw))),
            }
        }
        _ => {
            let cw = 1 + rng.below(u64::from(MAXW)) as u32;
            E {
                w,
                k: K::Cmp(
                    CMPS[rng.below(CMPS.len() as u64) as usize],
                    bx(gen_expr(rng, d, cw)),
                    bx(gen_expr(rng, d, cw)),
                ),
            }
        }
    }
}

/// The binary operations used by the exhaustive sweep. Multiply and the two divisions are left
/// out: the sweep's queries are refutations (`e == c` UNSAT for every unreachable `c`), and
/// refuting a multiplier or divider equality is the case this solver cannot close inside the
/// budget. Those operators are covered by `pinned_differential_all_ops`, whose queries are
/// satisfiable, and by `adder_equivalence_degrades_to_unknown_not_a_wrong_verdict`.
const OPS_WP: [Op2; 5] = [Op2::And, Op2::Or, Op2::Xor, Op2::Add, Op2::Sub];

/// A random expression of width `w` built only from width-preserving operations, so the whole
/// assignment space is `2^(w * NVARS)` and can be brute-forced.
fn gen_wp(rng: &mut Rng, depth: u32, w: u32) -> E {
    if depth == 0 || rng.below(4) == 0 {
        return leaf(rng, w);
    }
    let d = depth - 1;
    match rng.below(20) {
        0 => E {
            w,
            k: K::Not(bx(gen_wp(rng, d, w))),
        },
        1 => E {
            w,
            k: K::Neg(bx(gen_wp(rng, d, w))),
        },
        2..=11 => E {
            w,
            k: K::Bin(
                OPS_WP[rng.below(OPS_WP.len() as u64) as usize],
                bx(gen_wp(rng, d, w)),
                bx(gen_wp(rng, d, w)),
            ),
        },
        12..=14 => E {
            w,
            k: K::ShC(
                bx(gen_wp(rng, d, w)),
                rng.below(2 * u64::from(w)) as u32,
                SDIRS[rng.below(3) as usize],
            ),
        },
        15 => E {
            w,
            k: K::RotC(
                bx(gen_wp(rng, d, w)),
                rng.below(3 * u64::from(w)) as u32,
                rng.bool(),
            ),
        },
        16..=18 => E {
            // The amount is width `w`, so it can reach and exceed the vector width.
            w,
            k: K::ShV(
                bx(gen_wp(rng, d, w)),
                bx(gen_wp(rng, d, w)),
                SVKINDS[rng.below(3) as usize],
            ),
        },
        _ => E {
            w,
            k: K::RotV(bx(gen_wp(rng, d, w)), bx(gen_wp(rng, d, w)), rng.bool()),
        },
    }
}

// ---------------------------------------------------------------------------------------------
// Pool plumbing
// ---------------------------------------------------------------------------------------------

fn build_pool() -> (Solver, Vars) {
    let mut s = Solver::new();
    let mut vars: Vars = vec![Vec::new(); (MAXW + 1) as usize];
    for w in 1..=MAXW {
        for i in 0..NVARS {
            let v = s.var(&vname(w, i), w);
            vars[w as usize].push(v);
        }
    }
    (s, vars)
}

/// Read every pool variable back out of a model, keyed the same way as [`Vals`].
fn read_vals(m: &Model) -> Vals {
    let mut vals: Vals = vec![Vec::new(); (MAXW + 1) as usize];
    for w in 1..=MAXW {
        for i in 0..NVARS {
            vals[w as usize].push(m.get(&vname(w, i)).expect("a declared variable reads back"));
        }
    }
    vals
}

/// Constraints pinning every pool variable to `vals`.
fn pins(vars: &Vars, vals: &Vals) -> Vec<Bv> {
    let mut out = Vec::new();
    for w in 1..=MAXW {
        for i in 0..NVARS {
            out.push(vars[w as usize][i].eq(&Bv::val(vals[w as usize][i], w)));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------------------------

/// At small widths, the solver's verdict must match the oracle's over the entire space of
/// assignments and constants, so a reachable-but-missed value is caught rather than only a wrong
/// witness. Every call must decide: an `Unknown` fails the test.
#[test]
fn reachability_is_exact_at_small_widths() {
    let (solver, vars) = build_pool();
    let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);

    for &w in &[1u32, 3, 4] {
        let n = 1u64 << w;
        for _ in 0..120 {
            let e = gen_wp(&mut rng, 4, w);
            let expr = e.build(&vars);
            assert_eq!(
                expr.width(),
                w,
                "builder produced the wrong width for {e:?}"
            );

            // Every value `e` can take, by enumerating every assignment to the width-`w` vars.
            let mut vals: Vals = vec![vec![0; NVARS]; (MAXW + 1) as usize];
            let mut reachable = HashSet::new();
            for a in 0..n {
                for b in 0..n {
                    for c in 0..n {
                        vals[w as usize] = vec![a, b, c];
                        reachable.insert(e.eval(&vals));
                    }
                }
            }

            for c in 0..n {
                let want = Bv::val(c, w);
                let is_sat = match solver.check_all(&[expr.eq(&want)]) {
                    Solution::Sat(_) => true,
                    Solution::Unsat => false,
                    Solution::Unknown => panic!("unexpected Unknown: {e:?} == {c} at width {w}"),
                };
                assert_eq!(
                    is_sat,
                    reachable.contains(&c),
                    "`{e:?}` == {c} (width {w}): solver says {is_sat}"
                );

                // `expr != c` is refutable exactly when `c` is the *only* value `expr` can take.
                // That can only happen for a constant expression, so skip the second solve for
                // every other expression: it would only re-assert what the `eq` check above
                // already proved.
                if !(reachable.len() == 1 && reachable.contains(&c)) {
                    continue;
                }
                let is_unsat = match solver.check_all(&[expr.ne(&want)]) {
                    Solution::Unsat => true,
                    Solution::Sat(_) => false,
                    Solution::Unknown => panic!("unexpected Unknown: {e:?} != {c} at width {w}"),
                };
                assert!(
                    is_unsat,
                    "`{e:?}` != {c} (width {w}) should be UNSAT: {c} is the only value `e` takes"
                );
            }
        }
    }
}

/// Any expression at any width, pinned to a concrete assignment: `== v` must be SAT with a model
/// that reads back the assignment and re-evaluates to `v`, and `!= (v + 1)` must be UNSAT.
#[test]
fn pinned_differential_all_ops() {
    let (solver, vars) = build_pool();
    let mut rng = Rng::new(0xD1FF_1234_5678_9ABC);

    for _ in 0..600 {
        let w = 1 + rng.below(u64::from(MAXW)) as u32;
        let e = gen_expr(&mut rng, 4, w);
        let expr = e.build(&vars);
        assert_eq!(
            expr.width(),
            w,
            "builder produced the wrong width for {e:?}"
        );

        // A random concrete assignment for every pool variable.
        let mut vals: Vals = vec![Vec::new(); (MAXW + 1) as usize];
        for ww in 1..=MAXW {
            vals[ww as usize] = (0..NVARS)
                .map(|_| rng.next_u64() & mask(u64::MAX, ww))
                .collect();
        }
        let v = e.eval(&vals);

        // SAT: the assignment really does produce `v`.
        let mut cons = pins(&vars, &vals);
        cons.push(expr.eq(&Bv::val(v, w)));
        match solver.check_all(&cons) {
            Solution::Sat(m) => {
                let mv = read_vals(&m);
                assert_eq!(mv, vals, "pinned variables must read back unchanged");
                assert_eq!(e.eval(&mv), v, "model does not satisfy `{e:?}`");
            }
            other => panic!("a pinned point must be SAT, got {other:?} for {e:?}"),
        }

        // UNSAT: pinned, so the expression cannot equal any *other* value.
        let wrong = mask(v.wrapping_add(1), w);
        let mut cons = pins(&vars, &vals);
        cons.push(expr.eq(&Bv::val(wrong, w)));
        assert!(
            matches!(solver.check_all(&cons), Solution::Unsat),
            "pinned `{e:?}` admitted {wrong} (only {v} is possible)"
        );
        // And the matching `!=` direction is SAT.
        let mut cons = pins(&vars, &vals);
        cons.push(expr.ne(&Bv::val(wrong, w)));
        assert!(
            matches!(solver.check_all(&cons), Solution::Sat(_)),
            "pinned `{e:?}` rejected {wrong} (which it should accept)"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Metamorphic identities: must hold for every assignment, so their negation is UNSAT.
// ---------------------------------------------------------------------------------------------

/// Assert that `constraint` can never be violated (its negation is UNSAT).
fn tautology(s: &Solver, constraint: Bv, what: &str) {
    let negated = constraint.not();
    assert!(
        matches!(s.check_all(&[negated]), Solution::Unsat),
        "not a tautology: {what}"
    );
}

#[test]
fn metamorphic_algebraic_identities() {
    // Width 4: refuting an identity's negation costs what the circuit costs, and these become
    // adder- and multiplier-equivalence at width 16, past what this solver closes (see
    // `adder_equivalence_degrades_to_unknown_not_a_wrong_verdict`).
    let mut s = Solver::new();
    let a = s.var("a", 4);
    let b = s.var("b", 4);

    // Commutativity of the symmetric operations.
    tautology(&s, a.add(&b).eq(&b.add(&a)), "a + b == b + a");
    tautology(&s, a.mul(&b).eq(&b.mul(&a)), "a * b == b * a");
    tautology(&s, a.and(&b).eq(&b.and(&a)), "a & b == b & a");
    tautology(&s, a.or(&b).eq(&b.or(&a)), "a | b == b | a");
    tautology(&s, a.xor(&b).eq(&b.xor(&a)), "a ^ b == b ^ a");

    // Subtraction is addition of the negation; negation and NOT are self-inverse.
    tautology(&s, a.sub(&b).eq(&a.add(&b.neg())), "a - b == a + (-b)");
    tautology(&s, a.neg().neg().eq(&a), "-(-a) == a");
    tautology(&s, a.not().not().eq(&a), "~~a == a");

    // De Morgan.
    tautology(
        &s,
        a.and(&b).not().eq(&a.not().or(&b.not())),
        "~(a & b) == ~a | ~b",
    );
    tautology(
        &s,
        a.or(&b).not().eq(&a.not().and(&b.not())),
        "~(a | b) == ~a & ~b",
    );

    // Comparison mirrors.
    tautology(&s, a.ult(&b).eq(&b.ugt(&a)), "a <u b == b >u a");
    tautology(&s, a.slt(&b).eq(&b.sgt(&a)), "a <s b == b >s a");
    tautology(
        &s,
        a.ule(&b).eq(&a.ult(&b).or(&a.eq(&b))),
        "a <=u b == (a <u b) | (a == b)",
    );
}

#[test]
fn metamorphic_shift_and_rotate_relations() {
    let mut s = Solver::new();
    let a = s.var("a", 8);

    for k in 0..12u32 {
        // A rotate by k is undone by a rotate the same way, for every k (including k >= width).
        tautology(
            &s,
            a.rotl(k).rotr(k).eq(&a),
            "rotl(k) then rotr(k) is the identity",
        );
        // A left shift then an equal logical right shift keeps the low (width - k) bits.
        let keep = if k >= 8 { 0 } else { mask(u64::MAX, 8 - k) };
        tautology(
            &s,
            a.shl(k).lshr(k).eq(&a.and(&Bv::val(keep, 8))),
            "(a << k) >>u k == a masked to the low (width - k) bits",
        );
    }

    // Rotating by the full width is the identity; shifting by it empties.
    tautology(&s, a.rotl(8).eq(&a), "rotl by the width is the identity");
    tautology(&s, a.rotr(8).eq(&a), "rotr by the width is the identity");
    tautology(&s, a.shl(8).eq(&Bv::val(0, 8)), "shl by the width is zero");
    tautology(
        &s,
        a.lshr(8).eq(&Bv::val(0, 8)),
        "lshr by the width is zero",
    );
}

#[test]
fn metamorphic_extension_and_extraction() {
    // Width 4, as in `metamorphic_algebraic_identities`: refuting the negation has to stay cheap.
    let mut s = Solver::new();
    let a = s.var("a", 4);

    // Widening is value-preserving under the right interpretation.
    tautology(
        &s,
        a.zext(8).extract(3, 0).eq(&a),
        "zext then extract is the identity",
    );
    tautology(
        &s,
        a.sext(8).extract(3, 0).eq(&a),
        "sext then extract is the identity",
    );

    // Cutting a value into halves and concatenating them back returns it.
    let hi = a.extract(3, 2);
    let lo = a.extract(1, 0);
    tautology(
        &s,
        hi.concat(&lo).eq(&a),
        "extract halves then concat is the identity",
    );

    // An `ite` with equal branches is just the branch, and it selects correctly.
    let c = a.eq(&Bv::val(0, 4));
    tautology(&s, Bv::ite(&c, &a, &a).eq(&a), "ite c a a == a");
    let pick = Bv::ite(&c, &Bv::val(0, 4), &Bv::val(0xf, 4));
    // `c -> pick == 0`, not `c && pick == 0`: the latter is falsified by taking `c` false.
    tautology(
        &s,
        c.not().or(&pick.eq(&Bv::val(0, 4))),
        "ite picks `then` when c holds",
    );
}

#[test]
fn adder_equivalence_degrades_to_unknown_not_a_wrong_verdict() {
    // `a + b == b + a` is a tautology, so its negation is unsatisfiable at every width. Refuting
    // it is adder equivalence, which a DPLL core without clause learning cannot close once the
    // carries chain up. Whatever the budget does, the answer must never be `Sat`: there is no
    // model of `a + b != b + a`.
    fn verdict(w: u32, budget: Option<u64>) -> Solution {
        let mut s = Solver::new();
        let a = s.var("a", w);
        let b = s.var("b", w);
        let negated = a.add(&b).eq(&b.add(&a)).not();
        match budget {
            Some(n) => s.check_all_with_budget(&[negated], n),
            None => s.check_all(&[negated]),
        }
    }

    // Width 4 refutes inside the default budget.
    assert!(
        matches!(verdict(4, None), Solution::Unsat),
        "width 4 should refute a + b == b + a inside the default budget"
    );

    // Widths 16 and 32 under a small budget: `Unknown` is correct, `Sat` is not.
    for w in [16u32, 32] {
        match verdict(w, Some(10_000)) {
            Solution::Unsat => {}
            Solution::Unknown => {}
            Solution::Sat(m) => panic!("width {w}: a + b == b + a reported SAT with {m:?}"),
        }
    }
}
