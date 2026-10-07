//! The bit-blaster: lower a `Bv` DAG (and asserted 1-bit constraints) to CNF.
//!
//! Each bit of each node becomes a SAT literal; operations are encoded with Tseitin gates
//! (AND/OR/XOR), arithmetic with ripple-carry adders, and unsigned comparison via the
//! carry-out of `a + ¬b + 1` (which is 1 iff `a ≥ b`). Shared subgraphs are memoised on the
//! `Rc` pointer so a value used twice is encoded once. Bit vectors are little-endian:
//! index 0 is the least-significant bit.

use std::collections::HashMap;
use std::rc::Rc;

use crate::bv::{Bv, Cmp, Node, Op};
use crate::sat::Cnf;

pub(crate) struct Blaster<'b> {
    pub cnf: Cnf<'b>,
    true_lit: i32,
    memo: HashMap<usize, Vec<i32>>,
    /// Bit literals for each named variable id, so a model can be read back.
    pub var_bits: HashMap<usize, Vec<i32>>,
}

impl<'b> Blaster<'b> {
    pub fn new(bump: &'b bumpalo::Bump) -> Self {
        let mut cnf = Cnf::new(bump);
        let true_lit = cnf.new_var();
        cnf.add_clause(&[true_lit]); // pin it true, so its negation is a constant false
        Self {
            cnf,
            true_lit,
            memo: HashMap::new(),
            var_bits: HashMap::new(),
        }
    }

    fn t(&self) -> i32 {
        self.true_lit
    }
    fn f(&self) -> i32 {
        -self.true_lit
    }

    fn and(&mut self, a: i32, b: i32) -> i32 {
        let o = self.cnf.new_var();
        self.cnf.add_clause(&[-o, a]);
        self.cnf.add_clause(&[-o, b]);
        self.cnf.add_clause(&[o, -a, -b]);
        o
    }
    fn or(&mut self, a: i32, b: i32) -> i32 {
        let o = self.cnf.new_var();
        self.cnf.add_clause(&[o, -a]);
        self.cnf.add_clause(&[o, -b]);
        self.cnf.add_clause(&[-o, a, b]);
        o
    }
    fn xor(&mut self, a: i32, b: i32) -> i32 {
        let o = self.cnf.new_var();
        self.cnf.add_clause(&[-o, -a, -b]);
        self.cnf.add_clause(&[-o, a, b]);
        self.cnf.add_clause(&[o, -a, b]);
        self.cnf.add_clause(&[o, a, -b]);
        o
    }

    fn full_adder(&mut self, a: i32, b: i32, cin: i32) -> (i32, i32) {
        let axb = self.xor(a, b);
        let sum = self.xor(axb, cin);
        let ab = self.and(a, b);
        let axb_cin = self.and(axb, cin);
        let cout = self.or(ab, axb_cin);
        (sum, cout)
    }

    fn add_bits(&mut self, a: &[i32], b: &[i32], cin: i32) -> (Vec<i32>, i32) {
        let mut carry = cin;
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            let (s, c) = self.full_adder(a[i], b[i], carry);
            out.push(s);
            carry = c;
        }
        (out, carry)
    }

    /// Unsigned `a < b`: NOT the carry-out of `a + ¬b + 1`.
    fn ult_bits(&mut self, a: &[i32], b: &[i32]) -> i32 {
        let not_b: Vec<i32> = b.iter().map(|&l| -l).collect();
        let (_, cout) = self.add_bits(a, &not_b, self.t());
        -cout
    }

    fn eq_bits(&mut self, a: &[i32], b: &[i32]) -> i32 {
        let mut acc = self.t();
        for i in 0..a.len() {
            let x = self.xor(a[i], b[i]);
            acc = self.and(acc, -x); // -x is XNOR (bits equal)
        }
        acc
    }

    /// Encode a value to its bit literals (LSB first).
    pub fn encode(&mut self, bv: &Bv) -> Vec<i32> {
        let key = Rc::as_ptr(&bv.0) as usize;
        if let Some(bits) = self.memo.get(&key) {
            return bits.clone();
        }
        let bits = self.encode_node(bv);
        self.memo.insert(key, bits.clone());
        bits
    }

    fn encode_node(&mut self, bv: &Bv) -> Vec<i32> {
        let w = bv.width() as usize;
        match &*bv.0 {
            Node::Const(_, v) => (0..w)
                .map(|i| {
                    if (v >> i) & 1 == 1 {
                        self.t()
                    } else {
                        self.f()
                    }
                })
                .collect(),
            Node::Var(_, id) => {
                let bits: Vec<i32> = (0..w).map(|_| self.cnf.new_var()).collect();
                self.var_bits.insert(*id, bits.clone());
                bits
            }
            Node::Not(a) => self.encode(a).iter().map(|&l| -l).collect(),
            Node::Neg(a) => {
                let ab: Vec<i32> = self.encode(a).iter().map(|&l| -l).collect();
                let zero = vec![self.f(); w];
                self.add_bits(&ab, &zero, self.t()).0 // ¬a + 1
            }
            Node::Bin(op, a, b) => {
                let (ab, bb) = (self.encode(a), self.encode(b));
                match op {
                    Op::And => (0..w).map(|i| self.and(ab[i], bb[i])).collect(),
                    Op::Or => (0..w).map(|i| self.or(ab[i], bb[i])).collect(),
                    Op::Xor => (0..w).map(|i| self.xor(ab[i], bb[i])).collect(),
                    Op::Add => self.add_bits(&ab, &bb, self.f()).0,
                    Op::Sub => {
                        let not_b: Vec<i32> = bb.iter().map(|&l| -l).collect();
                        self.add_bits(&ab, &not_b, self.t()).0
                    }
                    Op::Mul => self.mul_bits(&ab, &bb),
                    Op::Udiv => self.udivrem_bits(&ab, &bb).0,
                    Op::Urem => self.udivrem_bits(&ab, &bb).1,
                    Op::Sdiv => self.sdiv_bits(&ab, &bb),
                    Op::Srem => self.srem_bits(&ab, &bb),
                }
            }
            Node::ShlC(a, k) => {
                let ab = self.encode(a);
                let k = *k as usize;
                (0..w)
                    .map(|i| if i < k { self.f() } else { ab[i - k] })
                    .collect()
            }
            Node::ShrC(a, k, arith) => {
                let ab = self.encode(a);
                let k = *k as usize;
                let fill = if *arith { ab[w - 1] } else { self.f() };
                (0..w)
                    .map(|i| if i + k < w { ab[i + k] } else { fill })
                    .collect()
            }
            Node::ShlV(a, k) => {
                let ab = self.encode(a);
                let kb = self.encode(k);
                self.shift_var_bits(&ab, &kb, ShKind::Left)
            }
            Node::ShrV(a, k, arith) => {
                let ab = self.encode(a);
                let kb = self.encode(k);
                let kind = if *arith {
                    ShKind::Aright
                } else {
                    ShKind::Lright
                };
                self.shift_var_bits(&ab, &kb, kind)
            }
            Node::RotC(a, k, left) => {
                let ab = self.encode(a);
                rot_const(&ab, *k, *left)
            }
            Node::RotV(a, k, left) => {
                let ab = self.encode(a);
                let kb = self.encode(k);
                self.rot_var_bits(&ab, &kb, *left)
            }
            Node::Compare(c, a, b) => {
                let (ab, bb) = (self.encode(a), self.encode(b));
                let bit = match c {
                    Cmp::Eq => self.eq_bits(&ab, &bb),
                    Cmp::Ne => -self.eq_bits(&ab, &bb),
                    Cmp::Ult => self.ult_bits(&ab, &bb),
                    Cmp::Ule => {
                        let lt = self.ult_bits(&ab, &bb);
                        let eq = self.eq_bits(&ab, &bb);
                        self.or(lt, eq)
                    }
                    Cmp::Slt => self.slt_bits(&ab, &bb),
                    Cmp::Sle => {
                        let lt = self.slt_bits(&ab, &bb);
                        let eq = self.eq_bits(&ab, &bb);
                        self.or(lt, eq)
                    }
                };
                vec![bit]
            }
            Node::Zext(a, _) => {
                let mut bits = self.encode(a);
                bits.resize(w, self.f());
                bits
            }
            Node::Sext(a, _) => {
                let ab = self.encode(a);
                let sign = *ab.last().unwrap_or(&self.f());
                let mut bits = ab.clone();
                bits.resize(w, sign);
                bits
            }
            Node::Extract(a, hi, lo) => {
                let ab = self.encode(a);
                ab[*lo as usize..=*hi as usize].to_vec()
            }
            Node::Concat(hi, lo) => {
                let mut bits = self.encode(lo);
                bits.extend(self.encode(hi));
                bits
            }
            Node::Ite(cond, then, els) => {
                let c = self.encode(cond)[0];
                let (tb, eb) = (self.encode(then), self.encode(els));
                (0..w)
                    .map(|i| {
                        let a = self.and(c, tb[i]);
                        let b = self.and(-c, eb[i]);
                        self.or(a, b)
                    })
                    .collect()
            }
        }
    }

    /// Signed `a < b`: flip the sign bits, then compare unsigned.
    fn slt_bits(&mut self, a: &[i32], b: &[i32]) -> i32 {
        let top = a.len() - 1;
        let mut aa = a.to_vec();
        let mut bb = b.to_vec();
        aa[top] = -aa[top];
        bb[top] = -bb[top];
        self.ult_bits(&aa, &bb)
    }

    /// Schoolbook multiply, truncated to the operand width.
    fn mul_bits(&mut self, a: &[i32], b: &[i32]) -> Vec<i32> {
        let w = a.len();
        let mut acc = vec![self.f(); w];
        for (j, &bj) in b.iter().enumerate() {
            let mut partial = vec![self.f(); w];
            for i in 0..(w - j) {
                partial[i + j] = self.and(a[i], bj);
            }
            acc = self.add_bits(&acc, &partial, self.f()).0;
        }
        acc
    }

    // ---- Division ----------------------------------------------------------------

    /// A 2:1 mux on two literals: `c ? t : e`.
    fn mux(&mut self, c: i32, t: i32, e: i32) -> i32 {
        let a = self.and(c, t);
        let b = self.and(-c, e);
        self.or(a, b)
    }

    /// Bitwise mux of two equal-length bit vectors.
    fn mux_vec(&mut self, c: i32, t: &[i32], e: &[i32]) -> Vec<i32> {
        let mut out = Vec::with_capacity(t.len());
        for i in 0..t.len() {
            let (ti, ei) = (t[i], e[i]);
            out.push(self.mux(c, ti, ei));
        }
        out
    }

    /// Bit literals for the constant `v`, zero-extended (or truncated) to `w` bits.
    fn const_bits(&mut self, v: u64, w: usize) -> Vec<i32> {
        (0..w)
            .map(|i| {
                if (v >> i) & 1 == 1 {
                    self.t()
                } else {
                    self.f()
                }
            })
            .collect()
    }

    /// Two's-complement absolute value: `sign ? -a : a`.
    fn abs_bits(&mut self, a: &[i32]) -> Vec<i32> {
        let w = a.len();
        let sign = a[w - 1];
        let na: Vec<i32> = a.iter().map(|&l| -l).collect();
        let zero = vec![self.f(); w];
        let neg = self.add_bits(&na, &zero, self.t()).0;
        self.mux_vec(sign, &neg, a)
    }

    /// Radix-2 restoring division: `(quotient, remainder)` of unsigned `a / b`.
    ///
    /// Each step shifts one dividend bit into a `w+1`-bit remainder and conditionally subtracts
    /// `b`. The carry-out of `rem + ¬b + 1` is exactly `rem >= b`, so a single adder per step
    /// yields both the difference and the comparison — no separate comparator is needed.
    ///
    /// A zero divisor needs no special case: `rem >= 0` always holds, so every quotient bit is 1
    /// (all-ones, SMT-LIB `bvudiv`) and the remainder ends up holding `a` (SMT-LIB `bvurem`).
    fn udivrem_bits(&mut self, a: &[i32], b: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let w = a.len();
        let mut rem = vec![self.f(); w + 1]; // one extra bit: the shifted-in bit always fits
        let mut bext = b.to_vec();
        bext.push(self.f()); // zero-extend b to w+1 so the adder widths match
        let not_b: Vec<i32> = bext.iter().map(|&l| -l).collect();
        let mut quot = vec![self.f(); w];
        for i in (0..w).rev() {
            // Fold the next dividend bit in. Little-endian, so it is *less* significant than
            // everything in `rem`: index 0, the rest slides up — `rem << 1 | a[i]` reversed.
            let mut shifted = Vec::with_capacity(w + 1);
            shifted.push(a[i]);
            shifted.extend_from_slice(&rem[0..w]);
            rem = shifted;
            let (diff, carry) = self.add_bits(&rem, &not_b, self.t());
            let ge = carry; // carry-out of rem + ¬b + 1  ==  rem >= b
            rem = self.mux_vec(ge, &diff, &rem);
            quot[i] = ge;
        }
        // The invariant rem < b <= 2^w - 1 keeps the remainder inside the low w bits.
        (quot, rem[0..w].to_vec())
    }

    /// Signed division: divide the magnitudes, then re-apply the sign (SMT-LIB `bvsdiv`).
    fn sdiv_bits(&mut self, a: &[i32], b: &[i32]) -> Vec<i32> {
        let w = a.len();
        let (aa, bb) = (self.abs_bits(a), self.abs_bits(b));
        let (q, _) = self.udivrem_bits(&aa, &bb);
        let nq: Vec<i32> = q.iter().map(|&l| -l).collect();
        let zero = vec![self.f(); w];
        let neg_q = self.add_bits(&nq, &zero, self.t()).0;
        let sign = self.xor(a[w - 1], b[w - 1]);
        let quotient = self.mux_vec(sign, &neg_q, &q);
        // A zero divisor yields all-ones for a non-negative dividend, and 1 otherwise.
        let bzero = self.eq_bits(b, &vec![self.f(); w]);
        let one = self.const_bits(1, w);
        let all_ones = vec![self.t(); w];
        let on_zero = self.mux_vec(a[w - 1], &one, &all_ones);
        self.mux_vec(bzero, &on_zero, &quotient)
    }

    /// Signed remainder, taking the sign of the **dividend** (SMT-LIB `bvsrem` — C's `%`).
    fn srem_bits(&mut self, a: &[i32], b: &[i32]) -> Vec<i32> {
        let w = a.len();
        let (aa, bb) = (self.abs_bits(a), self.abs_bits(b));
        let (_, r) = self.udivrem_bits(&aa, &bb);
        let nr: Vec<i32> = r.iter().map(|&l| -l).collect();
        let zero = vec![self.f(); w];
        let neg_r = self.add_bits(&nr, &zero, self.t()).0;
        let rem = self.mux_vec(a[w - 1], &neg_r, &r);
        // A zero divisor yields the dividend.
        let bzero = self.eq_bits(b, &vec![self.f(); w]);
        self.mux_vec(bzero, a, &rem)
    }

    // ---- Symbolic shifts and rotates ---------------------------------------------

    /// Barrel shifter for a symbolic amount: `stages` conditionally shift by 1, 2, 4, …
    ///
    /// An amount at or beyond the width does **not** wrap — every bit takes the fill (zero for
    /// `<<`/`>>u`, the sign bit for `>>s`), which is SMT-LIB shift semantics as opposed to a
    /// rotate's.
    fn shift_var_bits(&mut self, a: &[i32], k: &[i32], kind: ShKind) -> Vec<i32> {
        let w = a.len();
        let f = self.f();
        let stages = shift_stages(w);
        let mut cur = a.to_vec();
        for j in 0..stages {
            // k narrower than the barrel: its higher bits are zero, so those stages never fire.
            let Some(&sel) = k.get(j) else { break };
            let sh = 1usize << j;
            let mut moved = Vec::with_capacity(w);
            for i in 0..w {
                moved.push(match kind {
                    ShKind::Left => {
                        if i >= sh {
                            cur[i - sh]
                        } else {
                            f
                        }
                    }
                    ShKind::Lright => {
                        if i + sh < w {
                            cur[i + sh]
                        } else {
                            f
                        }
                    }
                    ShKind::Aright => {
                        if i + sh < w {
                            cur[i + sh]
                        } else {
                            cur[w - 1]
                        }
                    }
                });
            }
            cur = self.mux_vec(sel, &moved, &cur);
        }
        // `k >= w` replaces the entire result with the fill value.
        let mut overflow = f;
        for &bit in k.iter().skip(stages) {
            overflow = self.or(overflow, bit);
        }
        if stages > 0 && (1usize << stages) != w {
            // The width is not a power of two, so an amount below 2^stages can still reach it.
            let low: Vec<i32> = (0..stages)
                .map(|j| k.get(j).copied().unwrap_or(f))
                .collect();
            let wbits = self.const_bits(w as u64, stages);
            let ge = -self.ult_bits(&low, &wbits);
            overflow = self.or(overflow, ge);
        }
        let fill = match kind {
            ShKind::Aright => a[w - 1],
            ShKind::Left | ShKind::Lright => f,
        };
        let mut out = Vec::with_capacity(w);
        for &c in &cur {
            out.push(self.mux(overflow, fill, c));
        }
        out
    }

    /// Barrel rotator for a symbolic amount. A rotate is periodic in the width and never has a
    /// fill, so unlike a shift it has no out-of-range case to handle.
    fn rot_var_bits(&mut self, a: &[i32], k: &[i32], left: bool) -> Vec<i32> {
        let w = a.len();
        let f = self.f();
        let stages = shift_stages(w);
        if stages == 0 {
            return a.to_vec(); // width 1: every rotation is the identity
        }
        let amount: Vec<i32> = if (1usize << stages) == w {
            // Power-of-two width: the low `stages` bits already are `k mod w`.
            (0..stages)
                .map(|j| k.get(j).copied().unwrap_or(f))
                .collect()
        } else {
            // Otherwise reduce the amount modulo the width explicitly. The divider runs at the
            // wider operand rather than truncating `k` to `w` bits: `k mod w` depends on every bit
            // of `k`, and truncating is wrong whenever `w` does not divide `2^w` (w=5, k=32
            // truncates to 0, but 32 mod 5 = 2).
            let m = w.max(k.len());
            let mut kext = k.to_vec();
            kext.resize(m, f);
            let wc = self.const_bits(w as u64, m);
            let (_, r) = self.udivrem_bits(&kext, &wc);
            r[0..stages].to_vec() // r < w <= 2^stages, so it fits in `stages` bits
        };
        let mut cur = a.to_vec();
        // `amount` is exactly `stages` bits in both branches, so this walks the barrel one
        // stage at a time.
        for (j, &sel) in amount.iter().enumerate() {
            let sh = (1usize << j) % w;
            let mut rotated = Vec::with_capacity(w);
            for i in 0..w {
                rotated.push(if left {
                    cur[(i + w - sh) % w]
                } else {
                    cur[(i + sh) % w]
                });
            }
            cur = self.mux_vec(sel, &rotated, &cur);
        }
        cur
    }

    /// Require a 1-bit value to be true, as a unit clause on its single literal.
    ///
    /// # Panics
    ///
    /// Panics if `bv` is not 1 bit wide.
    pub fn assert_true(&mut self, bv: &Bv) {
        assert_eq!(
            bv.width(),
            1,
            "a constraint must be 1 bit, got {}",
            bv.width()
        );
        let bits = self.encode(bv);
        self.cnf.add_clause(&[bits[0]]);
    }
}

/// Which way a symbolic shift moves, and therefore what it fills vacated bits with.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShKind {
    Left,
    Lright,
    Aright,
}

/// The number of barrel stages needed to reach every amount below `w`: the smallest `s` with
/// `2^s >= w`. Width 1 needs none; 8 needs 3; 64 needs 6.
fn shift_stages(w: usize) -> usize {
    let mut stages = 0;
    let mut span = 1usize;
    while span < w {
        span <<= 1;
        stages += 1;
    }
    stages
}

/// Rotate a bit vector by a constant amount — a pure re-indexing, costing no gates.
fn rot_const(a: &[i32], k: u32, left: bool) -> Vec<i32> {
    let w = a.len();
    let sh = (k as usize) % w;
    (0..w)
        .map(|i| {
            if left {
                a[(i + w - sh) % w]
            } else {
                a[(i + sh) % w]
            }
        })
        .collect()
}
