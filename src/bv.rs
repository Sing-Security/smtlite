//! The bitvector expression DAG ([`Bv`]) that callers build and the blaster lowers.
//!
//! Nodes are `Rc`-shared so a value used twice is one subgraph, not two - the blaster
//! memoises on the `Rc` pointer, so shared subexpressions are encoded once. Widths are in
//! bits (1..=64); comparisons produce a 1-bit [`Bv`].
//!
//! Building an expression is cheap and never touches the solver: every operation here just
//! allocates a node. The formula is lowered to CNF only when a solve is asked for.

use std::fmt;
use std::rc::Rc;

/// Binary arithmetic/bitwise op (both operands the same width).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
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

/// Comparison op (both operands the same width; result is 1 bit).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cmp {
    Eq,
    Ne,
    Ult,
    Ule,
    Slt,
    Sle,
}

pub(crate) enum Node {
    Const(u32, u64),
    Var(u32, usize),
    Not(Bv),
    Neg(Bv),
    Bin(Op, Bv, Bv),
    ShlC(Bv, u32),
    ShrC(Bv, u32, bool), // right shift by a constant; bool = arithmetic (sign-filling)
    ShlV(Bv, Bv),        // shift left by a symbolic amount (k of any width)
    ShrV(Bv, Bv, bool),  // right by a symbolic amount; bool = arithmetic (sign-filling)
    RotC(Bv, u32, bool), // rotate by a constant; bool = toward the left
    RotV(Bv, Bv, bool),  // rotate by a symbolic amount; bool = toward the left
    Compare(Cmp, Bv, Bv),
    Zext(Bv, u32),
    Sext(Bv, u32),
    Extract(Bv, u32, u32), // hi, lo (inclusive)
    Concat(Bv, Bv),        // hi, lo
    Ite(Bv, Bv, Bv),       // cond(1), then, else
}

/// A bitvector value: a handle onto a shared expression node.
#[derive(Clone)]
pub struct Bv(pub(crate) Rc<Node>);

impl Bv {
    pub(crate) fn wrap(n: Node) -> Bv {
        match fold(&n) {
            Some(folded) => folded,
            None => Bv(Rc::new(n)),
        }
    }

    /// Do these two handles point at the SAME shared expression node? A cheap, recursion-free
    /// identity test - true when one value is reused (e.g. a size passed to both an allocator and a
    /// copy). Weaker than semantic equality (two structurally-identical-but-separately-built values
    /// return `false`), but it never blasts a formula, so a caller can use it as a fast pre-check.
    #[must_use]
    pub fn ptr_eq(&self, other: &Bv) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// The width of this value in bits.
    #[must_use]
    pub fn width(&self) -> u32 {
        // Iterative: width descends the left spine for the value-carrying nodes, so a deep chain
        // (a shift applied a hundred thousand times, say) reports its width without recursing.
        let mut node = &*self.0;
        loop {
            match node {
                Node::Const(w, _) | Node::Var(w, _) => return *w,
                Node::Not(a)
                | Node::Neg(a)
                | Node::Bin(_, a, _)
                | Node::ShlC(a, _)
                | Node::ShrC(a, _, _)
                | Node::ShlV(a, _)
                | Node::ShrV(a, _, _)
                | Node::RotC(a, _, _)
                | Node::RotV(a, _, _) => node = &*a.0,
                Node::Compare(_, _, _) => return 1,
                Node::Zext(_, w) | Node::Sext(_, w) => return *w,
                Node::Extract(_, hi, lo) => return hi - lo + 1,
                Node::Concat(a, b) => return a.width() + b.width(),
                Node::Ite(_, a, _) => return a.width(),
            }
        }
    }

    /// A width-`w` constant holding `v`. Bits of `v` at or above `w` are discarded, so
    /// `Bv::val(0x1ff, 8)` is `0xff`.
    ///
    /// # Panics
    ///
    /// Panics unless `w` is 1..=64 - see [`Solver::var`](crate::Solver::var).
    #[must_use]
    pub fn val(v: u64, w: u32) -> Bv {
        assert!(
            (1..=64).contains(&w),
            "constant width must be 1..=64, got {w}"
        );
        Bv::wrap(Node::Const(w, mask(v, w)))
    }

    /// The value as a plain number, if this handle *is* a constant.
    ///
    /// Constant expressions fold on construction, so this answers more often than "is this
    /// handle a `val`": `Bv::val(1, 8).add(&Bv::val(1, 8))` is already the constant `2`. An
    /// expression that mentions a variable still returns `None` here.
    #[must_use]
    pub fn as_const(&self) -> Option<u64> {
        if let Node::Const(_, v) = &*self.0 {
            Some(*v)
        } else {
            None
        }
    }

    /// Every variable id this expression references, deduplicated.
    ///
    /// Var ids are assigned in [`Solver::var`](crate::Solver::var) call order, so a caller can
    /// map them back to names to ask what a value actually depends on. The walk is iterative
    /// and dedups on node identity, so it terminates on any DAG however deep.
    #[must_use]
    pub fn var_ids(&self) -> Vec<usize> {
        // Iterative: a deep expression chain would overflow the stack. `seen` dedups by node
        // identity, so a shared subgraph is walked once.
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::new();
        let mut stack = vec![self.clone()];
        while let Some(cur) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&cur.0) as usize) {
                continue;
            }
            match &*cur.0 {
                Node::Const(_, _) => {}
                Node::Var(_, id) => ids.push(*id),
                Node::Not(a)
                | Node::Neg(a)
                | Node::ShlC(a, _)
                | Node::ShrC(a, _, _)
                | Node::RotC(a, _, _)
                | Node::Zext(a, _)
                | Node::Sext(a, _)
                | Node::Extract(a, _, _) => stack.push(a.clone()),
                Node::Bin(_, a, b)
                | Node::Compare(_, a, b)
                | Node::Concat(a, b)
                | Node::ShlV(a, b)
                | Node::ShrV(a, b, _)
                | Node::RotV(a, b, _) => {
                    stack.push(a.clone());
                    stack.push(b.clone());
                }
                Node::Ite(a, b, c) => {
                    stack.push(a.clone());
                    stack.push(b.clone());
                    stack.push(c.clone());
                }
            }
        }
        ids
    }

    /// Bitwise NOT.
    #[must_use]
    pub fn not(&self) -> Bv {
        Bv::wrap(Node::Not(self.clone()))
    }

    /// Two's-complement negation (`0 - self`, `0 - 1` at the width's bit count).
    #[must_use]
    pub fn neg(&self) -> Bv {
        Bv::wrap(Node::Neg(self.clone()))
    }

    /// Bitwise AND of two equal-width values.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width - see the [crate-level note](crate#width-and-threading-limits).
    #[must_use]
    pub fn and(&self, o: &Bv) -> Bv {
        self.bin(Op::And, o)
    }

    /// Bitwise OR of two equal-width values.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn or(&self, o: &Bv) -> Bv {
        self.bin(Op::Or, o)
    }

    /// Bitwise exclusive OR of two equal-width values.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn xor(&self, o: &Bv) -> Bv {
        self.bin(Op::Xor, o)
    }

    /// Wrapping addition. The carry out of the top bit is discarded, which is what makes
    /// `x.add(&Bv::val(1, w)).ult(&x)` a test for `x` at its maximum.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn add(&self, o: &Bv) -> Bv {
        self.bin(Op::Add, o)
    }

    /// Wrapping subtraction, i.e. addition of the two's-complement negation.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn sub(&self, o: &Bv) -> Bv {
        self.bin(Op::Sub, o)
    }

    /// Wrapping multiplication, truncated to the operand width.
    ///
    /// This is the expensive one: bit-blasting cost is quadratic in the width, so a 64-bit
    /// product is a thousand-odd gates per bit pair.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn mul(&self, o: &Bv) -> Bv {
        self.bin(Op::Mul, o)
    }

    /// Unsigned division, truncated. Division by zero yields all-ones (SMT-LIB `bvudiv`).
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn udiv(&self, o: &Bv) -> Bv {
        self.bin(Op::Udiv, o)
    }

    /// Unsigned remainder. A zero divisor yields the dividend (SMT-LIB `bvurem`).
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn urem(&self, o: &Bv) -> Bv {
        self.bin(Op::Urem, o)
    }

    /// Signed division, truncated toward zero. A zero divisor yields all-ones when the dividend
    /// is non-negative and `1` otherwise (SMT-LIB `bvsdiv`). `MIN / -1` wraps to `MIN`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn sdiv(&self, o: &Bv) -> Bv {
        self.bin(Op::Sdiv, o)
    }

    /// Signed remainder taking the sign of the **dividend** - this is C's `%` and Rust's `%`
    /// (SMT-LIB `bvsrem`, *not* `bvsmod`, which takes the sign of the divisor). A zero divisor
    /// yields the dividend.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn srem(&self, o: &Bv) -> Bv {
        self.bin(Op::Srem, o)
    }

    fn bin(&self, op: Op, o: &Bv) -> Bv {
        assert_eq!(
            self.width(),
            o.width(),
            "bitvector op needs equal widths, got {} and {}",
            self.width(),
            o.width()
        );
        Bv::wrap(Node::Bin(op, self.clone(), o.clone()))
    }

    /// Shift left by a constant `k`; vacated bits are zero.
    ///
    /// A shift empties rather than wraps: a `k` at or beyond the width yields zero
    /// (SMT-LIB `bvshl`). Contrast [`rotl`](Bv::rotl), which does wrap.
    #[must_use]
    pub fn shl(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShlC(self.clone(), k))
    }

    /// Logical (zero-filling) right shift by a constant `k`.
    #[must_use]
    pub fn lshr(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShrC(self.clone(), k, false))
    }

    /// Arithmetic (sign-filling) right shift by a constant `k`: vacated bits take the value of
    /// the sign bit, so the result keeps rounding toward negative infinity.
    #[must_use]
    pub fn ashr(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShrC(self.clone(), k, true))
    }

    /// Shift left by a **symbolic** amount. `k` may be any width; when its value is at least
    /// this value's width the result is zero (SMT-LIB `bvshl` - a shift is *not* a rotate).
    #[must_use]
    pub fn shl_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::ShlV(self.clone(), k.clone()))
    }
    /// Logical (zero-filling) right shift by a symbolic amount.
    #[must_use]
    pub fn lshr_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::ShrV(self.clone(), k.clone(), false))
    }
    /// Arithmetic (sign-filling) right shift by a symbolic amount.
    #[must_use]
    pub fn ashr_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::ShrV(self.clone(), k.clone(), true))
    }

    /// Rotate left by a constant. A rotation is a pure bit permutation, so unlike a shift it
    /// costs no gates at all. Amounts are taken modulo the width.
    #[must_use]
    pub fn rotl(&self, k: u32) -> Bv {
        Bv::wrap(Node::RotC(self.clone(), k, true))
    }
    /// Rotate right by a constant (a pure bit permutation).
    #[must_use]
    pub fn rotr(&self, k: u32) -> Bv {
        Bv::wrap(Node::RotC(self.clone(), k, false))
    }
    /// Rotate left by a symbolic amount. A rotate is periodic in the width, so the amount is
    /// reduced modulo the width - which for a non-power-of-two width costs a division.
    #[must_use]
    pub fn rotl_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::RotV(self.clone(), k.clone(), true))
    }
    /// Rotate right by a symbolic amount.
    #[must_use]
    pub fn rotr_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::RotV(self.clone(), k.clone(), false))
    }

    /// Equality, as a 1-bit value: `self == o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn eq(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Eq, o)
    }

    /// Inequality, as a 1-bit value.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn ne(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ne, o)
    }

    /// Unsigned `self < o`, as a 1-bit value.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn ult(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ult, o)
    }

    /// Unsigned `self <= o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn ule(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ule, o)
    }

    /// Unsigned `self > o`. Note the sense: at a fixed width, a spilled size is *smaller* than
    /// the element it should have held, so overflow usually shows up as `lt`, not `gt`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn ugt(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Ult, self)
    }

    /// Unsigned `self >= o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn uge(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Ule, self)
    }

    /// Signed `self < o`, treating the top bit as a sign.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn slt(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Slt, o)
    }

    /// Signed `self <= o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn sle(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Sle, o)
    }

    /// Signed `self > o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn sgt(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Slt, self)
    }

    /// Signed `self >= o`.
    ///
    /// # Panics
    ///
    /// Panics if `o` has a different width.
    #[must_use]
    pub fn sge(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Sle, self)
    }

    fn cmp(&self, c: Cmp, o: &Bv) -> Bv {
        assert_eq!(
            self.width(),
            o.width(),
            "comparison needs equal widths, got {} and {}",
            self.width(),
            o.width()
        );
        Bv::wrap(Node::Compare(c, self.clone(), o.clone()))
    }

    /// Zero-extend to `new_w` bits: pad with zeros, so an unsigned value is unchanged.
    ///
    /// # Panics
    ///
    /// Panics unless `new_w` is between this value's width and 64. Narrowing is
    /// [`extract`](Bv::extract)'s job, and 64 is the ceiling a model can read back.
    #[must_use]
    pub fn zext(&self, new_w: u32) -> Bv {
        assert!(
            new_w >= self.width(),
            "zext must widen: {} to {new_w}",
            self.width()
        );
        assert!(new_w <= 64, "width must be 1..=64, got {new_w}");
        Bv::wrap(Node::Zext(self.clone(), new_w))
    }

    /// Sign-extend to `new_w` bits: pad with copies of the sign bit, so a two's-complement
    /// value is unchanged.
    ///
    /// # Panics
    ///
    /// Panics unless `new_w` is between this value's width and 64.
    #[must_use]
    pub fn sext(&self, new_w: u32) -> Bv {
        assert!(
            new_w >= self.width(),
            "sext must widen: {} to {new_w}",
            self.width()
        );
        assert!(new_w <= 64, "width must be 1..=64, got {new_w}");
        Bv::wrap(Node::Sext(self.clone(), new_w))
    }

    /// Bits `lo` through `hi` inclusive, as a `hi - lo + 1`-bit value. Bits are little-endian,
    /// so bit 0 is the least significant.
    ///
    /// # Panics
    ///
    /// Panics unless `lo <= hi` and `hi` is a bit this value actually has
    /// (`hi < width`). The result width is therefore always within 1..=64.
    #[must_use]
    pub fn extract(&self, hi: u32, lo: u32) -> Bv {
        assert!(
            hi >= lo && hi < self.width(),
            "extract[{hi}:{lo}] out of range for width {}",
            self.width()
        );
        Bv::wrap(Node::Extract(self.clone(), hi, lo))
    }

    /// Concatenate: `self` becomes the high bits and `low` the low bits.
    ///
    /// # Panics
    ///
    /// Panics if the combined width exceeds 64.
    #[must_use]
    pub fn concat(&self, low: &Bv) -> Bv {
        let w = self.width() + low.width();
        assert!(
            w <= 64,
            "concatenation is {w} bits wide, over the 64-bit limit"
        );
        Bv::wrap(Node::Concat(self.clone(), low.clone()))
    }

    /// `cond ? then : els`, the one branch in the expression language.
    ///
    /// Combine with a comparison to build a case split: `Bv::ite(&x.ult(&bound), &safe, &unsafe)`.
    ///
    /// # Panics
    ///
    /// Panics if `cond` is not 1 bit wide, or if `then` and `els` differ in width.
    #[must_use]
    pub fn ite(cond: &Bv, then: &Bv, els: &Bv) -> Bv {
        assert_eq!(
            cond.width(),
            1,
            "ite condition must be 1 bit, got {}",
            cond.width()
        );
        assert_eq!(
            then.width(),
            els.width(),
            "ite branches must match: {} and {}",
            then.width(),
            els.width()
        );
        Bv::wrap(Node::Ite(cond.clone(), then.clone(), els.clone()))
    }

    /// Logical AND of two 1-bit values - the readable way to conjoin constraints
    /// ([`and`](Bv::and) does the same thing, but does not check that its operands really are
    /// propositions).
    ///
    /// # Panics
    ///
    /// Panics if either operand is wider than 1 bit.
    #[must_use]
    pub fn land(&self, o: &Bv) -> Bv {
        assert_eq!(
            self.width(),
            1,
            "land operand must be 1 bit, got {}",
            self.width()
        );
        assert_eq!(
            o.width(),
            1,
            "land operand must be 1 bit, got {}",
            o.width()
        );
        self.and(o)
    }
}

/// Truncate `v` to its low `w` bits. `w` at or above 64 leaves `v` unchanged, since there are
/// no more bits to drop.
#[must_use]
pub fn mask(v: u64, w: u32) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}

// ---------------------------------------------------------------------------------------------
// Simplification. Every combinator routes through `Bv::wrap`, so folding here - before the node
// is `Rc`-shared - means a constant expression is one node and a rewrite that returns an existing
// child keeps that child's identity, which is what the blaster's memoisation keys on.
// ---------------------------------------------------------------------------------------------

/// Reduce `n` to a constant or to an existing child, if it obviously is one. Returns `None` when
/// the node stands as built. Never recurses into children: it only peeks at the immediate
/// operands, so a deep chain costs one node's work here.
fn fold(n: &Node) -> Option<Bv> {
    match n {
        Node::Const(_, _) | Node::Var(_, _) => None,
        Node::Not(a) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(a.width(), mask(!v, a.width())));
            }
            if let Node::Not(b) = &*a.0 {
                return Some(b.clone());
            }
            None
        }
        Node::Neg(a) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(a.width(), mask(0u64.wrapping_sub(v), a.width())));
            }
            if let Node::Neg(b) = &*a.0 {
                return Some(b.clone());
            }
            None
        }
        Node::Bin(op, a, b) => fold_bin(*op, a, b),
        Node::ShlC(a, k) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(a.width(), shl_(v, u64::from(*k), a.width())));
            }
            if *k == 0 {
                return Some(a.clone());
            }
            None
        }
        Node::ShrC(a, k, arith) => {
            if let Some(v) = a.as_const() {
                let w = a.width();
                return Some(const_of(w, shr_(v, u64::from(*k), w, *arith)));
            }
            if *k == 0 {
                return Some(a.clone());
            }
            None
        }
        Node::ShlV(a, k) => {
            if let Some(kv) = k.as_const() {
                if let Some(v) = a.as_const() {
                    return Some(const_of(a.width(), shl_(v, kv, a.width())));
                }
                if kv == 0 {
                    return Some(a.clone());
                }
            }
            None
        }
        Node::ShrV(a, k, arith) => {
            if let Some(kv) = k.as_const() {
                if let Some(v) = a.as_const() {
                    let w = a.width();
                    return Some(const_of(w, shr_(v, kv, w, *arith)));
                }
                if kv == 0 {
                    return Some(a.clone());
                }
            }
            None
        }
        Node::RotC(a, k, left) => {
            if let Some(v) = a.as_const() {
                let w = a.width();
                return Some(const_of(w, rot_(v, u64::from(*k), w, *left)));
            }
            if *k == 0 {
                return Some(a.clone());
            }
            None
        }
        Node::RotV(a, k, left) => {
            if let Some(kv) = k.as_const() {
                if let Some(v) = a.as_const() {
                    let w = a.width();
                    return Some(const_of(w, rot_(v, kv, w, *left)));
                }
                if kv == 0 {
                    return Some(a.clone());
                }
            }
            None
        }
        Node::Compare(c, a, b) => fold_cmp(*c, a, b),
        Node::Zext(a, w) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(*w, v));
            }
            if *w == a.width() {
                return Some(a.clone());
            }
            None
        }
        Node::Sext(a, w) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(*w, mask(signed(v, a.width()) as u64, *w)));
            }
            if *w == a.width() {
                return Some(a.clone());
            }
            None
        }
        Node::Extract(a, hi, lo) => {
            if let Some(v) = a.as_const() {
                return Some(const_of(*hi - *lo + 1, mask(v >> *lo, *hi - *lo + 1)));
            }
            if *lo == 0 && *hi == a.width() - 1 {
                return Some(a.clone());
            }
            None
        }
        Node::Concat(a, b) => {
            if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
                let w = a.width() + b.width();
                return Some(const_of(w, mask((x << b.width()) | y, w)));
            }
            None
        }
        Node::Ite(c, a, b) => {
            if a.ptr_eq(b) {
                return Some(a.clone());
            }
            if let Some(cv) = c.as_const() {
                return Some(if cv != 0 { a.clone() } else { b.clone() });
            }
            None
        }
    }
}

/// Fold a binary op: both operands constant, one operand an identity, or an operand repeated.
fn fold_bin(op: Op, a: &Bv, b: &Bv) -> Option<Bv> {
    let w = a.width();
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return Some(const_of(w, eval_bin(op, x, y, w)));
    }
    let all = mask(u64::MAX, w);
    match op {
        Op::Add => {
            if a.as_const() == Some(0) {
                return Some(b.clone());
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
        }
        Op::Sub => {
            if a.ptr_eq(b) {
                return Some(const_of(w, 0));
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
        }
        Op::Mul => {
            if a.as_const() == Some(0) || b.as_const() == Some(0) {
                return Some(const_of(w, 0));
            }
            if a.as_const() == Some(1) {
                return Some(b.clone());
            }
            if b.as_const() == Some(1) {
                return Some(a.clone());
            }
        }
        Op::And => {
            if a.ptr_eq(b) {
                return Some(a.clone());
            }
            if a.as_const() == Some(0) || b.as_const() == Some(0) {
                return Some(const_of(w, 0));
            }
            if a.as_const() == Some(all) {
                return Some(b.clone());
            }
            if b.as_const() == Some(all) {
                return Some(a.clone());
            }
        }
        Op::Or => {
            if a.ptr_eq(b) {
                return Some(a.clone());
            }
            if a.as_const() == Some(0) {
                return Some(b.clone());
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
            if a.as_const() == Some(all) || b.as_const() == Some(all) {
                return Some(const_of(w, all));
            }
        }
        Op::Xor => {
            if a.ptr_eq(b) {
                return Some(const_of(w, 0));
            }
            if a.as_const() == Some(0) {
                return Some(b.clone());
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
            if a.as_const() == Some(all) {
                return Some(b.not());
            }
            if b.as_const() == Some(all) {
                return Some(a.not());
            }
        }
        Op::Udiv => {
            if b.as_const() == Some(1) {
                return Some(a.clone());
            }
            if b.as_const() == Some(0) {
                return Some(const_of(w, all));
            }
        }
        Op::Urem => {
            if a.ptr_eq(b) {
                return Some(const_of(w, 0));
            }
            if b.as_const() == Some(1) {
                return Some(const_of(w, 0));
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
        }
        Op::Sdiv => {
            if b.as_const() == Some(1) {
                return Some(a.clone());
            }
        }
        Op::Srem => {
            if a.ptr_eq(b) {
                return Some(const_of(w, 0));
            }
            if b.as_const() == Some(1) {
                return Some(const_of(w, 0));
            }
            if b.as_const() == Some(0) {
                return Some(a.clone());
            }
        }
    }
    None
}

/// Fold a comparison: both operands constant, a value compared with itself, or a symmetric
/// operation compared against its commuted twin (`a + b == b + a`).
fn fold_cmp(c: Cmp, a: &Bv, b: &Bv) -> Option<Bv> {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        let w = a.width();
        let bit = match c {
            Cmp::Eq => x == y,
            Cmp::Ne => x != y,
            Cmp::Ult => x < y,
            Cmp::Ule => x <= y,
            Cmp::Slt => signed(x, w) < signed(y, w),
            Cmp::Sle => signed(x, w) <= signed(y, w),
        };
        return Some(const_of(1, u64::from(bit)));
    }
    if a.ptr_eq(b) {
        return Some(const_of(
            1,
            match c {
                Cmp::Eq | Cmp::Ule | Cmp::Sle => 1,
                Cmp::Ne | Cmp::Ult | Cmp::Slt => 0,
            },
        ));
    }
    if matches!(c, Cmp::Eq | Cmp::Ne) {
        if let (Node::Bin(op1, p, q), Node::Bin(op2, r, s)) = (&*a.0, &*b.0) {
            if op1 == op2
                && is_commutative(*op1)
                && ((p.ptr_eq(r) && q.ptr_eq(s)) || (p.ptr_eq(s) && q.ptr_eq(r)))
            {
                return Some(const_of(1, u64::from(c == Cmp::Eq)));
            }
        }
    }
    None
}

fn is_commutative(op: Op) -> bool {
    matches!(op, Op::And | Op::Or | Op::Xor | Op::Add | Op::Mul)
}

/// A fresh constant node, value masked to `w`. Bypasses `wrap` so folding cannot recurse.
fn const_of(w: u32, v: u64) -> Bv {
    Bv(Rc::new(Node::Const(w, mask(v, w))))
}

fn eval_bin(op: Op, x: u64, y: u64, w: u32) -> u64 {
    match op {
        Op::And => mask(x & y, w),
        Op::Or => mask(x | y, w),
        Op::Xor => mask(x ^ y, w),
        Op::Add => mask(x.wrapping_add(y), w),
        Op::Sub => mask(x.wrapping_sub(y), w),
        Op::Mul => mask(x.wrapping_mul(y), w),
        Op::Udiv => udiv_(x, y, w),
        Op::Urem => urem_(x, y, w),
        Op::Sdiv => sdiv_(x, y, w),
        Op::Srem => srem_(x, y, w),
    }
}

/// The `w`-bit two's-complement value of `x`, as a signed integer.
fn signed(x: u64, w: u32) -> i128 {
    let x = mask(x, w);
    if (x >> (w - 1)) & 1 == 1 {
        x as i128 - (1i128 << w)
    } else {
        x as i128
    }
}

fn shl_(x: u64, k: u64, w: u32) -> u64 {
    if k >= u64::from(w) {
        0
    } else {
        mask(x << k, w)
    }
}

fn lshr_(x: u64, k: u64, w: u32) -> u64 {
    if k >= u64::from(w) {
        0
    } else {
        mask(x, w) >> k
    }
}

fn ashr_(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    if k >= u64::from(w) {
        if (x >> (w - 1)) & 1 == 1 {
            mask(u64::MAX, w)
        } else {
            0
        }
    } else {
        mask((signed(x, w) >> k) as u64, w)
    }
}

fn shr_(x: u64, k: u64, w: u32, arith: bool) -> u64 {
    if arith {
        ashr_(x, k, w)
    } else {
        lshr_(x, k, w)
    }
}

fn rotl_(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    let k = (k % u64::from(w)) as u32;
    if k == 0 {
        x
    } else {
        mask((x << k) | (x >> (w - k)), w)
    }
}

fn rotr_(x: u64, k: u64, w: u32) -> u64 {
    let x = mask(x, w);
    let k = (k % u64::from(w)) as u32;
    if k == 0 {
        x
    } else {
        mask((x >> k) | (x << (w - k)), w)
    }
}

fn rot_(x: u64, k: u64, w: u32, left: bool) -> u64 {
    if left { rotl_(x, k, w) } else { rotr_(x, k, w) }
}

fn udiv_(x: u64, y: u64, w: u32) -> u64 {
    x.checked_div(y).unwrap_or_else(|| mask(u64::MAX, w))
}

fn urem_(x: u64, y: u64, _w: u32) -> u64 {
    x.checked_rem(y).unwrap_or(x)
}

fn sdiv_(x: u64, y: u64, w: u32) -> u64 {
    let (a, b) = (signed(x, w), signed(y, w));
    // `a` and `b` sit in [-2^63, 2^63-1], so `checked_div` is `None` exactly when the divisor is
    // zero - the `MIN / -1` overflow cannot arise at these widths.
    match a.checked_div(b) {
        Some(q) => mask(q as u64, w),
        None => {
            if a >= 0 {
                mask(u64::MAX, w)
            } else {
                1
            }
        }
    }
}

fn srem_(x: u64, y: u64, w: u32) -> u64 {
    let (a, b) = (signed(x, w), signed(y, w));
    match a.checked_rem(b) {
        Some(r) => mask(r as u64, w),
        None => mask(x, w),
    }
}

impl fmt::Debug for Bv {
    // `Display`'s s-expression, wrapped so a debug print can tell the two apart.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bv({self})")
    }
}

impl fmt::Display for Bv {
    // Canonical s-expression: structurally identical expressions print identically, so this is
    // stable enough to key a memory slot by.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            Node::Const(w, v) => write!(f, "#{v:x}:{w}"),
            Node::Var(w, id) => write!(f, "v{id}:{w}"),
            Node::Not(a) => write!(f, "(~ {a})"),
            Node::Neg(a) => write!(f, "(- {a})"),
            Node::Bin(op, a, b) => {
                let s = match op {
                    Op::And => "&",
                    Op::Or => "|",
                    Op::Xor => "^",
                    Op::Add => "+",
                    Op::Sub => "-",
                    Op::Mul => "*",
                    Op::Udiv => "/u",
                    Op::Urem => "%u",
                    Op::Sdiv => "/s",
                    Op::Srem => "%s",
                };
                write!(f, "({s} {a} {b})")
            }
            Node::ShlC(a, k) => write!(f, "(<< {a} {k})"),
            Node::ShrC(a, k, arith) => {
                write!(f, "({} {a} {k})", if *arith { ">>s" } else { ">>u" })
            }
            Node::ShlV(a, k) => write!(f, "(<<v {a} {k})"),
            Node::ShrV(a, k, arith) => {
                write!(f, "({}v {a} {k})", if *arith { ">>s" } else { ">>u" })
            }
            Node::RotC(a, k, left) => {
                write!(f, "({} {a} {k})", if *left { "rotl" } else { "rotr" })
            }
            Node::RotV(a, k, left) => {
                write!(f, "({}v {a} {k})", if *left { "rotl" } else { "rotr" })
            }
            Node::Compare(c, a, b) => {
                let s = match c {
                    Cmp::Eq => "==",
                    Cmp::Ne => "!=",
                    Cmp::Ult => "<u",
                    Cmp::Ule => "<=u",
                    Cmp::Slt => "<s",
                    Cmp::Sle => "<=s",
                };
                write!(f, "({s} {a} {b})")
            }
            Node::Zext(a, w) => write!(f, "(zext{w} {a})"),
            Node::Sext(a, w) => write!(f, "(sext{w} {a})"),
            Node::Extract(a, hi, lo) => write!(f, "(ext[{hi}:{lo}] {a})"),
            Node::Concat(a, b) => write!(f, "(cat {a} {b})"),
            Node::Ite(c, a, b) => write!(f, "(ite {c} {a} {b})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(w: u32) -> Bv {
        Bv::wrap(Node::Var(w, 0))
    }

    #[test]
    fn constant_arithmetic_folds() {
        assert_eq!(Bv::val(1, 8).add(&Bv::val(1, 8)).as_const(), Some(2));
        assert_eq!(Bv::val(250, 8).add(&Bv::val(10, 8)).as_const(), Some(4)); // wraps
        assert_eq!(Bv::val(9, 8).udiv(&Bv::val(0, 8)).as_const(), Some(0xff));
        assert_eq!(Bv::val(9, 8).urem(&Bv::val(0, 8)).as_const(), Some(9));
        // Signed division by zero: all-ones for a non-negative dividend, 1 otherwise.
        assert_eq!(Bv::val(9, 8).sdiv(&Bv::val(0, 8)).as_const(), Some(0xff));
        assert_eq!(Bv::val(0x80, 8).sdiv(&Bv::val(0, 8)).as_const(), Some(1));
        // Arithmetic shift of a negative value fills with ones.
        assert_eq!(Bv::val(0x80, 8).ashr(2).as_const(), Some(0xe0));
        // Comparison of constants.
        assert_eq!(Bv::val(3, 8).ult(&Bv::val(4, 8)).as_const(), Some(1));
        assert_eq!(Bv::val(3, 8).slt(&Bv::val(0xff, 8)).as_const(), Some(0)); // 3 < -1 is false
    }

    #[test]
    fn identities_return_the_existing_child() {
        let x = var(8);
        assert!(x.add(&Bv::val(0, 8)).ptr_eq(&x));
        assert!(Bv::val(0, 8).add(&x).ptr_eq(&x));
        assert!(x.sub(&Bv::val(0, 8)).ptr_eq(&x));
        assert!(x.mul(&Bv::val(1, 8)).ptr_eq(&x));
        assert!(x.and(&Bv::val(0xff, 8)).ptr_eq(&x));
        assert!(x.or(&Bv::val(0, 8)).ptr_eq(&x));
        assert!(x.xor(&Bv::val(0, 8)).ptr_eq(&x));
        assert!(x.shl(0).ptr_eq(&x));
        assert!(x.rotl(0).ptr_eq(&x));
        assert!(x.neg().neg().ptr_eq(&x));
        assert!(x.not().not().ptr_eq(&x));
    }

    #[test]
    fn identities_return_constants() {
        let x = var(8);
        assert_eq!(x.sub(&x).as_const(), Some(0));
        assert_eq!(x.xor(&x).as_const(), Some(0));
        assert_eq!(x.mul(&Bv::val(0, 8)).as_const(), Some(0));
        assert_eq!(x.and(&Bv::val(0, 8)).as_const(), Some(0));
        assert_eq!(x.or(&Bv::val(0xff, 8)).as_const(), Some(0xff));
        assert_eq!(x.udiv(&Bv::val(0, 8)).as_const(), Some(0xff));
        assert_eq!(x.urem(&Bv::val(1, 8)).as_const(), Some(0));
    }

    #[test]
    fn a_value_compared_with_itself_folds() {
        let x = var(8);
        assert_eq!(x.eq(&x).as_const(), Some(1));
        assert_eq!(x.ne(&x).as_const(), Some(0));
        assert_eq!(x.ult(&x).as_const(), Some(0));
        assert_eq!(x.ule(&x).as_const(), Some(1));
        assert_eq!(x.slt(&x).as_const(), Some(0));
        assert_eq!(x.sle(&x).as_const(), Some(1));
    }

    #[test]
    fn a_symmetric_operation_equals_its_commuted_twin() {
        let a = var(8);
        let b = var(8);
        assert_eq!(a.add(&b).eq(&b.add(&a)).as_const(), Some(1));
        assert_eq!(a.mul(&b).eq(&b.mul(&a)).as_const(), Some(1));
        assert_eq!(a.and(&b).eq(&b.and(&a)).as_const(), Some(1));
        assert_eq!(a.add(&b).ne(&b.add(&a)).as_const(), Some(0));
    }
}
