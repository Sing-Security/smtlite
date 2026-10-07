//! The bitvector expression DAG (`Bv`) the executor builds and the blaster lowers.
//!
//! Nodes are `Rc`-shared so a value used twice is one subgraph, not two — the blaster
//! memoises on the `Rc` pointer, so shared subexpressions are encoded once. Widths are in
//! bits (1..=64); comparisons produce a 1-bit `Bv`.

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
    ShrC(Bv, u32, bool), // arithmetic?
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
        Bv(Rc::new(n))
    }

    /// Do these two handles point at the SAME shared expression node? A cheap, recursion-free
    /// identity test — true when one value is reused (e.g. a size passed to both an allocator and a
    /// copy). Weaker than semantic equality (two structurally-identical-but-separately-built values
    /// return `false`), but it never blasts a formula, so a caller can use it as a fast pre-check.
    #[must_use]
    pub fn ptr_eq(&self, other: &Bv) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// The width of this value in bits.
    #[must_use]
    pub fn width(&self) -> u32 {
        match &*self.0 {
            Node::Const(w, _) | Node::Var(w, _) => *w,
            Node::Not(a) | Node::Neg(a) | Node::Bin(_, a, _) => a.width(),
            Node::ShlC(a, _) | Node::ShrC(a, _, _) => a.width(),
            Node::ShlV(a, _) | Node::ShrV(a, _, _) | Node::RotC(a, _, _) | Node::RotV(a, _, _) => a.width(),
            Node::Compare(_, _, _) => 1,
            Node::Zext(_, w) | Node::Sext(_, w) => *w,
            Node::Extract(_, hi, lo) => hi - lo + 1,
            Node::Concat(a, b) => a.width() + b.width(),
            Node::Ite(_, a, _) => a.width(),
        }
    }

    /// A width-`w` constant.
    #[must_use]
    pub fn val(v: u64, w: u32) -> Bv {
        Bv::wrap(Node::Const(w, mask(v, w)))
    }

    /// The concrete value, if this is literally a constant (not an expression) — used by the
    /// executor to resolve a concrete indirect-call target.
    #[must_use]
    pub fn as_const(&self) -> Option<u64> {
        if let Node::Const(_, v) = &*self.0 { Some(*v) } else { None }
    }

    /// Every variable id this expression references (deduped) — so a caller can tell what a
    /// value actually depends on (a loaded field vs an untouched register).
    #[must_use]
    pub fn var_ids(&self) -> Vec<usize> {
        // Iterative worklist, not recursion: a deep expression chain (e.g. a long dataflow of a
        // pointer through many ops) would otherwise overflow the stack. `seen` (by Rc identity)
        // dedups shared subgraphs so a DAG is walked once.
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
                Node::Not(a) | Node::Neg(a) | Node::ShlC(a, _) | Node::ShrC(a, _, _) | Node::RotC(a, _, _)
                | Node::Zext(a, _) | Node::Sext(a, _) | Node::Extract(a, _, _) => stack.push(a.clone()),
                Node::Bin(_, a, b) | Node::Compare(_, a, b) | Node::Concat(a, b) | Node::ShlV(a, b)
                | Node::ShrV(a, b, _) | Node::RotV(a, b, _) => {
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

    #[must_use]
    pub fn not(&self) -> Bv {
        Bv::wrap(Node::Not(self.clone()))
    }
    #[must_use]
    pub fn neg(&self) -> Bv {
        Bv::wrap(Node::Neg(self.clone()))
    }

    #[must_use]
    pub fn and(&self, o: &Bv) -> Bv {
        self.bin(Op::And, o)
    }
    #[must_use]
    pub fn or(&self, o: &Bv) -> Bv {
        self.bin(Op::Or, o)
    }
    #[must_use]
    pub fn xor(&self, o: &Bv) -> Bv {
        self.bin(Op::Xor, o)
    }
    #[must_use]
    pub fn add(&self, o: &Bv) -> Bv {
        self.bin(Op::Add, o)
    }
    #[must_use]
    pub fn sub(&self, o: &Bv) -> Bv {
        self.bin(Op::Sub, o)
    }
    #[must_use]
    pub fn mul(&self, o: &Bv) -> Bv {
        self.bin(Op::Mul, o)
    }

    /// Unsigned division, truncated. Division by zero yields all-ones (SMT-LIB `bvudiv`).
    #[must_use]
    pub fn udiv(&self, o: &Bv) -> Bv {
        self.bin(Op::Udiv, o)
    }
    /// Unsigned remainder. A zero divisor yields the dividend (SMT-LIB `bvurem`).
    #[must_use]
    pub fn urem(&self, o: &Bv) -> Bv {
        self.bin(Op::Urem, o)
    }
    /// Signed division, truncated toward zero. A zero divisor yields all-ones when the dividend
    /// is non-negative and `1` otherwise (SMT-LIB `bvsdiv`). `MIN / -1` wraps to `MIN`.
    #[must_use]
    pub fn sdiv(&self, o: &Bv) -> Bv {
        self.bin(Op::Sdiv, o)
    }
    /// Signed remainder taking the sign of the **dividend** — this is C's `%` and Rust's `%`
    /// (SMT-LIB `bvsrem`, *not* `bvsmod`, which takes the sign of the divisor). A zero divisor
    /// yields the dividend.
    #[must_use]
    pub fn srem(&self, o: &Bv) -> Bv {
        self.bin(Op::Srem, o)
    }

    fn bin(&self, op: Op, o: &Bv) -> Bv {
        debug_assert_eq!(self.width(), o.width(), "bitvector op width mismatch");
        Bv::wrap(Node::Bin(op, self.clone(), o.clone()))
    }

    #[must_use]
    pub fn shl(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShlC(self.clone(), k))
    }
    #[must_use]
    pub fn lshr(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShrC(self.clone(), k, false))
    }
    #[must_use]
    pub fn ashr(&self, k: u32) -> Bv {
        Bv::wrap(Node::ShrC(self.clone(), k, true))
    }

    /// Shift left by a **symbolic** amount. `k` may be any width; when its value is at least
    /// this value's width the result is zero (SMT-LIB `bvshl` — a shift is *not* a rotate).
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
    /// reduced modulo the width — which for a non-power-of-two width costs a division.
    #[must_use]
    pub fn rotl_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::RotV(self.clone(), k.clone(), true))
    }
    /// Rotate right by a symbolic amount.
    #[must_use]
    pub fn rotr_var(&self, k: &Bv) -> Bv {
        Bv::wrap(Node::RotV(self.clone(), k.clone(), false))
    }

    #[must_use]
    pub fn eq(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Eq, o)
    }
    #[must_use]
    pub fn ne(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ne, o)
    }
    #[must_use]
    pub fn ult(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ult, o)
    }
    #[must_use]
    pub fn ule(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Ule, o)
    }
    #[must_use]
    pub fn ugt(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Ult, self)
    }
    #[must_use]
    pub fn uge(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Ule, self)
    }
    #[must_use]
    pub fn slt(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Slt, o)
    }
    #[must_use]
    pub fn sle(&self, o: &Bv) -> Bv {
        self.cmp(Cmp::Sle, o)
    }
    #[must_use]
    pub fn sgt(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Slt, self)
    }
    #[must_use]
    pub fn sge(&self, o: &Bv) -> Bv {
        o.cmp(Cmp::Sle, self)
    }

    fn cmp(&self, c: Cmp, o: &Bv) -> Bv {
        debug_assert_eq!(self.width(), o.width(), "comparison width mismatch");
        Bv::wrap(Node::Compare(c, self.clone(), o.clone()))
    }

    #[must_use]
    pub fn zext(&self, new_w: u32) -> Bv {
        debug_assert!(new_w >= self.width());
        Bv::wrap(Node::Zext(self.clone(), new_w))
    }
    #[must_use]
    pub fn sext(&self, new_w: u32) -> Bv {
        debug_assert!(new_w >= self.width());
        Bv::wrap(Node::Sext(self.clone(), new_w))
    }
    #[must_use]
    pub fn extract(&self, hi: u32, lo: u32) -> Bv {
        debug_assert!(hi >= lo && hi < self.width());
        Bv::wrap(Node::Extract(self.clone(), hi, lo))
    }
    #[must_use]
    pub fn concat(&self, low: &Bv) -> Bv {
        Bv::wrap(Node::Concat(self.clone(), low.clone()))
    }

    /// `cond ? self : other` — `cond` must be 1 bit, `self`/`other` the same width.
    #[must_use]
    pub fn ite(cond: &Bv, then: &Bv, els: &Bv) -> Bv {
        debug_assert_eq!(cond.width(), 1);
        debug_assert_eq!(then.width(), els.width());
        Bv::wrap(Node::Ite(cond.clone(), then.clone(), els.clone()))
    }

    /// Logical AND of two 1-bit values (for combining path constraints).
    #[must_use]
    pub fn land(&self, o: &Bv) -> Bv {
        debug_assert_eq!(self.width(), 1);
        debug_assert_eq!(o.width(), 1);
        self.and(o)
    }
}

/// Mask a value to `w` bits.
#[must_use]
pub fn mask(v: u64, w: u32) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}

impl fmt::Debug for Bv {
    /// The same canonical s-expression as [`Display`](fmt::Display).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bv({self})")
    }
}

impl fmt::Display for Bv {
    /// A canonical s-expression — stable enough to key a memory slot by, and readable.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            Node::Const(w, v) => write!(f, "#{v:x}:{w}"),
            Node::Var(w, id) => write!(f, "v{id}:{w}"),
            Node::Not(a) => write!(f, "(~ {a})"),
            Node::Neg(a) => write!(f, "(- {a})"),
            Node::Bin(op, a, b) => {
                let s = match op {
                    Op::And => "&", Op::Or => "|", Op::Xor => "^",
                    Op::Add => "+", Op::Sub => "-", Op::Mul => "*",
                    Op::Udiv => "/u", Op::Urem => "%u", Op::Sdiv => "/s", Op::Srem => "%s",
                };
                write!(f, "({s} {a} {b})")
            }
            Node::ShlC(a, k) => write!(f, "(<< {a} {k})"),
            Node::ShrC(a, k, arith) => write!(f, "({} {a} {k})", if *arith { ">>s" } else { ">>u" }),
            Node::ShlV(a, k) => write!(f, "(<<v {a} {k})"),
            Node::ShrV(a, k, arith) => write!(f, "({}v {a} {k})", if *arith { ">>s" } else { ">>u" }),
            Node::RotC(a, k, left) => write!(f, "({} {a} {k})", if *left { "rotl" } else { "rotr" }),
            Node::RotV(a, k, left) => write!(f, "({}v {a} {k})", if *left { "rotl" } else { "rotr" }),
            Node::Compare(c, a, b) => {
                let s = match c {
                    Cmp::Eq => "==", Cmp::Ne => "!=", Cmp::Ult => "<u", Cmp::Ule => "<=u",
                    Cmp::Slt => "<s", Cmp::Sle => "<=s",
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
