//! Finite-field arithmetic over a prime modulus, encoded as bit-vectors.
//!
//! The solver core is QF_BV (mod 2^w); a field element is just a `w`-bit `Bv`
//! that callers promise to keep canonical (`< p`). Add/sub are one conditional
//! ±p; multiply widens to 2w, multiplies, and reduces by the constant modulus.
//! Multiplication is the only division-bearing op and is correspondingly the
//! expensive one to bit-blast (see [`Fp::mul`]); a Barrett reduction is the
//! fast path if it ever bites.

use crate::{Bv, Solver};

/// BabyBear, the SP1 base field: `p = 2^31 - 2^27 + 1 = 2013265921`.
pub const BABY_BEAR: u32 = 0x7800_0001;

/// A canonical element of `F_p`, held as a 32-bit bit-vector.
///
/// Callers uphold the invariant `v < p`; the combinators preserve it.
#[derive(Clone, Debug)]
pub struct Fp {
    v: Bv,
}

impl Fp {
    /// The bit-width of a BabyBear element as a bit-vector.
    pub const WIDTH: u32 = 32;

    /// The underlying bit-vector.
    pub fn bits(&self) -> &Bv {
        &self.v
    }

    /// Consume the element into its underlying bit-vector.
    pub fn into_bits(self) -> Bv {
        self.v
    }

    /// A fresh symbolic canonical field element.
    pub fn var(s: &mut Solver, name: &str) -> Fp {
        Fp {
            v: s.var(name, Self::WIDTH),
        }
    }

    /// The constant `x`, which must already be canonical (`x < p`).
    pub fn val(x: u32) -> Fp {
        assert!(x < BABY_BEAR, "field constant {x} is not canonical");
        Fp {
            v: Bv::val(x as u64, Self::WIDTH),
        }
    }

    /// `self + rhs (mod p)`. Cheap: `a + b < 2p < 2^32`, one conditional subtract.
    pub fn add(&self, rhs: &Fp) -> Fp {
        let sum = self.v.add(&rhs.v);
        let p = Bv::val(BABY_BEAR as u64, Self::WIDTH);
        Fp {
            v: Bv::ite(&sum.uge(&p), &sum.sub(&p), &sum),
        }
    }

    /// `self - rhs (mod p)`. Cheap: one conditional add of `p`.
    pub fn sub(&self, rhs: &Fp) -> Fp {
        let d = self.v.sub(&rhs.v);
        let p = Bv::val(BABY_BEAR as u64, Self::WIDTH);
        Fp {
            v: Bv::ite(&self.v.uge(&rhs.v), &d, &d.add(&p)),
        }
    }

    /// `self * rhs (mod p)`.
    ///
    /// The 64-bit product is reduced by a constant division, which bit-blasts to
    /// roughly 110k clauses (see `Solver::with_max_clauses`); pair it with a
    /// clause cap and a deadline for anything more than a one-off query.
    pub fn mul(&self, rhs: &Fp) -> Fp {
        let prod = self.v.zext(64).mul(&rhs.v.zext(64));
        let r = prod.urem(&Bv::val(BABY_BEAR as u64, 64));
        Fp {
            v: r.extract(Self::WIDTH - 1, 0),
        }
    }

    /// 1 iff this value is canonical (`< p`). The witness a caller asserts to
    /// demand canonicality.
    pub fn is_canonical(&self) -> Bv {
        self.v.ult(&Bv::val(BABY_BEAR as u64, Self::WIDTH))
    }

    /// 1 iff the two canonical elements are equal.
    pub fn eq(&self, rhs: &Fp) -> Bv {
        self.v.eq(&rhs.v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Solution;

    #[test]
    fn add_sub_mul_known_values() {
        // 64-bit `urem` reductions need ~110k clauses; lift the cap for these laws.
        let mut s = Solver::new().with_max_clauses(300_000);
        let a = Fp::val(1_000_000_000);
        let b = Fp::val(1_000_000_000);
        let c = Fp::val(3);

        // 1e9 + 1e9 = 2e9 (still < p, no reduction).
        s.assert(a.add(&b).eq(&Fp::val(2_000_000_000)));
        // p-1 + 1 = 0.
        s.assert(Fp::val(BABY_BEAR - 1).add(&Fp::val(1)).eq(&Fp::val(0)));
        // 0 - 1 = p-1.
        s.assert(Fp::val(0).sub(&Fp::val(1)).eq(&Fp::val(BABY_BEAR - 1)));
        // 3 * 4 = 12.
        s.assert(c.mul(&Fp::val(4)).eq(&Fp::val(12)));
        // (p-1)^2 = 1 mod p.
        s.assert(
            Fp::val(BABY_BEAR - 1)
                .mul(&Fp::val(BABY_BEAR - 1))
                .eq(&Fp::val(1)),
        );

        match s.check() {
            Solution::Sat(_) => {}
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn mul_is_commutative() {
        let mut s = Solver::new().with_max_clauses(300_000);
        let a = Fp::var(&mut s, "a");
        let b = Fp::var(&mut s, "b");
        s.assert(a.is_canonical());
        s.assert(b.is_canonical());
        // a*b != b*a is unsatisfiable.
        s.assert(a.mul(&b).eq(&b.mul(&a)).not());
        match s.check() {
            Solution::Unsat => {}
            other => panic!("expected UNSAT (mul commutes), got {other:?}"),
        }
    }

    /// SP1 `FieldOpCols::eval_with_polynomials` (crates/core/machine/src/operations/
    /// field/field_op.rs) constrains only `op - result - carry*M == 0` plus u8 limb
    /// range-checks on `result`/`carry`. It never constrains `result < M`; the canonical
    /// bound is "assumed". This probe proves the gap: with fully canonical operands
    /// (`a, b < M`) the AIR still admits a non-canonical `result`.
    #[test]
    fn field_op_air_does_not_force_canonical_result() {
        use crate::{Bv, Solution, Solver};

        let mut s = Solver::new();
        let m = BABY_BEAR as u64;

        // Operands, under the chip's stated premise (canonical: a, b < M).
        let a = s.var("a", 32);
        let b = s.var("b", 32);
        // result/carry are u8-limbed columns: 4 limbs => < 2^32 each.
        let result = s.var("result", 32);
        let carry = s.var("carry", 32);

        let m32 = Bv::val(m, 32);
        let m64 = Bv::val(m, 64);
        s.assert(a.ult(&m32));
        s.assert(b.ult(&m32));

        // AIR (MUL case): a*b - result - carry*M == 0  =>  a*b == result + carry*M.
        let prod = a.zext(64).mul(&b.zext(64));
        let rhs = result.zext(64).add(&carry.zext(64).mul(&m64));
        s.assert(prod.eq(&rhs));

        // Query: is a non-canonical result accepted?
        s.assert(result.uge(&m32));

        match s.check() {
            Solution::Sat(model) => {
                let a = model.get("a").unwrap();
                let b = model.get("b").unwrap();
                let result = model.get("result").unwrap();
                let carry = model.get("carry").unwrap();
                assert!(a < m && b < m, "operand premise violated");
                assert_eq!(
                    a as u128 * b as u128,
                    result as u128 + carry as u128 * m as u128,
                    "AIR equation violated"
                );
                assert!(result >= m, "result should be non-canonical (>= M)");
                eprintln!(
                    "SAT: a={a}, b={b}, result={result} (M={m}), carry={carry} \
                     -> field_op AIR accepts a non-canonical result"
                );
            }
            Solution::Unsat => panic!("expected SAT: the AIR must admit a non-canonical result"),
            Solution::Unknown => panic!("solver gave up (budget/clause cap)"),
        }
    }
}
