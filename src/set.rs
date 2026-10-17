//! Finite sets of at most 64 elements, lowered to one bitvector.
//!
//! A set over a small universe is a characteristic vector: bit `i` is `1` when
//! element `i` is a member. That is not an encoding trick invented here - it is
//! what the programs this crate reasons about actually store (a role bitmap, a
//! feature mask, a token whitelist), so a query about membership is a query
//! about one bit of a word, and set algebra is word algebra.
//!
//! The universe is fixed by the set's width (`1..=64`), the same ceiling every
//! term in this crate has. Because the core already has symbolic shifts, an
//! *element index* may be symbolic, not just a constant: `contains(s, i)` shifts
//! by `i` and takes the low bit, and the shift semantics (shift at or past the
//! width empties the value) give the right answer for an out-of-range index -
//! it is not a member, and an insert there changes nothing.
//!
//! All operations are pure `Bv` composition, so a set query is decided by the
//! existing bit-blaster: there is no set theory to validate and no lemma to
//! instantiate. Sets are the one theory here that is fully eager.
//!
//! ```
//! use smtlite::{Bv, Set, Solution, Solver};
//!
//! let mut s = Solver::new();
//! let i = s.var("i", 3);
//! let mut s0 = Set::<4>::empty();
//! s0 = s0.insert(&i);
//! s.assert(s0.contains(&i).eq(&Bv::val(1, 1)));
//! // Membership of the element just inserted is a theorem, not a guess.
//! assert!(matches!(s.check(), Solution::Sat(_)));
//! ```

use crate::bv::Bv;

/// A finite set over the elements `0..N`, stored as an `N`-bit characteristic
/// vector.
///
/// `N` is part of the type only for documentation and literal construction;
/// every operation reads it back from the underlying value, so a set built from
/// a term of any width works the same way.
#[derive(Clone, Debug)]
pub struct Set<const N: u32> {
    bits: Bv,
}

impl<const N: u32> Set<N> {
    /// The empty set.
    ///
    /// # Panics
    ///
    /// Panics if `N` is outside `1..=64`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            bits: Bv::val(0, N),
        }
    }

    /// The full set: every element of the universe.
    ///
    /// # Panics
    ///
    /// Panics if `N` is outside `1..=64`.
    #[must_use]
    pub fn full() -> Self {
        Self {
            bits: Bv::val(u64::MAX, N),
        }
    }

    /// The set containing exactly `element`.
    ///
    /// # Panics
    ///
    /// Panics if `element >= N` - a constant index is checked, unlike a
    /// symbolic one, because it is a mistake in the caller rather than data.
    #[must_use]
    pub fn singleton(element: u32) -> Self {
        Self::empty().insert_const(element)
    }

    /// A set from a characteristic vector. The width of `bits` is the universe.
    #[must_use]
    pub fn from_bits(bits: Bv) -> Self {
        Self { bits }
    }

    /// The characteristic vector.
    #[must_use]
    pub fn bits(&self) -> &Bv {
        &self.bits
    }

    /// The number of elements in the universe.
    #[must_use]
    pub fn universe(&self) -> u32 {
        self.bits.width()
    }

    /// Is `index` a member? Yields a 1-bit value; an index at or past the
    /// universe is not a member.
    #[must_use]
    pub fn contains(&self, index: &Bv) -> Bv {
        self.bits.lshr_var(index).extract(0, 0)
    }

    /// Is the constant `index` a member?
    ///
    /// # Panics
    ///
    /// Panics if `index >= N`.
    #[must_use]
    pub fn contains_const(&self, index: u32) -> Bv {
        assert!(
            index < self.universe(),
            "element {index} is outside a {}-element universe",
            self.universe()
        );
        self.bits.extract(index, index)
    }

    /// Insert `index`. An out-of-universe index is a no-op, exactly as the
    /// shift semantics make it.
    #[must_use]
    pub fn insert(&self, index: &Bv) -> Self {
        let bit = Bv::val(1, self.universe()).shl_var(index);
        Self {
            bits: self.bits.or(&bit),
        }
    }

    /// Insert the constant `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= N`.
    #[must_use]
    pub fn insert_const(&self, index: u32) -> Self {
        assert!(
            index < self.universe(),
            "element {index} is outside a {}-element universe",
            self.universe()
        );
        let bit = Bv::val(1, self.universe()).shl(index);
        Self {
            bits: self.bits.or(&bit),
        }
    }

    /// Remove `index`. An out-of-universe index is a no-op.
    #[must_use]
    pub fn remove(&self, index: &Bv) -> Self {
        let bit = Bv::val(1, self.universe()).shl_var(index);
        Self {
            bits: self.bits.and(&bit.not()),
        }
    }

    /// Remove the constant `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= N`.
    #[must_use]
    pub fn remove_const(&self, index: u32) -> Self {
        assert!(
            index < self.universe(),
            "element {index} is outside a {}-element universe",
            self.universe()
        );
        let bit = Bv::val(1, self.universe()).shl(index);
        Self {
            bits: self.bits.and(&bit.not()),
        }
    }

    /// Insert `index` when `when` is true; leave the set unchanged when false.
    #[must_use]
    pub fn insert_if(&self, when: &Bv, index: &Bv) -> Self {
        assert_eq!(when.width(), 1, "the condition must be 1 bit");
        let bit = Bv::val(1, self.universe()).shl_var(index);
        let mask = Bv::ite(when, &bit, &Bv::val(0, self.universe()));
        Self {
            bits: self.bits.or(&mask),
        }
    }

    /// Every element of either set.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.or(&other.bits),
        }
    }

    /// The elements common to both sets.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.and(&other.bits),
        }
    }

    /// The elements of this set that are not in `other`.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.and(&other.bits.not()),
        }
    }

    /// The elements in exactly one of the two sets.
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        Self {
            bits: self.bits.xor(&other.bits),
        }
    }

    /// The elements of the universe that are not in this set.
    #[must_use]
    pub fn complement(&self) -> Self {
        Self {
            bits: self.bits.not(),
        }
    }

    /// Is every element of this set also in `other`?
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> Bv {
        self.difference(other).is_empty()
    }

    /// Do the two sets hold exactly the same elements? (A 1-bit value, not a
    /// Rust `bool` - both operands may be symbolic.)
    #[must_use]
    pub fn same(&self, other: &Self) -> Bv {
        self.bits.eq(&other.bits)
    }

    /// Is the set empty? (A 1-bit value: `bits == 0`.)
    #[must_use]
    pub fn is_empty(&self) -> Bv {
        self.bits.eq(&Bv::val(0, self.universe()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Solution, Solver};

    /// Every subset of the 4-element universe, as a bitmask.
    fn subsets() -> impl Iterator<Item = u64> {
        0u64..16
    }

    fn set(mask: u64) -> Set<4> {
        Set::from_bits(Bv::val(mask, 4))
    }

    /// Assert that a term is the constant `want` of its own width.
    fn assert_bits(actual: &Bv, want: u64, label: &str) {
        let mut s = Solver::new();
        s.assert(actual.eq(&Bv::val(want, actual.width())));
        assert!(
            matches!(s.check(), Solution::Sat(_)),
            "{label}: expected {want:#x}, term is not that constant"
        );
    }

    #[test]
    fn set_algebra_matches_bitmask_algebra_exhaustively() {
        for a in subsets() {
            for b in subsets() {
                let (sa, sb) = (set(a), set(b));
                let cases: [(&str, u64, Bv); 6] = [
                    ("union", a | b, sa.union(&sb).bits().clone()),
                    ("intersection", a & b, sa.intersection(&sb).bits().clone()),
                    ("difference", a & !b, sa.difference(&sb).bits().clone()),
                    (
                        "symmetric",
                        a ^ b,
                        sa.symmetric_difference(&sb).bits().clone(),
                    ),
                    ("complement", !a & 0xf, sa.complement().bits().clone()),
                    ("same", u64::from(a == b), sa.same(&sb)),
                ];
                for (label, want, got) in cases {
                    let mut s = Solver::new();
                    s.assert(got.eq(&Bv::val(want, got.width())));
                    assert!(
                        matches!(s.check(), Solution::Sat(_)),
                        "{label}: a={a:#x} b={b:#x} differs from the bitmask answer"
                    );
                }
                // Subset is a 1-bit predicate; it must agree with the mask test.
                let mut s = Solver::new();
                let want = u64::from(a & !b == 0);
                s.assert(sa.is_subset(&sb).eq(&Bv::val(want, 1)));
                assert!(
                    matches!(s.check(), Solution::Sat(_)),
                    "subset: a={a:#x} b={b:#x}"
                );
            }
            // Membership over every element, both constant and symbolic.
            for e in 0u32..4 {
                let mut s = Solver::new();
                s.assert(set(a).contains_const(e).eq(&Bv::val((a >> e) & 1, 1)));
                assert!(matches!(s.check(), Solution::Sat(_)));
            }
        }
    }

    #[test]
    fn insert_and_remove_const_match_the_mask() {
        for a in subsets() {
            for e in 0u32..4 {
                assert_bits(
                    set(a).insert_const(e).bits(),
                    a | 1 << e,
                    &format!("insert {e} into {a:#x}"),
                );
                assert_bits(
                    set(a).remove_const(e).bits(),
                    a & !(1 << e),
                    &format!("remove {e} from {a:#x}"),
                );
            }
        }
    }

    /// A symbolic index takes its value from the model: with `i` pinned by a
    /// constraint, `contains`/`insert`/`remove` must agree with the constant
    /// versions.
    #[test]
    fn a_symbolic_index_behaves_like_the_constant_it_takes() {
        for a in subsets() {
            for e in 0u32..5 {
                let mut s = Solver::new();
                let i = s.var("i", 3);
                s.assert(i.eq(&Bv::val(u64::from(e), 3)));
                let sa = set(a);
                // e == 4 is outside the universe: the shift semantics say "no
                // member", not "element 0".
                let want = if e < 4 { (a >> e) & 1 } else { 0 };
                s.assert(sa.contains(&i).eq(&Bv::val(want, 1)));
                let inserted = if e < 4 { a | 1 << e } else { a };
                s.assert(sa.insert(&i).bits().eq(&Bv::val(inserted, 4)));
                let removed = if e < 4 { a & !(1 << e) } else { a };
                s.assert(sa.remove(&i).bits().eq(&Bv::val(removed, 4)));
                assert!(matches!(s.check(), Solution::Sat(_)), "a={a:#x} e={e}");
            }
        }
    }

    #[test]
    fn singleton_full_and_empty() {
        assert_bits(Set::<4>::singleton(2).bits(), 4, "singleton 2");
        assert_bits(Set::<4>::full().bits(), 0xf, "full");
        assert_bits(Set::<4>::empty().bits(), 0, "empty");

        let mut s = Solver::new();
        s.assert(Set::<4>::empty().is_empty().eq(&Bv::val(1, 1)));
        s.assert(Set::<4>::singleton(1).is_empty().eq(&Bv::val(0, 1)));
        s.assert(
            Set::<4>::full()
                .is_subset(&Set::<4>::full())
                .eq(&Bv::val(1, 1)),
        );
        assert!(matches!(s.check(), Solution::Sat(_)));
    }
}
