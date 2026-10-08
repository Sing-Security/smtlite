# smtlite

A small, pure-Rust **QF_BV** SMT solver - quantifier-free bitvector formulas, solved by
bit-blasting to CNF and running a self-contained DPLL core over it.

No external solver. No C. No `unsafe`. One dependency (`bumpalo`, an arena allocator).

It exists for one job: answering **directed, single-function** queries - "is there an input that
makes this operation overflow?", "can this length field reach that copy?" - fast enough to sit
inside a symbolic executor that asks thousands of them. It is deliberately *not* a general-purpose
Z3 replacement.

```rust
use smtlite::{Solver, Bv, Solution};

let mut s = Solver::new();
let x = s.var("x", 64);
s.assert(x.add(&Bv::val(3, 64)).eq(&Bv::val(10, 64)));   // x + 3 == 10
match s.check() {
    Solution::Sat(m) => assert_eq!(m.get("x"), Some(7)),
    other => panic!("expected SAT, got {other:?}"),
}
```

## What it does

- Bitvector variables of any width from 1 to 64 bits, plus constants.
- Boolean: `and`, `or`, `xor`, `not`, `land` (logical and of two 1-bit values).
- Arithmetic: `add`, `sub`, `mul`, `neg` - all wraparound, as bitvectors are.
- Division and remainder: unsigned `udiv`/`urem` and signed `sdiv`/`srem`, with the SMT-LIB
  zero-divisor semantics - `bvudiv` by zero is all-ones, `bvurem` by zero is the dividend,
  `bvsdiv` by zero is all-ones for a non-negative dividend and `1` otherwise. `srem` takes the
  sign of the **dividend** (C's `%`), not the divisor's as `bvsmod` would.
- Shifts: `shl`, `lshr`, `ashr` by a **constant** amount, and `shl_var`, `lshr_var`, `ashr_var`
  by a **symbolic** one. Shifting at or past the width empties the value (zero-fill, or sign-fill
  for `ashr_var`) - a shift is not a rotate.
- Rotates: `rotl`/`rotr` by a constant and `rotl_var`/`rotr_var` by a symbolic amount. A rotate is
  a bit permutation, so it costs no gates at all and never empties the value.
- Comparison: `eq`, `ne`, `ult`, `ule`, `ugt`, `uge`, `slt`, `sle`, `sgt`, `sge` - each yields a
  1-bit value you can feed into other operations.
- Width: `zext`, `sext`, `extract(hi, lo)`, `concat`.
- `ite(cond, then, else)`.
- A model back out: `Model::get(name)` returns the concrete `u64` a variable took, or `None` if
  no variable by that name was ever declared.

`Solver::depends_on(&bv, "mem")` answers a question a caller asks constantly - did this value
actually derive from the data read out of the input, or is it still something the caller started
with? Name variables by origin (`mem...`, `init_...`) and the prefix tells you which.

## What it deliberately does not do

- **Arrays, uninterpreted functions, or quantifiers.** This is QF_BV only.
- **`bvsmod`**, whose remainder takes the sign of the *divisor*. `srem` gives C's `%`, which is
  what compiled code actually emits.
- **Overflow-detection builtins** (`bvuaddo`, `bvsaddo`, ...). Derive them from the primitives:
  `x.add(&y).ult(&x)` is the unsigned-add overflow test.
- **Widths above 64 bits** - a model is read back as a `u64`, so 64 is the ceiling. It is
  *enforced*, not merely documented: building a value wider than that, or combining two of
  different widths, panics rather than quietly blasting the wrong bits. Every method whose
  inputs have to satisfy something says so under `# Panics`.
- **An SMT-LIB front-end.** The name invites the assumption; the API is Rust only, there is no
  textual parser and no solver-on-a-pipe protocol.

Formulas are also **single-threaded**: a `Bv` is an `Rc`-shared node, so neither `Bv` nor `Solver`
is `Send`/`Sync`.

## Cost

Bit-blasting is where the formula size lives, and it is not uniform across operations:

| Operation (at width `w`) | Rough cost |
|---|---|
| constant rotate | **free** - pure re-indexing, no gates, no clauses |
| bitwise, constant shifts, extend/extract/concat | `O(w)` literals |
| `add`/`sub`, comparisons, `ite` | `O(w)` gates, 2-4 clauses each |
| `mul` | `O(w²)` |
| symbolic shift | `O(w log w)` - a barrel shifter |
| `div`/`rem` | `O(w²)`, a restoring divider |

So every operation at 32 bits lands under the default clause cap, but a **64-bit division reaches
~110,000 clauses** and needs the cap raised deliberately - see below. A symbolic rotate at a
power-of-two width never divides (the low bits of the amount already are the amount modulo the
width); at any other width it needs a divider as wide as its *amount*, so a 64-bit amount against
a 5-bit vector is the expensive combination, while a 5-bit amount is trivial.

## How it works

`Bv` values form an `Rc`-shared expression DAG; a value used twice is one subgraph, not two. On
`check`, the DAG is bit-blasted - each bit of each node becomes a SAT literal, operations encoded
through Tseitin gates, arithmetic through ripple-carry adders, unsigned comparison through the
carry-out of `a + ¬b + 1`, division through a restoring divider (whose per-step adder yields the
subtraction and the `rem >= b` test in one pass), and symbolic shifts and rotates through barrel
shifters. Bitvectors are little-endian (index 0 is the least-significant bit). Shared subgraphs are
memoised on the `Rc` pointer, so a value appearing in several constraints is encoded once.

CNF clauses are arena-allocated (`bumpalo`): a query produces thousands of tiny, same-lifetime
clauses, so one arena reset frees the whole formula instead of freeing clause-by-clause.

The SAT core is iterative DPLL - unit propagation to a fixpoint, then chronological
branch-and-flip backtracking. Correctness is preferred over raw speed, because the formulas a
directed query produces are small - hundreds to a few thousand clauses, with division the
notable exception (see **Cost** above).

## Bounded by construction

A solver embedded in a batch job must never hang it. Three guards, all of which degrade to
`Solution::Unknown` - "not proven either way" - rather than stalling:

- a **decision budget** (`check_with_budget`, `check_all_with_budget`),
- a **wall-clock deadline** (`check_all_within`), because a propagation-heavy formula can burn
  seconds between decisions, making a decision count a poor time proxy - the deadline is checked
  both between decisions and within unit propagation, so it bounds a query rather than a decision
  count,
- a **formula-size cap** (40,000 clauses by default): past that the instance is pathological and is
  better left to a manual walk-through than allowed to stall a sweep. Every operation at 32 bits
  fits under it; raise it with `Solver::with_max_clauses` for a 64-bit division, and pair that with
  a deadline.

`Solution::Unknown` is never a silent "unsat" - the three outcomes stay distinct so a caller can
tell a refutation from a timeout.

## Requirements

- Rust **1.85** or newer (the 2024 edition).
- No Cargo features - the crate has a single dependency and no optional parts.
- No `unsafe` (`unsafe_code = "forbid"`) and no build script.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
