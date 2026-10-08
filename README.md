# smtlite

A small, pure-Rust SMT solver for **QF_BV** - quantifier-free formulas over bitvectors. It takes a
bitvector expression, bit-blasts it to CNF, and runs its own DPLL SAT core over the result.

No external solver. No C. No `unsafe`. One dependency (`bumpalo`, an arena allocator).

It exists for one job: answering **directed, single-function** questions - "is there an input that
makes this operation overflow?", "can this length field reach that copy?" - fast enough to sit
inside a symbolic executor that asks thousands of them. It is not a general-purpose Z3
replacement, and it does not try to be.

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

## Expressions

Variables are bitvectors 1 to 64 bits wide. Widths are checked, not trusted: building a value
wider than 64 bits, or combining two values of different widths, panics rather than quietly
blasting the wrong bits. Every method whose inputs have to satisfy something says so under
`# Panics`. Constants fold eagerly as they are built - `Bv::val(1, 8).add(&Bv::val(1, 8))` is the
constant `2` before it ever meets the solver.

The usual operators are all there: boolean `and`/`or`/`xor`/`not`/`land`, wraparound
`add`/`sub`/`mul`/`neg`, and comparisons `eq`/`ne`/`ult`/`ule`/`ugt`/`uge`/`slt`/`sle`/`sgt`/`sge`,
each of which yields a 1-bit value you can feed into other operations. Width plumbing is
`zext`/`sext`/`extract`/`concat`, plus `ite(cond, then, else)`.

Division follows SMT-LIB, not Rust: dividing by zero is a defined value, not a panic. `x.udiv(0)`
is all ones, `x.urem(0)` is `x`, and signed division by zero is all ones for a non-negative
dividend and `1` otherwise. `srem` takes the sign of the **dividend** - C's `%`, not `bvsmod`'s
sign of the divisor.

Shifts come in two flavours: `shl`/`lshr`/`ashr` shift by a **constant**, and the `_var` forms by a
**symbolic** amount. Shifting at or past the width empties the value (zero-fill, or sign-fill for
`ashr_var`) - a shift is not a rotate. Rotates come in the same two flavours (`rotl`/`rotr`,
`rotl_var`/`rotr_var`): a constant rotate is a pure re-indexing of the bits, no gates at all, and
a symbolic one is free when the width is a power of two and costs a divider otherwise (see
**Cost**).

When the solver finds a model, `Model::get("x")` reads the value back as a `u64` - or `None` if
no variable by that name was ever declared.

## Working with the solver

`Solver::depends_on(&bv, "mem")` answers a question a caller asks constantly: did this value
actually derive from the data read out of the input, or is it still something the caller started
with? Name variables by origin (`mem...`, `init_...`) and the prefix tells you which.

`Solver::check_assumptions` solves the stored background plus a set of one-bit per-path
constraints, without re-blasting the background. A symbolic-execution sweep asserts its input
once, then fires per-path queries against it; `var` and `assert` invalidate the cached blast, so
later declarations still read back in the model.

## Where it stops

QF_BV means QF_BV: no arrays, no uninterpreted functions, no quantifiers. A formula that needs
them needs a different tool.

There is no SMT-LIB front-end. The name invites the assumption, so it is worth saying outright:
the API is Rust only, no textual parser, no solver-on-a-pipe protocol.

Widths top out at 64 bits because a model reads back as a `u64`. That ceiling is enforced with
panics, not documented as a trap for the caller - see above.

`bvsmod` is not included, because its remainder takes the sign of the divisor and `srem` gives
C's `%`, which is what compiled code actually emits. Overflow-detection builtins (`bvuaddo`,
`bvsaddo`, ...) are not included either; `x.add(&y).ult(&x)` is the unsigned-add overflow test.

Formulas are single-threaded: a `Bv` is an `Rc`-shared node, so neither `Bv` nor `Solver` is
`Send`/`Sync`.

## Cost

Bit-blasting is where the formula size lives, and operations do not cost the same:

| Operation (at width `w`) | Rough cost |
|---|---|
| constant rotate | **free** - pure re-indexing, no gates, no clauses |
| bitwise, constant shifts, extend/extract/concat | O(w) literals |
| `add`/`sub`, comparisons, `ite` | O(w) gates, 2-4 clauses each |
| `mul` | O(w^2) |
| symbolic shift | O(w log w) - a barrel shifter |
| `div`/`rem` | O(w^2), a restoring divider |

Every operation at 32 bits fits under the default clause cap. A 64-bit division reaches about
110,000 clauses and needs the cap raised deliberately - see below. A symbolic rotate at a
power-of-two width never divides (the low bits of the amount already are the amount modulo the
width); at any other width it needs a divider as wide as its *amount*, so a 64-bit amount against
a 5-bit vector is the expensive combination.

## How it works

`Bv` values form an `Rc`-shared expression DAG: a value used twice is one subgraph, not two. On
`check`, the DAG is bit-blasted - each bit of each node becomes a SAT literal. Operations are
encoded through Tseitin gates, arithmetic through ripple-carry adders, unsigned comparison
through the carry-out of `a + ~b + 1`, division through a restoring divider (whose per-step adder
yields the subtraction and the `rem >= b` test in one pass), and symbolic shifts and rotates
through barrel shifters. Bitvectors are little-endian (index 0 is the least-significant bit).
Shared subgraphs are memoised on the `Rc` pointer, so a value appearing in several constraints is
encoded once.

CNF clauses are arena-allocated (`bumpalo`): a query produces thousands of tiny, same-lifetime
clauses, so one arena reset frees the whole formula instead of freeing clause-by-clause.

The SAT core is iterative DPLL with two-watched-literal unit propagation and chronological
branch-and-flip backtracking. Before the search, one assignment-free pass drops tautological and
duplicate clauses and lets unit clauses force their literal. Branching then prefers the
unassigned variable with the most conflict activity (VSIDS) and tries its last-assigned value
first (phase saving). Correctness is preferred over raw speed, because the formulas a directed
query produces are small - hundreds to a few thousand clauses, with division the notable
exception (see **Cost** above).

## Bounded by construction

A solver embedded in a batch job must never hang it. Three guards, and each degrades to
`Solution::Unknown` - "not proven either way" - rather than stalling:

- a **decision budget** (`check_with_budget`, `check_all_with_budget`);
- a **wall-clock deadline** (`check_all_within`), checked both between decisions and inside unit
  propagation, because a propagation-heavy formula can burn seconds between decisions and a
  decision count is a poor time proxy;
- a **formula-size cap** (40,000 clauses by default). Every operation at 32 bits fits under it;
  raise it with `Solver::with_max_clauses` for a 64-bit division, and pair that with a deadline.

`Unknown` is never a silent "unsat": the three outcomes stay distinct, so a caller can tell a
refutation from a timeout.

## Requirements

- Rust **1.85** or newer (the 2024 edition).
- No Cargo features - the crate has a single dependency and no optional parts.
- No `unsafe` (`unsafe_code = "forbid"`) and no build script.

## Licence

Apache-2.0. See [LICENSE](LICENSE).
