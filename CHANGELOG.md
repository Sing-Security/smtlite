# Changelog

All notable changes to smtlite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.10] - 2026-10-09

### Changed

- **The conformance suite now proves the circuits, not the folder.** Operand constants fold on
  construction, so the operation tables in `tests/ops.rs` were proving a folded constant against
  a folded constant - correct, but no circuit ever reached the blaster (the 64-bit divider test
  even raised the formula cap for a formula that was never built). Each operand is now pinned
  into a fresh variable by an equality constraint, so every proof bit-blasts the operation's
  real circuit: the SAT half shows it can equal the oracle value and the UNSAT half refutes any
  difference. Every proof also asserts the expression it sends to the solver is still symbolic,
  so a regression back to folded constants fails loudly instead of passing vacuously.
- **Coverage is measured, not assumed.** Tests inside the blaster pin the clause counts of the
  wide circuits - a 64-bit divider is 108,160 clauses, the signed one 114,756, a 64-bit
  multiplier 75,872 - and a companion test shows the same division on constant operands blasts
  zero clauses. An end-to-end guard requires a pinned 64-bit division under the default cap to
  answer `Unknown` (the formula exceeds the cap), which the folded failure mode could not do.
- **Constant-folding semantics are tested as their own layer.** The folder is checked
  exhaustively against Rust's operators at width 4 - arithmetic and all ten comparisons - plus
  shifts and rotates at widths 4 and 7, directly via `as_const` with no solver involved, and
  labeled as folding coverage rather than circuit conformance.
- **Wider circuit coverage.** New edge tables for 64-bit signed division and remainder, 64-bit
  symbolic shifts and rotates, and 64-bit signed comparisons - operations that previously had
  no test whose operands survived folding. The suite runs 39 conformance tests (from 31), 10.7 s
  in debug and 2.3 s in release; the 64-bit division tables dominate, which is the dividers
  themselves being built and refuted.

## [0.1.9] - 2026-10-08

### Changed

- **The query bounds now cover the whole query, not just the search.** The wall-clock deadline
  and the formula-size cap take effect from API entry: the blaster checks the deadline while it
  encodes (memoising width-correct dummies once a bound trips, so the caller bails to `Unknown`
  before any solve sees them) and the cap stops clause generation at the limit instead of
  rejecting the finished formula. The cached background blast is bounded the same way, an
  over-cap background is remembered so later checks answer instantly, and `preprocess` checks
  the clock per clause. Results are unchanged - the fix removes wasted work, it does not change
  any verdict: an elapsed deadline still means `Unknown`, never a verdict.

## [0.1.8] - 2026-10-08

### Changed

- **README reworked into prose, same facts.** The operation catalog and the deliberate limits
  read as paragraphs instead of a spec sheet.
- **Assumption models are pinned as total.** A new test locks in that a model from
  `check_assumptions` reads back a variable only the stored background mentions - the property
  the cached blast's `var_bits` seeding exists for.

## [0.1.6] - 2026-10-08

### Added

- **`check_assumptions`, `check_assumptions_with_budget`, `check_assumptions_within`.** Solve the
  stored background plus a set of one-bit per-call assumptions, without re-blasting the background.
  `var` and `assert` invalidate the cached blast; `check_all` keeps its explicit whole-formula
  semantics.

### Changed

- **Expressions fold constants and identities when they are built.** Every combinator now folds
  constant operands (with SMT-LIB zero-divisor semantics) and width-aware identities - `x + 0`,
  `x - x`, `~~x`, self-comparisons - before a node is created, so a folded result is shared with
  every later use of the same subexpression. `as_const` reflects this: `val(1).add(val(1))` reads
  back as a constant, and `a + b == b + a` at width 16/32, which used to exhaust the decision
  budget, now folds to `Unsat`.
- **Unit propagation watches two literals per clause**, processing trail assignments instead of
  rescanning the clause set to a fixpoint. Pigeonhole(9, 8) went from 9.6 s to 0.12 s (release).
- **Branching prefers the unassigned variable with the most conflict activity** (VSIDS-style,
  bumped per conflict) and tries its last-assigned value first (phase saving).
- **The clause set is simplified once before the search.** Unit clauses force their literal, so
  clauses holding a forced literal are satisfied, the negation of a forced literal drops out, and
  tautological and duplicate clauses go; a clause whose literals all drop out settles the query
  unsat without any search.

## [0.1.1] - 2026-10-08

### Added

- **Three test suites beyond the per-operation conformance tests.** `tests/differential.rs` checks
  generated expressions against an independent evaluator - exhaustively at small widths, and by
  metamorphic identity elsewhere; `tests/contracts.rs` pins every documented panic; `tests/api.rs`
  covers model readback, budgeting and provenance.

### Fixed

- **`Model::get` reads back a declared variable no constraint mentioned.** The blaster recorded a
  variable's SAT bits only when a constraint encoded it, so an unused variable returned `None`
  from a model documented as total. Every declared variable's bits are now allocated up front.
- **The SAT core no longer branches on a variable that occurs in no clause.** Such a variable is
  free, so any value satisfies the formula, and deciding one only adds search the solver must
  unwind. With the fix above giving every declared variable SAT bits, unrelated declarations made
  refutations slower by orders of magnitude - one small width-4 formula went from 30 microseconds
  to over 2 seconds, and from `Unsat` to `Unknown`, once eight unrelated variables were declared.
- **The wall-clock deadline is checked inside unit propagation**, not only between decisions.
  `propagate` rescans the clause set to a fixpoint, so a propagation-heavy query could run past a
  `check_all_within` deadline before the next check; the deadline is now tested once per scan.

## [0.1.0] - 2026-10-07

First public release. Written for directed, single-function queries - "is there an input that
makes this overflow?", "can this length reach that copy?" - asked many times over.

### Added

- **`Solver`** - named bitvector variables (`var`), stored constraints (`assert`), and an
  explicit-constraint path (`check_all`) for a caller carrying per-path constraints against one
  shared variable namespace.
- **`Bv`** - bitvector expression combinators:
  - boolean `and`/`or`/`xor`/`not`/`land`;
  - arithmetic `add`/`sub`/`mul`/`neg`, all wraparound;
  - division and remainder - unsigned `udiv`/`urem`, signed `sdiv`/`srem`, with SMT-LIB
    zero-divisor semantics (`bvudiv` by zero is all-ones, `bvurem` by zero is the dividend,
    `bvsdiv` by zero is all-ones for a non-negative dividend and `1` otherwise). `srem` takes the
    sign of the dividend - C's `%`, not `bvsmod`'s;
  - shifts `shl`/`lshr`/`ashr` by a constant amount and `shl_var`/`lshr_var`/`ashr_var` by a
    symbolic one; a shift at or past the width empties the value rather than wrapping;
  - rotates `rotl`/`rotr` by a constant and `rotl_var`/`rotr_var` by a symbolic amount, where a
    power-of-two width reduces the amount for free and any other width costs a divider;
  - comparison `eq`/`ne`/`ult`/`ule`/`ugt`/`uge`/`slt`/`sle`/`sgt`/`sge`;
  - width `zext`/`sext`/`extract`/`concat`, plus `ite`;
  - and `val`, `as_const`, `width`, `ptr_eq`.
- **`Model`** - a satisfying assignment readable by variable name (`get`).
- **Bit-blaster** - Tseitin-encoded operations, ripple-carry adders, comparison via the carry-out
  of `a + ¬b + 1`, restoring division, barrel shifters and rotators, memoised shared subgraphs
  (`Rc` identity), little-endian bits.
- **SAT core** - iterative DPLL: unit propagation to a fixpoint with chronological branch-and-flip
  backtracking.
- **Bounded solving** - a decision budget, an optional wall-clock deadline (`check_all_within`),
  and a formula-size cap (`Solver::with_max_clauses`). All three degrade to `Solution::Unknown`
  rather than stalling the caller.
- **`Solver::depends_on`** - does a value derive from any variable whose name starts with a
  given prefix? Name variables by where their data came from and this tells you which of those
  origins a subexpression actually depends on, without threading provenance by hand.
- Arena-allocated CNF clauses via `bumpalo` - the crate's only dependency.
- No `unsafe` (`unsafe_code = "forbid"`), no C, no external solver.
- A conformance suite (`tests/ops.rs`) that checks every operation against Rust's native operators
  as an oracle: exhaustive at width 4, sampled at 8, edge cases at 32 and 64, with each result
  proven both satisfiable-equal and unsatisfiable-unequal to the expected constant.

### Notes

- **Widths are 1..=64, and the boundary is enforced.** A formula that would cross it - a
  variable or constant wider than 64, a concatenation past 64, operands of different widths
  - panics rather than producing a wrong answer. Release builds included: a silently
  mis-blasted formula is worse than a stopped caller. Every affected method documents this
  under `# Panics`.
- **A model reads back through `u64`.** `Model::get` returns the concrete value of a named
  variable, or `None` if no such variable was declared.
- `Bv` is an `Rc`-shared node, so `Bv` and `Solver` are neither `Send` nor `Sync` - solving is
  single-threaded by design.
- Bit-blasting cost varies sharply by operation: bitwise/constant-shift/constant-rotate are linear,
  `mul` and `div` are quadratic in the width, symbolic shifts are `O(w log w)`. A 64-bit division
  blasts to roughly 110,000 clauses, over the 40,000 default cap.
- There is no SMT-LIB front-end: the API is Rust only.
