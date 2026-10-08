//! A small, self-contained SAT core: iterative DPLL with two-watched-literal unit propagation,
//! chronological backtracking (branch-and-flip), and a step budget so a hard query degrades to
//! `Unknown` rather than running without a bound.
//!
//! Literals are `i32`: `+(v+1)` for the positive polarity of variable `v`, `-(v+1)` for the
//! negative. Variable 0 is literal `1` / `-1`.
//!
//! Not present: pure-literal elimination, clause learning and non-chronological backjumping.
//! Propagation keeps two watched literals per clause and processes each trail assignment once,
//! rather than re-scanning the clause set to a fixpoint - which is why the wall-clock deadline is
//! checked *inside* propagation, once per assignment, not only between decisions: see [`solve`].
//! Branching decides the unassigned variable with the most conflict activity (VSIDS-style) and
//! tries its last-assigned phase first (phase saving).
//!
//! The solver does not branch on a variable that occurs in no clause. Such a variable is free, so
//! any value satisfies the formula, and a [`crate::Solver`] allocates bits for every variable it
//! declares - including ones no constraint mentions.
//!
//! Before the search, one assignment-free pass simplifies the clause set: a unit clause forces its
//! literal, so clauses containing a forced literal are satisfied, the negation of a forced literal
//! drops out of a clause, and duplicate and tautological clauses go.

use std::collections::HashSet;
use std::time::Instant;

/// A CNF formula: `nvars` boolean variables and a conjunction of clauses (each a disjunction
/// of literals). Clauses are allocated in a bump arena `'b` - a directed query produces thousands
/// of tiny same-lifetime clauses, so arena-allocating them turns per-clause `malloc` into a pointer
/// bump and frees the whole formula at once when the arena drops.
pub struct Cnf<'b> {
    /// How many variables have been introduced. Variables are 0-based, so the literals range
    /// over ±1..=±`nvars`.
    pub nvars: usize,
    /// The conjunction, one clause per element, each borrowed from the arena.
    pub clauses: Vec<&'b [i32]>,
    /// The arena the clauses live in; kept so clauses can be allocated as they are added.
    bump: &'b bumpalo::Bump,
}

/// The result of a solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SatResult {
    /// Satisfiable, with a full assignment indexed by variable.
    Sat(Vec<bool>),
    /// Proven unsatisfiable.
    Unsat,
    /// The step budget was exhausted before a verdict - treat as "not proven either way".
    Unknown,
}

/// How a unit-propagation pass ended. `Deadline` is separate from `Conflict` so a deadline hit
/// mid-propagation becomes `Unknown`, never a mistaken `Unsat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Propagated {
    /// Reached a fixpoint with no conflict - the caller may branch.
    Fixpoint,
    /// A clause became all-false. The index is the conflicting clause, whose literals the caller
    /// bumps for the branching heuristic.
    Conflict(usize),
    /// The wall-clock deadline elapsed mid-propagation.
    Deadline,
}

/// The two-watched-literal index for one solve. Each clause keeps two watched positions; a clause
/// is only examined when one of those two literals becomes false, so propagation touches a clause
/// on assignment rather than on every pass.
struct Watches {
    /// For each literal (indexed by [`lit_index`]), the clauses currently watching it.
    lists: Vec<Vec<usize>>,
    /// For each clause, its two watched positions within the clause.
    watch: Vec<[usize; 2]>,
}

/// A literal's index in the watch table: `2*var + (0 if positive, 1 if negative)`.
fn lit_index(lit: i32) -> usize {
    (lit.unsigned_abs() - 1) as usize * 2 + usize::from(lit < 0)
}

/// Build the watch index: a clause of two or more literals watches its first two positions; a
/// unit clause watches its single literal twice (a false single literal is then a conflict).
fn build_watches(clauses: &[&[i32]], nvars: usize) -> Watches {
    let mut lists = vec![Vec::new(); 2 * nvars];
    let mut watch = Vec::with_capacity(clauses.len());
    for (ci, clause) in clauses.iter().enumerate() {
        let pair = if clause.len() >= 2 {
            [0usize, 1]
        } else {
            [0usize, 0]
        };
        lists[lit_index(clause[pair[0]])].push(ci);
        if clause.len() >= 2 {
            lists[lit_index(clause[pair[1]])].push(ci);
        }
        watch.push(pair);
    }
    Watches { lists, watch }
}

impl<'b> Cnf<'b> {
    /// An empty formula over zero variables, whose clauses will be allocated in `bump`.
    #[must_use]
    pub fn new(bump: &'b bumpalo::Bump) -> Self {
        Self {
            nvars: 0,
            clauses: Vec::new(),
            bump,
        }
    }

    /// Allocate a fresh variable, returning its positive literal.
    ///
    /// Never reuses an id, and never unassigns one, so literals stay valid for the formula's
    /// whole lifetime - which is what lets the blaster hand them around freely.
    pub fn new_var(&mut self) -> i32 {
        self.nvars += 1;
        self.nvars as i32 // literal for var (nvars-1) is +nvars
    }

    /// Add a clause: the disjunction of `lits`.
    ///
    /// The literals are copied into the arena, so the slice need not outlive the call. An
    /// empty clause is the empty disjunction - immediately false, and therefore the way to
    /// state "unsatisfiable" outright.
    pub fn add_clause(&mut self, lits: &[i32]) {
        self.clauses.push(self.bump.alloc_slice_copy(lits));
    }

    /// Solve, bounded by both a decision budget and an optional wall-clock `deadline`.
    ///
    /// The budget alone is a poor time proxy: propagation is cheap per assignment but unbounded in
    /// total, so a propagation-heavy formula can still burn time between decisions. The deadline is
    /// therefore checked *inside* propagation, once per trail assignment, which bounds the overshoot
    /// to a single assignment's watch list.
    #[must_use]
    pub fn solve_within(&self, budget: u64, deadline: Option<Instant>) -> SatResult {
        solve(&self.clauses, self.nvars, budget, deadline)
    }
}

/// One assignment-free cleanup pass over the clause set, run before the search starts.
///
/// Unit clauses force their literal, and that is the whole simplification:
/// - a clause containing a forced literal is already satisfied and goes;
/// - the negation of a forced literal can never satisfy a clause and drops out of it;
/// - a clause holding both a literal and its negation is always true and goes;
/// - duplicate clauses go (one copy per canonical form survives).
///
/// A clause whose literals all drop out is the empty clause: the formula is unsatisfiable, which
/// is reported as `None`. Nothing is assigned here, so the free-variable filter and model
/// readback see exactly the formula the caller built. The unit clauses themselves are kept, so
/// the forced values survive as seed assignments for the search.
fn preprocess(clauses: &[&[i32]], nvars: usize) -> Option<Vec<Vec<i32>>> {
    // What does each unit clause force? A later unit clause overwrites an earlier one; the
    // solver's own unit seeding rejects a conflicting pair, so either order is fine here.
    let mut forced = vec![None; nvars];
    for clause in clauses {
        if clause.len() == 1 {
            let lit = clause[0];
            forced[(lit.unsigned_abs() - 1) as usize] = Some(lit > 0);
        }
    }

    let mut out: Vec<Vec<i32>> = Vec::with_capacity(clauses.len());
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    for clause in clauses {
        let mut kept: Vec<i32> = Vec::with_capacity(clause.len());
        let mut satisfied = false;
        let mut tautology = false;
        for &lit in clause.iter() {
            let v = (lit.unsigned_abs() - 1) as usize;
            match forced[v] {
                Some(want) if want == (lit > 0) => {
                    if clause.len() == 1 {
                        kept.push(lit); // the forcing unit clause must survive
                    } else {
                        satisfied = true; // a forced literal already satisfies the clause
                        break;
                    }
                }
                Some(_) => {} // forced false: the literal never helps, drop it
                None => {
                    if kept.contains(&-lit) {
                        tautology = true; // x and -x in one clause: always true
                    }
                    kept.push(lit);
                }
            }
        }
        if satisfied || tautology {
            continue;
        }
        if kept.is_empty() {
            return None; // every literal was forced false: the empty clause
        }
        // Duplicates drop here: hash the sorted (canonical) form, first copy wins.
        let mut key = kept.clone();
        key.sort_unstable();
        if seen.insert(key) {
            out.push(kept);
        }
    }
    Some(out)
}

/// Solve a CNF formula given as borrowed clauses, bounded by a decision budget and an optional
/// wall-clock `deadline`.
///
/// This is the DPLL core shared by [`Cnf::solve_within`] and the incremental path in
/// [`crate::Solver`], which blasts its background once and appends per-call clauses in front of
/// the same search. The clauses may borrow from different arenas; only the slices themselves are
/// read here.
///
/// The budget alone is a poor time proxy: propagation is cheap per assignment but unbounded in
/// total, so a propagation-heavy formula can still burn time between decisions. The deadline is
/// therefore checked *inside* propagation, once per trail assignment, which bounds the overshoot
/// to a single assignment's watch list.
#[must_use]
pub fn solve(
    clauses: &[&[i32]],
    nvars: usize,
    budget: u64,
    deadline: Option<Instant>,
) -> SatResult {
    // An elapsed deadline answers Unknown, not a verdict the search has not actually earned.
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return SatResult::Unknown;
    }
    // The empty clause is the empty disjunction, i.e. false: unsatisfiable outright.
    if clauses.iter().any(|c| c.is_empty()) {
        return SatResult::Unsat;
    }
    // Simplify the clause set once, before any search state exists.
    let Some(cleaned) = preprocess(clauses, nvars) else {
        return SatResult::Unsat;
    };
    let cleaned: Vec<&[i32]> = cleaned.iter().map(Vec::as_slice).collect();
    let clauses = cleaned.as_slice();
    // The cleanup pass does not check the clock; do not let it eat the deadline silently.
    if deadline.is_some_and(|d| Instant::now() >= d) {
        return SatResult::Unknown;
    }

    let n = nvars;
    let mut assign: Vec<Option<bool>> = vec![None; n];
    let mut trail: Vec<usize> = Vec::new(); // variable indices, in assignment order
    let mut is_decision = vec![false; n];
    let mut flipped = vec![false; n];
    let mut decisions = 0u64;
    // Branching heuristic state: VSIDS-style conflict activity per variable - the next decision
    // prefers a variable that keeps appearing in conflicts - and phase saving: the value a
    // variable was last assigned, tried first when it is decided again.
    let mut activity = vec![0u64; n];
    let mut phase = vec![false; n];

    // Variables that occur in at least one clause. A variable outside this set is free, so
    // branching on it only adds search: a `Solver` allocates bits for every variable it
    // declares, including ones no constraint mentions.
    let mut active = vec![false; n];
    for clause in clauses {
        for &lit in clause.iter() {
            active[(lit.unsigned_abs() - 1) as usize] = true;
        }
    }

    let mut watches = build_watches(clauses, n);

    // Seed the unit clauses: nothing is assigned yet, so the only clauses already unit are the
    // single-literal ones (including the pinned-true literal every blast emits). Their literals
    // go on the trail before the main loop, and `propagate` processes them like any assignment.
    for clause in clauses {
        if clause.len() == 1 {
            let lit = clause[0];
            let v = (lit.unsigned_abs() - 1) as usize;
            match assign[v] {
                Some(b) if b != (lit > 0) => return SatResult::Unsat, // [x] and [-x]
                Some(_) => {}
                None => {
                    assign[v] = Some(lit > 0);
                    trail.push(v);
                }
            }
        }
    }

    // The next trail entry `propagate` has yet to process.
    let mut prop_head = 0usize;

    loop {
        match propagate(
            clauses,
            &mut watches,
            &mut assign,
            &mut trail,
            &mut is_decision,
            &mut prop_head,
            deadline,
        ) {
            Propagated::Deadline => return SatResult::Unknown,
            // Conflict: bump the conflicting clause's literals, then backtrack to the most
            // recent unflipped decision and flip it.
            Propagated::Conflict(ci) => {
                for &lit in clauses[ci].iter() {
                    activity[(lit.unsigned_abs() - 1) as usize] += 1;
                }
                loop {
                    let Some(&v) = trail.last() else {
                        return SatResult::Unsat; // no decisions left to undo
                    };
                    if is_decision[v] && !flipped[v] {
                        let tried = assign[v].unwrap_or(false);
                        assign[v] = Some(!tried);
                        flipped[v] = true;
                        break; // resume propagation with the flipped decision in place
                    }
                    // Undo this assignment, remembering its value as the phase to try next.
                    trail.pop();
                    phase[v] = assign[v].unwrap_or(false);
                    assign[v] = None;
                    is_decision[v] = false;
                    flipped[v] = false;
                }
            }
            // No conflict - branch on the unassigned variable a clause mentions whose conflict
            // activity is highest (ties to the lowest index, so the search stays deterministic),
            // trying its last-assigned phase first. Free variables stay unassigned and take a
            // default value in the model.
            Propagated::Fixpoint => {
                let best = (0..n)
                    .filter(|&v| active[v] && assign[v].is_none())
                    .min_by_key(|&v| (std::cmp::Reverse(activity[v]), v));
                match best {
                    None => {
                        return SatResult::Sat(assign.iter().map(|a| a.unwrap_or(false)).collect());
                    }
                    Some(v) => {
                        decisions += 1;
                        if decisions > budget {
                            return SatResult::Unknown;
                        }
                        assign[v] = Some(phase[v]);
                        is_decision[v] = true;
                        flipped[v] = false;
                        trail.push(v);
                    }
                }
            }
        }

        // After backtracking, the flipped decision is the last entry and its value changed, so it
        // must be re-processed. After branching, the new decision is the last entry and is
        // processed for the first time. Either way, resume from the last entry.
        prop_head = trail.len().saturating_sub(1);
    }
}

/// Two-watched-literal unit propagation. Processes each trail assignment from `prop_head` to the
/// end: the negation of the newly-true literal is false, so every clause watching that literal is
/// either satisfied (keep watching), has its watch moved to a still-possible literal, or has
/// become unit (enqueue the remaining literal) or conflicting.
fn propagate(
    clauses: &[&[i32]],
    watches: &mut Watches,
    assign: &mut [Option<bool>],
    trail: &mut Vec<usize>,
    is_decision: &mut [bool],
    prop_head: &mut usize,
    deadline: Option<Instant>,
) -> Propagated {
    while *prop_head < trail.len() {
        // Check the deadline once per assignment: cheap when absent, and it bounds the overshoot
        // to a single assignment's watch list rather than to the whole propagation.
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Propagated::Deadline;
        }
        let v = trail[*prop_head];
        *prop_head += 1;
        let value = assign[v].unwrap_or(false);
        // The literal this assignment made false is the negation of the newly-true one.
        let false_lit = if value { -(v as i32 + 1) } else { v as i32 + 1 };

        // Take the list out so watches moved to another literal can be appended there while this
        // one is rebuilt from the clauses that stay put.
        let pending = std::mem::take(&mut watches.lists[lit_index(false_lit)]);
        let mut keep = Vec::with_capacity(pending.len());
        let mut conflict = None;

        for (i, &ci) in pending.iter().enumerate() {
            let clause = clauses[ci];
            let w = watches.watch[ci];
            // Which of the two watched positions is the literal that just became false?
            let slot = if clause[w[0]] == false_lit { 0 } else { 1 };
            let other_slot = slot ^ 1;
            let other = clause[w[other_slot]];
            let other_v = (other.unsigned_abs() - 1) as usize;

            if assign[other_v] == Some(other > 0) {
                // The other watched literal is already true: the clause is satisfied, so leave it
                // watching the false literal and move on.
                keep.push(ci);
                continue;
            }

            // Look for another, non-false literal to watch instead.
            let mut moved = false;
            for (k, &l) in clause.iter().enumerate() {
                if k == w[0] || k == w[1] {
                    continue;
                }
                let lv = (l.unsigned_abs() - 1) as usize;
                let want = l > 0;
                if assign[lv] != Some(!want) {
                    let mut nw = w;
                    nw[slot] = k;
                    watches.watch[ci] = nw;
                    watches.lists[lit_index(l)].push(ci);
                    moved = true;
                    break;
                }
            }
            if moved {
                continue;
            }

            // No non-false replacement: the clause is unit (other unassigned) or all-false.
            match assign[other_v] {
                None => {
                    // Unit: force the remaining literal and keep watching the false literal.
                    assign[other_v] = Some(other > 0);
                    is_decision[other_v] = false;
                    trail.push(other_v);
                    keep.push(ci);
                }
                Some(_) => {
                    // `other` is false here (the true case was handled above): conflict.
                    keep.push(ci);
                    conflict = Some(ci);
                    // The watchers after this one were not examined; restore them untouched.
                    keep.extend_from_slice(&pending[i + 1..]);
                    break;
                }
            }
        }

        // Restore every clause still watching the false literal.
        watches.lists[lit_index(false_lit)] = keep;
        if let Some(ci) = conflict {
            return Propagated::Conflict(ci);
        }
    }
    Propagated::Fixpoint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_unit_chain() {
        // (a) ∧ (¬a ∨ b) ∧ (¬b ∨ c)  ==>  a=b=c=true.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let (a, b, c) = (cnf.new_var(), cnf.new_var(), cnf.new_var());
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a, b]);
        cnf.add_clause(&[-b, c]);
        match cnf.solve_within(10_000, None) {
            SatResult::Sat(m) => assert!(m[0] && m[1] && m[2]),
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn test_sat_contradiction_is_unsat() {
        // (a) ∧ (¬a)  ==>  UNSAT.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let a = cnf.new_var();
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a]);
        assert_eq!(cnf.solve_within(10_000, None), SatResult::Unsat);
    }

    #[test]
    fn test_sat_needs_a_decision() {
        // (a ∨ b) with nothing forcing either - satisfiable by branching.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let (a, b) = (cnf.new_var(), cnf.new_var());
        cnf.add_clause(&[a, b]);
        match cnf.solve_within(10_000, None) {
            SatResult::Sat(m) => assert!(m[0] || m[1]),
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    /// An `Instant` a second in the past, robust to a machine that booted under a second ago.
    fn elapsed() -> std::time::Instant {
        let now = std::time::Instant::now();
        now.checked_sub(std::time::Duration::from_secs(1))
            .unwrap_or(now)
    }

    #[test]
    fn test_elapsed_deadline_is_unknown_not_a_verdict() {
        // An expired deadline yields Unknown, not the Unsat this formula actually is.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let a = cnf.new_var();
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a]);
        assert_eq!(
            cnf.solve_within(10_000, Some(elapsed())),
            SatResult::Unknown
        );
    }

    #[test]
    fn test_future_deadline_does_not_block_a_verdict() {
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let a = cnf.new_var();
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a]);
        let far = std::time::Instant::now() + std::time::Duration::from_secs(3600);
        assert_eq!(cnf.solve_within(10_000, Some(far)), SatResult::Unsat);
    }

    #[test]
    fn test_sat_needs_several_decisions() {
        // Three independent two-way choices: satisfiable, but only by branching three times.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let v: Vec<i32> = (0..6).map(|_| cnf.new_var()).collect();
        cnf.add_clause(&[v[0], v[1]]);
        cnf.add_clause(&[v[2], v[3]]);
        cnf.add_clause(&[v[4], v[5]]);
        match cnf.solve_within(10_000, None) {
            SatResult::Sat(m) => {
                assert!((m[0] || m[1]) && (m[2] || m[3]) && (m[4] || m[5]));
            }
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn test_backtracking_flips_a_decision_and_recovers() {
        // (a ∨ b) ∧ (¬a ∨ b) ∧ (¬a ∨ ¬b): branching `a = true` propagates `b = true`, then
        // `¬a ∨ ¬b` conflicts, so the solver must flip `a` to false and settle on (a=false, b=true).
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let (a, b) = (cnf.new_var(), cnf.new_var());
        cnf.add_clause(&[a, b]);
        cnf.add_clause(&[-a, b]);
        cnf.add_clause(&[-a, -b]);
        match cnf.solve_within(10_000, None) {
            SatResult::Sat(m) => assert!(!m[0] && m[1], "expected a=false, b=true; got {m:?}"),
            other => panic!("expected SAT, got {other:?}"),
        }
    }

    #[test]
    fn test_pigeonhole_three_pigeons_two_holes_is_unsat() {
        // Search-heavy refutation: no short propagation proof, so this exercises backtracking
        // rather than a unit conflict.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let mut x = [[0i32; 2]; 3];
        for pigeon in &mut x {
            for slot in pigeon.iter_mut() {
                *slot = cnf.new_var();
            }
        }
        for pigeon in &x {
            cnf.add_clause(&[pigeon[0], pigeon[1]]); // every pigeon sits somewhere
        }
        // Group the literals by hole, then forbid every pair of pigeons in the same hole.
        let mut holes: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        for pigeon in &x {
            for (hole, &lit) in pigeon.iter().enumerate() {
                holes[hole].push(lit);
            }
        }
        for column in &holes {
            for (i, &p) in column.iter().enumerate() {
                for &q in &column[i + 1..] {
                    cnf.add_clause(&[-p, -q]); // no two pigeons share a hole
                }
            }
        }
        assert_eq!(cnf.solve_within(1_000_000, None), SatResult::Unsat);
    }

    #[test]
    fn test_pigeonhole_nine_pigeons_eight_holes_is_unsat() {
        // A larger search-heavy refutation: deep backtracking with the activity heuristic
        // steering the decisions, not a short unit-propagation proof.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let mut x = [[0i32; 8]; 9];
        for pigeon in &mut x {
            for slot in pigeon.iter_mut() {
                *slot = cnf.new_var();
            }
        }
        for pigeon in &x {
            cnf.add_clause(pigeon); // sits somewhere
        }
        let mut holes: [Vec<i32>; 8] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for pigeon in &x {
            for (hole, &lit) in pigeon.iter().enumerate() {
                holes[hole].push(lit);
            }
        }
        for column in &holes {
            for (i, &p) in column.iter().enumerate() {
                for &q in &column[i + 1..] {
                    cnf.add_clause(&[-p, -q]); // no two pigeons share a hole
                }
            }
        }
        assert_eq!(cnf.solve_within(1_000_000, None), SatResult::Unsat);
    }

    #[test]
    fn test_zero_budget_refutes_by_propagation_but_cannot_branch() {
        // Unsatisfiable with no branching: a zero budget still refutes it.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let a = cnf.new_var();
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a]);
        assert_eq!(cnf.solve_within(0, None), SatResult::Unsat);

        // Satisfiable only by branching: the first decision exceeds a zero budget.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let b = cnf.new_var();
        let c = cnf.new_var();
        cnf.add_clause(&[b, c]);
        assert_eq!(cnf.solve_within(0, None), SatResult::Unknown);
    }
}
