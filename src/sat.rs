//! A small, self-contained SAT core: iterative DPLL with unit propagation, chronological
//! backtracking (branch-and-flip), and a step budget so a hard query degrades to `Unknown`
//! rather than running without a bound.
//!
//! Literals are `i32`: `+(v+1)` for the positive polarity of variable `v`, `-(v+1)` for the
//! negative. Variable 0 is literal `1` / `-1`.
//!
//! Not present: pure-literal elimination, watched literals, clause learning and
//! non-chronological backjumping. The clause set is re-scanned to a fixpoint on every
//! propagation pass, which is why a sweep passes a wall-clock deadline alongside the decision
//! budget - and why that deadline is checked *inside* propagation, not only between decisions:
//! see [`Cnf::solve_within`].
//!
//! The solver does not branch on a variable that occurs in no clause. Such a variable is free, so
//! any value satisfies the formula, and a [`crate::Solver`] allocates bits for every variable it
//! declares - including ones no constraint mentions.

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
    /// A clause became all-false.
    Conflict,
    /// The wall-clock deadline elapsed mid-propagation.
    Deadline,
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
    /// The budget alone is a poor time proxy: `propagate` rescans every clause per fixpoint, so a
    /// propagation-heavy formula burns seconds between decisions. The deadline is therefore checked
    /// *inside* propagation as well as once per decision cycle, which bounds the overshoot to a
    /// single clause scan rather than to however long a whole fixpoint takes.
    #[must_use]
    pub fn solve_within(&self, budget: u64, deadline: Option<Instant>) -> SatResult {
        let n = self.nvars;
        let mut assign: Vec<Option<bool>> = vec![None; n];
        let mut trail: Vec<usize> = Vec::new(); // variable indices, in assignment order
        let mut is_decision = vec![false; n];
        let mut flipped = vec![false; n];
        let mut decisions = 0u64;

        // Variables that occur in at least one clause. A variable outside this set is free, so
        // branching on it only adds search: a `Solver` allocates bits for every variable it
        // declares, including ones no constraint mentions.
        let mut active = vec![false; n];
        for clause in &self.clauses {
            for &lit in clause.iter() {
                active[(lit.unsigned_abs() - 1) as usize] = true;
            }
        }

        loop {
            match self.propagate(&mut assign, &mut trail, &mut is_decision, deadline) {
                Propagated::Deadline => return SatResult::Unknown,
                // Conflict: backtrack to the most recent unflipped decision and flip it.
                Propagated::Conflict => loop {
                    let Some(&v) = trail.last() else {
                        return SatResult::Unsat; // no decisions left to undo
                    };
                    if is_decision[v] && !flipped[v] {
                        let tried = assign[v].unwrap_or(false);
                        assign[v] = Some(!tried);
                        flipped[v] = true;
                        break; // resume propagation with the flipped decision in place
                    }
                    // Undo this assignment and keep unwinding.
                    trail.pop();
                    assign[v] = None;
                    is_decision[v] = false;
                    flipped[v] = false;
                },
                // No conflict - branch on the next unassigned variable that a clause mentions.
                // Free variables stay unassigned and take a default value in the model.
                Propagated::Fixpoint => match (0..n).find(|&v| active[v] && assign[v].is_none()) {
                    None => {
                        return SatResult::Sat(assign.iter().map(|a| a.unwrap_or(false)).collect());
                    }
                    Some(v) => {
                        decisions += 1;
                        if decisions > budget {
                            return SatResult::Unknown;
                        }
                        assign[v] = Some(true);
                        is_decision[v] = true;
                        flipped[v] = false;
                        trail.push(v);
                    }
                },
            }
        }
    }

    /// Unit propagation to a fixpoint. Reports whether it hit a conflict, settled, or ran out of
    /// wall-clock `deadline` mid-scan - the last so a propagation-heavy formula cannot outlast a
    /// deadline the way a decision-count check alone would let it.
    fn propagate(
        &self,
        assign: &mut [Option<bool>],
        trail: &mut Vec<usize>,
        is_decision: &mut [bool],
        deadline: Option<Instant>,
    ) -> Propagated {
        loop {
            // Once per full rescan: a scan is this loop's unit of work, so checking here bounds the
            // overshoot to one scan rather than to the whole propagation. Cheap enough to ignore
            // when the passes are many and fast.
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Propagated::Deadline;
            }
            let mut changed = false;
            for clause in &self.clauses {
                let mut unit: Option<i32> = None;
                let mut unassigned = 0;
                let mut satisfied = false;
                for &lit in clause.iter() {
                    let v = (lit.unsigned_abs() - 1) as usize;
                    let want = lit > 0;
                    match assign[v] {
                        Some(b) => {
                            if b == want {
                                satisfied = true;
                                break;
                            }
                        }
                        None => {
                            unassigned += 1;
                            unit = Some(lit);
                        }
                    }
                }
                if satisfied {
                    continue;
                }
                if unassigned == 0 {
                    return Propagated::Conflict; // all literals false - conflict
                }
                if unassigned == 1 {
                    let lit = unit.unwrap_or(0);
                    let v = (lit.unsigned_abs() - 1) as usize;
                    assign[v] = Some(lit > 0);
                    is_decision[v] = false;
                    trail.push(v);
                    changed = true;
                }
            }
            if !changed {
                return Propagated::Fixpoint;
            }
        }
    }
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
