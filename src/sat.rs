//! A small, self-contained SAT core: iterative DPLL with unit propagation, chronological
//! backtracking (branch-and-flip), and a step budget so a hard query degrades to `Unknown`
//! rather than running without a bound.
//!
//! Literals are `i32`: `+(v+1)` for the positive polarity of variable `v`, `-(v+1)` for the
//! negative. Variable 0 is literal `1` / `-1`. Correctness is the priority here over raw
//! speed — the formulas the bit-blaster produces for a directed query are small (hundreds to
//! a few thousand clauses), and this solves those reliably.
//!
//! Deliberately absent: pure-literal elimination, watched literals, clause learning and
//! non-chronological backjumping. The clause set is re-scanned to a fixpoint on every
//! propagation pass, which is why the caller passes a wall-clock deadline alongside the
//! decision budget — see [`Cnf::solve_within`].

/// A CNF formula: `nvars` boolean variables and a conjunction of clauses (each a disjunction
/// of literals). Clauses are allocated in a bump arena `'b` — a directed query produces thousands
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
    /// The step budget was exhausted before a verdict — treat as "not proven either way".
    Unknown,
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
    /// whole lifetime — which is what lets the blaster hand them around freely.
    pub fn new_var(&mut self) -> i32 {
        self.nvars += 1;
        self.nvars as i32 // literal for var (nvars-1) is +nvars
    }

    /// Add a clause: the disjunction of `lits`.
    ///
    /// The literals are copied into the arena, so the slice need not outlive the call. An
    /// empty clause is the empty disjunction — immediately false, and therefore the way to
    /// state "unsatisfiable" outright.
    pub fn add_clause(&mut self, lits: &[i32]) {
        self.clauses.push(self.bump.alloc_slice_copy(lits));
    }

    /// Solve, bounded by BOTH a decision budget and an optional wall-clock `deadline`. The decision
    /// budget alone is a poor time proxy — a propagation-heavy formula burns seconds between
    /// decisions (`propagate` rescans every clause per fixpoint) — so a sweep passes a real deadline
    /// to cap the slowest single query and stop it flooring the batch's wall-clock.
    #[must_use]
    pub fn solve_within(&self, budget: u64, deadline: Option<std::time::Instant>) -> SatResult {
        let n = self.nvars;
        let mut assign: Vec<Option<bool>> = vec![None; n];
        let mut trail: Vec<usize> = Vec::new(); // variable indices, in assignment order
        let mut is_decision = vec![false; n];
        let mut flipped = vec![false; n];
        let mut decisions = 0u64;

        loop {
            // Check the clock once per decision/conflict cycle. When `propagate` dominates (few, slow
            // iterations) this still fires promptly; when iterations are many and fast the `Instant`
            // cost is negligible next to the propagation work each iteration already did.
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return SatResult::Unknown;
            }
            if self.propagate(&mut assign, &mut trail, &mut is_decision) {
                // Conflict: backtrack to the most recent unflipped decision and flip it.
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
                    // Undo this assignment and keep unwinding.
                    trail.pop();
                    assign[v] = None;
                    is_decision[v] = false;
                    flipped[v] = false;
                }
            } else {
                // No conflict — pick the next unassigned variable to branch on.
                match assign.iter().position(Option::is_none) {
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
                }
            }
        }
    }

    /// Unit propagation to a fixpoint. Returns `true` on conflict.
    fn propagate(
        &self,
        assign: &mut [Option<bool>],
        trail: &mut Vec<usize>,
        is_decision: &mut [bool],
    ) -> bool {
        loop {
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
                    return true; // all literals false — conflict
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
                return false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sat_unit_chain() {
        // (a) ∧ (¬a ∨ b) ∧ (¬b ∨ c)  ⇒  a=b=c=true.
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
        // (a) ∧ (¬a)  ⇒  UNSAT.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let a = cnf.new_var();
        cnf.add_clause(&[a]);
        cnf.add_clause(&[-a]);
        assert_eq!(cnf.solve_within(10_000, None), SatResult::Unsat);
    }

    #[test]
    fn test_sat_needs_a_decision() {
        // (a ∨ b) with nothing forcing either — satisfiable by branching.
        let bump = bumpalo::Bump::new();
        let mut cnf = Cnf::new(&bump);
        let (a, b) = (cnf.new_var(), cnf.new_var());
        cnf.add_clause(&[a, b]);
        match cnf.solve_within(10_000, None) {
            SatResult::Sat(m) => assert!(m[0] || m[1]),
            other => panic!("expected SAT, got {other:?}"),
        }
    }
}
