//! Alpha-beta search mirroring the C++ engine's first version (cpp/src/search.cpp at #13): iterative
//! deepening, negamax alpha-beta, quiescence, transposition table, ordering by hash move, MVV-LVA
//! captures, queen promotions and killers, repetition over the game history, mate scores.

use crate::eval::{evaluate, PIECE_VALUE};
use crate::movegen::{generate, generate_legal, is_legal, MoveList};
use crate::position::Position;
use crate::types::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MATE: i32 = 30000;
pub const MATE_BOUND: i32 = MATE - 1000;
pub const INF: i32 = MATE + 1;
pub const MAX_PLY: usize = 128;

const ORDER_HASH: i32 = 10_000_000;
const ORDER_CAPTURE: i32 = 1_000_000;
const ORDER_PROMOTION: i32 = 900_000;
const ORDER_KILLER: i32 = 800_000;

const HISTORY_MAX: i32 = 16384;

/// Late-move reduction in plies by remaining depth and moves searched (same formula as the C++ engine).
fn lmr_table() -> &'static [[i32; 64]; 64] {
    static T: std::sync::OnceLock<[[i32; 64]; 64]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0i32; 64]; 64];
        for (d, row) in t.iter_mut().enumerate().skip(1) {
            for (m, r) in row.iter_mut().enumerate().skip(1) {
                *r = (0.75 + (d as f64).ln() * (m as f64).ln() / 2.25) as i32;
            }
        }
        t
    })
}

const BOUND_EXACT: u8 = 1;
const BOUND_LOWER: u8 = 2;
const BOUND_UPPER: u8 = 3;

#[derive(Clone, Copy, Default)]
struct TTEntry {
    key: u64,
    mv: Move,
    score: i16,
    depth: i8,
    bound: u8,
}

pub struct Limits {
    pub depth: i32,
    pub time_ms: f64,
    pub nodes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            depth: 64,
            time_ms: 0.0,
            nodes: 0,
        }
    }
}

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

pub struct Searcher {
    tt: Vec<TTEntry>,
    tt_mask: usize,
    killers: [[Move; 2]; MAX_PLY],
    history: Box<[[[i32; 64]; 64]; 2]>, // quiet-move history: bonus for cutoffs, malus for quiets tried before one
    stack: Vec<u64>,
    nodes: u64,
    limits: Limits,
    start: Instant,
    aborted: bool,
    can_abort: bool,
    pub stop: Arc<AtomicBool>,
}

fn to_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_BOUND {
        score + ply
    } else if score < -MATE_BOUND {
        score - ply
    } else {
        score
    }
}

fn from_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_BOUND {
        score - ply
    } else if score < -MATE_BOUND {
        score + ply
    } else {
        score
    }
}

fn insufficient_material(pos: &Position) -> bool {
    for c in [WHITE, BLACK] {
        if pos.pieces[c][PAWN] | pos.pieces[c][ROOK] | pos.pieces[c][QUEEN] != 0 {
            return false;
        }
    }
    (pos.pieces[WHITE][KNIGHT]
        | pos.pieces[WHITE][BISHOP]
        | pos.pieces[BLACK][KNIGHT]
        | pos.pieces[BLACK][BISHOP])
        .count_ones()
        <= 1
}

fn mvv_lva(pos: &Position, m: Move) -> i32 {
    let victim = if move_flag(m) == EP_CAPTURE {
        PAWN
    } else {
        piece_type(pos.board[move_to(m)])
    };
    let attacker = piece_type(pos.board[move_from(m)]);
    PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker] / 10
}

pub fn format_score(score: i32) -> String {
    if score > MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score < -MATE_BOUND {
        format!("mate {}", -((MATE + score + 1) / 2))
    } else {
        format!("cp {score}")
    }
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher {
        let mut s = Searcher {
            tt: Vec::new(),
            tt_mask: 0,
            killers: [[NO_MOVE; 2]; MAX_PLY],
            history: Box::new([[[0; 64]; 64]; 2]),
            stack: Vec::new(),
            nodes: 0,
            limits: Limits::default(),
            start: Instant::now(),
            aborted: false,
            can_abort: true,
            stop: Arc::new(AtomicBool::new(false)),
        };
        s.resize(hash_mb);
        s
    }

    pub fn resize(&mut self, mb: usize) {
        let mut count = 1usize;
        while count * 2 * std::mem::size_of::<TTEntry>() <= mb * 1024 * 1024 {
            count *= 2;
        }
        self.tt = vec![TTEntry::default(); count];
        self.tt_mask = count - 1;
    }

    pub fn new_game(&mut self) {
        self.tt.iter_mut().for_each(|e| *e = TTEntry::default());
        self.killers = [[NO_MOVE; 2]; MAX_PLY];
        *self.history = [[[0; 64]; 64]; 2];
    }

    /// History gravity: bonuses saturate at +/-HISTORY_MAX, so old information fades as new arrives.
    fn update_history(&mut self, side: usize, m: Move, bonus: i32) {
        let h = &mut self.history[side][move_from(m)][move_to(m)];
        *h += bonus - *h * bonus.abs() / HISTORY_MAX;
    }

    fn probe(&self, key: u64) -> Option<TTEntry> {
        let e = self.tt[key as usize & self.tt_mask];
        (e.bound != 0 && e.key == key).then_some(e)
    }

    fn store(&mut self, key: u64, mv: Move, score: i32, depth: i32, bound: u8) {
        let e = &mut self.tt[key as usize & self.tt_mask];
        if e.key == key && depth < e.depth as i32 && bound != BOUND_EXACT {
            return;
        }
        let mv = if mv == NO_MOVE && e.key == key {
            e.mv
        } else {
            mv
        };
        *e = TTEntry {
            key,
            mv,
            score: score as i16,
            depth: depth as i8,
            bound,
        };
    }

    fn time_up(&mut self) -> bool {
        if !self.can_abort {
            return false;
        }
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }
        if self.limits.nodes > 0 && self.nodes >= self.limits.nodes {
            return true;
        }
        self.limits.time_ms > 0.0
            && self.nodes & 2047 == 0
            && self.start.elapsed().as_secs_f64() * 1000.0 >= self.limits.time_ms
    }

    fn is_draw(&self, pos: &Position) -> bool {
        if pos.halfmove >= 100 || insufficient_material(pos) {
            return true;
        }
        let n = self.stack.len() as i64 - 1; // stack.last() is this position
        let mut i = n - 2;
        while i >= 0 && i >= n - pos.halfmove as i64 {
            if self.stack[i as usize] == pos.hash {
                return true;
            }
            i -= 2;
        }
        false
    }

    fn order(&self, pos: &Position, list: &mut MoveList, hash_move: Move, ply: usize) {
        let n = list.len;
        let mut scores = [0i32; MAX_MOVES];
        for (score, &m) in scores.iter_mut().zip(list.moves[..n].iter()) {
            *score = if m == hash_move {
                ORDER_HASH
            } else if is_capture(m) {
                ORDER_CAPTURE + mvv_lva(pos, m)
            } else if is_promotion(m) && promotion_type(m) == QUEEN {
                ORDER_PROMOTION
            } else if ply < MAX_PLY && m == self.killers[ply][0] {
                ORDER_KILLER
            } else if ply < MAX_PLY && m == self.killers[ply][1] {
                ORDER_KILLER - 1
            } else {
                self.history[pos.side][move_from(m)][move_to(m)] // in [-HISTORY_MAX, HISTORY_MAX], below killers
            };
        }
        // Insertion sort, descending (lists are short).
        for i in 1..n {
            let (m, s) = (list.moves[i], scores[i]);
            let mut j = i;
            while j > 0 && scores[j - 1] < s {
                list.moves[j] = list.moves[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }
            list.moves[j] = m;
            scores[j] = s;
        }
    }

    fn quiesce(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            self.aborted = true;
            return 0;
        }
        let in_check = pos.in_check();
        let mut list = MoveList::new();
        if in_check {
            generate(pos, &mut list, false); // all evasions; no standing pat in check
        } else {
            let stand_pat = evaluate(pos);
            if stand_pat >= beta {
                return beta;
            }
            alpha = alpha.max(stand_pat);
            generate(pos, &mut list, true);
        }
        self.order(pos, &mut list, NO_MOVE, MAX_PLY);
        let mut any_legal = false;
        for i in 0..list.len {
            let m = list.moves[i];
            if !in_check && !is_capture(m) {
                continue; // quiet queen promotions are not searched here (as in the C++ first version)
            }
            if !is_legal(pos, m) {
                continue;
            }
            any_legal = true;
            let next = pos.make(m);
            self.stack.push(next.hash);
            let score = -self.quiesce(&next, -beta, -alpha, ply + 1);
            self.stack.pop();
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
            alpha = alpha.max(score);
        }
        if in_check && !any_legal {
            return -(MATE - ply);
        }
        alpha
    }

    /// Principal variation search for one child: the first move gets the full window; later moves a
    /// zero-width window, re-searched with the full window only if they beat alpha.
    fn pvs(
        &mut self,
        next: &Position,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: i32,
        first: bool,
    ) -> i32 {
        if first || beta - alpha == 1 {
            return -self.negamax(next, depth, -beta, -alpha, ply, true);
        }
        let mut score = -self.negamax(next, depth, -alpha - 1, -alpha, ply, true);
        if score > alpha && score < beta && !self.aborted {
            score = -self.negamax(next, depth, -beta, -alpha, ply, true);
        }
        score
    }

    fn negamax(
        &mut self,
        pos: &Position,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        allow_null: bool,
    ) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            self.aborted = true;
            return 0;
        }
        if self.is_draw(pos) {
            return 0;
        }
        // Check extension: checks are forcing and have few replies, so look one ply further.
        let in_check = pos.in_check();
        if in_check && (ply as usize) < MAX_PLY - 1 {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiesce(pos, alpha, beta, ply);
        }
        let mut hash_move = NO_MOVE;
        if let Some(e) = self.probe(pos.hash) {
            hash_move = e.mv;
            if e.depth as i32 >= depth {
                let s = from_tt(e.score as i32, ply);
                if e.bound == BOUND_EXACT
                    || (e.bound == BOUND_LOWER && s >= beta)
                    || (e.bound == BOUND_UPPER && s <= alpha)
                {
                    return s;
                }
            }
        }
        let pv_node = beta - alpha > 1;
        let static_eval = if in_check { -INF } else { evaluate(pos) };

        // Reverse futility pruning: near the leaves, a static eval far above beta is very unlikely to drop.
        if !pv_node
            && !in_check
            && depth <= 6
            && beta.abs() < MATE_BOUND
            && static_eval - 80 * depth >= beta
        {
            return static_eval;
        }
        // Null-move pruning: if passing still holds beta after a reduced search, a real move would too.
        if allow_null
            && !pv_node
            && depth >= 3
            && !in_check
            && pos.has_non_pawn_material(pos.side)
            && static_eval >= beta
        {
            let r = 2 + depth / 6;
            let next = pos.make_null();
            self.stack.push(next.hash);
            let score = -self.negamax(&next, depth - 1 - r, -beta, -beta + 1, ply + 1, false);
            self.stack.pop();
            if self.aborted {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        let mut list = MoveList::new();
        generate(pos, &mut list, false);
        self.order(pos, &mut list, hash_move, ply as usize);
        let alpha_orig = alpha;
        let mut best = NO_MOVE;
        let (mut searched, mut legal) = (0i32, 0i32);
        let mut quiets_tried = [NO_MOVE; MAX_MOVES];
        let mut n_quiets = 0usize;
        for i in 0..list.len {
            let m = list.moves[i];
            if !is_legal(pos, m) {
                continue;
            }
            legal += 1;
            let next = pos.make(m);
            let quiet = !is_capture(m) && !is_promotion(m);
            let gives_check = next.in_check();
            // Futility pruning: near the leaves, skip quiet moves that can't lift the eval to alpha.
            if !pv_node
                && !in_check
                && depth <= 3
                && searched > 0
                && quiet
                && !gives_check
                && static_eval + 100 + 100 * depth <= alpha
            {
                continue;
            }
            self.stack.push(next.hash);
            let p = ply as usize;
            let killer = p < MAX_PLY && (m == self.killers[p][0] || m == self.killers[p][1]);
            let score;
            // Late-move reductions: late quiet moves first get a reduced zero-window search.
            if searched >= 3 && depth >= 3 && quiet && !killer && !in_check && !gives_check {
                let mut r = lmr_table()[depth.min(63) as usize][searched.min(63) as usize];
                if pv_node {
                    r = (r - 1).max(0);
                }
                let reduced = (depth - 1 - r).max(1);
                let mut s = -self.negamax(&next, reduced, -alpha - 1, -alpha, ply + 1, true);
                if s > alpha && reduced < depth - 1 && !self.aborted {
                    s = self.pvs(&next, depth - 1, alpha, beta, ply + 1, false);
                }
                score = s;
            } else {
                score = self.pvs(&next, depth - 1, alpha, beta, ply + 1, searched == 0);
            }
            searched += 1;
            self.stack.pop();
            if self.aborted {
                return 0;
            }
            if score >= beta {
                self.store(pos.hash, m, to_tt(beta, ply), depth, BOUND_LOWER);
                if quiet {
                    if p < MAX_PLY && self.killers[p][0] != m {
                        self.killers[p][1] = self.killers[p][0];
                        self.killers[p][0] = m;
                    }
                    let bonus = (depth * depth).min(400);
                    self.update_history(pos.side, m, bonus);
                    for &q in &quiets_tried[..n_quiets] {
                        self.update_history(pos.side, q, -bonus);
                    }
                }
                return beta;
            }
            if quiet {
                quiets_tried[n_quiets] = m;
                n_quiets += 1;
            }
            if score > alpha {
                alpha = score;
                best = m;
            }
        }
        if legal == 0 {
            return if in_check { -(MATE - ply) } else { 0 };
        }
        if searched == 0 {
            return alpha; // every legal move was pruned: a fail low, not a mate or stalemate
        }
        let bound = if alpha > alpha_orig {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        self.store(pos.hash, best, to_tt(alpha, ply), depth, bound);
        alpha
    }

    /// Searches every root move inside (alpha, beta): returns alpha with NO_MOVE on a fail low, beta with
    /// the move on a fail high.
    fn root(
        &mut self,
        pos: &Position,
        depth: i32,
        prev_best: Move,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Move) {
        let mut list = MoveList::new();
        generate_legal(pos, &mut list);
        self.order(pos, &mut list, prev_best, 0);
        let mut best = NO_MOVE;
        for i in 0..list.len {
            let m = list.moves[i];
            let next = pos.make(m);
            self.stack.push(next.hash);
            let score = self.pvs(&next, depth - 1, alpha, beta, 1, i == 0);
            self.stack.pop();
            if self.aborted {
                return (0, NO_MOVE);
            }
            if score > alpha {
                alpha = score;
                best = m;
                if alpha >= beta {
                    return (beta, best);
                }
            }
        }
        (alpha, best)
    }

    /// Searches `root` (history: hashes of earlier positions in the game, oldest first) and reports each
    /// finished depth through `on_info(depth, score, nodes, millis, best_move)`.
    pub fn search(
        &mut self,
        root: &Position,
        history: &[u64],
        limits: Limits,
        mut on_info: impl FnMut(i32, i32, u64, f64, Move),
    ) -> SearchResult {
        self.limits = limits;
        self.start = Instant::now();
        self.nodes = 0;
        self.aborted = false;
        self.killers = [[NO_MOVE; 2]; MAX_PLY];
        self.stack = history.to_vec();
        self.stack.push(root.hash);
        let mut legal = MoveList::new();
        generate_legal(root, &mut legal);
        let mut result = SearchResult {
            best: NO_MOVE,
            score: 0,
            depth: 0,
            nodes: 0,
        };
        if legal.len == 0 {
            return result;
        }
        result.best = legal.moves[0];
        for depth in 1..=self.limits.depth {
            self.can_abort = depth > 1; // depth 1 always completes
            let mut prev = if result.depth > 0 {
                result.best
            } else {
                NO_MOVE
            };
            let (mut score, mut best);
            if depth >= 5 {
                // Aspiration window around the previous score; widen the failing side on a miss.
                let mut delta = 50;
                let (mut a, mut b) = (
                    (result.score - delta).max(-INF),
                    (result.score + delta).min(INF),
                );
                loop {
                    (score, best) = self.root(root, depth, prev, a, b);
                    if self.aborted {
                        break;
                    }
                    if score <= a {
                        a = (a - delta).max(-INF);
                    } else if score >= b {
                        b = (b + delta).min(INF);
                        prev = best; // the fail-high move goes first in the re-search
                    } else {
                        break;
                    }
                    delta *= 2;
                    if delta > 1000 {
                        a = -INF;
                        b = INF;
                    }
                }
            } else {
                (score, best) = self.root(root, depth, prev, -INF, INF);
            }
            if self.aborted {
                break;
            }
            result = SearchResult {
                best,
                score,
                depth,
                nodes: self.nodes,
            };
            on_info(
                depth,
                score,
                self.nodes,
                self.start.elapsed().as_secs_f64() * 1000.0,
                best,
            );
            if (score > MATE_BOUND && MATE - score <= depth)
                || (score < -MATE_BOUND && MATE + score <= depth)
            {
                break; // a mate the search has fully proven won't change
            }
        }
        result.nodes = self.nodes;
        result
    }
}
