//! Hand-written evaluation with identical numbers to cpp/src/eval.cpp (and engine/eval.py): material,
//! piece-square tables generated from simple rules, tapered middlegame/endgame, bishop pair.

use crate::position::Position;
use crate::types::*;
use std::sync::OnceLock;

pub const PIECE_VALUE: [i32; 6] = [100, 320, 330, 500, 900, 0];
const PHASE_WEIGHT: [i32; 6] = [0, 1, 1, 2, 4, 0];
const MAX_PHASE: i32 = 24;
const BISHOP_PAIR: i32 = 30;

struct TablesPst {
    mg: [[i32; 64]; 6],
    eg: [[i32; 64]; 6],
}

static PST: OnceLock<TablesPst> = OnceLock::new();

/// 0 on the corners .. 6 on the four centre squares.
fn centrality(sq: usize) -> i32 {
    let f2 = (2 * file_of(sq) as i32 - 7).abs();
    let r2 = (2 * rank_of(sq) as i32 - 7).abs();
    7 - (f2 + r2) / 2
}

fn pst() -> &'static TablesPst {
    PST.get_or_init(|| {
        let pawn_mg_rank = [0, 0, 2, 6, 14, 30, 50, 0];
        let pawn_eg_rank = [0, 0, 5, 15, 30, 55, 90, 0];
        let king_home_file = [10, 20, 5, -5, -5, 5, 20, 10];
        let mut t = TablesPst {
            mg: [[0; 64]; 6],
            eg: [[0; 64]; 6],
        };
        for sq in 0..64 {
            let (f, r, c) = (file_of(sq), rank_of(sq), centrality(sq));
            let centre_file = f == 3 || f == 4;
            t.mg[PAWN][sq] = pawn_mg_rank[r]
                + if centre_file && (r == 3 || r == 4) {
                    10
                } else {
                    0
                };
            t.eg[PAWN][sq] = pawn_eg_rank[r];
            t.mg[KNIGHT][sq] = -30 + 8 * c;
            t.eg[KNIGHT][sq] = -25 + 6 * c;
            t.mg[BISHOP][sq] = -10 + 4 * c;
            t.eg[BISHOP][sq] = -8 + 3 * c;
            t.mg[ROOK][sq] = if r == 6 { 15 } else { 0 } + if centre_file { 5 } else { 0 };
            t.eg[ROOK][sq] = if r == 6 { 5 } else { 0 };
            t.mg[QUEEN][sq] = -5 + 2 * c;
            t.eg[QUEEN][sq] = -10 + 3 * c;
            t.mg[KING][sq] = if r == 0 {
                king_home_file[f]
            } else {
                -25 * (r.min(3) as i32)
            };
            t.eg[KING][sq] = -18 + 6 * c;
        }
        t
    })
}

/// Centipawns from White's point of view.
pub fn evaluate_white(pos: &Position) -> i32 {
    let t = pst();
    let (mut mg, mut eg, mut phase) = (0i32, 0i32, 0i32);
    for color in [WHITE, BLACK] {
        let sign = if color == WHITE { 1 } else { -1 };
        for kind in PAWN..=KING {
            let mut b = pos.pieces[color][kind];
            phase += PHASE_WEIGHT[kind] * b.count_ones() as i32;
            while b != 0 {
                let sq = pop_lsb(&mut b);
                let idx = if color == WHITE { sq } else { sq ^ 56 };
                mg += sign * (PIECE_VALUE[kind] + t.mg[kind][idx]);
                eg += sign * (PIECE_VALUE[kind] + t.eg[kind][idx]);
            }
        }
    }
    let phase = phase.min(MAX_PHASE);
    let mut score = (mg * phase + eg * (MAX_PHASE - phase)).div_euclid(MAX_PHASE);
    if pos.pieces[WHITE][BISHOP].count_ones() >= 2 {
        score += BISHOP_PAIR;
    }
    if pos.pieces[BLACK][BISHOP].count_ones() >= 2 {
        score -= BISHOP_PAIR;
    }
    score
}

/// Centipawns from the side to move's point of view.
pub fn evaluate(pos: &Position) -> i32 {
    let s = evaluate_white(pos);
    if pos.side == WHITE {
        s
    } else {
        -s
    }
}
