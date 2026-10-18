//! Move generation: pseudo-legal moves; a move is legal if the mover's king isn't attacked after it.
//! Castling out of or through check is rejected during generation (same design as cpp/src/movegen.cpp).

use crate::bitboard::tables;
use crate::position::*;
use crate::types::*;

pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList {
            moves: [NO_MOVE; MAX_MOVES],
            len: 0,
        }
    }
    #[inline]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn add_pawn_moves(
    list: &mut MoveList,
    from: usize,
    to: usize,
    capture: bool,
    promotes: bool,
    noisy_only: bool,
) {
    if !promotes {
        list.push(make_move(from, to, if capture { CAPTURE } else { QUIET }));
        return;
    }
    let base = if capture { PROMO_CAPTURE } else { PROMO };
    list.push(make_move(from, to, base + 3)); // queen
    if noisy_only {
        return;
    }
    for p in 0..3 {
        list.push(make_move(from, to, base + p));
    }
}

#[inline]
fn add_piece_moves(list: &mut MoveList, from: usize, mut targets: Bitboard, enemies: Bitboard) {
    while targets != 0 {
        let to = pop_lsb(&mut targets);
        list.push(make_move(
            from,
            to,
            if enemies & bit(to) != 0 {
                CAPTURE
            } else {
                QUIET
            },
        ));
    }
}

/// Pseudo-legal moves; `noisy_only`: captures and queen promotions (for quiescence search).
pub fn generate(pos: &Position, list: &mut MoveList, noisy_only: bool) {
    let t = tables();
    let us = pos.side;
    let them = us ^ 1;
    let occupied = pos.all();
    let enemies = pos.occ[them];
    let empty = !occupied;
    let targets = if noisy_only { enemies } else { !pos.occ[us] };

    let up: isize = if us == WHITE { 8 } else { -8 };
    let promo_from: u64 = if us == WHITE {
        0x00FF_0000_0000_0000
    } else {
        0x0000_0000_0000_FF00
    };
    let start_rank: u64 = if us == WHITE {
        0x0000_0000_0000_FF00
    } else {
        0x00FF_0000_0000_0000
    };
    let mut pawns = pos.pieces[us][PAWN];
    while pawns != 0 {
        let from = pop_lsb(&mut pawns);
        let promotes = bit(from) & promo_from != 0;
        let to = (from as isize + up) as usize;
        if empty & bit(to) != 0 {
            if !noisy_only || promotes {
                add_pawn_moves(list, from, to, false, promotes, noisy_only);
            }
            if !noisy_only && bit(from) & start_rank != 0 {
                let to2 = (to as isize + up) as usize;
                if empty & bit(to2) != 0 {
                    list.push(make_move(from, to2, DOUBLE_PUSH));
                }
            }
        }
        let mut caps = t.pawn[us][from] & enemies;
        while caps != 0 {
            add_pawn_moves(list, from, pop_lsb(&mut caps), true, promotes, noisy_only);
        }
        if pos.ep != NO_SQ && t.pawn[us][from] & bit(pos.ep as usize) != 0 {
            list.push(make_move(from, pos.ep as usize, EP_CAPTURE));
        }
    }

    let mut b = pos.pieces[us][KNIGHT];
    while b != 0 {
        let from = pop_lsb(&mut b);
        add_piece_moves(list, from, t.knight[from] & targets, enemies);
    }
    let mut b = pos.pieces[us][BISHOP];
    while b != 0 {
        let from = pop_lsb(&mut b);
        add_piece_moves(
            list,
            from,
            t.bishop_attacks(from, occupied) & targets,
            enemies,
        );
    }
    let mut b = pos.pieces[us][ROOK];
    while b != 0 {
        let from = pop_lsb(&mut b);
        add_piece_moves(
            list,
            from,
            t.rook_attacks(from, occupied) & targets,
            enemies,
        );
    }
    let mut b = pos.pieces[us][QUEEN];
    while b != 0 {
        let from = pop_lsb(&mut b);
        add_piece_moves(
            list,
            from,
            t.queen_attacks(from, occupied) & targets,
            enemies,
        );
    }
    let ksq = pos.king_sq(us);
    add_piece_moves(list, ksq, t.king[ksq] & targets, enemies);

    if noisy_only {
        return;
    }
    let rank = if us == WHITE { 0 } else { 56 };
    let (oo, ooo) = if us == WHITE {
        (WHITE_OO, WHITE_OOO)
    } else {
        (BLACK_OO, BLACK_OOO)
    };
    if pos.castling & oo != 0
        && occupied & (bit(rank + 5) | bit(rank + 6)) == 0
        && !pos.attacked(rank + 4, them)
        && !pos.attacked(rank + 5, them)
    {
        list.push(make_move(rank + 4, rank + 6, KING_CASTLE));
    }
    if pos.castling & ooo != 0
        && occupied & (bit(rank + 1) | bit(rank + 2) | bit(rank + 3)) == 0
        && !pos.attacked(rank + 4, them)
        && !pos.attacked(rank + 3, them)
    {
        list.push(make_move(rank + 4, rank + 2, QUEEN_CASTLE));
    }
}

#[inline]
pub fn is_legal(pos: &Position, m: Move) -> bool {
    let next = pos.make(m);
    !next.attacked(next.king_sq(pos.side), next.side)
}

pub fn generate_legal(pos: &Position, list: &mut MoveList) {
    let mut all = MoveList::new();
    generate(pos, &mut all, false);
    for &m in all.as_slice() {
        if is_legal(pos, m) {
            list.push(m);
        }
    }
}

/// Number of leaf nodes of the legal move tree to `depth` (move-generator test).
pub fn perft(pos: &Position, depth: u32) -> u64 {
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    if depth <= 1 {
        return if depth == 1 { list.len as u64 } else { 1 };
    }
    let mut nodes = 0;
    for &m in list.as_slice() {
        nodes += perft(&pos.make(m), depth - 1);
    }
    nodes
}
