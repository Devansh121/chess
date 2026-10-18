//! Board state with copy-make and an incrementally updated Zobrist hash. Same design and the same
//! Zobrist seed as cpp/src/position.cpp, so both engines hash a position identically.

use crate::bitboard::tables;
use crate::types::*;
use std::sync::OnceLock;

pub const WHITE_OO: u8 = 1;
pub const WHITE_OOO: u8 = 2;
pub const BLACK_OO: u8 = 4;
pub const BLACK_OOO: u8 = 8;

const PIECE_CHARS: &[u8; 12] = b"PNBRQKpnbrqk";

pub struct Zobrist {
    pub piece_sq: [[u64; 64]; 12],
    pub castling: [u64; 16],
    pub ep_file: [u64; 8],
    pub side: u64,
    castle_keep: [u8; 64],
}

static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

#[inline]
pub fn zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            // splitmix64
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut z = Zobrist {
            piece_sq: [[0; 64]; 12],
            castling: [0; 16],
            ep_file: [0; 8],
            side: 0,
            castle_keep: [15; 64],
        };
        for p in z.piece_sq.iter_mut() {
            for k in p.iter_mut() {
                *k = next();
            }
        }
        for k in z.castling.iter_mut() {
            *k = next();
        }
        for k in z.ep_file.iter_mut() {
            *k = next();
        }
        z.side = next();
        z.castle_keep[0] &= !WHITE_OOO;
        z.castle_keep[7] &= !WHITE_OO;
        z.castle_keep[4] &= !(WHITE_OO | WHITE_OOO);
        z.castle_keep[56] &= !BLACK_OOO;
        z.castle_keep[63] &= !BLACK_OO;
        z.castle_keep[60] &= !(BLACK_OO | BLACK_OOO);
        z
    })
}

#[derive(Clone, Copy)]
pub struct Position {
    pub pieces: [[Bitboard; 6]; 2],
    pub occ: [Bitboard; 2],
    pub board: [u8; 64],
    pub side: usize,
    pub castling: u8,
    pub ep: u8, // NO_SQ unless an en passant capture is actually possible
    pub halfmove: u16,
    pub fullmove: u16,
    pub hash: u64,
}

impl Position {
    fn empty() -> Position {
        Position {
            pieces: [[0; 6]; 2],
            occ: [0; 2],
            board: [NO_PIECE; 64],
            side: WHITE,
            castling: 0,
            ep: NO_SQ,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        }
    }

    #[inline]
    fn put(&mut self, code: u8, sq: usize) {
        self.pieces[piece_color(code)][piece_type(code)] |= bit(sq);
        self.occ[piece_color(code)] |= bit(sq);
        self.board[sq] = code;
        self.hash ^= zobrist().piece_sq[code as usize][sq];
    }

    #[inline]
    fn remove(&mut self, sq: usize) {
        let code = self.board[sq];
        self.pieces[piece_color(code)][piece_type(code)] &= !bit(sq);
        self.occ[piece_color(code)] &= !bit(sq);
        self.board[sq] = NO_PIECE;
        self.hash ^= zobrist().piece_sq[code as usize][sq];
    }

    #[inline]
    pub fn all(&self) -> Bitboard {
        self.occ[WHITE] | self.occ[BLACK]
    }
    #[inline]
    pub fn king_sq(&self, color: usize) -> usize {
        self.pieces[color][KING].trailing_zeros() as usize
    }

    pub fn attacked(&self, sq: usize, by: usize) -> bool {
        let t = tables();
        let occ = self.all();
        let p = &self.pieces[by];
        (t.pawn[by ^ 1][sq] & p[PAWN]) != 0
            || (t.knight[sq] & p[KNIGHT]) != 0
            || (t.king[sq] & p[KING]) != 0
            || (t.bishop_attacks(sq, occ) & (p[BISHOP] | p[QUEEN])) != 0
            || (t.rook_attacks(sq, occ) & (p[ROOK] | p[QUEEN])) != 0
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.attacked(self.king_sq(self.side), self.side ^ 1)
    }

    pub fn compute_hash(&self) -> u64 {
        let z = zobrist();
        let mut h = 0;
        for sq in 0..64 {
            if self.board[sq] != NO_PIECE {
                h ^= z.piece_sq[self.board[sq] as usize][sq];
            }
        }
        h ^= z.castling[self.castling as usize];
        if self.ep != NO_SQ {
            h ^= z.ep_file[file_of(self.ep as usize)];
        }
        if self.side == BLACK {
            h ^= z.side;
        }
        h
    }

    pub fn from_fen(fen: &str) -> Option<Position> {
        let mut p = Position::empty();
        let mut it = fen.split_whitespace();
        let placement = it.next()?;
        let stm = it.next()?;
        let rights = it.next().unwrap_or("-");
        let eps = it.next().unwrap_or("-");
        let (mut rank, mut file) = (7i32, 0i32);
        for c in placement.bytes() {
            match c {
                b'/' => {
                    rank -= 1;
                    file = 0;
                }
                b'1'..=b'8' => file += (c - b'0') as i32,
                _ => {
                    let idx = PIECE_CHARS.iter().position(|&x| x == c)?;
                    if rank < 0 || file > 7 {
                        return None;
                    }
                    p.put(idx as u8, (rank * 8 + file) as usize);
                    file += 1;
                }
            }
        }
        p.side = if stm == "b" { BLACK } else { WHITE };
        for c in rights.bytes() {
            p.castling |= match c {
                b'K' => WHITE_OO,
                b'Q' => WHITE_OOO,
                b'k' => BLACK_OO,
                b'q' => BLACK_OOO,
                _ => 0,
            };
        }
        let eb = eps.as_bytes();
        if eb.len() == 2 && (b'a'..=b'h').contains(&eb[0]) {
            let sq = ((eb[1] - b'1') * 8 + (eb[0] - b'a')) as usize;
            // Keep the ep square only if a pawn of the side to move can actually capture there.
            if tables().pawn[p.side ^ 1][sq] & p.pieces[p.side][PAWN] != 0 {
                p.ep = sq as u8;
            }
        }
        p.halfmove = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        p.fullmove = it.next().and_then(|x| x.parse().ok()).unwrap_or(1);
        if p.pieces[WHITE][KING].count_ones() != 1 || p.pieces[BLACK][KING].count_ones() != 1 {
            return None;
        }
        p.hash = p.compute_hash();
        Some(p)
    }

    pub fn fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let code = self.board[rank * 8 + file];
                if code == NO_PIECE {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    s.push((b'0' + empty) as char);
                    empty = 0;
                }
                s.push(PIECE_CHARS[code as usize] as char);
            }
            if empty > 0 {
                s.push((b'0' + empty) as char);
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.side == WHITE { " w " } else { " b " });
        let mut r = String::new();
        for (flag, c) in [
            (WHITE_OO, 'K'),
            (WHITE_OOO, 'Q'),
            (BLACK_OO, 'k'),
            (BLACK_OOO, 'q'),
        ] {
            if self.castling & flag != 0 {
                r.push(c);
            }
        }
        s.push_str(if r.is_empty() { "-" } else { &r });
        s.push(' ');
        s.push_str(&if self.ep == NO_SQ {
            "-".to_string()
        } else {
            square_name(self.ep as usize)
        });
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }

    /// Pass the turn (for null-move pruning).
    pub fn make_null(&self) -> Position {
        let z = zobrist();
        let mut n = *self;
        if n.ep != NO_SQ {
            n.hash ^= z.ep_file[file_of(n.ep as usize)];
        }
        n.ep = NO_SQ;
        n.halfmove += 1;
        n.side ^= 1;
        n.hash ^= z.side;
        n
    }

    pub fn has_non_pawn_material(&self, color: usize) -> bool {
        self.pieces[color][KNIGHT]
            | self.pieces[color][BISHOP]
            | self.pieces[color][ROOK]
            | self.pieces[color][QUEEN]
            != 0
    }

    pub fn make(&self, m: Move) -> Position {
        let z = zobrist();
        let mut n = *self;
        let (from, to, flag) = (move_from(m), move_to(m), move_flag(m));
        let us = self.side;
        let them = us ^ 1;
        let code = self.board[from];
        let kind = piece_type(code);

        if n.ep != NO_SQ {
            n.hash ^= z.ep_file[file_of(n.ep as usize)];
        }
        n.ep = NO_SQ;
        n.halfmove = if kind == PAWN || is_capture(m) {
            0
        } else {
            self.halfmove + 1
        };

        if flag == EP_CAPTURE {
            n.remove(to ^ 8); // the captured pawn sits one rank behind the target square
        } else if is_capture(m) {
            n.remove(to);
        }
        n.remove(from);
        n.put(
            if is_promotion(m) {
                piece_code(us, promotion_type(m))
            } else {
                code
            },
            to,
        );

        if flag == KING_CASTLE {
            n.remove(from + 3);
            n.put(piece_code(us, ROOK), from + 1);
        } else if flag == QUEEN_CASTLE {
            n.remove(from - 4);
            n.put(piece_code(us, ROOK), from - 1);
        } else if flag == DOUBLE_PUSH {
            let ep_sq = (from + to) / 2;
            if tables().pawn[us][ep_sq] & n.pieces[them][PAWN] != 0 {
                n.ep = ep_sq as u8;
                n.hash ^= z.ep_file[file_of(ep_sq)];
            }
        }

        let rights = self.castling & z.castle_keep[from] & z.castle_keep[to];
        if rights != self.castling {
            n.hash ^= z.castling[self.castling as usize] ^ z.castling[rights as usize];
            n.castling = rights;
        }
        n.side = them;
        n.hash ^= z.side;
        if us == BLACK {
            n.fullmove += 1;
        }
        n
    }
}
