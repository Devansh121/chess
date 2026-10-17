//! Basic types: bitboards, squares, pieces and the 16-bit move encoding (same layout as cpp/src/types.h).

pub type Bitboard = u64;
pub type Move = u16;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;

/// Piece code = colour * 6 + type; `NO_PIECE` marks an empty square.
pub const NO_PIECE: u8 = 12;
pub const NO_SQ: u8 = 64;
pub const MAX_MOVES: usize = 256;

#[inline]
pub fn piece_code(color: usize, kind: usize) -> u8 {
    (color * 6 + kind) as u8
}
#[inline]
pub fn piece_color(code: u8) -> usize {
    code as usize / 6
}
#[inline]
pub fn piece_type(code: u8) -> usize {
    code as usize % 6
}

#[inline]
pub fn file_of(sq: usize) -> usize {
    sq & 7
}
#[inline]
pub fn rank_of(sq: usize) -> usize {
    sq >> 3
}
#[inline]
pub fn bit(sq: usize) -> Bitboard {
    1u64 << sq
}
#[inline]
pub fn pop_lsb(b: &mut Bitboard) -> usize {
    let s = b.trailing_zeros() as usize;
    *b &= *b - 1;
    s
}

// Move flags: bit 2 = capture, bit 3 = promotion; promotion piece = KNIGHT + (flag & 3).
pub const QUIET: u16 = 0;
pub const DOUBLE_PUSH: u16 = 1;
pub const KING_CASTLE: u16 = 2;
pub const QUEEN_CASTLE: u16 = 3;
pub const CAPTURE: u16 = 4;
pub const EP_CAPTURE: u16 = 5;
pub const PROMO: u16 = 8;
pub const PROMO_CAPTURE: u16 = 12;
pub const NO_MOVE: Move = 0;

#[inline]
pub fn make_move(from: usize, to: usize, flag: u16) -> Move {
    (from as u16) | ((to as u16) << 6) | (flag << 12)
}
#[inline]
pub fn move_from(m: Move) -> usize {
    (m & 63) as usize
}
#[inline]
pub fn move_to(m: Move) -> usize {
    ((m >> 6) & 63) as usize
}
#[inline]
pub fn move_flag(m: Move) -> u16 {
    m >> 12
}
#[inline]
pub fn is_capture(m: Move) -> bool {
    move_flag(m) & 4 != 0
}
#[inline]
pub fn is_promotion(m: Move) -> bool {
    move_flag(m) & 8 != 0
}
#[inline]
pub fn promotion_type(m: Move) -> usize {
    KNIGHT + (move_flag(m) & 3) as usize
}

pub fn square_name(sq: usize) -> String {
    format!(
        "{}{}",
        (b'a' + file_of(sq) as u8) as char,
        (b'1' + rank_of(sq) as u8) as char
    )
}

pub fn move_to_uci(m: Move) -> String {
    if m == NO_MOVE {
        return "0000".to_string();
    }
    let mut s = square_name(move_from(m)) + &square_name(move_to(m));
    if is_promotion(m) {
        s.push(['n', 'b', 'r', 'q'][promotion_type(m) - KNIGHT]);
    }
    s
}
