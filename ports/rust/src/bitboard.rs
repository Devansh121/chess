//! Precomputed attack tables. Sliding pieces use magic bitboards with the magic numbers shared with the
//! C++ engine; each is verified when the tables are built, and a seeded search replaces any that fail.

use crate::magics::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::types::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Default)]
struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

pub struct Tables {
    pub knight: [Bitboard; 64],
    pub king: [Bitboard; 64],
    pub pawn: [[Bitboard; 64]; 2], // squares a pawn of that colour on sq attacks
    rook: [Magic; 64],
    bishop: [Magic; 64],
    rook_table: Vec<Bitboard>,
    bishop_table: Vec<Bitboard>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

#[inline]
pub fn tables() -> &'static Tables {
    TABLES.get_or_init(build)
}

impl Tables {
    #[inline]
    pub fn rook_attacks(&self, sq: usize, occ: Bitboard) -> Bitboard {
        let m = &self.rook[sq];
        self.rook_table[m.offset + (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize]
    }
    #[inline]
    pub fn bishop_attacks(&self, sq: usize, occ: Bitboard) -> Bitboard {
        let m = &self.bishop[sq];
        self.bishop_table[m.offset + (((occ & m.mask).wrapping_mul(m.magic)) >> m.shift) as usize]
    }
    #[inline]
    pub fn queen_attacks(&self, sq: usize, occ: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occ) | self.bishop_attacks(sq, occ)
    }
}

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn step(sq: usize, df: i32, dr: i32) -> Bitboard {
    let f = file_of(sq) as i32 + df;
    let r = rank_of(sq) as i32 + dr;
    if (0..8).contains(&f) && (0..8).contains(&r) {
        bit((r * 8 + f) as usize)
    } else {
        0
    }
}

/// Slow ray-walking reference (builds the tables; used by tests).
pub fn slider_attacks_slow(sq: usize, occ: Bitboard, rook: bool) -> Bitboard {
    let mut att = 0;
    for &(df, dr) in if rook { &ROOK_DIRS } else { &BISHOP_DIRS } {
        let (mut f, mut r) = (file_of(sq) as i32 + df, rank_of(sq) as i32 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as usize;
            att |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    att
}

fn relevant_mask(sq: usize, rook: bool) -> Bitboard {
    let mut mask = 0;
    for &(df, dr) in if rook { &ROOK_DIRS } else { &BISHOP_DIRS } {
        let (mut f, mut r) = (file_of(sq) as i32 + df, rank_of(sq) as i32 + dr);
        while (0..8).contains(&(f + df)) && (0..8).contains(&(r + dr)) {
            mask |= bit((r * 8 + f) as usize);
            f += df;
            r += dr;
        }
    }
    mask
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2685821657736338717)
    }
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

fn init_magics(
    rook: bool,
    known: &[u64; 64],
    magics: &mut [Magic; 64],
    table: &mut Vec<Bitboard>,
    rng: &mut Rng,
) {
    let total: usize = (0..64)
        .map(|sq| 1usize << relevant_mask(sq, rook).count_ones())
        .sum();
    table.clear();
    table.resize(total, 0);
    let mut offset = 0;
    let mut stamp: Vec<u32> = Vec::new();
    for sq in 0..64 {
        let mask = relevant_mask(sq, rook);
        let bits = mask.count_ones();
        let size = 1usize << bits;
        // Every subset of the mask (carry-rippler enumeration) with its true attack set.
        let mut occs = Vec::with_capacity(size);
        let mut atts = Vec::with_capacity(size);
        let mut sub: u64 = 0;
        loop {
            occs.push(sub);
            atts.push(slider_attacks_slow(sq, sub, rook));
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }
        let shift = 64 - bits;
        stamp.clear();
        stamp.resize(size, 0);
        let mut attempt: u32 = 0;
        let try_magic =
            |magic: u64, attempt: u32, table: &mut Vec<Bitboard>, stamp: &mut Vec<u32>| -> bool {
                for i in 0..occs.len() {
                    let idx = ((occs[i].wrapping_mul(magic)) >> shift) as usize;
                    // Two blocker sets may share a slot only if they give the same attacks.
                    if stamp[idx] != attempt {
                        stamp[idx] = attempt;
                        table[offset + idx] = atts[i];
                    } else if table[offset + idx] != atts[i] {
                        return false;
                    }
                }
                true
            };
        attempt += 1;
        let mut magic = known[sq];
        if !try_magic(magic, attempt, table, &mut stamp) {
            loop {
                attempt += 1;
                magic = rng.sparse();
                if (mask.wrapping_mul(magic) & 0xFF00_0000_0000_0000).count_ones() < 6 {
                    continue;
                }
                if try_magic(magic, attempt, table, &mut stamp) {
                    break;
                }
            }
        }
        magics[sq] = Magic {
            mask,
            magic,
            shift,
            offset,
        };
        offset += size;
    }
}

fn build() -> Tables {
    let mut t = Tables {
        knight: [0; 64],
        king: [0; 64],
        pawn: [[0; 64]; 2],
        rook: [Magic::default(); 64],
        bishop: [Magic::default(); 64],
        rook_table: Vec::new(),
        bishop_table: Vec::new(),
    };
    for sq in 0..64 {
        t.knight[sq] = step(sq, 1, 2)
            | step(sq, 2, 1)
            | step(sq, 2, -1)
            | step(sq, 1, -2)
            | step(sq, -1, -2)
            | step(sq, -2, -1)
            | step(sq, -2, 1)
            | step(sq, -1, 2);
        t.king[sq] = step(sq, 1, 0)
            | step(sq, 1, 1)
            | step(sq, 0, 1)
            | step(sq, -1, 1)
            | step(sq, -1, 0)
            | step(sq, -1, -1)
            | step(sq, 0, -1)
            | step(sq, 1, -1);
        t.pawn[WHITE][sq] = step(sq, -1, 1) | step(sq, 1, 1);
        t.pawn[BLACK][sq] = step(sq, -1, -1) | step(sq, 1, -1);
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    init_magics(true, &ROOK_MAGICS, &mut t.rook, &mut t.rook_table, &mut rng);
    init_magics(
        false,
        &BISHOP_MAGICS,
        &mut t.bishop,
        &mut t.bishop_table,
        &mut rng,
    );
    t
}
