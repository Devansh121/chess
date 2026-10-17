//! Same checks as cpp/tests/perft_test.cpp: published perft counts, the incremental hash equals a
//! from-scratch hash at every node to depth 3, and FEN round trips.

use chess_rs::movegen::{generate_legal, perft, MoveList};
use chess_rs::position::Position;
use chess_rs::PERFT_CASES;

fn check_hashes(pos: &Position, depth: u32) {
    assert_eq!(
        pos.hash,
        pos.compute_hash(),
        "hash mismatch at {}",
        pos.fen()
    );
    if depth == 0 {
        return;
    }
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    for &m in list.as_slice() {
        check_hashes(&pos.make(m), depth - 1);
    }
}

#[test]
fn perft_counts_match_published_values() {
    for (name, fen, depth, expected) in PERFT_CASES {
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&pos, depth), expected, "{name}");
    }
}

#[test]
fn incremental_hash_matches_recomputed_hash() {
    for (_, fen, _, _) in PERFT_CASES {
        check_hashes(&Position::from_fen(fen).unwrap(), 3);
    }
}

#[test]
fn fen_round_trips() {
    for (name, fen, _, _) in PERFT_CASES {
        assert_eq!(Position::from_fen(fen).unwrap().fen(), fen, "{name}");
    }
}

#[test]
fn magic_tables_match_ray_walking() {
    let t = chess_rs::bitboard::tables();
    let mut s: u64 = 12345;
    for _ in 0..20000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let occ = s & (s >> 3);
        let sq = (s % 64) as usize;
        assert_eq!(
            t.rook_attacks(sq, occ),
            chess_rs::bitboard::slider_attacks_slow(sq, occ, true)
        );
        assert_eq!(
            t.bishop_attacks(sq, occ),
            chess_rs::bitboard::slider_attacks_slow(sq, occ, false)
        );
    }
}
