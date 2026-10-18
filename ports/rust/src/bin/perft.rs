//! Perft benchmark: runs the shared test positions and reports node counts and speed
//! (the same work as cpp/tests/perft_test.cpp without its hash checks).

use chess_rs::movegen::perft;
use chess_rs::position::Position;
use chess_rs::PERFT_CASES;
use std::time::Instant;

fn main() {
    // Build the tables before timing.
    let _ = chess_rs::bitboard::tables();
    let _ = chess_rs::position::zobrist();
    let mut total = 0u64;
    let mut ok = true;
    let start = Instant::now();
    for (name, fen, depth, expected) in PERFT_CASES {
        let pos = Position::from_fen(fen).expect("valid fen");
        let n = perft(&pos, depth);
        total += n;
        ok &= n == expected;
        println!(
            "{name:<9} depth {depth}: {n} (expected {expected}){}",
            if n == expected { "" } else { "  MISMATCH" }
        );
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "{total} nodes in {secs:.2} s ({:.1} M nodes/s)",
        total as f64 / secs / 1e6
    );
    if !ok {
        std::process::exit(1);
    }
}
