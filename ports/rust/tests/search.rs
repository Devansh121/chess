//! Search and evaluation checks mirroring cpp/tests/search_test.cpp.

use chess_rs::eval::evaluate_white;
use chess_rs::position::Position;
use chess_rs::search::{Limits, Searcher, MATE};
use chess_rs::types::move_to_uci;

fn best(fen: &str, depth: i32) -> (String, i32) {
    let mut s = Searcher::new(16);
    let r = s.search(
        &Position::from_fen(fen).unwrap(),
        &[],
        Limits {
            depth,
            ..Default::default()
        },
        |_, _, _, _, _| {},
    );
    (move_to_uci(r.best), r.score)
}

#[test]
fn mates_and_tactics() {
    assert_eq!(
        best("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 4),
        ("a1a8".to_string(), MATE - 1)
    );
    assert_eq!(
        best(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            4
        )
        .0,
        "h5f7"
    );
    assert_eq!(best("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1", 5).1, MATE - 3);
    assert_eq!(best("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3).0, "d1d5");
    let (m, s) = best("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1", 1);
    assert!(m != "d1d5" && s > 600, "quiescence sees the recapture");
    assert_eq!(
        best("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3).0,
        "0000",
        "stalemate: no move"
    );
}

#[test]
fn evaluation_matches_the_cpp_and_python_values() {
    assert_eq!(
        evaluate_white(
            &Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .unwrap()
        ),
        0
    );
    assert_eq!(
        evaluate_white(&Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap()),
        899
    );
    assert_eq!(
        evaluate_white(
            &Position::from_fen(
                "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8"
            )
            .unwrap()
        ),
        81
    );
}
