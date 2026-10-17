//! UCI front end for the Rust port (`chess-rs`), with `bench [depth]` for node-count comparison with the
//! C++ engine. The search runs on a worker thread so "stop" and "quit" work mid-search.

use chess_rs::movegen::{generate_legal, MoveList};
use chess_rs::position::Position;
use chess_rs::search::{format_score, Limits, Searcher};
use chess_rs::types::*;
use std::io::{self, BufRead, Write};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Same positions as the C++ engine's bench.
const BENCH_FENS: [&str; 8] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 8",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "8/8/4k3/8/2p5/8/B2K4/8 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
];

fn send(line: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

fn parse_move(pos: &Position, s: &str) -> Option<Move> {
    let mut list = MoveList::new();
    generate_legal(pos, &mut list);
    list.as_slice()
        .iter()
        .copied()
        .find(|&m| move_to_uci(m) == s)
}

fn bench(depth: i32) {
    let mut s = Searcher::new(16);
    let mut total = 0u64;
    let start = Instant::now();
    for fen in BENCH_FENS {
        let pos = Position::from_fen(fen).unwrap();
        s.new_game();
        let r = s.search(
            &pos,
            &[],
            Limits {
                depth,
                ..Default::default()
            },
            |_, _, _, _, _| {},
        );
        total += r.nodes;
    }
    let secs = start.elapsed().as_secs_f64();
    println!(
        "bench depth {depth}: {total} nodes {secs:.2} s {} nps",
        (total as f64 / secs.max(1e-9)) as u64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "bench" {
        bench(args.get(2).and_then(|d| d.parse().ok()).unwrap_or(7));
        return;
    }
    let searcher = Arc::new(Mutex::new(Searcher::new(64)));
    let stop = searcher.lock().unwrap().stop.clone();
    let mut pos = Position::from_fen(START_FEN).unwrap();
    let mut history: Vec<u64> = Vec::new();
    let mut worker: Option<thread::JoinHandle<()>> = None;
    let stop_search = |worker: &mut Option<thread::JoinHandle<()>>| {
        stop.store(true, Ordering::Relaxed);
        if let Some(w) = worker.take() {
            let _ = w.join();
        }
    };

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => {
                send("id name chess-rs");
                send("id author Devansh Wassista");
                send("option name Hash type spin default 64 min 1 max 4096");
                send("uciok");
            }
            Some("isready") => send("readyok"),
            Some("setoption") => {
                if let (Some(&"Hash"), Some(v)) = (tokens.get(2), tokens.get(4)) {
                    stop_search(&mut worker);
                    if let Ok(mb) = v.parse::<usize>() {
                        searcher.lock().unwrap().resize(mb.max(1));
                    }
                }
            }
            Some("ucinewgame") => {
                stop_search(&mut worker);
                searcher.lock().unwrap().new_game();
            }
            Some("position") => {
                stop_search(&mut worker);
                let moves_at = tokens.iter().position(|&t| t == "moves");
                let fen = if tokens.get(1) == Some(&"fen") {
                    tokens[2..moves_at.unwrap_or(tokens.len())].join(" ")
                } else {
                    START_FEN.to_string()
                };
                pos = Position::from_fen(&fen)
                    .unwrap_or_else(|| Position::from_fen(START_FEN).unwrap());
                history.clear();
                if let Some(i) = moves_at {
                    for s in &tokens[i + 1..] {
                        let Some(m) = parse_move(&pos, s) else { break };
                        history.push(pos.hash);
                        pos = pos.make(m);
                    }
                }
            }
            Some("go") => {
                stop_search(&mut worker);
                let mut limits = Limits::default();
                let (mut wtime, mut btime, mut winc, mut binc, mut mtg) = (0.0, 0.0, 0.0, 0.0, 0.0);
                let mut i = 1;
                while i < tokens.len() {
                    let v = tokens
                        .get(i + 1)
                        .and_then(|x| x.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    match tokens[i] {
                        "depth" => limits.depth = v as i32,
                        "movetime" => limits.time_ms = v,
                        "nodes" => limits.nodes = v as u64,
                        "wtime" => wtime = v,
                        "btime" => btime = v,
                        "winc" => winc = v,
                        "binc" => binc = v,
                        "movestogo" => mtg = v,
                        _ => {}
                    }
                    i += 2;
                }
                if limits.time_ms == 0.0 {
                    let (remaining, inc) = if pos.side == WHITE {
                        (wtime, winc)
                    } else {
                        (btime, binc)
                    };
                    if remaining > 0.0 {
                        let t = remaining / if mtg > 0.0 { mtg } else { 30.0 } + 0.8 * inc;
                        limits.time_ms = (t.min(remaining / 3.0) - 10.0).max(5.0);
                    }
                }
                stop.store(false, Ordering::Relaxed);
                let (searcher, root, hist) = (searcher.clone(), pos, history.clone());
                worker = Some(thread::spawn(move || {
                    let mut s = searcher.lock().unwrap();
                    let r = s.search(&root, &hist, limits, |depth, score, nodes, ms, best| {
                        let nps = if ms > 0.0 {
                            (nodes as f64 * 1000.0 / ms) as u64
                        } else {
                            0
                        };
                        send(&format!(
                            "info depth {depth} score {} nodes {nodes} nps {nps} time {} pv {}",
                            format_score(score),
                            ms as u64,
                            move_to_uci(best)
                        ));
                    });
                    send(&format!("bestmove {}", move_to_uci(r.best)));
                }));
            }
            Some("stop") => stop_search(&mut worker),
            Some("quit") => break,
            _ => {}
        }
    }
    stop_search(&mut worker);
}
