// Hand-written evaluation, ported from engine/eval.py with identical numbers: material, piece-square
// tables generated from simple rules, tapered between middlegame and endgame, bishop pair bonus.
#pragma once

#include "position.h"

namespace eval {

constexpr int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 0};

void init();
int evaluate_white(const Position& pos);  // centipawns from White's view
int evaluate(const Position& pos);        // centipawns from the side to move's view

}  // namespace eval
