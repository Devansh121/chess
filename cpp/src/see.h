// Static exchange evaluation: the material outcome of a capture sequence on one square, each side
// recapturing with its least valuable attacker and free to stop when continuing would lose material.
// Sliders behind the capturing pieces (x-rays) join in as the square's attackers are removed.
#pragma once

#include "position.h"

// Material gain in centipawns for the side making capture `m` (negative: the exchange loses material).
int see(const Position& pos, Move m);
