#include "see.h"

#include <algorithm>

#include "eval.h"

namespace {
// The king's value only matters as "can't be recaptured into"; it is never actually traded.
constexpr int SEE_VALUE[6] = {100, 320, 330, 500, 900, 20000};
}  // namespace

int see(const Position& pos, Move m) {
  int from = move_from(m), to = move_to(m);
  bool ep = move_flag(m) == EP_CAPTURE;
  int gain[40];
  int d = 0;
  U64 occupied = pos.all() ^ bit(from);
  if (ep) occupied ^= bit(to ^ 8);
  gain[0] = SEE_VALUE[ep ? PAWN : piece_type(pos.board[to])];
  int on_square = piece_type(pos.board[from]);  // the piece that would be captured next
  if (is_promotion(m)) {
    gain[0] += SEE_VALUE[promotion_type(m)] - SEE_VALUE[PAWN];
    on_square = promotion_type(m);
  }
  int side = pos.side ^ 1;
  U64 attackers = pos.attackers_to(to, occupied) & occupied;

  for (;;) {
    U64 mine = attackers & pos.occ[side];
    if (!mine) break;
    int type = PAWN;
    while (!(mine & pos.pieces[side][type])) type++;
    // The king may only recapture if the opponent has nothing left attacking the square.
    if (type == KING && (attackers & pos.occ[side ^ 1])) break;
    d++;
    gain[d] = SEE_VALUE[on_square] - gain[d - 1];
    if (std::max(-gain[d - 1], gain[d]) < 0) break;  // neither side wants to continue from here
    occupied ^= bit(lsb(mine & pos.pieces[side][type]));
    attackers = pos.attackers_to(to, occupied) & occupied;  // re-scan to pick up x-rays
    on_square = type;
    side ^= 1;
  }
  // Each side may stop instead of recapturing: fold the sequence back to the first capture.
  while (d > 0) {
    gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    d--;
  }
  return gain[0];
}
