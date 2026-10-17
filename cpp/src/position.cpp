#include "position.h"

#include <cstring>
#include <sstream>

namespace zobrist {
U64 piece_sq[12][64];
U64 castling[16];
U64 ep_file[8];
U64 side;
}  // namespace zobrist

namespace {

// Rights that survive a move touching this square (as from- or to-square).
int castle_keep[64];

const char PIECE_CHARS[] = "PNBRQKpnbrqk";

}  // namespace

void Position::init() {
  static bool done = false;
  if (done) return;
  done = true;
  bb::init();
  U64 s = 0x2545F4914F6CDD1DULL;
  auto next = [&s]() {  // splitmix64
    U64 z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  };
  for (auto& p : zobrist::piece_sq)
    for (auto& k : p) k = next();
  for (auto& k : zobrist::castling) k = next();
  for (auto& k : zobrist::ep_file) k = next();
  zobrist::side = next();

  for (int sq = 0; sq < 64; sq++) castle_keep[sq] = 15;
  castle_keep[0] &= ~WHITE_OOO;  // a1
  castle_keep[7] &= ~WHITE_OO;   // h1
  castle_keep[4] &= ~(WHITE_OO | WHITE_OOO);
  castle_keep[56] &= ~BLACK_OOO;  // a8
  castle_keep[63] &= ~BLACK_OO;   // h8
  castle_keep[60] &= ~(BLACK_OO | BLACK_OOO);
}

void Position::clear() {
  std::memset(pieces, 0, sizeof(pieces));
  occ[WHITE] = occ[BLACK] = 0;
  std::memset(board, NO_PIECE, sizeof(board));
  side = WHITE;
  castling = 0;
  ep = NO_SQ;
  halfmove = 0;
  fullmove = 1;
  hash = 0;
}

void Position::put(int code, int sq) {
  pieces[piece_color(code)][piece_type(code)] |= bit(sq);
  occ[piece_color(code)] |= bit(sq);
  board[sq] = code;
  hash ^= zobrist::piece_sq[code][sq];
}

void Position::remove(int sq) {
  int code = board[sq];
  pieces[piece_color(code)][piece_type(code)] &= ~bit(sq);
  occ[piece_color(code)] &= ~bit(sq);
  board[sq] = NO_PIECE;
  hash ^= zobrist::piece_sq[code][sq];
}

U64 Position::compute_hash() const {
  U64 h = 0;
  for (int sq = 0; sq < 64; sq++)
    if (board[sq] != NO_PIECE) h ^= zobrist::piece_sq[board[sq]][sq];
  h ^= zobrist::castling[castling];
  if (ep != NO_SQ) h ^= zobrist::ep_file[file_of(ep)];
  if (side == BLACK) h ^= zobrist::side;
  return h;
}

bool Position::set_fen(const std::string& fen) {
  clear();
  std::istringstream in(fen);
  std::string placement, stm, rights, eps;
  if (!(in >> placement >> stm)) return false;
  in >> rights >> eps;
  int rank = 7, file = 0;
  for (char c : placement) {
    if (c == '/') {
      rank--;
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
    } else {
      const char* p = std::strchr(PIECE_CHARS, c);
      if (!p || rank < 0 || file > 7) return false;
      put(int(p - PIECE_CHARS), rank * 8 + file);
      file++;
    }
  }
  side = stm == "b" ? BLACK : WHITE;
  for (char c : rights) {
    if (c == 'K') castling |= WHITE_OO;
    if (c == 'Q') castling |= WHITE_OOO;
    if (c == 'k') castling |= BLACK_OO;
    if (c == 'q') castling |= BLACK_OOO;
  }
  if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h') {
    int sq = (eps[1] - '1') * 8 + (eps[0] - 'a');
    // Keep the ep square only if a pawn of the side to move can actually capture there.
    if (bb::pawn_attacks[side ^ 1][sq] & pieces[side][PAWN]) ep = sq;
  }
  if (in >> halfmove) in >> fullmove;
  if (popcount(pieces[WHITE][KING]) != 1 || popcount(pieces[BLACK][KING]) != 1) return false;
  hash = compute_hash();
  return true;
}

std::string Position::fen() const {
  std::string s;
  for (int rank = 7; rank >= 0; rank--) {
    int empty = 0;
    for (int file = 0; file < 8; file++) {
      int code = board[rank * 8 + file];
      if (code == NO_PIECE) {
        empty++;
        continue;
      }
      if (empty) s += char('0' + empty);
      empty = 0;
      s += PIECE_CHARS[code];
    }
    if (empty) s += char('0' + empty);
    if (rank) s += '/';
  }
  s += side == WHITE ? " w " : " b ";
  std::string r;
  if (castling & WHITE_OO) r += 'K';
  if (castling & WHITE_OOO) r += 'Q';
  if (castling & BLACK_OO) r += 'k';
  if (castling & BLACK_OOO) r += 'q';
  s += r.empty() ? "-" : r;
  s += ' ';
  s += ep == NO_SQ ? "-" : square_name(ep);
  s += ' ' + std::to_string(halfmove) + ' ' + std::to_string(fullmove);
  return s;
}

bool Position::attacked(int sq, int by) const {
  U64 occupied = all();
  const U64* p = pieces[by];
  return (bb::pawn_attacks[by ^ 1][sq] & p[PAWN]) || (bb::knight_attacks[sq] & p[KNIGHT]) ||
         (bb::king_attacks[sq] & p[KING]) || (bb::bishop_attacks(sq, occupied) & (p[BISHOP] | p[QUEEN])) ||
         (bb::rook_attacks(sq, occupied) & (p[ROOK] | p[QUEEN]));
}

U64 Position::attackers_to(int sq, U64 occupied) const {
  return (bb::pawn_attacks[BLACK][sq] & pieces[WHITE][PAWN]) | (bb::pawn_attacks[WHITE][sq] & pieces[BLACK][PAWN]) |
         (bb::knight_attacks[sq] & (pieces[WHITE][KNIGHT] | pieces[BLACK][KNIGHT])) |
         (bb::king_attacks[sq] & (pieces[WHITE][KING] | pieces[BLACK][KING])) |
         (bb::bishop_attacks(sq, occupied) &
          (pieces[WHITE][BISHOP] | pieces[BLACK][BISHOP] | pieces[WHITE][QUEEN] | pieces[BLACK][QUEEN])) |
         (bb::rook_attacks(sq, occupied) &
          (pieces[WHITE][ROOK] | pieces[BLACK][ROOK] | pieces[WHITE][QUEEN] | pieces[BLACK][QUEEN]));
}

Position Position::make(Move m) const {
  Position n = *this;
  int from = move_from(m), to = move_to(m), flag = move_flag(m);
  int us = side, them = side ^ 1;
  int code = board[from];
  int type = piece_type(code);

  if (n.ep != NO_SQ) n.hash ^= zobrist::ep_file[file_of(n.ep)];
  n.ep = NO_SQ;
  n.halfmove = (type == PAWN || is_capture(m)) ? 0 : halfmove + 1;

  if (flag == EP_CAPTURE)
    n.remove(to ^ 8);  // the captured pawn sits one rank behind the target square
  else if (is_capture(m))
    n.remove(to);

  n.remove(from);
  n.put(is_promotion(m) ? piece_code(us, promotion_type(m)) : code, to);

  if (flag == KING_CASTLE) {
    n.remove(from + 3);
    n.put(piece_code(us, ROOK), from + 1);
  } else if (flag == QUEEN_CASTLE) {
    n.remove(from - 4);
    n.put(piece_code(us, ROOK), from - 1);
  } else if (flag == DOUBLE_PUSH) {
    int ep_sq = (from + to) / 2;
    if (bb::pawn_attacks[us][ep_sq] & n.pieces[them][PAWN]) {
      n.ep = ep_sq;
      n.hash ^= zobrist::ep_file[file_of(ep_sq)];
    }
  }

  int rights = castling & castle_keep[from] & castle_keep[to];
  if (rights != castling) {
    n.hash ^= zobrist::castling[castling] ^ zobrist::castling[rights];
    n.castling = rights;
  }

  n.side = them;
  n.hash ^= zobrist::side;
  if (us == BLACK) n.fullmove++;
  return n;
}

Position Position::make_null() const {
  Position n = *this;
  if (n.ep != NO_SQ) n.hash ^= zobrist::ep_file[file_of(n.ep)];
  n.ep = NO_SQ;
  n.halfmove++;
  n.side ^= 1;
  n.hash ^= zobrist::side;
  return n;
}

std::string square_name(int sq) { return {char('a' + file_of(sq)), char('1' + rank_of(sq))}; }

std::string move_to_uci(Move m) {
  if (m == NO_MOVE) return "0000";
  std::string s = square_name(move_from(m)) + square_name(move_to(m));
  if (is_promotion(m)) s += "nbrq"[promotion_type(m) - KNIGHT];
  return s;
}
