#pragma once
// Search margins, exposed as UCI spin options so tools/spsa.py can tune them by self-play.

#include <string>

namespace tune {

// name, default, min, max
#define TUNE_PARAMS(X)            \
  X(RazorMargin, 250, 50, 600)    \
  X(RfpMargin, 80, 20, 200)       \
  X(NmpEvalDiv, 200, 50, 500)     \
  X(FutBase, 100, 0, 300)         \
  X(FutMul, 100, 30, 250)         \
  X(SeeQuiet, 60, 10, 150)        \
  X(SeeCapture, 100, 20, 250)     \
  X(LmrBase, 75, 0, 200)          \
  X(LmrDiv, 225, 100, 400)        \
  X(LmpBase, 3, 1, 8)             \
  X(AspDelta, 50, 10, 150)        \
  X(HistBonusMax, 400, 100, 1200)

#define TUNE_DECLARE(name, def, lo, hi) inline int name = def;
TUNE_PARAMS(TUNE_DECLARE)
#undef TUNE_DECLARE

struct Param {
  const char* name;
  int* value;
  int def, lo, hi;
};

#define TUNE_ENTRY(name, def, lo, hi) {#name, &name, def, lo, hi},
inline Param params[] = {TUNE_PARAMS(TUNE_ENTRY)};
#undef TUNE_ENTRY

}  // namespace tune
