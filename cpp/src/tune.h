#pragma once
// Search margins, exposed as UCI spin options so tools/spsa.py can tune them by self-play.

#include <string>

namespace tune {

// name, default, min, max
#define TUNE_PARAMS(X)         \
  X(RazorMargin, 199, 50, 600) \
  X(RfpMargin, 71, 20, 200)    \
  X(NmpEvalDiv, 218, 50, 500)  \
  X(FutBase, 106, 0, 300)      \
  X(FutMul, 94, 30, 250)       \
  X(SeeQuiet, 55, 10, 150)     \
  X(SeeCapture, 90, 20, 250)   \
  X(LmrBase, 97, 0, 200)       \
  X(LmrDiv, 197, 100, 400)     \
  X(LmpBase, 3, 1, 8)          \
  X(AspDelta, 51, 10, 150)     \
  X(HistBonusMax, 397, 100, 1200)

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
