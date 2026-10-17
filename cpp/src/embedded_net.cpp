#include "nnue.h"

#if defined(CHESS_EMBEDDED_NET)
extern "C" const char embedded_net_data[];
extern "C" const char embedded_net_end[];
#endif

namespace nnue {

std::string Network::load_embedded() {
#if defined(CHESS_EMBEDDED_NET)
  return load_from_memory(embedded_net_data, size_t(embedded_net_end - embedded_net_data));
#else
  return "this build has no embedded network";
#endif
}

}  // namespace nnue
