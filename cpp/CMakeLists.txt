cmake_minimum_required(VERSION 3.16)
project(chess_engine CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Tune for the build machine by default; CI turns this off so binaries run anywhere.
option(NATIVE "Compile with -march=native" ON)
option(SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer (-O1, debug info)" OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra)
if(NATIVE)
  add_compile_options(-march=native)
endif()
if(SANITIZE)
  add_compile_options(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(engine_core STATIC
  src/bitboard.cpp
  src/position.cpp
  src/movegen.cpp
  src/eval.cpp
  src/tt.cpp
  src/search.cpp
  src/see.cpp
  src/nnue.cpp
  src/embedded_net.cpp
)
target_include_directories(engine_core PUBLIC src)

# Embed the default network (nets/default.nnue) into the binary when it exists.
set(EMBED_NET_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../nets/default.nnue" CACHE FILEPATH "Network to embed")
if(EXISTS "${EMBED_NET_FILE}")
  configure_file(src/embedded_net.S.in "${CMAKE_CURRENT_BINARY_DIR}/embedded_net.S" @ONLY)
  set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/embedded_net.S" PROPERTIES OBJECT_DEPENDS "${EMBED_NET_FILE}")
  target_sources(engine_core PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/embedded_net.S")
  target_compile_definitions(engine_core PRIVATE CHESS_EMBEDDED_NET)
  message(STATUS "Embedding network ${EMBED_NET_FILE}")
else()
  message(STATUS "No network at ${EMBED_NET_FILE}; building without an embedded network")
endif()

add_executable(chess-engine src/main.cpp)
target_link_libraries(chess-engine engine_core pthread)

add_executable(perft_test tests/perft_test.cpp)
target_link_libraries(perft_test engine_core)

add_executable(search_test tests/search_test.cpp)
target_link_libraries(search_test engine_core)

add_executable(nnue_test tests/nnue_test.cpp)
target_link_libraries(nnue_test engine_core)

enable_testing()
add_test(NAME perft COMMAND perft_test)
add_test(NAME search COMMAND search_test)
add_test(NAME nnue COMMAND nnue_test)
