#include "libtorch_graph_cleanup.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

struct Keeper { ~Keeper() { ++scoped_reclaims; } };

int main(int argc, char** argv) {
  assert(argc == 2);
  fail_stage = std::atoi(argv[1]);
  auto graph = std::make_unique<vlaforge::backends::CheckedCUDAGraph>(true);
  auto keeper = std::make_unique<Keeper>();
  bool success = false;
  try {
    if (fail_stage == 4) throw std::runtime_error("injected stream drain failure");
    graph->ResetChecked();
    graph->ResetChecked();
    graph.reset();
    keeper.reset();
    success = true;
  } catch (...) {
    assert(graph != nullptr);
    assert(graph->owns_exec() == (fail_stage == 1 || fail_stage >= 4));
    assert(graph->owns_graph() == (fail_stage != 3));
    assert(graph->owns_pool_ref());
    (void)graph.release();
    (void)keeper.release();
  }
  assert(current_device == 5 && unsafe_cleanup == 0);
  assert(success == (fail_stage == 0));
  assert(base_destructors == (success ? 1 : 0));
  assert(scoped_reclaims == (success ? 1 : 0));
  assert(exec_calls == (fail_stage >= 4 ? 0 : 1));
  assert(graph_calls == (fail_stage == 1 || fail_stage >= 4 ? 0 : 1));
  assert(release_calls == (fail_stage == 0 || fail_stage == 3 ? 1 : 0));
  std::cout << "CPU-only checked cleanup stage " << fail_stage << ": passed\n";
}
