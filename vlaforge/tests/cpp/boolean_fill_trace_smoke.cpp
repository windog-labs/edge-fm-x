#include <torch/script.h>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  torch::NoGradGuard no_grad;
  auto model = torch::jit::load(argv[1], torch::kCPU);
  const auto result = model.forward({torch::ones({3})}).toTuple()->elements();
  if (result.size() != 2) return 3;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto value = result[i].toTensor();
    if (value.scalar_type() != torch::kBool || value.numel() != 3 ||
        !value.eq(static_cast<bool>(i)).all().item<bool>()) return 4;
  }
  return 0;
}
