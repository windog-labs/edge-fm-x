#pragma once
#include <stdexcept>
#define AT_CUDA_CHECK(expression) do { if ((expression) != 0) throw std::runtime_error("injected CUDA destroy failure"); } while (false)
