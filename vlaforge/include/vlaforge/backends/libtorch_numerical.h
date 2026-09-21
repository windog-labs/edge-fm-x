#ifndef VLAFORGE_BACKENDS_LIBTORCH_NUMERICAL_H_
#define VLAFORGE_BACKENDS_LIBTORCH_NUMERICAL_H_

#include "vlaforge/runtime/numerical.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Validation-only provider for libtorch.python_context_v2/1. No setters or
 * Python runtime. Covers 22 declared flags plus release/API identifiers, not
 * all execution settings. Requires one matching LibTorch instance per worker.
 */
const VLAForgeNumericalProviderApi* vlaforge_libtorch_numerical_provider_api(void);

#define VLAFORGE_LIBTORCH_NUMERICAL_WORKER_ABI_VERSION 1u
#define VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS 1u
#define VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD 2u

typedef struct VLAForgeLibTorchNumericalWorkerOptions {
  size_t struct_size;
  uint32_t abi_version;
  uint32_t acknowledgements;
} VLAForgeLibTorchNumericalWorkerOptions;

/* Explicit worker bootstrap, never called by the provider or Session. Caller
 * must own the process and calling thread before starting model/device work.
 * Both acknowledgements are mandatory, not proof of external-thread isolation.
 * One policy may be initialized before the first lease; later retuning is
 * refused. On error the declared flags are restored, or the worker is poisoned
 * and must exit. Unlisted execution settings are outside this contract.
 */
VLAForgeStatus vlaforge_libtorch_numerical_initialize_worker(
    const VLAForgeNumericalRequirementView* requirement,
    const VLAForgeLibTorchNumericalWorkerOptions* options);

#ifdef __cplusplus
}
#endif
#endif
