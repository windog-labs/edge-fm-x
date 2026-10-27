#ifndef VLAFORGE_BACKENDS_TENSORRT_PRECISION_PLUGINS_H_
#define VLAFORGE_BACKENDS_TENSORRT_PRECISION_PLUGINS_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Registers optional FLOAT-carrier BF16 rounding, MatMul, and scalar F64 sin/cos
 * plugins; returns 1 on success. VLAForgeScalarSinCosF64 v1 (empty namespace)
 * takes one static volume-one FLOAT/LINEAR tensor (rank 0 or all-one dimensions)
 * and returns FLOAT/LINEAR [1, 2*N], sin first, cos second. Its sole creator
 * field, frequency_words, is INT32[2*N], 1 <= N <= 16384: each binary64 frequency
 * is encoded as signed low then high 32-bit words, preserving bits with memcpy.
 * Frequencies must be finite. Math is binary64 multiply/sin/cos, then FLOAT RNE.
 * Initialize/attach before capture; retire captured graphs before destroy/detach.
 * CUDA cleanup fences pending work; failed fences poison/leak until process exit.
 */
int vlaforge_register_tensorrt_precision_plugins(void);

#ifdef __cplusplus
}
#endif

#endif  /* VLAFORGE_BACKENDS_TENSORRT_PRECISION_PLUGINS_H_ */
