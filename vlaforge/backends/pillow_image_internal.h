#ifndef VLAFORGE_PILLOW_IMAGE_INTERNAL_H_
#define VLAFORGE_PILLOW_IMAGE_INTERNAL_H_
#include <stdint.h>

struct VLAForgePillowResize {
  const uint8_t* data;
  uint8_t* output;
  int source_width, source_height, width, height;
  int pad_to_square, filter;
  uint8_t background[3];
};
#ifdef __cplusplus
extern "C" {
#endif
int vlaforge_pillow_resize(const struct VLAForgePillowResize* request, const char** error);
#ifdef __cplusplus
}
#endif
#endif
