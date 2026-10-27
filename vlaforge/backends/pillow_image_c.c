#include "pillow_image_internal.h"
#include "Imaging.h"

static _Thread_local const char* imaging_error;

/* The C++ caller serializes access to the standalone memory arena. */
void ImagingSectionEnter(ImagingSectionCookie* cookie) { if (cookie) *cookie = NULL; }
void ImagingSectionLeave(ImagingSectionCookie* cookie) { (void)cookie; }
void* ImagingError_OSError(void) { imaging_error = "image library OS error"; return NULL; }
void* ImagingError_MemoryError(void) { imaging_error = "image library allocation failed"; return NULL; }
void* ImagingError_ModeError(void) { imaging_error = "image library mode error"; return NULL; }
void* ImagingError_Mismatch(void) { imaging_error = "image library mismatch"; return NULL; }
void* ImagingError_ValueError(const char* message) { imaging_error = message ? message : "image library value error"; return NULL; }
void ImagingError_Clear(void) { imaging_error = NULL; }

int vlaforge_pillow_resize(const struct VLAForgePillowResize* p, const char** error) {
  const int width = p->pad_to_square && p->source_height > p->source_width ? p->source_height : p->source_width;
  const int height = p->pad_to_square ? width : p->source_height;
  const int left = (width - p->source_width) / 2, top = (height - p->source_height) / 2;
  Imaging input = NULL, output = NULL;
  int success = 0;
  float box[4] = {0, 0, (float)width, (float)height};
  ImagingError_Clear();
  input = ImagingNew("RGB", width, height);
  if (!input) goto done;
  for (int y = 0; y < height; ++y) {
    uint8_t* row = (uint8_t*)input->image[y];
    for (int x = 0; x < width; ++x) {
      const int actual = x >= left && x - left < p->source_width && y >= top && y - top < p->source_height;
      for (int channel = 0; channel < 3; ++channel) {
        row[x * 4 + channel] = actual ? p->data[((size_t)(y - top) * p->source_width + x - left) * 3 + channel] : p->background[channel];
      }
      row[x * 4 + 3] = 0;
    }
  }
  output = ImagingResample(input, p->width, p->height, p->filter, box);
  if (!output) goto done;
  for (int y = 0; y < p->height; ++y) {
    const uint8_t* row = (const uint8_t*)output->image[y];
    for (int x = 0; x < p->width; ++x) {
      for (int channel = 0; channel < 3; ++channel) p->output[((size_t)y * p->width + x) * 3 + channel] = row[x * 4 + channel];
    }
  }
  success = 1;
done:
  ImagingDelete(output);
  ImagingDelete(input);
  if (error) *error = imaging_error ? imaging_error : "native image preparation failed";
  return success;
}
