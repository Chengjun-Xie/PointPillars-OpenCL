#pragma once

// using MACRO to allocate memory inside kernel
#define NUM_3D_BOX_CORNERS_MACRO 8
#define NUM_2D_BOX_CORNERS_MACRO 4

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// Returns the next power of 2 for a given number
uint32_t inline NextPower(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}
