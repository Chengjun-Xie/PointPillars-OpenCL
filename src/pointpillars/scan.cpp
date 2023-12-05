#include "pointpillars/scan.hpp"

namespace pointpillars {
void ScanX(int *dev_output, const int *dev_input, int w, int h, int n) {
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (j == 0) {
        dev_output[i * w + j] = dev_input[i * w + j];
      } else {
        dev_output[i * w + j] =
            dev_input[i * w + j] + dev_output[i * w + j - 1];
      }
    }
  }
}

void ScanY(int *dev_output, const int *dev_input, int w, int h, int n) {
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {
      if (j == 0) {
        dev_output[i + j * w] = dev_input[i + j * w];
      } else {
        dev_output[i + j * w] =
            dev_input[i + j * w] + dev_output[i + (j - 1) * w];
      }
    }
  }
}
}  // namespace pointpillars
