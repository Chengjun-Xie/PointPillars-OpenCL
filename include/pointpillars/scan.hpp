#pragma once
#include <cstdint>

namespace pointpillars {

// Prefix sum in 2D coordinates
//
// These functions calculate the cumulative sum along X or Y in a 2D array
//
//          X--->
//            W
//  Y    o-------------
//  |    |
//  |  H |
//  v    |
//       |
//
// For details about the algorithm please check:
//   Sengupta, Shubhabrata & Lefohn, Aaron & Owens, John. (2006). A
//   Work-Efficient Step-Efficient Prefix Sum Algorithm.
//

// Prefix in x-direction, calculates the cumulative sum along x
void ScanX(int *dev_output, const int *dev_input, int w, int h, int n);

// Prefix in y-direction, calculates the cumulative sum along y
void ScanY(int *dev_output, const int *dev_input, int w, int h, int n);

}  // namespace pointpillars
