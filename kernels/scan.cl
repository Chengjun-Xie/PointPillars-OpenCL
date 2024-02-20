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
__kernel void ScanXKernel(__global int* dev_output,
                          const __global int* dev_input,
                          const int n,
                          __local int* temp){
  int thid = get_local_id(0);
  int bid = get_group_id(0);
  int bdim = get_local_size(0);
  int offset = 1;

  temp[2 * thid] = dev_input[bid * bdim * 2 + 2 * thid];  // load input into shared memory
  temp[2 * thid + 1] = dev_input[bid * bdim * 2 + 2 * thid + 1];

  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) {
    temp[n - 1] = 0;
  }                                 // clear the last element
  for (int d = 1; d < n; d *= 2) {  // traverse down tree & build scan
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  dev_output[bid * bdim * 2 + 2 * thid] = temp[2 * thid + 1];  // write results to device memory
  if (thid + 1 == bdim) {
    dev_output[bid * bdim * 2 + 2 * thid + 1] = temp[2 * thid + 1] + dev_input[bid * bdim * 2 + 2 * thid + 1];
  } else {
    dev_output[bid * bdim * 2 + 2 * thid + 1] = temp[2 * thid + 2];
  }
}

// Prefix in y-direction, calculates the cumulative sum along y
__kernel void ScanYKernel(__global int *dev_output,
                          const __global int *dev_input,
                          const int n,
                          __local int* temp){
  int thid = get_local_id(0);
  int bid = get_group_id(0);
  int bdim = get_local_size(0);
  int gdim = get_num_groups(0);
  int offset = 1;
  temp[2 * thid] = dev_input[bid + 2 * thid * gdim];  // load input into shared memory
  temp[2 * thid + 1] = dev_input[bid + 2 * thid * gdim + gdim];
  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0) {
    temp[n - 1] = 0;
  }                                 // clear the last element
  for (int d = 1; d < n; d *= 2) {  // traverse down tree & build scan
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  dev_output[bid + 2 * thid * gdim] = temp[2 * thid + 1];  // write results to device memory
  int second_ind = 2 * thid + 2;
  if (second_ind == bdim * 2) {
    dev_output[bid + 2 * thid * gdim + gdim] = temp[2 * thid + 1] + dev_input[bid + 2 * thid * gdim + gdim];
  } else {
    dev_output[bid + 2 * thid * gdim + gdim] = temp[2 * thid + 2];
  }
}
