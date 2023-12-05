__kernel void MaskAnchorsSimpleKernel(const __global float *anchors_px,
                                      const __global float *anchors_py,
                                      const __global int *pillar_map, 
                                      __global int *anchor_mask,
                                      const __global float *anchors_rad, 
                                      const float min_x_range,
                                      const float min_y_range,
                                      const float pillar_x_size,
                                      const float pillar_y_size, 
                                      const int grid_x_size,
                                      const int grid_y_size, 
                                      const int C,
                                      const int R,
                                      const int H,
                                      const int W,
                                      const int length) {
  const int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (index >= length) {return;}

  const int y = index / (H * C * R);
  const int x = (index - y * H * C * R) / (C * R);
  const int c = (index - y * H * C * R - x * C * R) / R;

  float rad = anchors_rad[c];

  float x_anchor = anchors_px[index];
  float y_anchor = anchors_py[index];

  int anchor_coordinates_min_x = (x_anchor - rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_min_y = (y_anchor - rad - min_y_range) / pillar_y_size;
  int anchor_coordinates_max_x = (x_anchor + rad - min_x_range) / pillar_x_size;
  int anchor_coordinates_max_y = (y_anchor + rad - min_y_range) / pillar_y_size;

  anchor_coordinates_min_x = max(anchor_coordinates_min_x, 0);
  anchor_coordinates_min_y = max(anchor_coordinates_min_y, 0);
  anchor_coordinates_max_x = min(anchor_coordinates_max_x, (int)(grid_x_size - 1));
  anchor_coordinates_max_y = min(anchor_coordinates_max_y, (int)(grid_y_size - 1));

  // cumulative sum difference
  int bottom_left = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_min_x];
  int top_left = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_min_x];

  int bottom_right = pillar_map[anchor_coordinates_max_y * grid_x_size + anchor_coordinates_max_x];
  int top_right = pillar_map[anchor_coordinates_min_y * grid_x_size + anchor_coordinates_max_x];

  // Area calculation
  int area = bottom_right - top_right - bottom_left + top_left;

  if (area >= 1) {
    anchor_mask[index] = 1;
  } else {
    anchor_mask[index] = 0;
  }
}