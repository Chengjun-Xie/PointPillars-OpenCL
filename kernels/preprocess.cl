

// This kernel is called on each point of the Point Cloud.  
// It calculates the coordinates of the point in the 2D pillar
// map and adds it to the corresponding pillar.
__kernel void MakePillarHistoKernel(const __global float* dev_points,
                                    __global float* dev_pillar_x_in_coors,
                                    __global float* dev_pillar_y_in_coors,
                                    __global float* dev_pillar_z_in_coors,
                                    __global float* dev_pillar_i_in_coors,
                                    __global int* pillar_count_histo,
                                    const int num_points,
                                    const int max_points_per_pillar,
                                    const int grid_x_size,
                                    const int grid_y_size,
                                    const int grid_z_size,
                                    const float min_x_range,
                                    const float min_y_range,
                                    const float min_z_range,
                                    const float pillar_x_size,
                                    const float pillar_y_size,
                                    const float pillar_z_size) 
{
  int point_index = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (point_index >= num_points) {
    return;
  } 

  // Indexes in the pillar map
  int xIndex = floor((float)((dev_points[point_index * 4 + 0] - min_x_range) / pillar_x_size));
  int yIndex = floor((float)((dev_points[point_index * 4 + 1] - min_y_range) / pillar_y_size));
  int zIndex = floor((float)((dev_points[point_index * 4 + 2] - min_z_range) / pillar_z_size));

  // Check if it is within grid range
  if (xIndex >= 0 && xIndex < grid_x_size && 
      yIndex >= 0 && yIndex < grid_y_size && 
      zIndex >= 0 && zIndex < grid_z_size) {
    // increase the point count
    int count = atomic_add(&pillar_count_histo[yIndex * grid_x_size + xIndex], 1);
    if (count < max_points_per_pillar) {
      // pillar index
      int pillarIndex = yIndex * grid_x_size * max_points_per_pillar + xIndex * max_points_per_pillar;

      // pointPillarIndex is the m-point in the n-pillar
      int pointPillarIndex = pillarIndex + count;

      // add point to pillar data
      dev_pillar_x_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 0];
      dev_pillar_y_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 1];
      dev_pillar_z_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 2];
      dev_pillar_i_in_coors[pointPillarIndex] = dev_points[point_index * 4 + 3];
    }
  }
}


// This kernel is executed on a specific location in the pillar map.
// It will test if the corresponding pillar has points.
// In such case it will mark the pillar for use as input to the PillarFeatureExtraction
// A pillar mask is also generated and can be used to optimize the decoding.
__kernel void MakePillarIndexKernel(const __global int* dev_pillar_count_histo,
                                    __global int* dev_counter,
                                    __global int* dev_pillar_count,
                                    __global int* dev_x_coors,
                                    __global int* dev_y_coors,
                                    __global float* dev_x_coors_for_sub,
                                    __global float* dev_y_coors_for_sub,
                                    __global float* dev_num_points_per_pillar,
                                    __global int* dev_sparse_pillar_map,
                                    const int max_pillars,
                                    const int max_points_per_pillar,
                                    const int grid_x_size,
                                    const int grid_y_size,
                                    const float min_x_range,
                                    const float min_y_range,
                                    const float pillar_x_size,
                                    const float pillar_y_size)
{
  int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (index >= grid_y_size * grid_x_size) {
    return;
  }

  int y = index / grid_x_size;
  int x = index % grid_x_size;

  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomic_add(dev_counter, 1);
  if (count < max_pillars) {
    atomic_add(dev_pillar_count, 1);
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }

    // grid coordinates of this pillar
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;

    // metric position of this pillar
    dev_x_coors_for_sub[count] = x * pillar_x_size + 0.5f * pillar_x_size + min_x_range;
    dev_y_coors_for_sub[count] = y * pillar_y_size + 0.5f * pillar_y_size + min_y_range;

    // map of pillars with at least one point
    dev_sparse_pillar_map[y * grid_x_size + x] = 1;
  }
}

// This kernel generates the input feature map to the PillarFeatureExtraction network.
// It takes the pillars that were marked for use 
// and stores the first 4 features (x,y,z,i) in the input feature map.
__kernel void MakePillarFeatureKernel(const __global float* dev_pillar_x_in_coors,
                                      const __global float* dev_pillar_y_in_coors,
                                      const __global float* dev_pillar_z_in_coors,
                                      const __global float* dev_pillar_i_in_coors,
                                      __global float* dev_pillar_x,
                                      __global float* dev_pillar_y,
                                      __global float* dev_pillar_z,
                                      __global float* dev_pillar_i,
                                      const __global int* dev_x_coors,
                                      const __global int* dev_y_coors,
                                      const __global float* dev_num_points_per_pillar,
                                      const int max_points,
                                      const int grid_x_size,
                                      const int total_point) 
{ 
  int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (tid >= total_point) {
    return;
  }

  int ith_pillar = tid / max_points;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = tid % max_points;
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }

  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  int pillar_ind = ith_pillar * max_points + ith_point;
  int coors_ind = y_ind * grid_x_size * max_points + x_ind * max_points + ith_point;
  dev_pillar_x[pillar_ind] = dev_pillar_x_in_coors[coors_ind];
  dev_pillar_y[pillar_ind] = dev_pillar_y_in_coors[coors_ind];
  dev_pillar_z[pillar_ind] = dev_pillar_z_in_coors[coors_ind];
  dev_pillar_i[pillar_ind] = dev_pillar_i_in_coors[coors_ind];
}

// This kernel takes the pillars that were marked for use and stores the features: 
// (pillar_center_x, pillar_center_y, pillar_mask) in the input feature map.
__kernel void MakeExtraNetworkInputKernel(const __global float* dev_x_coors_for_sub,
                                          const __global float* dev_y_coors_for_sub,
                                          const __global float* dev_num_points_per_pillar,
                                          __global float* dev_x_coors_for_sub_shaped,
                                          __global float* dev_y_coors_for_sub_shaped,
                                          __global float* dev_pillar_feature_mask,
                                          const int max_num_points_per_pillar,
                                          const int total_point) 
{
  int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (tid >= total_point) {
    return;
  }

  int ith_pillar = tid / max_num_points_per_pillar;
  int ith_point = tid % max_num_points_per_pillar;

  float x = dev_x_coors_for_sub[ith_pillar];
  float y = dev_y_coors_for_sub[ith_pillar];
  int num_points_for_a_pillar = dev_num_points_per_pillar[ith_pillar];
  int ind = ith_pillar * max_num_points_per_pillar + ith_point;
  
  dev_x_coors_for_sub_shaped[ind] = x;
  dev_y_coors_for_sub_shaped[ind] = y;

  if (ith_point < num_points_for_a_pillar) {
    dev_pillar_feature_mask[ind] = 1.0f;
  } else {
    dev_pillar_feature_mask[ind] = 0.0f;
  }
}

