__kernel void ScatterKernel(const __global int* x_coors,
                            const __global int* y_coors,
                            const __global float* pfe_output,
                            __global float* scattered_feature,
                            const int max_num_pillars_,
                            const int grid_x_size,
                            const int grid_y_size,
                            const int num_features,
                            const int max_index) 
{
  int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
  if (tid >= max_index) {
    return;
  }
  // Get pillar index and feature index from current group and local id
  int i_pillar = tid / num_features;
  int i_feature = tid % num_features;

  // Read (x,y) indices in the sparse feature map of the corresponding pillar
  int x_ind = x_coors[i_pillar];
  int y_ind = y_coors[i_pillar];
  float feature = pfe_output[i_feature * max_num_pillars_ + i_pillar];

  // Copy the i feature from pillar to the sparse feature map
  scattered_feature[i_feature * grid_y_size * grid_x_size + y_ind * grid_x_size + x_ind] = feature;
}