__kernel void PointpillarMemset(__global float* dev_pillar_x_auto,
                                __global float* dev_pillar_y_auto,
                                __global float* dev_pillar_z_auto,
                                __global float* dev_pillar_i_auto,
                                __global float* dev_x_coors_for_sub_shaped_auto,
                                __global float* dev_y_coors_for_sub_shaped_auto,
                                __global float* dev_pillar_feature_mask_auto,
                                const int max_index) 
{
    const int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if (index >= max_index) {
        return;
    }

    dev_pillar_x_auto[index] = 0.0;
    dev_pillar_y_auto[index] = 0.0;
    dev_pillar_z_auto[index] = 0.0;
    dev_pillar_i_auto[index] = 0.0;
    dev_x_coors_for_sub_shaped_auto[index] = 0.0;
    dev_y_coors_for_sub_shaped_auto[index] = 0.0;
    dev_pillar_feature_mask_auto[index] = 0.0;
}

__kernel void PreprocessMemset(__global float* dev_pillar_x_in_coors_auto,
                               __global float* dev_pillar_y_in_coors_auto,
                               __global float* dev_pillar_z_in_coors_auto,
                               __global float* dev_pillar_i_in_coors_auto,
                               const int max_index)                                
{
    int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if (index >= max_index) {
        return;
    }

    dev_pillar_x_in_coors_auto[index] = 0.0;
    dev_pillar_y_in_coors_auto[index] = 0.0;
    dev_pillar_z_in_coors_auto[index] = 0.0;
    dev_pillar_i_in_coors_auto[index] = 0.0; 
}

__kernel void ScatterMemset(__global float* dev_scattered_feature_auto,
                            const int max_index) 
{
    int index = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if (index >= max_index) {
        return;
    }

    dev_scattered_feature_auto[index] = 0.0;    
}