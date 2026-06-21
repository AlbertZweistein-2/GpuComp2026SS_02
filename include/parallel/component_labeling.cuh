#pragma once

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "types.hpp"

struct ConnectedComponentsCudaScratch
{
    thrust::device_vector<int> parents;
};

struct RegionBuildCudaScratch
{
    thrust::device_vector<int> areas;
    thrust::device_vector<int> min_x;
    thrust::device_vector<int> min_y;
    thrust::device_vector<int> max_x;
    thrust::device_vector<int> max_y;
    thrust::device_vector<Region> regions;
    thrust::device_vector<int> region_count;
};

void allocate_component_labeling_cuda_scratch(
    int width,
    int height,
    ConnectedComponentsCudaScratch &components,
    RegionBuildCudaScratch &region_build
);

void build_regions_from_labels_cuda(
    const int *d_compact_labels,
    int width,
    int height,
    int min_area,
    RegionBuildCudaScratch &scratch,
    std::vector<Region> &regions);

void connected_components_buf_cuda_device(
    const uint8_t *d_binary,
    int *d_labels,
    int width,
    int height,
    ConnectedComponentsCudaScratch &scratch);
