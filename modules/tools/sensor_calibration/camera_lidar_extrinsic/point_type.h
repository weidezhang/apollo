#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/impl/project_inliers.hpp> 
#include <pcl/common/common_headers.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/pca.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/features/organized_edge_detection.h>
//#include <pcl/features/integral_image_normal.h>

namespace pcl {
 struct PointXYZIR
 {
  PCL_ADD_POINT4D;                    
  float    intensity;                 
  uint16_t ring;                      
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     
 }EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (uint16_t, ring, ring))

typedef pcl::PointXYZIR PointXYZIR;
typedef pcl::PointCloud<pcl::PointXYZIR> PointCloudIR; 
typedef PointCloudIR::Ptr PointCloudIRPtr;
typedef pcl::visualization::PCLVisualizer PointCloudVisualizer;
typedef pcl::visualization::PCLVisualizer::Ptr PointCloudVisualizerPtr;



