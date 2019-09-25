#pragma once 

#include "calibration_data.h"
#include "point_type.h"
#include "config.h"
#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class PointCloudFeatureExtractor  {
public:
	struct Bound {
        float x_min, x_max, y_min, y_max, z_min, z_max; 
	}; 
	void CalFeature(PointCloudIRPtr cloud, VelodyneCalibrationData* data);

private:
    PointCloudIRPtr ExtractROI(PointCloudIRPtr cloud);
    void ExtractPlane(PointCloudIRPtr cloud_passthrough, VelodyneCalibrationData *data);
    PointCloudVisualizerPtr InitViewer(PointCloudIRPtr cloud);
    void VisualizePlane(PointCloudVisualizer::Ptr viewer, PointCloudIRPtr cloud_filtered, PointCloudIRPtr basic_cloud_ptr);
    Bound bound = {1, 3.5, -1.5, 2, -3, 1}; 
    void GetCorners(pcl::PCA<PointXYZIR> &pca, std::vector<PointXYZIR> *corners);
    PointCloudIRPtr ApplyFilter(PointCloudIRPtr cloud);
    //std::vector<Eigen::Vector3f> FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_);
    std::vector<Eigen::Vector3f> FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_, PointCloudIRPtr *saved);
    PointCloudIRPtr plane_ptr_;
    PointCloudIRPtr projected_ptr_;
    PointCloudIRPtr original_ptr_;
    void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void);
};


class CameraFeatureExtractor {
public:
	void CalFeature(Config &cfg, cv::Mat cv_image);
};