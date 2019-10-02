#pragma once 

#include "calibration_data.h"
#include "point_type.h"
#include "config.h"
#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

struct Bound {
    float x_min, x_max, y_min, y_max, z_min, z_max; 
    void debug() {
         std::cout<<"bound is "<<x_min<<" "<<x_max<<" "<<y_min<<" "<<y_max<<" "<<z_min<<" "<<z_max<<std::endl;
    }
}; 

class PointCloudFeatureExtractor  {
public:
	void CalFeature(PointCloudIRPtr cloud, VelodyneCalibrationData* data, bool vis=false);
    static Bound bound;

private:
    PointCloudIRPtr ExtractROI(PointCloudIRPtr cloud);
    void ExtractPlane(PointCloudIRPtr cloud_passthrough, VelodyneCalibrationData *data);
    PointCloudVisualizerPtr InitViewer(PointCloudIRPtr cloud);
    void VisualizePlane(PointCloudVisualizer::Ptr viewer, PointCloudIRPtr cloud_filtered, PointCloudIRPtr basic_cloud_ptr);
    void GetCorners(pcl::PCA<PointXYZIR> &pca, std::vector<PointXYZIR> *corners);
    PointCloudIRPtr ApplyFilter(PointCloudIRPtr cloud);
    //std::vector<Eigen::Vector3f> FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_);
    std::vector<Eigen::Vector3f> FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_, PointCloudIRPtr *saved);
    void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void);
    void RemoveOrAddPC(pcl::visualization::PCLVisualizer *viewer, PointCloudIRPtr cloud, double r, double g, double b, 
                   std::string id);

    PointCloudIRPtr plane_ptr_;
    PointCloudIRPtr projected_ptr_;
    PointCloudIRPtr original_ptr_;
    PointCloudIRPtr all_ptr_;
    PointXYZIR plane_normal_;
    PointXYZIR plane_center_;
    bool vis_;
};


class CameraFeatureExtractor {
public:
	bool CalFeature(Config &cfg, cv::Mat cv_image, CameraCalibrationData &data, bool vis=false);
};