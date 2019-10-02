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
    struct LidarRectInfo {
        int min_idx;
    };
    struct VisualizeDebug {
        int min_idx_left;
        int min_idx_right;
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
    //std::vector<Eigen::Vector3f> FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_, PointCloudIRPtr *saved);
    void KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void);
    std::vector<std::deque<pcl::PointXYZIR>> GetBoundaryPoints(PointCloudIRPtr cloud_projected, PointCloudIRPtr* max_points_ptr, PointCloudIRPtr* min_points_ptr); 
    void ProjectPlane(pcl::ModelCoefficients::Ptr coefficients, PointCloudIRPtr cloud_filtered, PointCloudIRPtr *cloud_projected_ptr);
    void FitScanLine(pcl::ModelCoefficients::Ptr coefficients); //fit scan line
    std::vector<cv::Point2f> Get2DPoints(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_); //get 2d points on a plane
    cv::Point2f GetProjectedPointOnLine(cv::Point2f v1, cv::Point2f v2, cv::Point2f p);
    std::vector<cv::Point2f> ProjectScanLine(std::vector<cv::Point2f>& point2d);
    std::vector<cv::Vec4f> FitTopAndBottomLines(PointCloudIRPtr points, pcl::ModelCoefficients::Ptr coeff, LidarRectInfo &info);
    void Init2DCoord(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_);
    void Visualize2D(PointCloudIRPtr min_points, PointCloudIRPtr max_points, cv::Point2f r1, cv::Point2f r2, 
                     pcl::ModelCoefficients::Ptr coeff, VisualizeDebug &info);
    Eigen::Vector3f Get3DPoint(cv::Point2f& pt);
    bool FindIntersection(cv::Vec4f &line1, cv::Vec4f &line2, cv::Point2f &r);

    //variables
    std::vector<std::deque<pcl::PointXYZIR>> candidate_segments_; 
    std::map<int, cv::Vec4f> scan_line_coefficients_;
    PointCloudIRPtr inlier_pts_;
    PointCloudIRPtr projected_ptr_;
    PointCloudIRPtr original_ptr_;
    //2d plane coordinates
    Eigen::Vector3f p0_;
    Eigen::Vector3f u_;
    Eigen::Vector3f v_;


};


class CameraFeatureExtractor {
public:
	void CalFeature(Config &cfg, cv::Mat cv_image);
};