#pragma once 
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

struct VelodyneCalibrationData {
    double velodynepoint[3];
    double velodynenormal[3];
    double velodynecorner[3];
};

struct CameraCalibrationData {
    double camerapoint[3];
    double cameranormal[3];
    double pixeldata;
};

struct CameraVelodyneCalibrationData {
    std::vector<std::vector< double >> velodynenormals;
    std::vector<std::vector< double >> velodynepoints;
    std::vector<std::vector< double >> cameranormals;
    std::vector<std::vector< double >> camerapoints;
    std::vector<std::vector< double >> velodynecorners;
    std::vector<double> pixeldata;
    cv::Mat cameranormals_mat; //n*3
    cv::Mat camerapoints_mat; //n*3
    cv::Mat velodynepoints_mat; //n*3
    cv::Mat velodynenormals_mat; //n*3
    cv::Mat velodynecorners_mat; //n*3
    cv::Mat pixeldata_mat;
    int sample_size;

    void debug() {
         std::cout<<"print velodyne parameters:\n"
                  <<"normal:\n"
                  <<velodynenormals_mat
                  <<"\n"
                  <<"norm of normal\n"
                  <<"1st row:"<<cv::norm(velodynenormals_mat.row(0))
                  <<"2nd row:"<<cv::norm(velodynenormals_mat.row(1))
                  <<"3rd row:"<<cv::norm(velodynenormals_mat.row(2))
                  <<"\n"
                  <<"center:\n"
                  <<velodynepoints_mat
                  <<"\nprint camera parameters:\n"
                  <<"normal:\n"
                  <<cameranormals_mat
                  <<"\n"
                  <<"norm of normal\n"
                  <<"1st row:"<<cv::norm(cameranormals_mat.row(0))
                  <<"2nd row:"<<cv::norm(cameranormals_mat.row(1))
                  <<"3rd row:"<<cv::norm(cameranormals_mat.row(2))
                  <<"\n"
                  <<"center:\n"
                  <<camerapoints_mat
                  <<"\n";
    }
};