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


