#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

class Config {
public:
    static constexpr int grid_size[2] = {6, 8};
	static constexpr float square_length = static_cast<float>(3.9375 * 0.0254 * 1000); // 3 inch + 7/8 inch + 1/16 inch and converts to mm
    static constexpr float board_dimension[2] = {static_cast<float>(27.5 * 0.0254 * 1000), static_cast<float>(39.25 * 0.0254 * 1000)}; //in mm
    static constexpr float cb_translation_error[2] = {0,0}; //2+1/16 inches from each side
    static constexpr bool fisheye_model = false;
    
    Config() {
        //TODO: if in inches, convert to mm
        //cameramat = (cv::Mat_<double>(3,3) << 1977.99959, 0, 941.951859, 0, 1980.44291, 523.704889, 0, 0, 1);
        //cfg.distcoeff_num = 8; 
        //distcoeff = (cv::Mat_<double>(1,8) <<-1.01882078e+01, 2.72324327e+01, -3.13564463e-04,5.19166521e-03, 4.62080728e+00,
        //                                -9.77136872e+00, 2.29067012e+01, 1.67344873e+01);
        distcoeff = (cv::Mat_<double>(1,5) << -0.35930196, 0.05752256, 0.00168984, -0.00248133, 0.14960282);
        cameramat = (cv::Mat_<double>(3,3) << 1.95439959e+03, 0.00000000e+00, 9.91970870e+02, 0.00000000e+00, 1.95533124e+03, 4.96773088e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00);
        //distcoeff = (cv::Mat_<double>(1,5) << -4.02919915e-01, 1.43554547e-01, -3.32907781e-04, 5.12762862e-03, -2.01594785e-02);
        //cameramat = (cv::Mat_<double>(3,3) << 1.97742775e+03, 0.00000000e+00, 9.43269202e+02, 0.00000000e+00, 1.98005776e+03, 5.24090553e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00);

    }

    cv::Mat cameramat;
    static constexpr int distcoeff_num = 8; 
    cv::Mat distcoeff;
};