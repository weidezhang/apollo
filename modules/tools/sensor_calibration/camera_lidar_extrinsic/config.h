#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

class Config {
public:
    static constexpr int grid_size[2] = {6, 8};
	static constexpr float square_length = static_cast<float>(3.9375 * 0.0254 * 1000); // 3 inch + 7/8 inch + 1/16 inch
    static constexpr float board_dimension[2] = {static_cast<float>(27.5 * 0.0254 * 1000), static_cast<float>(39.25 * 0.0254 * 1000)}; //in inches
    static constexpr float cb_translation_error[2] = {0,0}; //2+1/16 inches from each side
    static constexpr bool fisheye_model = false;

    cv::Mat cameramat;
    static constexpr int distcoeff_num = 8; 
    cv::Mat distcoeff;
};