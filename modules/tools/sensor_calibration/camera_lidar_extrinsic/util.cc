#include "util.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

double * CalibrationUtil::ConvertoImgpts(double x, double y, double z) {
  Config cfg;  
  double tmpxC = x/z;
  double tmpyC = y/z;
  cv::Point2d planepointsC;
  planepointsC.x = tmpxC;
  planepointsC.y = tmpyC;
  double r2 = tmpxC*tmpxC + tmpyC*tmpyC;
  //std::vector<cv::Point2f> image_points2;

  if (cfg.fisheye_model)
  {
    double r1 = pow(r2,0.5);
    double a0 = std::atan(r1);
    // distortion function for a fisheye lens
    double a1 = a0*(1 + cfg.distcoeff.at<double>(0)*pow(a0,2) + cfg.distcoeff.at<double>(1)*pow(a0,4)
                    + cfg.distcoeff.at<double>(2)*pow(a0,6) + cfg.distcoeff.at<double>(3)*pow(a0,8));
    planepointsC.x = (a1/r1)*tmpxC;
    planepointsC.y = (a1/r1)*tmpyC;
    planepointsC.x = cfg.cameramat.at<double>(0,0)*planepointsC.x + cfg.cameramat.at<double>(0,2);
    planepointsC.y = cfg.cameramat.at<double>(1,1)*planepointsC.y + cfg.cameramat.at<double>(1,2);
  }
  else // For pinhole camera model
  {
  	  /*std::vector<cv::Point3f> pts;
      pts.push_back(cv::Point3f{static_cast<float>(x),static_cast<float>(y),static_cast<float>(z)});
      cv::Mat rvec = cv::Mat::eye(3,3, cv::DataType<double>::type);
      cv::Mat tvec = cv::Mat::zeros(3,1, cv::DataType<double>::type);
      cv::projectPoints(pts, rvec, tvec,cfg.cameramat, cfg.distcoeff, image_points2);*/
    double tmpdist = 1 + cfg.distcoeff.at<double>(0)*r2 + cfg.distcoeff.at<double>(1)*r2*r2 +
        cfg.distcoeff.at<double>(4)*r2*r2*r2;
    planepointsC.x = tmpxC*tmpdist + 2*cfg.distcoeff.at<double>(2)*tmpxC*tmpyC +
        cfg.distcoeff.at<double>(3)*(r2+2*tmpxC*tmpxC);
    planepointsC.y = tmpyC*tmpdist + cfg.distcoeff.at<double>(2)*(r2+2*tmpyC*tmpyC) +
        2*cfg.distcoeff.at<double>(3)*tmpxC*tmpyC;
    planepointsC.x = cfg.cameramat.at<double>(0,0)*planepointsC.x + cfg.cameramat.at<double>(0,2);
    planepointsC.y = cfg.cameramat.at<double>(1,1)*planepointsC.y + cfg.cameramat.at<double>(1,2);
  }

  double * img_coord = new double[2];
  *(img_coord) = planepointsC.x;
  *(img_coord+1) = planepointsC.y;
  //*(img_coord) = image_points2[0].x;
  //*(img_coord+1) = image_points2[0].y;
  return img_coord;
}
