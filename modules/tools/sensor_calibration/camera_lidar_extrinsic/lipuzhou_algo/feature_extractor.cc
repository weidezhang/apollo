#include "feature_extractor.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <numeric>

void PointCloudFeatureExtractor::CalFeature(PointCloudIRPtr cloud, VelodyneCalibrationData *data) {
    PointCloudIRPtr out = ExtractROI(cloud);
    std::cout<<"finished extracting roi out size " << out->points.size()<<std::endl;
    assert(out->points.size() > 0);
    ExtractPlane(out, data); 
    std::cout<<"finished extracting plane"<<std::endl;
}

PointCloudIRPtr PointCloudFeatureExtractor::ExtractROI(PointCloudIRPtr cloud) {
  PointCloudIRPtr cloud_passthrough1(new PointCloudIR);
  // Filter out the experimental region
  pcl::PassThrough<PointXYZIR> pass1;
  pass1.setInputCloud (cloud);
  pass1.setFilterFieldName ("x");
  pass1.setFilterLimits (bound.x_min, bound.x_max);
  pass1.filter (*cloud_passthrough1);
  pcl::PassThrough<pcl::PointXYZIR> pass_z1;
  pass_z1.setInputCloud (cloud_passthrough1);
  pass_z1.setFilterFieldName ("z");
  pass_z1.setFilterLimits (bound.z_min, bound.z_max);
  pass_z1.filter (*cloud_passthrough1);
  pcl::PassThrough<pcl::PointXYZIR> pass_final1;
  pass_final1.setInputCloud (cloud_passthrough1);
  pass_final1.setFilterFieldName ("y");
  pass_final1.setFilterLimits (bound.y_min, bound.y_max);
  pass_final1.filter (*cloud_passthrough1);
  return cloud_passthrough1; 
}

PointCloudIRPtr ExtractInlier(PointCloudIRPtr cloud, pcl::PointIndices::Ptr inliers, bool negative=false) {
    PointCloudIRPtr saved (new PointCloudIR);
    pcl::ExtractIndices<pcl::PointXYZIR> extract;
    extract.setInputCloud (cloud);
    extract.setIndices (inliers);
    std::cout<<"inliher size 1 "<<inliers->indices.size()<<std::endl;
    extract.setNegative(false);
    extract.filter(*saved);
    return saved;
}



void PointCloudFeatureExtractor::ExtractPlane(PointCloudIRPtr cloud_passthrough, VelodyneCalibrationData *data) {
    //VelodyneCalibrationData& feature_data = *data;
    
    original_ptr_ = cloud_passthrough;
    pcl::PassThrough<pcl::PointXYZIR> pass_z;
    PointCloudIRPtr cloud_filtered(new PointCloudIR),
        corrected_plane(new PointCloudIR);
    // Filter out the board point cloud
    // find the point with max height(z val) in cloud_passthrough
    double z_max = cloud_passthrough->points[0].z;
    //size_t pt_index;
    for (size_t i = 0; i < cloud_passthrough->points.size(); ++i)
    {
      if (cloud_passthrough->points[i].z > z_max)
      {
        //pt_index = i;
        z_max = cloud_passthrough->points[i].z;
      }
    }
        std::cout<<"finished extracting roi1"<<std::endl;
    // 20inch x 30inch board with multiplier 0.0254 to meter
    double diagonal = sqrt(pow(20 * 0.0254,2) + pow(30 * 0.0254,2));
    // subtract by approximate diagonal length (in metres)
    double z_min = z_max - diagonal;

    pass_z.setInputCloud (cloud_passthrough);
    pass_z.setFilterFieldName ("z");
    pass_z.setFilterLimits (static_cast<float>(z_min), static_cast<float>(z_max));
    pass_z.filter (*cloud_filtered); // board point cloud

    //apply statistical outlier filter
    //PointCloudIRPtr cloud_filtered2 = ApplyFilter(cloud_filtered);
    PointCloudIRPtr cloud_filtered2 = cloud_filtered;
    // Fit a plane through the board point cloud
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    //int i = 0, nr_points = static_cast<int>(cloud_filtered->points.size());
    pcl::SACSegmentation<pcl::PointXYZIR> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.4);
    //seg.setDistanceThreshold(0.004);
    pcl::ExtractIndices<pcl::PointXYZIR> extract;
    seg.setInputCloud (cloud_filtered2);
    seg.segment (*inliers, *coefficients);
    
    //extract inliered point from segmented inliers
    inlier_pts_ = ExtractInlier(cloud_filtered2, inliers); //save fitted plane point cloud
    // Plane normal vector magnitude
    //float mag = static_cast<float>(sqrt(pow(coefficients->values[0], 2) + pow(coefficients->values[1], 2)
    //    + pow(coefficients->values[2], 2)));

    PointCloudIRPtr max_points, min_points;
    candidate_segments_ = GetBoundaryPoints(inlier_pts_, &max_points, &min_points); //furter calculate the boundary points for inlihers
    Init2DCoord(inlier_pts_, coefficients);
    FitScanLine(coefficients);
    std::cout<<"after fit scan line"<<std::endl;  
    LidarRectInfo info;
    VisualizeDebug debug;
    std::vector<cv::Vec4f> lines = FitTopAndBottomLines(min_points, coefficients, info);
    debug.min_idx_left = info.min_idx;
    cv::Point2f r1, r2;
    //FindIntersection(lines[0], lines[1], r1);
    lines = FitTopAndBottomLines(max_points, coefficients, info);
    debug.min_idx_right = info.min_idx;
    //FindIntersection(lines[0], lines[1], r2);
    //Eigen::Vector3f r13d =  Get3DPoint(r1);
    //Eigen::Vector3f r23d = Get3DPoint(r2);
    Visualize2D(min_points, max_points, r1, r2, coefficients, debug);
    //std::cout<<"basic cloud size is " << basic_cloud_ptr->size() << std::endl; 

    //PointCloudVisualizerPtr viewer = InitViewer(cloud_filtered2);

    //VisualizePlane(viewer, cloud_projected,  basic_cloud_ptr);
    //pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler1(saved, 0, 0, 255);
    //viewer->addPointCloud(saved, color_handler1, "projected1");
    //viewer->saveScreenshot(std::string("./tmp.png"));
    
    //viewer->spin();
}


void PointCloudFeatureExtractor::Visualize2D(PointCloudIRPtr min_points, PointCloudIRPtr max_points, 
                                             cv::Point2f r1, cv::Point2f r2, pcl::ModelCoefficients::Ptr coeff,
                                             VisualizeDebug &info) {
    std::vector<cv::Point2f> minpts = Get2DPoints(min_points,coeff);
    std::vector<cv::Point2f> maxpts = Get2DPoints(max_points,coeff);
    float minx = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::min();
    float miny = std::numeric_limits<float>::max();
    float maxy = std::numeric_limits<float>::min();
    for(int i=0; i<static_cast<int>(minpts.size()); ++i) {
      minx = std::min(minpts[i].x, minx);
      maxx = std::max(minpts[i].x, maxx);
      miny = std::min(minpts[i].y, miny);
      maxy = std::max(minpts[i].y, maxy);
    }

    for(int i=0; i<static_cast<int>(maxpts.size()); ++i) {
      minx = std::min(maxpts[i].x, minx);
      maxx = std::max(maxpts[i].x, maxx);
      miny = std::min(maxpts[i].y, miny);
      maxy = std::max(maxpts[i].y, maxy);
    }

    float resolution1 = ( maxx - minx)/800.0f;
    int width = static_cast<int> (( maxx - minx)  / resolution1);
    int height = static_cast<int> (( maxy - miny)  / resolution1);

    cv::Mat img(height+300, width+300, CV_8UC3, cv::Scalar(0,0,0));
    for(int i=0; i<static_cast<int>(minpts.size()); ++i) {
      int x = static_cast<int>((minpts[i].x - minx) /resolution1) + 50;
      int y = static_cast<int>((minpts[i].y - miny) / resolution1) + 50;
      img.at<cv::Vec3b>(y,x) = cv::Vec3b(255,255,255);
    }
    int x_c = static_cast<int>((minpts[info.min_idx_left].x - minx) /resolution1) + 50;
    int y_c = static_cast<int>((minpts[info.min_idx_left].y - miny) /resolution1) + 50;
    cv::Point center(x_c, y_c);
    cv::circle(img, center, 5, cv::Scalar(255,0,0));

    for(int i=0; i<static_cast<int>(maxpts.size()); ++i) {
      int x = static_cast<int>((maxpts[i].x - minx) /resolution1) + 50;
      int y = static_cast<int>((maxpts[i].y - miny) / resolution1) + 50;
      img.at<cv::Vec3b>(y,x) = cv::Vec3b(255,255,255);
    }
    
    x_c = static_cast<int>((maxpts[info.min_idx_right].x - minx) /resolution1) + 50;
    y_c = static_cast<int>((maxpts[info.min_idx_right].y - miny) /resolution1) + 50;
    center = cv::Point(x_c, y_c);
    cv::circle(img, center, 5, cv::Scalar(255,0,0));
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img);                   // Show our image inside it.
    cv::waitKey(0);                                          // Wait for a keystroke in the window
}





float cosine_similarity(cv::Point2f &a, cv::Point2f &b)
{ 
    float ab = a.dot(b);
    float aa = a.dot(a);
    float bb = b.dot(b);
    return ab / static_cast<float>(sqrt(aa*bb));
}

std::vector<cv::Vec4f> PointCloudFeatureExtractor::FitTopAndBottomLines(PointCloudIRPtr points, 
                                                                        pcl::ModelCoefficients::Ptr coeff,
                                                                        LidarRectInfo &info) {
  std::vector<cv::Point2f> point2d = Get2DPoints(points, coeff);
  std::vector<cv::Point2f> dirs;
  cv::Point2f pt = point2d[0];
  for(int i=1; i<static_cast<int>(point2d.size()); ++i) {
     cv::Point2f dir = point2d[i] - pt;
     dirs.push_back(dir);
     pt = point2d[i]; 
  }
  float small_cos = 1.0;
  int small_idx = 0; //finding the pivoting point 
  //find point idx with largest directional change
  for (int i=1; i<static_cast<int>(dirs.size()-1);++i) {
     float cos1 = cosine_similarity(dirs[i-1], dirs[i]);
     //float cos2 = cosine_similarity(dirs[i], dirs[i+1]);
     if (cos1 < small_cos) {
        small_cos = cos1;
        small_idx = i;
     }
  }
  info.min_idx = small_idx;
  std::cout<<"small idx found is "<<small_idx <<std::endl;

  std::vector<cv::Point2f> projected2d = ProjectScanLine(point2d);
  std::cout<<"after projecting scan line"<<std::endl;
  std::vector<cv::Point2f> topline(projected2d.begin(), projected2d.begin()+small_idx);
  std::vector<cv::Point2f> bottomline(projected2d.begin()+small_idx, projected2d.end());
  std::vector<cv::Vec4f> lines;
  cv::Vec4f line; 
  std::cout<<"before fit top line" << std::endl; 
  if (topline.size() > 1) {
    cv::fitLine(topline, line, CV_DIST_L1, 1, 0.001, 0.001);
    lines.push_back(line);
  }
  std::cout<<"before fit bottom line "<<std::endl;
  if (bottomline.size() > 1) {
    cv::fitLine(bottomline, line, CV_DIST_L1, 1, 0.001, 0.001);
    lines.push_back(line);
  }
  return lines;
}


// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool PointCloudFeatureExtractor::FindIntersection(cv::Vec4f &lines1, cv::Vec4f &lines2, cv::Point2f &r) {
    cv::Point2f o1(lines1[2], lines1[3]);
    cv::Point2f o2(lines2[2], lines2[3]);
    cv::Point2f d1(lines1[0], lines1[1]);
    cv::Point2f d2(lines2[0], lines2[1]);
    cv::Point2f x = o2 - o1;
    float cross = d1.x*d2.y - d1.y*d2.x;
    if (std::fabs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}



/**
* Get projected point P' of P on line e1. 
* @return projected point p.
*/
cv::Point2f PointCloudFeatureExtractor::GetProjectedPointOnLine(cv::Point2f v1, cv::Point2f v2, cv::Point2f p)
{
  // get dot product of e1, e2
  cv::Point2f e1 = v2 - v1;
  cv::Point2f e2 = p - v1;
  float valDp = e1.dot(e2);
  // get length of vectors
  float lenLineE1 = static_cast<float>(sqrt(e1.x * e1.x + e1.y * e1.y));
  float lenLineE2 = static_cast<float>(sqrt(e2.x * e2.x + e2.y * e2.y));
  float cos = valDp / (lenLineE1 * lenLineE2);
  // length of v1P'
  float projLenOfLine = cos * lenLineE2;
  return cv::Point2f(v1.x + (projLenOfLine * e1.x) / lenLineE1,
                      v1.y + (projLenOfLine * e1.y) / lenLineE1);
}

std::vector<cv::Point2f> PointCloudFeatureExtractor::ProjectScanLine(std::vector<cv::Point2f>& point2d) {
  std::vector<cv::Point2f> projected;
  for(int i=0; i<static_cast<int>(point2d.size()); ++i) {
      cv::Vec4f& line = scan_line_coefficients_[i];
      float t = 100;
      cv::Point2f pt1(line[2] - line[0] * t, line[3] - line[1] * t);
      cv::Point2f pt2(line[2] + line[0] * t, line[3] + line[1] * t);
      cv::Point2f pt =  GetProjectedPointOnLine(pt1, pt2, point2d[i]);
      projected.push_back(pt); 
  }
  return projected;
}

void PointCloudFeatureExtractor::FitScanLine(pcl::ModelCoefficients::Ptr coefficients) {
     //for each scan line, project to plane and use ransac to fit line
     for (int i=0; i<static_cast<int>(candidate_segments_.size()); ++i) {
         std::deque<PointXYZIR>& points = candidate_segments_[i];
         PointCloudIRPtr pc (new PointCloudIR);
         for(int j=0; j<static_cast<int>(points.size()); ++j) {
             pc->points.push_back(points[j]);
         }
         if (pc->points.size()<=2) {
            std::cout<<"scan line size less than 2 size: " << pc->points.size() << "index: "<<i<<std::endl; 
            continue;
         }

         std::cout<<"print ring in fitscanline "<<i << " ";
         for(int m=0; m<static_cast<int>(candidate_segments_[i].size()); ++m) {
           pcl::PointXYZIR temp = candidate_segments_[i][m];
           std::cout<<temp.x << " " << temp.y << " " << temp.z << ", ";
         } 
         std::cout<<std::endl;
         for (int h=0; h<static_cast<int>(pc->size());++h) {
             std::cout<<"x y z"<<(*pc)[h].x << " " << (*pc)[h].y << " " << (*pc)[h].z << std::endl;
         }

         std::vector<cv::Point2f> point2d = Get2DPoints(pc, coefficients);
         cv::Vec4f line;
         for(int k=0; k<static_cast<int>(point2d.size()); ++k) {
            std::cout<<"p x " << point2d[k].x << " p y " <<point2d[k].y << std::endl;
         }
         std::cout<<"before fit line";
         cv::fitLine(point2d, line, CV_DIST_L1, 1, 0.001, 0.001);
         scan_line_coefficients_[i] = line;
     }
}

/*
pcl::ModelCoefficients::Ptr PointCloudFeatureExtractor::FitLine(PointCloudIRPtr cloud) {
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    pcl::SACSegmentation<pcl::PointXYZIR> seg;
    seg.setOptimizeCoefficients (true);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.004);
    seg.setModelType (pcl::SACMODEL_LINE);
    seg.setMethodType (pcl::SAC_RANSAC);
    //seg.setDistanceThreshold (0.02);
    seg.setDistanceThreshold (0.02);
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients); // Fitting line1 through max points
    return pcl::ModelCoefficients::Ptr;
}*/

void PointCloudFeatureExtractor::ProjectPlane(pcl::ModelCoefficients::Ptr coefficients, 
                    PointCloudIRPtr cloud_filtered, PointCloudIRPtr *cloud_projected_ptr) {
    // Project the inliers on the fit plane
    pcl::PointCloud<pcl::PointXYZIR>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZIR>);
    pcl::ProjectInliers<pcl::PointXYZIR> proj;
    proj.setModelType (pcl::SACMODEL_PLANE); 
    proj.setInputCloud (cloud_filtered); 
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);
    *cloud_projected_ptr = cloud_projected;
}




//separate left and right boundary
std::vector<std::deque<pcl::PointXYZIR>> PointCloudFeatureExtractor::GetBoundaryPoints(PointCloudIRPtr cloud_projected, PointCloudIRPtr* max_points_ptr, PointCloudIRPtr* min_points_ptr) {
    // First: Sort out the points in the point cloud according to their ring numbers
    std::vector<std::deque<pcl::PointXYZIR>> candidate_segments(64); //todo make 64 configurable

    double x_projected = 0; double y_projected = 0; double z_projected = 0;
    for (size_t i = 0; i < cloud_projected->points.size(); ++i)
    {
      x_projected += cloud_projected->points[i].x;
      y_projected += cloud_projected->points[i].y;
      z_projected += cloud_projected->points[i].z;

      int ring_number = static_cast<int>(cloud_projected->points[i].ring);

      //push back the points in a particular ring number
      candidate_segments[ring_number].push_back((cloud_projected->points[i]));
    }

    // Second: Arrange points in every ring in descending order of y coordinate
    pcl::PointXYZIR max, min;
    pcl::PointCloud<pcl::PointXYZIR>::Ptr max_points(new pcl::PointCloud<pcl::PointXYZIR>);
    pcl::PointCloud<pcl::PointXYZIR>::Ptr min_points(new pcl::PointCloud<pcl::PointXYZIR>);
    for (int i = 0; static_cast<size_t>(i) < candidate_segments.size(); i++)
    {
      if (candidate_segments[i].size()==0) // If no points belong to a aprticular ring number
      {
        std::cout<<"ring " << i << " is empty " << std::endl;
        continue;
      }
      for(int j = 0; j < static_cast<int>(candidate_segments[i].size()); j++)
      {
        for(int k = j+1; k < static_cast<int>(candidate_segments[i].size()); k++)
        {
          //If there is a larger element found on right of the point, swap
          if(candidate_segments[i][j].y < candidate_segments[i][k].y)
          {
            pcl::PointXYZIR temp;
            temp = candidate_segments[i][k];
            candidate_segments[i][k] = candidate_segments[i][j];
            candidate_segments[i][j] = temp;
          }
        }
      }
    }

    // Third: Find minimum and maximum points in a ring
    for (int i = 0; static_cast<size_t>(i) < candidate_segments.size(); i++)
    {
      if (candidate_segments[i].size()==0)
      {
        continue;
      }
      max = candidate_segments[i][0];
      min = candidate_segments[i][candidate_segments[i].size()-1];
      min_points->push_back(min);
      max_points->push_back(max);
    }

    for(int i=0; i<64; ++i) {
       std::cout<<"print ring "<<i << " ";
       for(int j=0; j<static_cast<int>(candidate_segments[i].size()); ++j) {
           pcl::PointXYZIR temp = candidate_segments[i][j];
           std::cout<<temp.x << " " << temp.y << " " << temp.z << ", ";
       } 
       std::cout<<std::endl;
    }

    *min_points_ptr = min_points;
    *max_points_ptr = max_points;
    return candidate_segments;
}

PointCloudIRPtr PointCloudFeatureExtractor::ApplyFilter(PointCloudIRPtr cloud) {
    // Create the filtering object
     PointCloudIRPtr cloud_filtered (new PointCloudIR);
     PointCloudIRPtr cloud_filtered2 (new PointCloudIR);
     pcl::VoxelGrid<PointXYZIR> sor;
     sor.setInputCloud (cloud);
     sor.setLeafSize (0.01f, 0.01f, 0.01f);
     sor.filter (*cloud_filtered);
     pcl::StatisticalOutlierRemoval<pcl::PointXYZIR> sor2;
     sor2.setInputCloud (cloud_filtered);
     sor2.setMeanK (50);
     sor2.setStddevMulThresh (1.0);
     sor2.filter (*cloud_filtered2);
     return cloud_filtered2;
}

void PointCloudFeatureExtractor::GetCorners(pcl::PCA<PointXYZIR> &pca, std::vector<PointXYZIR> *corners) {
    Eigen::MatrixXf &coeff = pca.getCoefficients();
    Eigen::Vector3f &egv = pca.getEigenValues();
    std::cout<<"eigen value is "<<egv[0] << " " << egv[1] << " " << egv[2] << std::endl;
    std::cout<<"coeff rows and cols is " << coeff.rows() << " " << coeff.cols() << std::endl; 

    float minx = coeff.row(1).minCoeff();
    float maxx = coeff.row(1).maxCoeff();
    float miny = coeff.row(0).minCoeff();
    float maxy = coeff.row(0).maxCoeff();
    std::vector<PointXYZIR> tcorners; 
    tcorners.push_back(PointXYZIR{miny, minx, 0, 0.0, 0});
    tcorners.push_back(PointXYZIR{miny, maxx, 0, 0.0, 0});
    tcorners.push_back(PointXYZIR{maxy, minx, 0, 0.0, 0});
    tcorners.push_back(PointXYZIR{maxy, maxx, 0, 0.0, 0});
    for (auto pt: tcorners) {
        PointXYZIR input;
        pca.reconstruct(pt, input);
        corners->push_back(input);
    }
}

PointCloudVisualizerPtr PointCloudFeatureExtractor::InitViewer(PointCloudIRPtr cloud) {
    
    PointCloudVisualizer::Ptr viewer (new PointCloudVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->registerKeyboardCallback (&PointCloudFeatureExtractor::KeyboardEventOccurred, *this, (void*)viewer.get ());
    return (viewer);
}

void PointCloudFeatureExtractor::VisualizePlane(PointCloudVisualizer::Ptr viewer, PointCloudIRPtr cloud_filtered, PointCloudIRPtr basic_cloud_ptr) {
     //pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler(cloud_filtered,  255, 0, 0);
     //viewer->addPointCloud(cloud_filtered, color_handler, "projected");
     for (int i=0; i < static_cast<int>(basic_cloud_ptr->points.size()-1); ++i) {
       std::cout<<"x y z" <<basic_cloud_ptr->points[i].x <<" " << basic_cloud_ptr->points[i].y << " " << basic_cloud_ptr->points[i].z << std::endl;
       std::string id = std::to_string(i);
       viewer->addLine<PointXYZIR>(basic_cloud_ptr->points[i], basic_cloud_ptr->points[i+1], 255, 0, 0, id);
     }
     viewer->addLine<PointXYZIR>(basic_cloud_ptr->points[0], basic_cloud_ptr->points[basic_cloud_ptr->size()-1], 255, 0, 0, "last");
}


void calcHist(const Eigen::VectorXf& data, const float min_x, const float max_x, const int num_bins, std::vector<int>& indices)
{
  std::vector<std::vector<int> > hist;
  hist.resize(num_bins, std::vector<int>()); //save the indexes corresponding to bin
  std::vector<int> hist_counts(num_bins, 0);
  float bin_size = (max_x-min_x)/static_cast<float>(num_bins);
  for (int i=0; i<static_cast<int>(data.size()); ++i) {
    // Assign bin
    int bin = static_cast<int>((data[i]-min_x)/bin_size);
    // Avoid out of range
    bin = std::min(std::max(bin, 0), num_bins-1);
    hist[bin].push_back(i);
    hist_counts[bin]++;
  }
  float totalcnt = 0.0f, totalvalidbin=0.0f;
  float avgcnt = 0.0f; 
  // Compute average
  for (int i = 0;i < num_bins; ++i) {
    if(hist_counts[i]>0) {
        totalcnt += static_cast<float>(hist_counts[i]);
        totalvalidbin++;
      }
  }
  avgcnt = totalcnt / totalvalidbin;
  std::cout<<"avg cnt is "<<avgcnt << std::endl;
  for(int i=0;i<num_bins;++i) {
     if(hist_counts[i] < avgcnt) {
         indices.insert(indices.end(), hist[i].begin(), hist[i].end());
     }
  }
}



void RemoveOrAddPC(pcl::visualization::PCLVisualizer *viewer, PointCloudIRPtr cloud, double r, double g, double b, std::string id) {
   bool status = viewer->removePointCloud(id);
   if (!status) {
      pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler(cloud, r, g, b);
      viewer->addPointCloud<PointXYZIR> (cloud, color_handler, id);  
   }
}

void PointCloudFeatureExtractor::KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  static std::string original = "original", plane="plane", project="project";

  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);


  if (event.getKeySym () == "p" && event.keyDown()) {
      std::cout<<"visualization of p" << std::endl;
      RemoveOrAddPC(viewer, inlier_pts_, 0, 255, 0, plane);
  }

  if (event.getKeySym () == "o" && event.keyDown()) {
      std::cout<<"visualization of o" << std::endl;
      RemoveOrAddPC(viewer, original_ptr_, 255, 255, 255, original);
  }  

  if (event.getKeySym () == "r" && event.keyDown()) {
      std::cout<<"visualization of r" << std::endl;
      RemoveOrAddPC(viewer, projected_ptr_, 255, 0, 0, project);
  }
}





void calcHist(const Eigen::VectorXf& data, const float min_x, const float max_x, const int num_bins, std::set<int>& indiceset)
{
  std::vector<int> indices;
  calcHist(data, min_x, max_x, num_bins, indices);
  for(auto i:indices) {
     indiceset.insert(i);
  }
}

PointCloudIRPtr ApplyPCAHistogramFilter(PointCloudIR& cloud_projected) {
    PointCloudIRPtr saved(new PointCloudIR);
    /*pcl::PCA<PointXYZIR> pca (cloud_projected);
    Eigen::MatrixXf &coeff = pca.getCoefficients();
    //Eigen::Vector3f &egv = pca.getEigenValues();
    pcl::PointIndices::Ptr indices (new pcl::PointIndices);
    Eigen::VectorXf dir1 = coeff.row(0);
    Eigen::VectorXf dir2 = coeff.row(1);
    calcHist(dir1, dir1.minCoeff(), dir1.maxCoeff(), 10, indices->indices);
    calcHist(dir2, dir2.minCoeff(), dir2.maxCoeff(), 10, indices->indices);
    std::cout<<"pca filter indices size is " << indices->indices.size() << "with total point size " 
             <<cloud_projected.size() << std::endl;
    pcl::ExtractIndices<pcl::PointXYZIR> extract;
    extract.setInputCloud (cloud_projected.makeShared());
    extract.setIndices (indices);
    extract.setNegative(true);
    extract.filter(*saved);
    return saved;*/
    return saved;     
}


void PointCloudFeatureExtractor::Init2DCoord(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_) {
  // Project points onto the table plane 
  PointCloudIRPtr projected_cloud;
  ProjectPlane(table_coefficients_const_, cloud, &projected_cloud);
  // Apply PCA filter to remove noise points
  //*saved = ApplyPCAHistogramFilter(projected_cloud);
  //projected_cloud = *(*saved); 
  // store the table top plane parameters 
  Eigen::Vector3f plane_normal; 
  plane_normal.x() = table_coefficients_const_->values[0]; 
  plane_normal.y() = table_coefficients_const_->values[1]; 
  plane_normal.z() = table_coefficients_const_->values[2]; 
  // compute an orthogonal normal to the plane normal 
  v_ = plane_normal.unitOrthogonal(); 
  // take the cross product of the two normals to get 
  // a thirds normal, on the plane 
  u_ = plane_normal.cross(v_);  
  // choose a point on the plane 
  p0_ = Eigen::Vector3f(projected_cloud->points[0].x, 
                      projected_cloud->points[0].y, 
                      projected_cloud->points[0].z);    
}


std::vector<cv::Point2f> PointCloudFeatureExtractor::Get2DPoints(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_) {
   
  PointCloudIRPtr projected_cloud;
  ProjectPlane(table_coefficients_const_, cloud, &projected_cloud);
  // project the 3D point onto a 2D plane 
  std::vector<cv::Point2f> points;
  for(unsigned int ii=0; ii<projected_cloud->points.size(); ii++) 
  { 
    Eigen::Vector3f p3d(projected_cloud->points[ii].x, 
                         projected_cloud->points[ii].y, 
                         projected_cloud->points[ii].z); 

    // subtract all 3D points with a point in the plane 
    // this will move the origin of the 3D coordinate system 
    // onto the plane 
    p3d = p3d - p0_; 
    cv::Point2f p2d; 
    p2d.x = p3d.dot(u_); 
    p2d.y = p3d.dot(v_); 
    points.push_back(p2d); 
  } 
  return points;
}

Eigen::Vector3f PointCloudFeatureExtractor::Get3DPoint(cv::Point2f& pt) {
    Eigen::Vector3f pt3d(pt.x*u_ + pt.y*v_ + p0_); 
    return pt3d;
}

/*
std::vector<Eigen::Vector3f> 
PointCloudFeatureExtractor::FitTableTopBbx(PointCloudIRPtr &cloud, pcl::ModelCoefficients::Ptr table_coefficients_const_, PointCloudIRPtr *saved) { 
  std::vector<Eigen::Vector3f> table_top_bbx; 
  // Project points onto the table plane 
  pcl::ProjectInliers<PointXYZIR> proj; 
  proj.setModelType(pcl::SACMODEL_PLANE); 
  pcl::PointCloud<PointXYZIR> projected_cloud; 
  proj.setInputCloud(cloud); 
  proj.setModelCoefficients(table_coefficients_const_); 
  proj.filter(projected_cloud); 

  std::cout<<"cloud after projection is "<<cloud->size()<<std::endl;
  // Apply PCA filter to remove noise points
  //*saved = ApplyPCAHistogramFilter(projected_cloud);
  //projected_cloud = *(*saved); 
  // store the table top plane parameters 
  Eigen::Vector3f plane_normal; 
  plane_normal.x() = table_coefficients_const_->values[0]; 
  plane_normal.y() = table_coefficients_const_->values[1]; 
  plane_normal.z() = table_coefficients_const_->values[2]; 
  // compute an orthogonal normal to the plane normal 
  Eigen::Vector3f v = plane_normal.unitOrthogonal(); 
  // take the cross product of the two normals to get 
  // a thirds normal, on the plane 
  Eigen::Vector3f u = plane_normal.cross(v); 

  // project the 3D point onto a 2D plane 
  std::vector<cv::Point2f> points; 
  // choose a point on the plane 
  Eigen::Vector3f p0(projected_cloud.points[0].x, 
                      projected_cloud.points[0].y, 
                      projected_cloud.points[0].z); 
  for(unsigned int ii=0; ii<projected_cloud.points.size(); ii++) 
  { 
    Eigen::Vector3f p3d(projected_cloud.points[ii].x, 
                         projected_cloud.points[ii].y, 
                         projected_cloud.points[ii].z); 

    // subtract all 3D points with a point in the plane 
    // this will move the origin of the 3D coordinate system 
    // onto the plane 
    p3d = p3d - p0; 

    cv::Point2f p2d; 
    p2d.x = p3d.dot(u); 
    p2d.y = p3d.dot(v); 
    points.push_back(p2d); 
  } 

  cv::Mat points_mat(points); 
  cv::RotatedRect rrect = cv::minAreaRect(points_mat); 
  cv::Point2f rrPts[4]; 
  rrect.points(rrPts); 
  
  //obtaining directional vector for RotatedRect Edges
  cv::Point2f vec1 =  rrPts[0] - rrPts[1];
  cv::Point2f vec2 =  rrPts[1] - rrPts[2];
  float dist1 = static_cast<float>(std::sqrt(vec1.x * vec1.x + vec1.y * vec1.y));
  float dist2 = static_cast<float>(std::sqrt(vec2.x * vec2.x + vec2.y * vec2.y));
  vec1  *= 1.0 / dist1; 
  vec2  *= 1.0 / dist2; 
  Eigen::VectorXf projected1(points.size()), projected2(points.size());
  for(int i=0; i<static_cast<int>(points.size()); ++i) {
     projected1[i] = points[i].dot(vec1);
     projected2[i] = points[i].dot(vec2);
  }
  std::set<int> filter_inds;
  calcHist(projected1, projected1.minCoeff(), projected1.maxCoeff(), 50, filter_inds);
  calcHist(projected2, projected2.minCoeff(), projected2.maxCoeff(), 50, filter_inds);
  
  std::cout<<"num points filtered is " << filter_inds.size()<<std::endl;

  std::vector<cv::Point2f> point_filtered;
  for(int i=0; i<static_cast<int>(points.size()); ++i) {
      if(filter_inds.find(i) ==filter_inds.end()) {
         point_filtered.push_back(points[i]); 
      }
  }
  std::cout<<"number of points after filter is " << point_filtered.size()<<std::endl; 
  //do the minrect again for cleansed point set on 2d
  cv::Mat points_mat2(point_filtered); 
  rrect = cv::minAreaRect(points_mat2);  
  rrect.points(rrPts); 

  //store the table top bounding points in a vector 
  for(unsigned int ii=0; ii<4; ii++) 
  { 
    Eigen::Vector3f pbbx(rrPts[ii].x*u + rrPts[ii].y*v + p0); 
    table_top_bbx.push_back(pbbx); 
  } 
  //Eigen::Vector3f center(rrect.center.x*u + rrect.center.y*v + p0); 
  //table_top_bbx.push_back(center); 

  return table_top_bbx; 
} */


double * ConvertoImgpts(double x, double y, double z, const Config& cfg)
{  
  double tmpxC = x/z;
  double tmpyC = y/z;
  cv::Point2d planepointsC;
  planepointsC.x = tmpxC;
  planepointsC.y = tmpyC;
  double r2 = tmpxC*tmpxC + tmpyC*tmpyC;

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

  return img_coord;
}


void CameraFeatureExtractor::CalFeature(Config &cfg, cv::Mat cv_image) {

  //TODO: if in inches, convert to mm
  cfg.cameramat = (cv::Mat_<double>(3,3) << 1977.99959, 0, 941.951859, 0, 1980.44291, 523.704889, 0, 0, 1);
  //cfg.distcoeff_num = 8; 
  cfg.distcoeff = (cv::Mat_<double>(1,8) <<-1.01882078e+01, 2.72324327e+01, -3.13564463e-04,5.19166521e-03, 4.62080728e+00,
                                        -9.77136872e+00, 2.29067012e+01, 1.67344873e+01);
  cv::Mat corner_vectors = cv::Mat::eye(3,5,CV_64F);
  cv::Mat chessboard_normal = cv::Mat(1,3,CV_64F);
  // checkerboard corners, middle square corners, board corners and centre
  std::vector< cv::Point2f > image_points, imagePoints1, imagePoints;

  //////////////// IMAGE FEATURES //////////////////

  cv::Size2i patternNum(cfg.grid_size[1],cfg.grid_size[0]); //width, height
  cv::Size2i patternSize(static_cast<int>(cfg.square_length), static_cast<int>(cfg.square_length));

  //TODO: make image to grayscale and do chessboard corner detection processing
  cv::Mat gray;
  cv::cvtColor(cv_image, gray, CV_BGR2GRAY);
  std::vector<cv::Point2f> corners, corners_undistorted;
  std::vector<cv::Point3f> grid3dpoint;
 
  // Find checkerboard pattern in the image
  bool patternfound = cv::findChessboardCorners(gray, patternNum, corners,
                                                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

  std::cout<<"patternfound " << patternfound << std::endl;

  if(patternfound) {
    // Find corner points with sub-pixel accuracy
    cv::cornerSubPix(gray, corners, cv::Size(11,11), cv::Size(-1,-1),
                 cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    cv::Size imgsize;
    imgsize.height = cv_image.rows;
    imgsize.width = cv_image.cols;
    float tx, ty; // Translation values
    // Location of board frame origin from the bottom left inner corner of the checkerboard
    //tx = static_cast<float>((patternNum.height) * patternSize.height)/2.0f;
    //ty = static_cast<float>((patternNum.width) * patternSize.width)/2.0f;
    std::cout<<"1" << std::endl;
    tx = static_cast<float>((patternNum.height - 1) * patternSize.height)/2.0f;
    ty = static_cast<float>((patternNum.width - 1) * patternSize.width)/2.0f;
    // Board corners w.r.t board frame
    for(int i = 0; i < patternNum.height; i++)
    {
      for(int j = 0; j < patternNum.width; j++)
      {
        cv::Point3f tmpgrid3dpoint;
        // Translating origin from bottom left corner to the centre of the checkerboard
        tmpgrid3dpoint.x = static_cast<float>(i*patternSize.height) - tx; //subpixel location for board center 
        tmpgrid3dpoint.y = static_cast<float>(j*patternSize.width) - ty; //subpixel location for board center
        tmpgrid3dpoint.z = 0;
        grid3dpoint.push_back(tmpgrid3dpoint);
      }
    }
        std::cout<<"1" << std::endl;
    std::vector< cv::Point3f > boardcorners;
    // Board corner coordinates from the centre of the checkerboard
    boardcorners.push_back(cv::Point3f((cfg.board_dimension[0])/2,
                                       (cfg.board_dimension[1])/2, 0.0));
    boardcorners.push_back(cv::Point3f(-(cfg.board_dimension[0])/2,
                                       (cfg.board_dimension[1])/2, 0.0));
    boardcorners.push_back(cv::Point3f(-(cfg.board_dimension[0])/2,
                                       -(cfg.board_dimension[1])/2, 0.0));
    boardcorners.push_back(cv::Point3f((cfg.board_dimension[0])/2,
                                       -(cfg.board_dimension[1])/2, 0.0));
    // Board centre coordinates from the centre of the checkerboard (due to incorrect placement of checkerbord on board)
    boardcorners.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));

    std::vector< cv::Point3f > square_edge;
    // centre checkerboard square corner coordinates wrt the centre of the checkerboard (origin)
    square_edge.push_back(cv::Point3f(-cfg.square_length/2, -cfg.square_length/2, 0.0));
    square_edge.push_back(cv::Point3f(cfg.square_length/2, cfg.square_length/2, 0.0));
    cv::Mat rvec(3,3,cv::DataType<double>::type); // Initialization for pinhole and fisheye cameras
    cv::Mat tvec(3,1,cv::DataType<double>::type);
    std::cout<<"1" << std::endl;
    
    if (cfg.fisheye_model)
    {
      // Undistort the image by applying the fisheye intrinsic parameters
      // the final input param is the camera matrix in the new or rectified coordinate frame.
      // We put this to be the same as cfg.cameramat or else it will be set to empty matrix by default.
      /*cv::fisheye::undistortPoints(corners, corners_undistorted, cfg.cameramat, cfg.distcoeff,
                                   cfg.cameramat);
      cv::Mat fake_distcoeff = (Mat_<double>(4,1) << 0, 0, 0, 0);
      cv::solvePnP(grid3dpoint, corners_undistorted, cfg.cameramat, fake_distcoeff, rvec, tvec);
      cv::fisheye::projectPoints(grid3dpoint, image_points, rvec, tvec, cfg.cameramat, cfg.distcoeff);
      // Mark the centre square corner points
      cv::fisheye::projectPoints(square_edge, imagePoints1, rvec, tvec, cfg.cameramat, cfg.distcoeff);
      cv::fisheye::projectPoints(boardcorners, imagePoints, rvec, tvec, cfg.cameramat, cfg.distcoeff);
      for (int i = 0; i < grid3dpoint.size(); i++)
        cv::circle(cv_ptr->image, image_points[i], 5, CV_RGB(255,0,0), -1);
      */
//        for (int i = 0; i < square_edge.size(); i++)
//          cv::circle(cv_ptr->image, imagePoints1[i], 5, CV_RGB(255,0,0), -1);
//        // Mark the board corner points and centre point
//        for (int i = 0; i < boardcorners.size(); i++)
//          cv::circle(cv_ptr->image, imagePoints[i], 5, CV_RGB(255,0,0), -1);

    }
    // Pinhole model
    else
    {
          std::cout<<"1" << std::endl;
      cv::solvePnP(grid3dpoint, corners, cfg.cameramat, cfg.distcoeff, rvec, tvec);
      cv::projectPoints(grid3dpoint, rvec, tvec, cfg.cameramat, cfg.distcoeff, image_points);
    std::cout<<"1" << std::endl;
      // Mark the centre square corner points
      cv::projectPoints(square_edge, rvec, tvec, cfg.cameramat, cfg.distcoeff, imagePoints1);
      cv::projectPoints(boardcorners, rvec, tvec, cfg.cameramat, cfg.distcoeff, imagePoints);

        for (int i = 0; i < static_cast<int>(square_edge.size()); i++)
          cv::circle(cv_image, imagePoints1[i], 5, CV_RGB(0,255,0), -1);
//        // Mark the board corner points and centre point
        for (int i = 0; i < static_cast<int>(boardcorners.size()); i++)
            cv::circle(cv_image, imagePoints[i], 5, CV_RGB(0,0,255), -1);
      for (int i=0; i<static_cast<int>(image_points.size()); ++i) {
          cv::circle(cv_image, image_points[i], 5, CV_RGB(255,0,0), -1);
      }
      cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
      cv::imshow( "Display window", cv_image);                   // Show our image inside it.
      cv::waitKey(0); 
    }
    
    // chessboardpose is a 3*4 transform matrix that transforms points in board frame to camera frame | R&T
    /*cv::Mat chessboardpose = cv::Mat::eye(4,4,CV_64F);
    cv::Mat tmprmat = cv::Mat(3,3,CV_64F); // rotation matrix
    cv::Rodrigues(rvec,tmprmat); // Euler angles to rotation matrix

    for(int j = 0; j < 3; j++)
    {
      for(int k = 0; k < 3; k++)
      {
        chessboardpose.at<double>(j,k) = tmprmat.at<double>(j,k);
      }
      chessboardpose.at<double>(j,3) = tvec.at<double>(j);
    }

    chessboard_normal.at<double>(0) = 0;
    chessboard_normal.at<double>(1) = 0;
    chessboard_normal.at<double>(2) = 1;
    chessboard_normal = chessboard_normal * chessboardpose(cv::Rect(0,0,3,3)).t();

    for (int k = 0; k < static_cast<int>(boardcorners.size()); k++)
    {
      // take every point in boardcorners set
      cv::Point3f pt(boardcorners[k]);
      for (int i = 0; i < 3; i++)
      {
        // Transform it to obtain the coordinates in cam frame
        corner_vectors.at<double>(i,k) = chessboardpose.at<double>(i,0)*pt.x +
            chessboardpose.at<double>(i,1)*pt.y + chessboardpose.at<double>(i,3);
      }

      // convert 3D coordinates to image coordinates
      double * img_coord = ConvertoImgpts(corner_vectors.at<double>(0,k),
                                                               corner_vectors.at<double>(1,k),
                                                               corner_vectors.at<double>(2,k), cfg);
      // Mark the corners and the board centre
      if (k==0)
        cv::circle(cv_ptr->image, cv::Point(static_cast<int>(img_coord[0]),static_cast<int>(img_coord[1])),
            8, CV_RGB(0,255,0),-1); //green
      else if (k==1)
        cv::circle(cv_ptr->image, cv::Point(static_cast<int>(img_coord[0]),static_cast<int>(img_coord[1])),
            8, CV_RGB(255,255,0),-1); //yellow
      else if (k==2)
        cv::circle(cv_ptr->image, cv::Point(static_cast<int>(img_coord[0]),static_cast<int>(img_coord[1])),
            8, CV_RGB(0,0,255),-1); //blue
      else if (k==3)
        cv::circle(cv_ptr->image, cv::Point(static_cast<int>(img_coord[0]),static_cast<int>(img_coord[1])),
            8, CV_RGB(255,0,0),-1); //red
      else
        cv::circle(cv_ptr->image, cv::Point(static_cast<int>(img_coord[0]),static_cast<int>(img_coord[1])),
            8, CV_RGB(255,255,255),-1); //white for centre

      delete[] img_coord;
    }*/
  }

}