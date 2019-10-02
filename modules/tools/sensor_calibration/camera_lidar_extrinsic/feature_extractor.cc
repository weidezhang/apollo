#include "feature_extractor.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

Bound PointCloudFeatureExtractor::bound;

void PointCloudFeatureExtractor::CalFeature(PointCloudIRPtr cloud, VelodyneCalibrationData *data, bool vis) {
    vis_ = vis;
    all_ptr_ = cloud;
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
    VelodyneCalibrationData& feature_data = *data;
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
    Config cfg;
    double diagonal = sqrt(pow(cfg.board_dimension[0]/1000.0f,2) + pow(cfg.board_dimension[1]/1000.0f,2));
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
    seg.setDistanceThreshold (0.04);
    //seg.setDistanceThreshold(0.1);
    pcl::ExtractIndices<pcl::PointXYZIR> extract;
    seg.setInputCloud (cloud_filtered2);
    seg.segment (*inliers, *coefficients);
    
    //extract inliered point from segmented inliers
    plane_ptr_ = ExtractInlier(cloud_filtered2, inliers); //save fitted plane point cloud

    // Plane normal vector magnitude
    float mag = static_cast<float>(sqrt(pow(coefficients->values[0], 2) + pow(coefficients->values[1], 2)
        + pow(coefficients->values[2], 2)));
    //std::cout<<"finished extracting roi3"<<std::endl;
    // Project the inliers on the fit plane
    pcl::PointCloud<pcl::PointXYZIR>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZIR>);
    pcl::ProjectInliers<pcl::PointXYZIR> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (plane_ptr_);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);

    projected_ptr_ = cloud_projected; //save projected point cloud 

    PointCloudIRPtr saved; 
    std::vector<Eigen::Vector3f> pts = FitTableTopBbx(plane_ptr_, coefficients, &saved);

    assert(pts.size()==5); //4 corners plus center of the plane detected
    //center point in millimeters
    feature_data.velodynepoint[0] = pts[4].x();
    feature_data.velodynepoint[1] = pts[4].y();
    feature_data.velodynepoint[2] = pts[4].z();
    feature_data.velodynenormal[0] = -coefficients->values[0]/mag;
    feature_data.velodynenormal[1] = -coefficients->values[1]/mag;
    feature_data.velodynenormal[2] = -coefficients->values[2]/mag;


    double top_down_radius = sqrt(pow(feature_data.velodynepoint[0],2)
        + pow(feature_data.velodynepoint[1],2));
    double x_comp = feature_data.velodynepoint[0] + feature_data.velodynenormal[0]/2;
    double y_comp = feature_data.velodynepoint[1] + feature_data.velodynenormal[1]/2;
    double vector_dist = sqrt(pow(x_comp,2) + pow(y_comp,2));
    if (vector_dist > top_down_radius)
    {
      std::cout<<"revert sign for normal"<<std::endl;
      feature_data.velodynenormal[0] = -feature_data.velodynenormal[0];
      feature_data.velodynenormal[1] = -feature_data.velodynenormal[1];
      feature_data.velodynenormal[2] = -feature_data.velodynenormal[2];
    }

    //corner
    feature_data.velodynecorner[0] = pts[0].x();
    feature_data.velodynecorner[1] = pts[0].y();
    feature_data.velodynecorner[2] = pts[0].z();
     
    //purely for visualization purposes
    PointCloudIRPtr basic_cloud_ptr (new PointCloudIR);
    for(int i=0; i<static_cast<int>(pts.size());++i) {
        basic_cloud_ptr->points.push_back(PointXYZIR{pts[i].x(), pts[i].y(), pts[i].z(), 0.0, 0});
    } 

    std::cout<<"basic cloud size is " << basic_cloud_ptr->size() << std::endl; 

    if (vis_) {
       PointCloudVisualizerPtr viewer = InitViewer(cloud_passthrough);
       plane_normal_ = PointXYZIR{static_cast<float>(feature_data.velodynenormal[0]), 
                               static_cast<float>(feature_data.velodynenormal[1]), 
                               static_cast<float>(feature_data.velodynenormal[2]), 0, 0};
       plane_center_ = PointXYZIR{static_cast<float>(feature_data.velodynepoint[0]), 
                               static_cast<float>(feature_data.velodynepoint[1]), 
                               static_cast<float>(feature_data.velodynepoint[2]), 0, 0};
       VisualizePlane(viewer, cloud_projected,  basic_cloud_ptr);
    
       //pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler1(saved, 0, 0, 255);
       //viewer->addPointCloud(saved, color_handler1, "projected1");
       viewer->spin();
     }
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

void PointCloudFeatureExtractor::VisualizePlane(PointCloudVisualizer::Ptr viewer, PointCloudIRPtr cloud_filtered, 
                                                PointCloudIRPtr basic_cloud_ptr) {
     //pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler(cloud_filtered,  255, 0, 0);
     //viewer->addPointCloud(cloud_filtered, color_handler, "projected");
     int psize = static_cast<int>(basic_cloud_ptr->points.size());
     for (int i=0; i < (psize-2); ++i) {
       std::cout<<"x y z" <<basic_cloud_ptr->points[i].x <<" " << basic_cloud_ptr->points[i].y << " " << basic_cloud_ptr->points[i].z << std::endl;
       std::string id = std::to_string(i);
       viewer->addLine<PointXYZIR>(basic_cloud_ptr->points[i], basic_cloud_ptr->points[i+1], 255, 0, 0, id);
     }
     viewer->addLine<PointXYZIR>(basic_cloud_ptr->points[0], basic_cloud_ptr->points[psize-2], 255, 0, 0, "last");
     viewer->addSphere<PointXYZIR>(basic_cloud_ptr->points[psize-1], 0.03);
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
     if(hist_counts[i] < (avgcnt/3.0f)) {
         indices.insert(indices.end(), hist[i].begin(), hist[i].end());
     }
  }
}



void PointCloudFeatureExtractor::RemoveOrAddPC(pcl::visualization::PCLVisualizer *viewer, PointCloudIRPtr cloud, double r, double g, double b, 
                   std::string id) {
   bool status = viewer->removePointCloud(id);
   if (!status) {
      pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> color_handler(cloud, r, g, b);
      viewer->addPointCloud<PointXYZIR> (cloud, color_handler, id); 
      if(id == "project") {
        /*pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        normals->push_back(normal);
        viewer->addPointCloudNormals<PointXYZIR, pcl::Normal> (cloud, normals,10, 0.05f, "normals");*/
        PointXYZIR dest = plane_normal_;
        dest.x += plane_center_.x;
        dest.y += plane_center_.y;
        dest.z += plane_center_.z;
        viewer->addArrow<PointXYZIR>(dest, plane_center_, 1.0, 1.0, 1.0, "normals");
      } 
   }
}

void PointCloudFeatureExtractor::KeyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void) {
  static std::string original = "original", plane="plane", project="project", all="all";

  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);


  if (event.getKeySym () == "p" && event.keyDown()) {
      std::cout<<"visualization of p" << std::endl;
      RemoveOrAddPC(viewer, plane_ptr_, 0, 255, 0, plane);
  }

  if (event.getKeySym () == "o" && event.keyDown()) {
      std::cout<<"visualization of o" << std::endl;
      RemoveOrAddPC(viewer, original_ptr_, 0, 0, 255, original);
  }  

  if (event.getKeySym () == "r" && event.keyDown()) {
      std::cout<<"visualization of r" << std::endl;
      RemoveOrAddPC(viewer, projected_ptr_, 255, 0, 0, project);
  }

  if (event.getKeySym () == "a" && event.keyDown()) {
      std::cout<<"visualization of a" << std::endl;
      RemoveOrAddPC(viewer, all_ptr_, 255, 255, 255, all);
  }

  if (event.getKeySym () == "q" && event.keyDown()) {
      std::cout<<"closing" << std::endl;
      viewer->close();
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
  Eigen::Vector3f center(rrect.center.x*u + rrect.center.y*v + p0); 
  table_top_bbx.push_back(center); 

  return table_top_bbx; 
} 


bool CameraFeatureExtractor::CalFeature(Config &cfg, cv::Mat cv_image, CameraCalibrationData &data, bool vis) {
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
      //one to one correspondence between corners and grid3d points
      cv::solvePnP(grid3dpoint, corners, cfg.cameramat, cfg.distcoeff, rvec, tvec);
      cv::projectPoints(grid3dpoint, rvec, tvec, cfg.cameramat, cfg.distcoeff, image_points);
      // Mark the centre square corner points
      cv::projectPoints(square_edge, rvec, tvec, cfg.cameramat, cfg.distcoeff, imagePoints1);
      cv::projectPoints(boardcorners, rvec, tvec, cfg.cameramat, cfg.distcoeff, imagePoints);

        for (int i = 0; i < static_cast<int>(square_edge.size()); i++)
            cv::circle(cv_image, imagePoints1[i], 5, CV_RGB(0,255,0), -1);
        // Mark the board corner points and centre point
        for (int i = 0; i < static_cast<int>(boardcorners.size()); i++)
            cv::circle(cv_image, imagePoints[i], 5, CV_RGB(0,0,255), -1);
        for (int i=0; i<static_cast<int>(image_points.size()); ++i) {
            cv::circle(cv_image, image_points[i], 5, CV_RGB(255,0,0), -1);
            cv::putText(cv_image, std::to_string(i), corners[i], cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
            1.0, // Scale. 2.0 = 2x bigger
            cv::Scalar(255,255,255));
      }

      //print normal point
      std::vector<cv::Point3f> normals;
      std::vector<cv::Point2f> image_points2;
      normals.push_back(cv::Point3f{0,0,100});
      cv::projectPoints(normals, rvec, tvec,cfg.cameramat, cfg.distcoeff, image_points2);

      for (int i=0; i<static_cast<int>(image_points2.size()); ++i) {
          cv::circle(cv_image, image_points2[i], 5, CV_RGB(0,255,0), -1);
      }

      if (vis) {
         cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
         cv::imshow( "Display window", cv_image);                   // Show our image inside it.
         cv::waitKey(0);
      } 
    }
    
    // chessboardpose is a 3*4 transform matrix that transforms points in board frame to camera frame | R&T
    cv::Mat chessboardpose = cv::Mat::eye(4,4,CV_64F);
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

      /*// convert 3D coordinates to image coordinates
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
      */
    }

    data.camerapoint[0] = corner_vectors.at<double>(0,4)/1000; //coverts back to meter
    data.camerapoint[1] = corner_vectors.at<double>(1,4)/1000;
    data.camerapoint[2] = corner_vectors.at<double>(2,4)/1000;
    data.cameranormal[0] = chessboard_normal.at<double>(0);
    data.cameranormal[1] = chessboard_normal.at<double>(1);
    data.cameranormal[2] = chessboard_normal.at<double>(2);
    data.pixeldata = sqrt(pow((imagePoints1[1].x - imagePoints1[0].x), 2) +
        pow((imagePoints1[1].y - imagePoints1[0].y),2));
    return true;
  }
 
  return false;
}