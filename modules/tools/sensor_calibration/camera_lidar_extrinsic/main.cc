#include "feature_extractor.h"
#include "cyber/common/file.h"
#include "modules/common/util/string_util.h"
#include <fstream>
#include "optimize.h"
#include <gflags/gflags.h>

DEFINE_bool(savefilelist, true, "save selected pcd or image list");
DEFINE_double(xmin, 1.0f, "xmin");
DEFINE_double(xmax, 3.0f, "xmax");
DEFINE_double(ymin, -2.0f, "ymin");
DEFINE_double(ymax, 2.0f, "ymax");
DEFINE_double(zmin, -1.0f, "zmin");
DEFINE_double(zmax, 1.0f, "zmax");

using namespace apollo::cyber::common;

std::vector<std::pair<double, std::string>> load_timestamp(std::map<std::string, double>& data, std::string folder) {
     std::string path = folder + "timestamps.txt";
     std::string prefix;
     double ts;
     std::vector<std::pair<double, std::string>> indx; 
     std::ifstream infile(path);
     while (infile >> prefix >> ts) {
       // process pair (a,b)
       data[prefix] = ts;
       indx.push_back(std::pair<double,std::string>(ts, prefix));
     }
     return indx;
}

template<typename InputIterator, typename ValueType>
InputIterator closest(InputIterator first, InputIterator last, ValueType value)
{
    return std::min_element(first, last, [&](ValueType x, ValueType y)
    {   
        return std::abs(x.first - value.first) < std::abs(y.first - value.first);
    });
}

bool load_pcd_feature(std::string file, std::string folder, 
                      CameraVelodyneCalibrationData *calibrationdata,
                      int sample_idx, bool need_decision=false) {
    PointCloudIRPtr cloud(new PointCloudIR);
    if (pcl::io::loadPCDFile<PointXYZIR>(folder+file, *cloud) != -1) {
       std::cout << "loaded file " << file <<"size " << cloud->width << std::endl;
       VelodyneCalibrationData data;
       PointCloudFeatureExtractor extractor;
       extractor.CalFeature(cloud, &data, true);
       char decision = 'y';
       if (need_decision) {
         std::cout<<"please input accept this pcd (y/n):";
         std::cin >> decision;
       }
       if (decision == 'y') {
          //collect the timestamp and file locat 
          for (int j=0; j<3; ++j) {
             calibrationdata->velodynepoints_mat.at<double>(sample_idx,j)
               = data.velodynepoint[j];
             calibrationdata->velodynenormals_mat.at<double>(sample_idx,j)
               = data.velodynenormal[j];  
             calibrationdata->velodynecorners_mat.at<double>(sample_idx,j)
               = data.velodynecorner[j];
          }
          return true;
       }
    }
    return false;
}

bool load_img_feature(std::string file, std::string folder, 
                      CameraVelodyneCalibrationData *calibrationdata,
                      int sample_idx) {
    std::cout<<"loading "<<folder+file<<std::endl;
    cv::Mat img = cv::imread(folder+file);
    CameraCalibrationData data;
    Config cfg;
    CameraFeatureExtractor extractor_img;
    if (extractor_img.CalFeature(cfg, img, data)) {
        for (int j = 0; j < 3; j++) {
           calibrationdata->camerapoints_mat.at<double>(sample_idx,j)
             = data.camerapoint[j];
           calibrationdata->cameranormals_mat.at<double>(sample_idx,j)
             = data.cameranormal[j];
        }
        calibrationdata->pixeldata_mat.at<double>(sample_idx) = data.pixeldata;
        return true; 
    }
    return false;
}

//save valid list
void save_valid_file(std::string &path, std::vector<std::string> &valid_file) {
    if(valid_file.size()==0)
        return; 

    std::ofstream ofs(path);
    for(auto file:valid_file) {
         ofs<<file<<std::endl;
     }
     ofs.close();
}

int main(int argc, char **argv) {
   ::google::ParseCommandLineFlags(&argc, &argv, true);
   PointCloudFeatureExtractor::bound.x_min = static_cast<float>(FLAGS_xmin);
   PointCloudFeatureExtractor::bound.x_max = static_cast<float>(FLAGS_xmax);
   PointCloudFeatureExtractor::bound.y_min = static_cast<float>(FLAGS_ymin);
   PointCloudFeatureExtractor::bound.y_max = static_cast<float>(FLAGS_ymax);
   PointCloudFeatureExtractor::bound.z_min = static_cast<float>(FLAGS_zmin);
   PointCloudFeatureExtractor::bound.z_max = static_cast<float>(FLAGS_zmax);
   PointCloudFeatureExtractor::bound.debug();

   std::string pcd_folder = "/apollo/data/extracted_data/Camera_Lidar_Calibration-2019-10-04-21-40/_apollo_sensor_velodyne64_PointCloud2/";
   //take the image from  
   std::string img_folder = "/apollo/data/extracted_data/Camera_Lidar_Calibration-2019-10-04-21-40/_apollo_sensor_camera_front_6mm_image/";
   std::string precached_pcd_path = "/apollo/data/saved_pcd_list.txt";
   std::string precached_img_path = "/apollo/data/saved_img_list.txt";

   std::vector<std::string> res = ListSubPaths(pcd_folder, DT_REG);
   const int sample = 9;
   //initialize data structure for collecting samples
   CameraVelodyneCalibrationData calibrationdata;
   //camera specific 
   calibrationdata.cameranormals_mat = cv::Mat(sample , 3, CV_64F);
   calibrationdata.camerapoints_mat = cv::Mat(sample , 3, CV_64F);
   //velodyne specific
   calibrationdata.velodynepoints_mat = cv::Mat(sample , 3, CV_64F);
   calibrationdata.velodynenormals_mat = cv::Mat(sample , 3, CV_64F);
   calibrationdata.velodynecorners_mat = cv::Mat(sample , 3, CV_64F);
   calibrationdata.pixeldata_mat = cv::Mat(1 , sample, CV_64F);

   //int sample_idx = 0;
   std::map<std::string, double> lidar_timestamp, camera_timestamp;
   std::vector<std::pair<double, std::string>> lidar_idx = load_timestamp(lidar_timestamp, pcd_folder);
   load_timestamp(camera_timestamp, img_folder);
   std::vector<double> saved_ts;
   int sample_idx = 0, total_sample;
   std::vector<std::string> valid_pcd_file, valid_img_file;
   sample_idx = 0;
   if (apollo::cyber::common::PathExists(precached_img_path)) {
      std::string file;
      std::ifstream ifs(precached_img_path);
      while(std::getline(ifs, file)) {
         if(load_img_feature(file,img_folder, &calibrationdata, sample_idx)) {
              ++sample_idx;
              std::vector<std::string> splits;
              apollo::common::util::Split(file, '.', &splits);
              saved_ts.push_back(camera_timestamp[splits[0]]);
              valid_img_file.push_back(file);
         }
      }
      ifs.close();
   }
   else {
     std::vector<std::string> res = ListSubPaths(img_folder, DT_REG);
     //manually take 10 pcds and finds corresponding images to extract features 
     std::vector<std::string> valid_file;
     for (int i=0; i<static_cast<int>(res.size()); ++i) {     
       if(apollo::common::util::EndWith(res[i], std::string("png"))) {
          if(load_img_feature(res[i],img_folder, &calibrationdata, sample_idx)) {
             ++sample_idx;
             valid_img_file.push_back(res[i]);
             std::vector<std::string> splits;
             apollo::common::util::Split(res[i], '.', &splits);
             saved_ts.push_back(camera_timestamp[splits[0]]);
          }
       }
     }
   }

   std::cout<<"total image sample collected "<<sample_idx<<std::endl;
   sample_idx = 0;
   if(apollo::cyber::common::PathExists(precached_pcd_path)) {
     std::ifstream ifs(precached_pcd_path);
     std::string file;
     while(ifs>>file) {
        if(load_pcd_feature(file,pcd_folder, &calibrationdata, sample_idx, false)) {
             ++sample_idx;
             valid_pcd_file.push_back(file);
        }
     }
   }
   else {
      for(auto i: saved_ts) {
         std::cout<<"checking ts " << i <<std::endl;
         auto pcd_ts = closest(lidar_idx.begin(), lidar_idx.end(), std::pair<double, std::string>(i, ""));
         double ts = pcd_ts->first;
         std::string file=pcd_ts->second + ".pcd";
         std::cout.precision(17);
         std::cout<<"process pcd file " << file << " ts "
                <<ts << " lookup ts is "<<i << std::endl;
         if(load_pcd_feature(file, pcd_folder, &calibrationdata, sample_idx, true)) {
             ++sample_idx;
             valid_pcd_file.push_back(file);
         }
      }
    }

   total_sample = sample_idx;
  
   std::cout<<"total sample collected "<<total_sample;
   if (total_sample < sample) {
      std::cout<<"not meeting minimum requirement";  
      exit(1); 
   }
   /*res = ListSubPaths(img_folder, DT_REG);
   CameraFeatureExtractor extractor_img;
   Config cfg;
   for (int i=0; i<static_cast<int>(res.size()); ++i) {  
     if(apollo::common::util::EndWith(res[i], std::string("png"))) {
        std::cout<<"process " << res[i] << std::endl;
        cv::Mat img = cv::imread(img_folder+res[i]);
        extractor_img.CalFeature(cfg, img);
     }
   }*/

   if(FLAGS_savefilelist) {
      assert(valid_img_file.size() == valid_pcd_file.size());
      //save valid file
      save_valid_file(precached_pcd_path, valid_pcd_file);
      save_valid_file(precached_img_path, valid_img_file);
    }

   calibrationdata.debug();
   calibrationdata.sample_size = total_sample;
   //generate calibration data 
   std::cout << "total sample collected collected successfully" << std::endl;
    
   //optimizing parameters 
   Optimizer opt;
   opt.optimize(calibrationdata);
   std::string outputfolder="/apollo/data/projection";
   apollo::cyber::common::CreateDir(outputfolder);
   assert(valid_img_file.size() == valid_pcd_file.size());
   for(int i=0; i<static_cast<int>(valid_img_file.size());++i) {
      std::string img_file = img_folder + valid_img_file[i];
      std::string pcd_file = pcd_folder + valid_pcd_file[i];
      std::cout<<"projecting "<<pcd_file << " onto "<<img_file <<std::endl;
      opt.dumpprojection(img_file, pcd_file, outputfolder); 
   }
}