#include "feature_extractor.h"
#include "cyber/common/file.h"
#include "modules/common/util/string_util.h"
using namespace apollo::cyber::common;

int main() {

   std::string pcd_folder = "/apollo/data/extracted_data/Camera_Lidar_Calibration-2019-09-17-11-08/_apollo_sensor_velodyne64_PointCloud2/";
   
   /*std::vector<std::string> res = ListSubPaths(pcd_folder, DT_REG);
   PointCloudFeatureExtractor extractor; 
   for (int i=0; i<static_cast<int>(res.size()); ++i) {
     
     if(apollo::common::util::EndWith(res[i], std::string("pcd"))) {
        PointCloudIRPtr cloud(new PointCloudIR);
        if (pcl::io::loadPCDFile<PointXYZIR>(pcd_folder+res[i], *cloud) != -1) {
           std::cout << "loaded file " << res[i] <<"size " << cloud->width << std::endl;
           VelodyneCalibrationData data; 
           extractor.CalFeature(cloud, &data);      
        }
     }
   }*/

   std::string img_folder = "/apollo/data/extracted_data/Camera_Lidar_Calibration-2019-09-24-15-14/_apollo_sensor_camera_front_6mm_image/";
   std::vector<std::string> res = ListSubPaths(img_folder, DT_REG);
   CameraFeatureExtractor extractor;
   Config cfg; 
   for (int i=0; i<static_cast<int>(res.size()); ++i) {
     
     if(apollo::common::util::EndWith(res[i], std::string("png"))) {
        std::cout<<"process " << res[i] << std::endl;
        cv::Mat img = cv::imread(img_folder+res[i]);
        extractor.CalFeature(cfg, img);
     }
   }
}