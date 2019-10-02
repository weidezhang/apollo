#include <string>
#include <iostream>
#include <fstream>
#include "calibration_data.h"

using std::string;
using std::cout;
using std::endl;

struct Rot_Trans
{
  double e1; // Joint (Rotation and translation) optimization variables
  double e2;
  double e3;
  double x;
  double y;
  double z;
  std::string to_string() const
  {
    return
        std::string("{")
        +  "e1:"+std::to_string(e1)
        +", e2:"+std::to_string(e2)
        +", e3:"+std::to_string(e3)
        +", x:"+std::to_string(x)
        +", y:"+std::to_string(y)
        +", z:"+std::to_string(z)
        +"}";
  }
};


class Optimizer
{
public:
  void optimize(CameraVelodyneCalibrationData &data);
  void dumpprojection(std::string& img_file,
  	                  std::string& pcd_file,
  	                  std::string& output);
private:
  Rot_Trans extrinsics_;
};
