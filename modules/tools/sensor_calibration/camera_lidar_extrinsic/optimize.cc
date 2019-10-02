// Optimization node
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include "openga.h"
#include "optimize.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <tf/transform_datatypes.h>
#include "util.h"
#include "point_type.h"


#define PI 3.141592653589793238463

int sample = 0;

bool output = 0; //FLAGS_output;
bool output2 = 0; //FLAGS_output2;

double colmap[50][3] =  {{0,0,0.5385},{0,0,0.6154},{0,0,0.6923},
                         {0,0,0.7692},{0,0,0.8462},{0,0,0.9231},
                         {0,0,1.0000},{0,0.0769,1.0000},{0,0.1538,1.0000},
                         {0,0.2308,1.0000},{0,0.3846,1.0000},{0,0.4615,1.0000},
                         {0,0.5385,1.0000},{0,0.6154,1.0000},{0,0.6923,1.0000},
                         {0,0.7692,1.0000},{0,0.8462,1.0000},{0,0.9231,1.0000},
                         {0,1.0000,1.0000},{0.0769,1.0000,0.9231},{0.1538,1.0000,0.8462},
                         {0.2308,1.0000,0.7692},{0.3077,1.0000,0.6923},{0.3846,1.0000,0.6154},
                         {0.4615,1.0000,0.5385},{0.5385,1.0000,0.4615},{0.6154,1.0000,0.3846},
                         {0.6923,1.0000,0.3077},{0.7692,1.0000,0.2308},{0.8462,1.0000,0.1538},
                         {0.9231,1.0000,0.0769},{1.0000,1.0000,0},{1.0000,0.9231,0},
                         {1.0000,0.8462,0},{1.0000,0.7692,0},{1.0000,0.6923,0},
                         {1.0000,0.6154,0},{1.0000,0.5385,0},{1.0000,0.4615,0},
                         {1.0000,0.3846,0},{1.0000,0.3077,0},{1.0000,0.2308,0},
                         {1.0000,0.1538,0},{1.0000,0.0769,0},{1.0000,0,0},
                         {0.9231,0,0},{0.8462,0,0},{0.7692,0,0},{0.6923,0,0}};


struct Rotation
{
  double e1; // Rotation optimization variables
  double e2;
  double e3;
  std::string to_string() const
  {
    return
        std::string("{")
        +  "e1:"+std::to_string(e1)
        +", e2:"+std::to_string(e2)
        +", e3:"+std::to_string(e3)
        +"}";
  }
};

struct Rot_Trans_cost // equivalent to y in matlab
{
  double objective2; // This is where the results of simulation is stored but not yet finalized.
};

struct Rotationcost // equivalent to y in matlab
{
  double objective1; // This is where the results of simulation is stored but not yet finalized.
};

Rotation eul;
Rot_Trans eul_t, eul_it;
void image_projection (Rot_Trans rot_trans);

typedef EA::Genetic<Rotation,Rotationcost> GA_Type;
typedef EA::Genetic<Rot_Trans, Rot_Trans_cost> GA_Type2;

CameraVelodyneCalibrationData * calibrationdata;


void init_genes2(Rot_Trans& p,const std::function<double(void)> &rnd01)
{
  std::vector< double > pi_vals;
  pi_vals.push_back(PI/18);
  pi_vals.push_back(-PI/18);
  int RandIndex = rand() % 2;
  p.e1 = eul_t.e1 + pi_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.e2 = eul_t.e2 + pi_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.e3 = eul_t.e3 + pi_vals.at(RandIndex)*rnd01();

  std::vector< double > trans_vals;
  trans_vals.push_back(0.05);
  trans_vals.push_back(-0.05);
  RandIndex = rand() % 2;
  p.x = eul_t.x + trans_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.y = eul_t.y + trans_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.z = eul_t.z + trans_vals.at(RandIndex)*rnd01();
}

double rotation_fitness_func(double e1, double e2, double e3)
{
  tf::Matrix3x3 rot;
  rot.setRPY(e1, e2, e3);
  cv::Mat tmp_rot = (cv::Mat_<double>(3,3) << rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2],
      rot.getRow(1)[0], rot.getRow(1)[1], rot.getRow(1)[2],
      rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2]);

  cv::Mat normals = calibrationdata->cameranormals_mat * tmp_rot.t(); // camera normals in lidar frame
  cv::Mat normal_diff = normals - calibrationdata->velodynenormals_mat;
  cv::Mat normal_square = normal_diff.mul(normal_diff); // square the xyz components of normal_diff
  cv::Mat summed_norm_diff;
  cv::reduce(normal_square, summed_norm_diff, 1, CV_REDUCE_SUM, CV_64F); // add the squared terms
  cv::Mat sqrt_norm_diff; sqrt(summed_norm_diff, sqrt_norm_diff); // take the square root
  double sqrt_norm_sum = 0.0;
  for (int i = 0; i < sample; i++)
    sqrt_norm_sum += sqrt_norm_diff.at<double>(i); // Add the errors involved in all the vectors

  double ana = sqrt_norm_sum/sample; // divide this sum by the total number of samples

  // vectors on the board plane (w.r.t lidar frame)
  cv::Mat plane_vectors = calibrationdata->velodynepoints_mat - calibrationdata->velodynecorners_mat;
  double error_dot = 0.0;
  for (int i = 0; i < sqrt_norm_diff.rows; i++)
  {
    cv::Mat plane_vector = plane_vectors.row(i);
    plane_vector = plane_vector/norm(plane_vector);
    double temp_err_dot = pow(normals.row(i).dot(plane_vector), 2);
    error_dot += temp_err_dot;
  }
  error_dot = error_dot/sample; // dot product average

  if (output) {
    std::cout << "sqrt_norm_sum " << sqrt_norm_sum << std::endl;
    std::cout << "sqrt_norm_diff.rows " << sqrt_norm_diff.rows << std::endl;
    std::cout << "rotation " << tmp_rot << std::endl;
    std::cout << "normals " << normals << std::endl;
    std::cout << "normal_diff " << normal_diff << std::endl;
    std::cout << "normal_square " << normal_square << std::endl;
    std::cout << "summed_norm_diff " << summed_norm_diff << std::endl;
    std::cout << "sqrt_norm_diff " << sqrt_norm_diff << std::endl;
    std::cout << "sqrt_norm_sum " << sqrt_norm_sum << std::endl;
    std::cout << "ana " << ana << std::endl;
    std::cout << "error_dot " << error_dot << std::endl;
  }
  return error_dot + ana;
}

bool eval_solution2 (const Rot_Trans& p, Rot_Trans_cost &c)
{
  const double& e1 = p.e1; const double& e2 = p.e2; const double& e3 = p.e3;
  const double& x = p.x; const double& y = p.y; const double& z = p.z;

  tf::Matrix3x3 rot;
  rot.setRPY(e1, e2, e3);
  cv::Mat tmp_rot = (cv::Mat_<double>(3,3) << rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2],
      rot.getRow(1)[0], rot.getRow(1)[1], rot.getRow(1)[2],
      rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2]);

  double rot_error = rotation_fitness_func(e1, e2, e3);

  cv::Mat translation_ana = (cv::Mat_<double>(1,3) << x, y, z);
  cv::Mat rt, t_fin, vpoints, cpoints; cv::Mat l_row = (cv::Mat_<double>(1,4) << 0.0, 0.0, 0.0, 1.0);
  cv::hconcat(tmp_rot, translation_ana.t(), rt);
  cv::vconcat(rt, l_row, t_fin);
  cv::hconcat(calibrationdata->velodynepoints_mat, cv::Mat::ones(sample, 1, CV_64F), vpoints);
  cv::hconcat(calibrationdata->camerapoints_mat, cv::Mat::ones(sample, 1, CV_64F), cpoints);
  cv::Mat cp_rot = t_fin.inv()*vpoints.t(); cp_rot = cp_rot.t();
  cv::Mat trans_diff = cp_rot - cpoints;
  trans_diff = trans_diff.mul(trans_diff);
  cv::Mat summed_norm_diff, sqrt_norm_sum, sqrt_norm_diff;
  cv::reduce(trans_diff, summed_norm_diff, 1, CV_REDUCE_SUM, CV_64F);
  sqrt(summed_norm_diff, sqrt_norm_diff);
  double summed_sqrt = 0.0;
  for (int i = 0; i < sample; i++)
  {
    summed_sqrt += sqrt_norm_diff.at<double>(i);
  }
  double error_trans = summed_sqrt/sample;

  cv::Mat meanValue, stdValue;
  cv::meanStdDev(sqrt_norm_diff, meanValue, stdValue);

  double var; var = stdValue.at<double>(0);

  std::vector<double> pixel_error;
  for (int i = 0; i < sample; i++)
  {
    double * my_cp = CalibrationUtil::ConvertoImgpts(cp_rot.at<double>(i,0),
                                     cp_rot.at<double>(i,1),
                                     cp_rot.at<double>(i,2));
    double * my_vp = CalibrationUtil::ConvertoImgpts(cpoints.at<double>(i,0),
                                     cpoints.at<double>(i,1),
                                     cpoints.at<double>(i,2));
    double pix_e = sqrt(pow((my_cp[0]-my_vp[0]),2) + pow((my_cp[1]-my_vp[1]),2))*
        calibrationdata->pixeldata_mat.at<double>(i);
    pixel_error.push_back(pix_e);
  }

  double error_pix = *std::max_element(pixel_error.begin(), pixel_error.end());

  c.objective2 = rot_error + var + error_trans + error_pix;

  if (output2) {
    std::cout << "sample " << sample << std::endl;
    std::cout << "tmp_rot " << tmp_rot << std::endl;
    std::cout << "cp_rot " << cp_rot << std::endl;
    std::cout << "t_fin " << t_fin << std::endl;
    std::cout << "translation_ana " << translation_ana << std::endl;
    std::cout << "cp_rot " << cp_rot << std::endl;
    std::cout << "calibrationdata.camerapoints_mat " << calibrationdata->camerapoints_mat << std::endl;
    std::cout << "trans_diff " << trans_diff << std::endl;
    std::cout << "summed_norm_diff " << summed_norm_diff << std::endl;
    std::cout << "sqrt_norm_diff " << sqrt_norm_diff << std::endl;
    std::cout << "summed_sqrt " << summed_sqrt << std::endl;
    std::cout << "error_trans " << error_trans << std::endl;
    std::cout << "c.objective2 " << c.objective2 << std::endl;
    std::cout << "error_pix " << error_pix << std::endl;
  }
  output2 = 0;
  return true; // solution is accepted
}

Rot_Trans mutate2 (const Rot_Trans& X_base, const std::function<double(void)> &rnd01, double shrink_scale)
{
  Rot_Trans X_new;
  bool in_range;
  do {
    in_range = true;
    X_new = X_base;
    X_new.e1 += 0.2 * (rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.e1 >= (eul_t.e1-PI/18) && X_new.e1 < (eul_t.e1+PI/18));
    X_new.e2 += 0.2 * (rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.e2 >= (eul_t.e2-PI/18) && X_new.e2 < (eul_t.e2+PI/18));
    X_new.e3 += 0.2*(rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.e3 >= (eul_t.e3-PI/18) && X_new.e3 < (eul_t.e3+PI/18));

    X_new.x += 0.2 * (rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.x >= (eul_t.x-0.05) && X_new.x < (eul_t.x+0.05));
    X_new.y += 0.2 * (rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.y >= (eul_t.y-0.05) && X_new.y < (eul_t.y+0.05));
    X_new.z += 0.2*(rnd01() - rnd01()) * shrink_scale;
    in_range = in_range && (X_new.z >= (eul_t.z-0.05) && X_new.z < (eul_t.z+0.05));

  } while(!in_range);
  return X_new;
}

Rot_Trans crossover2 (const Rot_Trans& X1,
                      const Rot_Trans& X2,
                      const std::function<double(void)> &rnd01)
{
  Rot_Trans X_new;
  double r;
  r = rnd01();
  X_new.e1 = r*X1.e1 + (1.0-r)*X2.e1;
  r = rnd01();
  X_new.e2 = r*X1.e2 + (1.0-r)*X2.e2;
  r = rnd01();
  X_new.e3 = r*X1.e3 + (1.0-r)*X2.e3;
  r = rnd01();
  X_new.x = r*X1.x + (1.0-r)*X2.x;
  r = rnd01();
  X_new.y = r*X1.y + (1.0-r)*X2.y;
  r = rnd01();
  X_new.z = r*X1.z + (1.0-r)*X2.z;
  return X_new;
}

double calculate_SO_total_fitness2 (const GA_Type2::thisChromosomeType &X)
{
  // finalize the cost
  double final_cost = 0.0;
  final_cost += X.middle_costs.objective2;
  return final_cost;
}

// A function to show/store the results of each generation.
void SO_report_generation2 (int generation_number,
                            const EA::GenerationType<Rot_Trans,
                            Rot_Trans_cost> &last_generation,
                            const Rot_Trans& best_genes)
{
  eul_it.e1 = best_genes.e1;
  eul_it.e2 = best_genes.e2;
  eul_it.e3 = best_genes.e3;
  eul_it.x = best_genes.x;
  eul_it.y = best_genes.y;
  eul_it.z = best_genes.z;
}

void init_genes(Rotation& p, const std::function<double(void)> &rnd01)
{
  std::vector<double> pi_vals;
  pi_vals.push_back(PI/8);
  pi_vals.push_back(-PI/8);
  int RandIndex = rand() % 2;
  p.e1 = eul.e1 + pi_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.e2 = eul.e2 + pi_vals.at(RandIndex)*rnd01();
  RandIndex = rand() % 2;
  p.e3 = eul.e3 + pi_vals.at(RandIndex)*rnd01();
}

bool eval_solution (const Rotation& p, Rotationcost &c)
{
  const double& e1 = p.e1;
  const double& e2 = p.e2;
  const double& e3 = p.e3;

  c.objective1 = rotation_fitness_func(e1, e2, e3);

  return true; // solution is accepted
}

Rotation mutate (const Rotation& X_base,
                 const std::function<double(void)> &rnd01,
                 double shrink_scale)
{
  Rotation X_new;
  bool in_range;
  do{
    in_range = true;
    X_new = X_base;
    X_new.e1 += 0.2*(rnd01() - rnd01())*shrink_scale;
    in_range = in_range && (X_new.e1 >= (eul.e1-PI/8) && X_new.e1 < (eul.e1+PI/8));
    X_new.e2 += 0.2*(rnd01() - rnd01())*shrink_scale;
    in_range = in_range && (X_new.e2 >= (eul.e2-PI/8) && X_new.e2 < (eul.e2+PI/8));
    X_new.e3 += 0.2*(rnd01() - rnd01())*shrink_scale;
    in_range = in_range && (X_new.e3 >= (eul.e3-PI/8) && X_new.e3 < (eul.e3+PI/8));
  } while(!in_range);
  return X_new;
}

Rotation crossover (const Rotation& X1,
                    const Rotation& X2,
                    const std::function<double(void)> &rnd01)
{
  Rotation X_new;
  double r = rnd01();
  X_new.e1 = r*X1.e1 + (1.0-r)*X2.e1;
  r = rnd01();
  X_new.e2 = r*X1.e2 + (1.0-r)*X2.e2;
  r=rnd01();
  X_new.e3 = r*X1.e3 + (1.0-r)*X2.e3;
  return X_new;
}

double calculate_SO_total_fitness (const GA_Type::thisChromosomeType &X)
{
  double final_cost = 0.0; // finalize the cost
  final_cost += X.middle_costs.objective1;
  return final_cost;
}

// A function to show/store the results of each generation.
void SO_report_generation (int generation_number,
                           const EA::GenerationType<Rotation,Rotationcost> &last_generation,
                           const Rotation& best_genes)
{
//  std::cout
//      <<"Generation ["<<generation_number<<"], "
//     <<"Best ="<<last_generation.best_total_cost<<", "
//    <<"Average ="<<last_generation.average_cost<<", "
//   <<"Best genes =("<<best_genes.to_string()<<")"<<", "
//  <<"Exe_time ="<<last_generation.exe_time
//  << std::endl;
  eul_t.e1 = best_genes.e1; eul_t.e2 = best_genes.e2; eul_t.e3 = best_genes.e3;

//  std::cout << "eul_t assign " << eul_t.e1 << " "
//            << eul_t.e2 << " "
//            <<  eul_t.e3 << std::endl;
}

// Function converts rotation matrix to corresponding euler angles
std::vector<double> rotm2eul(cv::Mat mat)
{
  std::vector<double> euler(3);
  euler[0] = atan2(mat.at<double>(2,1), mat.at<double>(2,2)); //rotation about x axis: roll
  euler[1] = atan2(-mat.at<double>(2,0), sqrt(mat.at<double>(2,1)*mat.at<double>(2,1) + mat.at<double>(2,2)*mat.at<double>(2,2)));
  euler[2] = atan2(mat.at<double>(1,0), mat.at<double>(0,0)); //rotation about z axis: yaw
  return euler;
}

void Optimizer::optimize(CameraVelodyneCalibrationData& data) {
  calibrationdata = &data;
  sample = data.sample_size;
  //at least 3 normals needed to continue since NN has to be invertible to estimate initial value of rotation matrix
  /*cv::Mat NN = calibrationdata->cameranormals_mat.t()*calibrationdata->cameranormals_mat;
  cv::Mat NM = calibrationdata->cameranormals_mat.t()*calibrationdata->velodynenormals_mat;
  if (cv::determinant(NN) < 0.001) {
     std::cout<<"NN matrix is not invertible"<<std::endl;
     exit(1);
  }

  std::cout<<"NN is " << NN << std::endl;
  std::cout<<"NM is " << NM << std::endl;
  std::cout<<"NN inv is "<<NN.inv()<<std::endl;
  cv::Mat UNR = (NN.inv()*NM).t(); // Analytical rotation matrix for real data
  */
  cv::Mat UNR = (cv::Mat_<double>(3,3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
  //obtain initial value from measurements

  std::cout << "Analytical rotation matrix " << UNR << std::endl;
  std::vector<double> euler;
  euler = rotm2eul(UNR); // rpy wrt original axes
  std::cout << "Analytical Euler angles " << euler.at(0) << " " << euler.at(1) << " " << euler.at(2) << " " << std::endl;
  eul.e1 = euler[0]; eul.e2 = euler[1]; eul.e3 = euler[2];

  //debug print
  // Optimized rotation
  tf::Matrix3x3 rotd;
  rotd.setRPY(eul_t.e1, eul_t.e2, eul_t.e3);
  tf::Quaternion qd;
  rotd.getRotation(qd);
  //TODO: need investigation
  std::cout<<"quaternion for initial val is "<<qd.x()<<" "<<qd.y() << " "<<qd.z()<< " " <<qd.w()<<std::endl;


  EA::Chronometer timer;
  timer.tic();

  // Optimization for rotation alone
  GA_Type ga_obj;
  ga_obj.problem_mode = EA::GA_MODE::SOGA;
  ga_obj.multi_threading = false;
  ga_obj.verbose = false;
  ga_obj.population = 200;
  ga_obj.generation_max = 1000;
  ga_obj.calculate_SO_total_fitness = calculate_SO_total_fitness;
  ga_obj.init_genes = init_genes;
  ga_obj.eval_solution = eval_solution;
  ga_obj.mutate = mutate;
  ga_obj.crossover = crossover;
  ga_obj.SO_report_generation = SO_report_generation;
  ga_obj.best_stall_max = 100;
  ga_obj.average_stall_max = 100;
  ga_obj.tol_stall_average = 1e-8;
  ga_obj.tol_stall_best = 1e-8;
  ga_obj.elite_count = 10;
  ga_obj.crossover_fraction = 0.8;
  ga_obj.mutation_rate = 0.2;
  ga_obj.best_stall_max = 10;
  ga_obj.elite_count = 10;
  ga_obj.solve();

  // Optimized rotation
  tf::Matrix3x3 rot;
  rot.setRPY(eul_t.e1, eul_t.e2, eul_t.e3);
  cv::Mat tmp_rot = (cv::Mat_<double>(3,3) <<
                     rot.getRow(0)[0], rot.getRow(0)[1], rot.getRow(0)[2],
      rot.getRow(1)[0], rot.getRow(1)[1], rot.getRow(1)[2],
      rot.getRow(2)[0], rot.getRow(2)[1], rot.getRow(2)[2]);

  std::cout<<"tmp rotation is " << tmp_rot<<std::endl;
  //exit(1);
 

  // Analytical Translation
  /*cv::Mat cp_trans = tmp_rot*calibrationdata->camerapoints_mat.t();
  cv::Mat trans_diff = calibrationdata->velodynepoints_mat.t() - cp_trans;
  cv::Mat summed_diff; cv::reduce(trans_diff, summed_diff, 1, CV_REDUCE_SUM, CV_64F);
  summed_diff = summed_diff/trans_diff.cols;
  eul_t.x = summed_diff.at<double>(0);
  eul_t.y = summed_diff.at<double>(1);
  eul_t.z = summed_diff.at<double>(2);
  */
  //use the measured translation difference
  //x: 0.8637672084168291
  //z: -0.3331645055101623
  //y: -0.1409059968513362
  eul_t.x = 0.3048;
  eul_t.y = -0.137;
  eul_t.z = -0.42;
  std::cout << "Rotation and Translation after first optimization " << eul_t.e1 << " "
            << eul_t.e2 << " " << eul_t.e3 << " " << eul_t.x << " " << eul_t.y << " " << eul_t.z << std::endl;

  // extrinsics stored the vector of extrinsic parameters in every iteration
  std::vector<std::vector<double>> extrinsics;
  for (int i = 0; i < 20; i++)
  {
    // Joint optimization for Rotation and Translation (Perform this 10 times and take the average of the extrinsics)
    GA_Type2 ga_obj2;
    ga_obj2.problem_mode = EA::GA_MODE::SOGA;
    ga_obj2.multi_threading = false;
    ga_obj2.verbose = false;
    ga_obj2.population = 200;
    ga_obj2.generation_max = 1000;
    ga_obj2.calculate_SO_total_fitness = calculate_SO_total_fitness2;
    ga_obj2.init_genes = init_genes2;
    ga_obj2.eval_solution = eval_solution2;
    ga_obj2.mutate = mutate2;
    ga_obj2.crossover = crossover2;
    ga_obj2.SO_report_generation = SO_report_generation2;
    ga_obj2.best_stall_max = 100;
    ga_obj2.average_stall_max = 100;
    ga_obj2.tol_stall_average = 1e-8;
    ga_obj2.tol_stall_best = 1e-8;
    ga_obj2.elite_count = 10;
    ga_obj2.crossover_fraction = 0.8;
    ga_obj2.mutation_rate = 0.2;
    ga_obj2.best_stall_max = 10;
    ga_obj2.elite_count = 10;
    ga_obj2.solve();
    std::vector<double> ex_it;
    ex_it.push_back(eul_it.e1); ex_it.push_back(eul_it.e2); ex_it.push_back(eul_it.e3);
    ex_it.push_back(eul_it.x); ex_it.push_back(eul_it.y); ex_it.push_back(eul_it.z);
    extrinsics.push_back(ex_it);
    std::cout << "Extrinsics for iteration" << i << " " << extrinsics[i][0] << " " << extrinsics[i][1] << " " << extrinsics[i][2]
              << " " << extrinsics[i][3] << " " << extrinsics[i][4] << " " << extrinsics[i][5] << std::endl;
  }
  // Perform the average operation
  double e_x = 0.0; double e_y = 0.0; double e_z = 0.0; double e_e1 = 0.0; double e_e2 = 0.0; double e_e3 = 0.0;
  for (int i = 0; i < 10; i++)
  {
    e_e1 += extrinsics[i][0]; e_e2 += extrinsics[i][1]; e_e3 += extrinsics[i][2];
    e_x += extrinsics[i][3]; e_y += extrinsics[i][4]; e_z += extrinsics[i][5];
  }

  Rot_Trans rot_trans;
  rot_trans.e1 = e_e1/10; rot_trans.e2 = e_e2/10; rot_trans.e3 = e_e3/10;
  rot_trans.x = e_x/10; rot_trans.y = e_y/10; rot_trans.z = e_z/10;
  std::cout << "Extrinsic Parameters " << " " << rot_trans.e1 << " " << rot_trans.e2 << " " << rot_trans.e3
            << " " << rot_trans.x << " " << rot_trans.y << " " << rot_trans.z << std::endl;
  std::cout << "The problem is optimized in " << timer.toc() << " seconds." << std::endl;
  
  extrinsics_ = rot_trans;
  tf::Matrix3x3 rot2;
  rot2.setRPY(rot_trans.e1, rot_trans.e2, rot_trans.e3);
  tf::Quaternion q; 
  rot.getRotation(q);
  std::cout<<"quaternion obtained is xyzw: "<<q.x()<<" " << q.y() <<" " << q.z() << " " << q.w()<<std::endl;
  std::cout<<"translation is " << " " << rot_trans.x << " " << rot_trans.y << " " << rot_trans.z << std::endl;
}


pcl::PointCloud<pcl::PointXYZIR> organized_pointcloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_pointcloud)
{
  pcl::PointCloud<pcl::PointXYZIR> organized_pc;
  pcl::KdTreeFLANN<pcl::PointXYZIR> kdtree;

  // Kdtree to sort the point cloud
  kdtree.setInputCloud (input_pointcloud);

  pcl::PointXYZIR searchPoint;// camera position as target
  searchPoint.x = 0.0f;
  searchPoint.y = 0.0f;
  searchPoint.z = 0.0f;

  int K = static_cast<int>(input_pointcloud->points.size());
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  // Sort the point cloud based on distance to the camera
  if (kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
  {
    for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
    {
      pcl::PointXYZIR point;
      point.x = input_pointcloud->points[ pointIdxNKNSearch[i] ].x;
      point.y = input_pointcloud->points[ pointIdxNKNSearch[i] ].y;
      point.z = input_pointcloud->points[ pointIdxNKNSearch[i] ].z;
      point.intensity = input_pointcloud->points[ pointIdxNKNSearch[i] ].intensity;
      point.ring = input_pointcloud->points[ pointIdxNKNSearch[i] ].ring;
      organized_pc.push_back(point);
    }
  }

  //Return sorted point cloud
  return(organized_pc);
}


void Optimizer::dumpprojection(std::string& img_file, std::string& pcd_file, std::string &output)
{
  Rot_Trans rot_trans = extrinsics_;

  cv::Mat new_image_raw = cv::imread(img_file);
  //Extrinsic parameter: Transform Velodyne -> cameras
  tf::Matrix3x3 rot;
  rot.setRPY(rot_trans.e1, rot_trans.e2, rot_trans.e3);

  Eigen::MatrixXf t1(4,4),t2(4,4);
  t1 << static_cast<float>(rot.getRow(0)[0]), static_cast<float>(rot.getRow(0)[1]), static_cast<float>(rot.getRow(0)[2]), static_cast<float>(rot_trans.x),
      static_cast<float>(rot.getRow(1)[0]), static_cast<float>(rot.getRow(1)[1]), static_cast<float>(rot.getRow(1)[2]), static_cast<float>(rot_trans.y),
      static_cast<float>(rot.getRow(2)[0]), static_cast<float>(rot.getRow(2)[1]), static_cast<float>(rot.getRow(2)[2]), static_cast<float>(rot_trans.z),
      0, 0, 0, 1;

  //    x: 0.8637672084168291
  //  z: -0.3331645055101623
  //  y: -0.1409059968513362
  /*t1 << -0.0069811f,  0.0122169f,  0.9999010f, 0.8637672084168f,
        -0.9999756f, -0.0000426f, -0.0069811f, -0.1409059968513f,
        -0.0000426f, -0.9999254f,  0.0122169f, -0.3331645055101f,
        0,0,0,1;*/

  t2 = t1.inverse();

  Eigen::Affine3f transform_A = Eigen::Affine3f::Identity();
  transform_A.matrix() << t2(0,0), t2(0,1), t2(0,2), t2(0,3),
      t2(1,0), t2(1,1), t2(1,2), t2(1,3),
      t2(2,0), t2(2,1), t2(2,2), t2(2,3),
      t2(3,0), t2(3,1), t2(3,2), t2(3,3);

  PointCloudIRPtr cloud(new PointCloudIR);
  if (pcl::io::loadPCDFile<PointXYZIR>(pcd_file, *cloud) == -1 || cloud->size() < 1) {
    std::cout<<"pcd size not expected" << pcd_file <<std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZIR> organized;
  //organized = organized_pointcloud(cloud);
  organized = *cloud;

  for (pcl::PointCloud<pcl::PointXYZIR>::const_iterator it = organized.begin(); it != organized.end(); it++)
  {
    pcl::PointXYZIR itA;
    itA = pcl::transformPoint (*it, transform_A);
    if (itA.z < 0 or std::abs(itA.x/itA.z) > 1.2)
      continue;

    double * img_pts = CalibrationUtil::ConvertoImgpts(itA.x, itA.y, itA.z);
    double length = sqrt(pow(itA.x,2) + pow(itA.y,2) + pow(itA.z,2)); //range of every point
    if(length > 10) continue; //remove points after 10 meter 
    int color = static_cast<int>(std::min(round((length/10.0)*49), 49.0));

    if (img_pts[1] >=0 and img_pts[1] < new_image_raw.rows
        and img_pts[0] >=0 and img_pts[0] < new_image_raw.cols)
    {
      cv::circle(new_image_raw, cv::Point(static_cast<int>(img_pts[0]), static_cast<int>(img_pts[1])), 3,
          CV_RGB(255*colmap[color][0], 255*colmap[color][1], 255*colmap[color][2]), -1);
    }
    delete[] img_pts; 
  }

  // Publish the image projection
  static int id = 0;
  std::string file = std::to_string(id++);
  std::string new_img_filename = output+"/"+file+".png";
  cv::imwrite(new_img_filename, new_image_raw);
}