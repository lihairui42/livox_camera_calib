#include "include/lidar_camera_calib.hpp"
#include "ceres/ceres.h"
#include "include/common.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

// Data path
string image_file;
string pcd_file;
string opencv_file;
string lidarIMUExtPara_file;

string result_file;
string pnp_file;

string init_file;
string rough_file;
string opt_file;
string opt_pnp_file;
string residual_file;

// Camera config
double camera_matrix[9];
double dist_coeffs[5];
double width;
double height;

// Calib config
bool use_rough_calib;
bool use_ada_voxel;
bool calib_en;
bool T_opt;
cv::Mat imageCalibration;
cv::Mat imageCalibration_pnp;

// instrins matrix
Eigen::Matrix3d inner;
// Distortion coefficient
Eigen::Vector4d distor;
Eigen::Vector4d quaternion;
Eigen::Vector3d transation;

// Normal pnp solution
// Normal pnp solution 代价函数计算模型
class pnp_calib {
public:
  pnp_calib(PnPData p) { pd = p; }         //残差计算
  template <typename T>                   //模板  T为泛型
  bool operator()(const T *_q, const T *_t, T *residuals) const {  // _q _t lidar位姿参数，残差  首先根据相机的外参将世界坐标系转换到相机坐标系下并归一化
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>(); //声明一个类型为T的3x3的内参矩阵  cx cy fx 
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>(); //声明一个类型为T的4X1的畸变矩阵  k1 k2 p1 p2（k3） 
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};//姿态四元数 增量Quaternion (const Scalar &w, const Scalar &x, const Scalar &y, const Scalar &z)
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};    //位置增量矩阵
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));  //p_1位置 lidar 
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * (p_l + t_incre);  //p_c 中心投影
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c; //P_2 整体投影
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];  // uo,vo是归一化坐标，  其次根据相机的内参对归一化的相机坐标系进行畸变的计算和投影点计算
    const T &fx = innerT.coeffRef(0, 0);   //径向畸变系数 应用二阶和四阶径向畸变
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;//获取的点通常是图像的像素点，所以需要先通过小孔相机模型转换到归一化坐标系下          
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;//经过畸变矫正或理想点的坐标
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;//径向畸变 x_{distorted} = x( 1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    residuals[0] = ud - T(pd.u);//误差是预测位置和观测位置之间的差值
    residuals[1] = vd - T(pd.v);//最后将计算出来的3D投影点坐标与观察的图像图像坐标  重投影误差
    return true;
  }
  static ceres::CostFunction *Create(PnPData p) {
    return (
        new ceres::AutoDiffCostFunction<pnp_calib, 2, 4, 3>(new pnp_calib(p))); //使用自动求导，将之前的代价函数结构体传入，第一个2是输出维度，即残差的维度，第二个4是输入维度，即待寻优参数x的维度。
  }

private:
  PnPData pd;
};

// pnp calib with direction vector
class vpnp_calib {
public:
  vpnp_calib(VPnPData p) { pd = p; }
  template <typename T>
  bool operator()(const T *_q, const T *_t, T *residuals) const {
    Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
    Eigen::Matrix<T, 4, 1> distorT = distor.cast<T>();
    Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
    Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
    Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
    Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
    Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;
    T uo = p_2[0] / p_2[2];
    T vo = p_2[1] / p_2[2];
    const T &fx = innerT.coeffRef(0, 0);
    const T &cx = innerT.coeffRef(0, 2);
    const T &fy = innerT.coeffRef(1, 1);
    const T &cy = innerT.coeffRef(1, 2);
    T xo = (uo - cx) / fx;
    T yo = (vo - cy) / fy;
    T r2 = xo * xo + yo * yo;
    T r4 = r2 * r2;
    T distortion = 1.0 + distorT[0] * r2 + distorT[1] * r4;
    T xd = xo * distortion + (distorT[2] * xo * yo + distorT[2] * xo * yo) +
           distorT[3] * (r2 + xo * xo + xo * xo);
    T yd = yo * distortion + distorT[3] * xo * yo + distorT[3] * xo * yo +
           distorT[2] * (r2 + yo * yo + yo * yo);
    T ud = fx * xd + cx;
    T vd = fy * yd + cy;
    if (T(pd.direction(0)) == T(0.0) && T(pd.direction(1)) == T(0.0)) {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
    } else {
      residuals[0] = ud - T(pd.u);
      residuals[1] = vd - T(pd.v);
      Eigen::Matrix<T, 2, 2> I =
          Eigen::Matrix<float, 2, 2>::Identity().cast<T>();//Identity 自增 cast类型转换
      Eigen::Matrix<T, 2, 1> n = pd.direction.cast<T>();//位姿方向转换2X1矩阵
      Eigen::Matrix<T, 1, 2> nt = pd.direction.transpose().cast<T>();//位姿位置转换1X2矩阵
      Eigen::Matrix<T, 2, 2> V = n * nt;
      V = I - V;//位姿矩阵
      Eigen::Matrix<T, 2, 2> R = Eigen::Matrix<float, 2, 2>::Zero().cast<T>();
      R.coeffRef(0, 0) = residuals[0];
      R.coeffRef(1, 1) = residuals[1];
      R = V * R * V.transpose();
      residuals[0] = R.coeffRef(0, 0);
      residuals[1] = R.coeffRef(1, 1);
    }
    return true;
  }
  static ceres::CostFunction *Create(VPnPData p) {
    return (new ceres::AutoDiffCostFunction<vpnp_calib, 2, 4, 3>(
        new vpnp_calib(p)));
  }

private:
  VPnPData pd;
};
/*
  粗配准阶段是大致关联三维线和二维线特征，关联的函数是：
  P.c=Nmatch/Nsum;
  Nsum表示LiDAR边缘点的数量，
  Nmatch表示的是匹配上的LiDAR边缘点数量。
  匹配是基于LiDAR点投影到图像上后距离直线的方向和距离来评价的（match_dis），初始匹配阶段基于网格搜索遍历所有种可能的转换参数，旋转网格大小为0.1*50度（默认，平移网格大小（？？没有平移）
*/
void roughCalib(Calibration &calibra, Vector6d &calib_params, //定义粗校准函数相机畸变  外参  搜索范围  迭代最大次数
                double search_resolution, int max_iter) {
  float match_dis = 25;   //发现距离？默认值25
  int const_num= 0;
  Eigen::Vector3d fix_adjust_euler(0, 0, 0);    //声明欧拉初始矩阵
  for (int n = 0; n < 2; n++) {
    // if(n>0)
    // {
    //   search_resolution=search_resolution/10;
    //   max_iter=max_iter*10/2;
    // }

    for (int round = 0; round < 3; round++) {
      Eigen::Matrix3d rot;//待优化变量rot
      rot = Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());
      // std::cout << "初始化影像待优化矩阵" << rot << std::endl;
      float min_cost = 1000;
      for (int iter = 0; iter < max_iter; iter++) {
        Eigen::Vector3d adjust_euler = fix_adjust_euler;
        adjust_euler[round] = fix_adjust_euler[round] +
                              pow(-1, iter) * int(iter / 2) * search_resolution;
        Eigen::Matrix3d adjust_rotation_matrix;
        adjust_rotation_matrix =
            Eigen::AngleAxisd(adjust_euler[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(adjust_euler[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(adjust_euler[2], Eigen::Vector3d::UnitX());
        Eigen::Matrix3d test_rot = rot * adjust_rotation_matrix;
       // Eigen::Matrix3d test_rot = adjust_rotation_matrix*rot;

        // std::cout << "adjust_rotation_matrix " << adjust_rotation_matrix
        //           << std::endl;
        Eigen::Vector3d test_euler = test_rot.eulerAngles(2, 1, 0);
        // std::cout << "test euler: " << test_euler << std::endl;
        Vector6d test_params;
        test_params << test_euler[0], test_euler[1], test_euler[2],
            calib_params[3], calib_params[4], calib_params[5];
        //调整外参安置角（每次调整安置角单个安置角0.1°），匹配图像与雷达边线，存于pnplist中。通过判断（lidar边线总数-配对数）/边线总数 确定当前安置角参数是否可用
        std::vector<VPnPData> pnp_list;
        calibra.buildVPnp(test_params, match_dis, false,
                          calibra.rgb_egde_cloud_, calibra.lidar_edge_clouds,
                          pnp_list);
                          
        float cost = (calibra.lidar_edge_clouds->size() - pnp_list.size()) *
                     1.0 / calibra.lidar_edge_clouds->size();
        if (cost < min_cost) {
          std::cout << "粗拼特征最小占比::" << cost << std::endl;
          min_cost = cost;
          calib_params[0] = test_params[0];
          calib_params[1] = test_params[1];
          calib_params[2] = test_params[2];
          calibra.buildVPnp(calib_params, match_dis, true,
                            calibra.rgb_egde_cloud_, calibra.lidar_edge_clouds,
                            pnp_list);
          cv::Mat projection_img = calibra.getProjectionImg(calib_params,imageCalibration);
          cv::Mat projection_img_show;
          cv::resize(projection_img,projection_img_show,cv::Size(1280,720));
          cv::imshow("Rough Optimization", projection_img_show);
          cv::waitKey(50);
          std::cout<<"pnp_list_num "<< pnp_list.size()<<std::endl;
          const_num=0;
        }
        else
        {
          const_num++;;
        }
       if(0)//const_num>=3)
       {
         break;
       }
       ROS_INFO(" const_numr %d",const_num);
       ROS_INFO(" Rougn calibration number %d",iter);
       std::cout << "校准角度:"<<round <<" 粗校步进 (ypr):" << RAD2DEG(adjust_euler(0)) <<' '<< RAD2DEG(adjust_euler(1)) <<' '<< RAD2DEG(adjust_euler(2))<<endl;
       std::cout << "校准角度:"<<round <<" 粗校姿态角(ypr):" << RAD2DEG(calib_params(0)) <<' '<< RAD2DEG(calib_params(1)) <<' '<< RAD2DEG(calib_params(2))<<endl;
      }
    }
  }
}

/**********************************************/
/*相机去畸变:针孔模型*****************************/
/*********************************************/
int Image_PinHole_Distort(cv::Mat camera_matrix_, cv::Size &imageSize, cv::Mat dist_coeffs_, cv::Mat &image_src, cv::Mat &image_des)
{
  cv::Mat map1, map2;
  imageSize = image_src.size();

  cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
  cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, imageSize, 1, imageSize, 0),
    imageSize, CV_16SC2, map1, map2);
  cv::remap(image_src, image_des, map1, map2, cv::INTER_LINEAR);    //使用INTER_LINEAR进行remap(双线性插值)

  if(0)
  {
    cv::Mat imageCalibration_show;
    cv::resize(image_des, imageCalibration_show, cv::Size(1920, 1080));
    cv::imshow("imageCalibration", imageCalibration_show);
    cv::waitKey(100);
  }
    
  return 1;
}



/**********************************************/
/*Yaml参数读取*********************************/
/*********************************************/
int Yaml_Para_Deal(ros::NodeHandle &nh, Config_OutDoor &outConf, LidarIMU_ExtPara &paraL, CameraIMU_ExtPara &paraC)
{
  nh.param<string>("common/image_file", image_file, "");
  nh.param<string>("common/pcd_file", pcd_file, "");
  nh.param<string>("common/opencv_file", opencv_file, "");
  nh.param<string>("common/Lidar_IMU_file", lidarIMUExtPara_file, "");

  nh.param<string>("common/result_file", result_file, "");
  nh.param<string>("common/pnp_file", pnp_file, "");

  nh.param<bool>("calib/use_rough_calib", use_rough_calib, false);
  nh.param<bool>("calib/calib_en", calib_en, false);
  nh.param<bool>("calib/T_opt", T_opt, false);
  nh.param<bool>("calib/use_ada_voxel", use_ada_voxel, false);

  std::cout << "image_file path:" << image_file << std::endl;
  std::cout << "pcd_file path:" << pcd_file << std::endl;

  nh.param<string>("common/init_file", init_file, "");
  nh.param<string>("common/rough_file", rough_file, "");
  nh.param<string>("common/opt_file", opt_file, "");
  nh.param<string>("common/opt_pnp_file", opt_pnp_file, "");
  nh.param<string>("common/residual_file", residual_file, "");

  //原outdoor参数
  nh.param<vector<double>>("ExtrinsicMat/data", outConf.init_extrinsic_, vector<double>());

  nh.param<int>("Canny/gray_threshold", outConf.rgb_canny_threshold_, 10);
  nh.param<int>("Canny/len_threshold", outConf.rgb_edge_minLen_, 200);

  nh.param<float>("Voxel/size", outConf.voxel_size_, 0.5);  
  nh.param<float>("Voxel_auto/size", outConf.voxel_auto_size_, 4.0); 
  nh.param<float>("Voxel/down_sample_size", outConf.down_sample_size_, 0.05);  

  nh.param<float>("Plane/min_points_size", outConf.plane_size_threshold_, 30); 
  nh.param<float>("Plane/max_size", outConf.plane_max_size_, 8);  
  nh.param<float>("Plane/normal_theta_min", outConf.theta_min_, 30); 
  nh.param<float>("Plane/normal_theta_max", outConf.theta_max_, 150); 

  nh.param<float>("Ransac/dis_threshold", outConf.ransac_dis_threshold_, 0.02); 

  nh.param<float>("Edge/min_dis_threshold", outConf.min_line_dis_threshold_, 0.03); 
  nh.param<float>("Edge/max_dis_threshold", outConf.max_line_dis_threshold_, 0.06);  

  nh.param<int>("Color/intensity_threshold", outConf.color_intensity_threshold_, 5);

  outConf.theta_min_ = cos(DEG2RAD(outConf.theta_min_));
  outConf.theta_max_ = cos(DEG2RAD(outConf.theta_max_));
  
  outConf.direction_theta_min_ = cos(DEG2RAD(30.0));
  outConf.direction_theta_max_ = cos(DEG2RAD(150.0));

  //Lidar IMU外参
  nh.param<double>("LidarIMU/roll", paraL.roll, 0); 
  nh.param<double>("LidarIMU/pitch", paraL.pitch, 0); 
  nh.param<double>("LidarIMU/yaw", paraL.yaw, -90);

  double D2R = M_PI / 180;
  paraL.roll = paraL.roll * D2R;
  paraL.pitch = paraL.pitch * D2R;
  paraL.yaw = paraL.yaw * D2R;

  //Camara IMU外参
  nh.param<double>("CameraIMU/roll", paraC.roll, 0); 
  nh.param<double>("CameraIMU/pitch", paraC.pitch, 0); 
  nh.param<double>("CameraIMU/yaw", paraC.yaw, -90);

  paraC.roll = paraC.roll * D2R;
  paraC.pitch = paraC.pitch * D2R;
  paraC.yaw = paraC.yaw * D2R;

  return 0;
}

/**********************************************/
/*Opencv XML文件读取内参*************************/
/**********************************************/
int Camera_OpencvPara_Deal(string opencv_file)
{
  cv::FileStorage fs;   
  fs.open(opencv_file, FileStorage::READ);
  if(!fs.isOpened())
  {
    std::cout << "Camera para .xml file open failed!";
    return -1;
  }

  cv::Mat M1, M2;
  fs["Camera_Matrix"] >> M1;
  camera_matrix[0] = M1.at<double>(0 ,0);
  camera_matrix[4] = M1.at<double>(1 ,1);
  camera_matrix[2] = M1.at<double>(0 ,2);
  camera_matrix[5] = M1.at<double>(1 ,2);
  camera_matrix[8] = 1;

  fs["Distortion_Coefficients"] >> M2;
  dist_coeffs[0] = M2.at<double>(0);
  dist_coeffs[1] = M2.at<double>(1);
  dist_coeffs[2] = M2.at<double>(2);
  dist_coeffs[3] = M2.at<double>(3);
  dist_coeffs[4] = M2.at<double>(4);

  std::cout << "Camera_Matrix:" << std::endl;
  std::cout << camera_matrix[0] << "    "
            << camera_matrix[1] << "    "
            << camera_matrix[2] << "    " << std::endl;
  std::cout << camera_matrix[3] << "    "
            << camera_matrix[4] << "    "
            << camera_matrix[5] << "    " << std::endl;
  std::cout << camera_matrix[6] << "    "
            << camera_matrix[7] << "    "
            << camera_matrix[8] << "    "<< std::endl;

  std::cout << "dist_coeffs:" << std::endl;
  std::cout << dist_coeffs[0] << "    "
            << dist_coeffs[1] << "    "
            << dist_coeffs[2] << "    "
            << dist_coeffs[3] << "    "
            << dist_coeffs[4] << "    "<< std::endl;

  return 0;
}

/**********************************************/
/*Opencv XML文件读取内参*************************/
/**********************************************/
int Lidar_IMU_ExtPara_Deal(string lidarIMUExtPara_file, LidarIMU_ExtPara &paraL)
{
  vector<double> dd;

  std::fstream file_;
  file_.open(lidarIMUExtPara_file, ios::in);
  if (!file_) {
    cout << "Lidar-IMU extern para file: " << lidarIMUExtPara_file << " open failed" << endl;
    return -1;
  }

  std::string line, line0;
  int equ_i = 0;
  std::getline(file_, line);
  if(!file_.eof())
  {
    std::getline(file_, line);

    //寻找等号
    for(int i=0; i<line.length(); i++)
    {
      if(line[i] == '=')
      {
        equ_i = i;
        break;
      } 
    }

    line0 = line.substr(equ_i+1, line.length()-1);

    istringstream devide(line0);
    string word;
    while(devide >> word)
    {
      dd.push_back(std::atof(word.c_str()));
    }
  }

  if(isfinite(dd[0]) && isfinite(dd[1]) && isfinite(dd[2]))
  {
    double D2R = M_PI / 180;
    paraL.deltaRoll = dd[2] * D2R;
    paraL.deltaPitch = dd[1] * D2R;
    paraL.deltaYaw = dd[0] * D2R;
    std::cout << "Lidar-IMU Extern Para(roll-pitch-yaw):  "<< dd[2] << "  "<< dd[1] << "  "<< dd[0] << endl;
  }
  else
  {
    paraL.deltaRoll = 0;
    paraL.deltaPitch = 0;
    paraL.deltaYaw = 0;
    std::cout << "Lidar-IMU Extern Para invalid, all set to 0:  "<< 0 << "  "<< 0 << "  "<< 0 << endl;
  }

  return 0;
}

/**********************************************/
/*相机参数处理**********************************/
/*********************************************/
int Calibration_Para_Deal(Calibration &calibra, cv::Mat &camera_matrix_, cv::Mat &dist_coeffs_)
{
  calibra.fx_ = camera_matrix[0];
  calibra.cx_ = camera_matrix[2];
  calibra.fy_ = camera_matrix[4];
  calibra.cy_ = camera_matrix[5];
  calibra.k1_ = dist_coeffs[0];
  calibra.k2_ = dist_coeffs[1];
  calibra.p1_ = dist_coeffs[2];
  calibra.p2_ = dist_coeffs[3];
  calibra.k3_ = dist_coeffs[4];

  camera_matrix_.at<double>(0, 0) = camera_matrix[0];//Fx
	camera_matrix_.at<double>(0, 1) = 0;
	camera_matrix_.at<double>(0, 2) = camera_matrix[2];//Cx
	camera_matrix_.at<double>(1, 1) = camera_matrix[4];//Fy
	camera_matrix_.at<double>(1, 2) = camera_matrix[5];//cy

  dist_coeffs_.at<double>(0, 0) = dist_coeffs[0];
	dist_coeffs_.at<double>(1, 0) = dist_coeffs[1] ;
	dist_coeffs_.at<double>(2, 0) = dist_coeffs[2];
	dist_coeffs_.at<double>(3, 0) = dist_coeffs[3];
	dist_coeffs_.at<double>(4, 0) = dist_coeffs[4];     //k1,k2,p1,p2,k3     

  return 0;
}

/**********************************************/
/*iamge边缘提取*******************************/
/*********************************************/
int  Image_Edge_Deal(Calibration &calibra)
{
    // check rgb or gray
  if (calibra.image_.type() == CV_8UC1) {
    calibra.grey_image_ = calibra.image_;
  } else if (calibra.image_.type() == CV_8UC3) {
    cv::cvtColor(calibra.image_, calibra.grey_image_, cv::COLOR_BGR2GRAY);
  } else {
    std::string msg = "Unsupported image type, please use CV_8UC3 or CV_8UC1";
    ROS_ERROR_STREAM(msg.c_str());
    exit(-1);
  }

  cv::Mat edge_image;
  calibra.edgeDetector(calibra.rgb_canny_threshold_, calibra.rgb_edge_minLen_, calibra.grey_image_, edge_image,
               calibra.rgb_egde_cloud_);
  std::string msg = "Sucessfully extract edge from image, edge size:" +
                    std::to_string(calibra.rgb_egde_cloud_->size());
  ROS_INFO_STREAM(msg.c_str());
  std::cout << "rgb_egde size:" << calibra.rgb_egde_cloud_->size() << std::endl;

  return 0;
}

/**********************************************/
/*点云数据边缘提取*******************************/
/*********************************************/
int Scan_Edge_Deal(Calibration &calibra)
{
  calibra.lidar_edge_clouds = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

  Eigen::Vector3d lwh(50, 50, 30);
  Eigen::Vector3d origin(0, -25, -10);
  std::vector<VoxelGrid> voxel_list;
  std::unordered_map<VOXEL_LOC, OctoTree *> adapt_voxel_map;
  time_t t1 = clock();

  if (use_ada_voxel)
  {
    calibra.adaptVoxel(adapt_voxel_map, calibra.voxel_auto_size_, 0.0009);
    calibra.debugVoxel(adapt_voxel_map);
    down_sampling_voxel(*calibra.lidar_edge_clouds, 0.05);
    ROS_INFO_STREAM("Adaptive voxel sucess!");
    time_t t2 = clock();
    std::cout << "adaptive time:" << (double)(t2 - t1) / (CLOCKS_PER_SEC) << "s" << std::endl;
  }
  else
  {
    std::unordered_map<VOXEL_LOC, Voxel *> voxel_map; 
    calibra.initVoxel(calibra.raw_lidar_cloud_, calibra.voxel_size_, voxel_map);
    calibra.LiDAREdgeExtraction(voxel_map, calibra.ransac_dis_threshold_, calibra.plane_size_threshold_,
                        calibra.lidar_edge_clouds);
    time_t t3 = clock();
    std::cout << "voxel time:" << (double)(t3 - t1) / (CLOCKS_PER_SEC) << std::endl;
  }
  std::cout << "lidar edge size:" << calibra.lidar_edge_clouds->size() << std::endl;
  return 0;
}


/**********************************************/
/*image和点云初次映射****************************/
/*********************************************/
int Scan_Camera_Projection_First(Calibration &calibra, cv::Mat &imageCalibration, Vector6d &calib_params, Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
  ROS_INFO_STREAM("Finish prepare!");
  inner << calibra.fx_, 0.0, calibra.cx_, 0.0, calibra.fy_, calibra.cy_, 0.0, 0.0, 1.0;
  distor << calibra.k1_, calibra.k2_, calibra.p1_, calibra.p2_;
  R = calibra.init_rotation_matrix_;
  T = calibra.init_translation_vector_;
  std::cout << "Initial rotation matrix:" << std::endl
            << calibra.init_rotation_matrix_ << std::endl;
  std::cout << "Initial translation:"
            << calibra.init_translation_vector_.transpose() << std::endl;

  Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
  calib_params[0] = euler[0];
  calib_params[1] = euler[1];
  calib_params[2] = euler[2];
  calib_params[3] = T[0];
  calib_params[4] = T[1];
  calib_params[5] = T[2];

  cv::Mat init_img = calibra.getProjectionImg(calib_params, imageCalibration);
  cv::Mat init_img_show;
  init_img_show=cv::Mat::zeros(init_img.size(),init_img.type());
  cv::resize(init_img,init_img_show,cv::Size(1280,720));
  cv::imshow("Initial extrinsic", init_img_show);
  cv::imwrite(init_file, init_img_show);
  cv::waitKey(1000);

  return 0;
}

/**********************************************/
/*精校准***************************************/
/*********************************************/
void OptCalib(Calibration &calibra, 
              Vector6d &calib_params, 
              cv::Mat &imageCalibration, 
              std::vector<VPnPData> &vpnp_list, 
              bool &opt_flag,
              Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
  int iter = 0;
  // Maximum match distance threshold: 15 pixels
  // If initial extrinsic lead to error over 15 pixels, the algorithm will not work
  int dis_threshold = 30; 

  //不是通过迭代次数决定处理时间，而是通过dis_threshold（边缘点云与图像边缘点云距离最大值）
  // Iteratively reducve the matching distance threshold
  for (dis_threshold = 30; dis_threshold > 10; dis_threshold -= 1) {
    
    // For each distance, do twice optimization
    for (int cnt = 0; cnt < 2; cnt++) {
      std::cout << "Iteration:" << iter++ << " Dis:" << dis_threshold << std::endl;
      //生成vpnp_list,用于迭代优化
      calibra.buildVPnp(calib_params, dis_threshold, true,
                        calibra.rgb_egde_cloud_, calibra.lidar_edge_clouds,
                        vpnp_list);

      cv::Mat projection_img = calibra.getProjectionImg(calib_params,imageCalibration);
      cv::Mat projection_img_show;
      cv::resize(projection_img,projection_img_show,cv::Size(1280,720));
      cv::imshow("Optimization", projection_img_show);
      cv::waitKey(100);
      Eigen::Vector3d euler_angle(calib_params[0], calib_params[1],
                                  calib_params[2]);
      Eigen::Matrix3d opt_init_R;
      opt_init_R = Eigen::AngleAxisd(euler_angle[0], Eigen::Vector3d::UnitZ()) *
                   Eigen::AngleAxisd(euler_angle[1], Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(euler_angle[2], Eigen::Vector3d::UnitX());
      Eigen::Quaterniond q(opt_init_R);
      Eigen::Vector3d ori_t = T;
      double ext[7];
      ext[0] = q.x();
      ext[1] = q.y();
      ext[2] = q.z();
      ext[3] = q.w();
      ext[4] = T[0];
      ext[5] = T[1];
      ext[6] = T[2];
      Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);  //姿态相关四元数
      Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);     //位置偏移量


      ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
      ceres::Problem problem;

      problem.AddParameterBlock(ext, 4, q_parameterization);
      problem.AddParameterBlock(ext + 4, 3);
    
      for (auto val : vpnp_list) {
        ceres::CostFunction *cost_function;
        cost_function = vpnp_calib::Create(val);
        problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
        //problem.AddResidualBlock(cost_function, NULL, ext);
      }

      if(!T_opt)
      {
        problem.SetParameterBlockConstant(ext+4);
      }

      //SetParameterBlockVariable(ext+4)
      ceres::Solver::Options options;//最后配置并运行求解器
      options.preconditioner_type = ceres::JACOBI;
      options.linear_solver_type = ceres::SPARSE_SCHUR;//配置增量方程的解法 SPARSE_SCHUR DENSE_QR
      options.minimizer_progress_to_stdout = true;//输出优化过程信息
      options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;//信任域策略 LM算法
      // options.max_num_iterations= 10;  //迭代次数设置
      options.num_threads = 6;//解算线程数
      options.max_num_line_search_direction_restarts=0.5; //线性角度限制
      
      ceres::Solver::Summary summary;//优化信息
      ceres::Solve(options, &problem, &summary);//求解!!!
      std::cout << summary.BriefReport() << std::endl;
      
      Eigen::Matrix3d rot = m_q.toRotationMatrix();
      euler_angle = rot.eulerAngles(2, 1, 0);
      // std::cout << m_t << std::endl;
      calib_params[0] = euler_angle[0];
      calib_params[1] = euler_angle[1];
      calib_params[2] = euler_angle[2];
      calib_params[3] = m_t(0);
      calib_params[4] = m_t(1);
      calib_params[5] = m_t(2);

      // calib_params[3] = T(0);
      // calib_params[4] = T(1);
      // calib_params[5] = T(2);
      R = rot;
      T[0] = m_t(0);
      T[1] = m_t(1);
      T[2] = m_t(2);

      Eigen::Quaterniond opt_q(R);
      std::cout << "旋转偏差:" << RAD2DEG(opt_q.angularDistance(q))
                << " ,平移偏差:" << (T - ori_t).norm() << std::endl;
      std::cout << "精细调整旋转矩阵后:"<< euler_angle[0] <<' '<<euler_angle[1] <<' '<< euler_angle[2] << std::endl;   //输出旋转矩阵
      std::cout << "精细调整位置后:"<< T[0] <<' '<< T[1] <<' '<< T[2] << std::endl;      

      // if(iter>1)
      // {
      //   if ( 0 && ((opt_q.angularDistance(q) < DEG2RAD(0.01) &&
      //       (T - ori_t).norm() < 0.005)||( opt_q.angularDistance(q) > DEG2RAD(0.5))) )
      //   {
      //       opt_flag = false;
      //   }
      //   else
      //   {
      //     if ((opt_q.angularDistance(q) < DEG2RAD(0.03)) || (opt_q.angularDistance(q) > DEG2RAD(0.08))) {
      //         opt_flag = false;
      //     }
      //   }
      // }
      if (!opt_flag) {
            break;
      }
    }
    if (!opt_flag) {
      break;
    }
  }
  ros::Rate loop(0.5);
}

/**********************************************/
/*精校准结果*************************************/
/**********************************************/
int  OptResult(Calibration &calibra,
                Vector6d &calib_params, 
                Eigen::Matrix3d &R, Eigen::Vector3d &T, 
                time_t &t1, 
               cv::Mat &opt_img, cv::Mat &opt_img_show,
               Eigen::Vector3d &vcs, Eigen::Matrix3d &Csc0)    
{
  R = Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());


  std::ofstream outfile(result_file);

  outfile << "Camera-Lisar的相对位姿:" <<  std::endl;
  for (int i = 0; i < 3; i++) {
    outfile << R(i, 0) << "," << R(i, 1) << "," << R(i, 2) << "," << T[i]
            << std::endl;
  }
  outfile << 0 << "," << 0 << "," << 0 << "," << 1 << std::endl << std::endl;

  outfile << "Camera-Lisar欧拉角(yaw-pitch-roll):" <<  std::endl;
  Eigen::Vector3d euler_ori = R.eulerAngles(2, 1, 0);
  outfile << RAD2DEG(euler_ori[0]) << "," << RAD2DEG(euler_ori[1]) << ","
          << RAD2DEG(euler_ori[2]) << "," << 0 << "," << 0 << "," << 0
          << std::endl << std::endl;    
  
  opt_img = calibra.getProjectionImg(calib_params,imageCalibration);
  cv::resize(opt_img,opt_img_show,cv::Size(1280,720));
  cv::imshow("Optimization result", opt_img_show);
  cv::imwrite(opt_file, opt_img);

  cv::waitKey(1000);

  Eigen::Matrix3d init_rotation = Csc0;
  Eigen::Matrix3d adjust_rotation;
  
  outfile << "Camera-Lisar的小角度旋转矩阵:" <<  std::endl;
  adjust_rotation =  init_rotation.transpose() * R;
  for (int i = 0; i < 3; i++) {
    outfile << adjust_rotation(i, 0) << "," << adjust_rotation(i, 1) << "," <<adjust_rotation(i, 2) << "," << T[i]
            << std::endl;
  }
  outfile << 0 << "," << 0 << "," << 0 << "," << 1 << std::endl << std::endl;

  outfile << "Camera-Lisar的小角度欧拉角(yaw-pitch-roll):" <<  std::endl;
  Eigen::Vector3d adjust_euler;
  adjust_euler(2) = atan2(adjust_rotation(2,1), adjust_rotation(2,2));
  adjust_euler(1) = -asin(adjust_rotation(2,0));
  adjust_euler(0) = atan2(adjust_rotation(1,0), adjust_rotation(0,0));
  outfile << RAD2DEG(adjust_euler[0]) << "," << RAD2DEG(adjust_euler[1]) << ","
          << RAD2DEG(adjust_euler[2]) << "," << 0 << "," << 0 << "," << 0
          << std::endl << std::endl;

  outfile << "Camera-IMU的小角度:" <<  std::endl;
  outfile << "delta Roll : " <<  vcs(2) << std::endl;
  outfile << "delta pitch: " <<  vcs(1) << std::endl;
  outfile << "delta yaw  : " <<  vcs(0) << std::endl;

  time_t t3 = clock();
  std::cout << "总校准时间:" << (double)(t3 - t1) / (CLOCKS_PER_SEC) << "s" << std::endl;
  outfile.close();

  return 0;
}

/**********************************************/
/*内参优化**************************************/
/**********************************************/
int  OptInner(vector<Point3f> &points_3D, vector<Point2f> &points_2D, 
              cv::Mat &camera_matrix_, cv::Mat &dist_coeffs_, 
              std::vector<VPnPData> &vpnp_list, cv::Size &imageSize)
{
  vector<vector<Point3f>> object_points_seq;
  vector<vector<Point2f>> image_points_seq;
  cv::Mat rvecsMat;                                                // 存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
  cv::Mat tvecsMat;  

  for (size_t i = 0; i < vpnp_list.size(); i++)
  {
    points_3D.push_back(Point3f(vpnp_list[i].x,vpnp_list[i].y,vpnp_list[i].z));
    points_2D.push_back(Point2f(vpnp_list[i].u,vpnp_list[i].v));
  }

  object_points_seq.push_back(points_3D);
  image_points_seq.push_back(points_2D);
  double err_first=cv::calibrateCamera(object_points_seq, image_points_seq, imageSize,camera_matrix_, dist_coeffs_, rvecsMat, tvecsMat,CALIB_USE_INTRINSIC_GUESS);
  Mat rvecsMat_cv;
  cv::Rodrigues(rvecsMat, rvecsMat_cv);
  std::cout<<"vector<Mat>:"<<camera_matrix_<< std::endl;
  std::cout<<"dist_coeffs_:"<<dist_coeffs_<< std::endl;
  std::cout<<"rvecsMat:"<< rvecsMat_cv << std::endl;
  std::cout<<"tvecsMat:"<< tvecsMat << std::endl;  
  std::cout << "重投影误差1：" << err_first << "像素" << endl << endl; 

  return 0;
}


/**********************************************/
/*计算外参**************************************/
/**********************************************/
void OptOuter(Calibration &calibra, vector<Point3f> &points_3D, vector<Point2f> &points_2D, 
              cv::Mat &camera_matrix_, cv::Mat &dist_coeffs_, 
              Eigen::Matrix3d &R, Eigen::Vector3d &T, 
              cv::Size &imageSize,
              cv::Mat &opt_img, cv::Mat &opt_img_show,
              std::ofstream &pnpfile)
{
  cv::Mat imageCalibration_show;
  cv::Mat map1, map2;

  Mat r, t;
  solvePnP(points_3D, points_2D, camera_matrix_, dist_coeffs_, r, t);

  Mat R_cv;
  cv::Rodrigues(r, R_cv);  //旋转向量转化为旋转矩阵

  cv::cv2eigen(R_cv, R);
  cout << "R=" << endl << R << endl;
  cv::cv2eigen(t, T);
  cout << "t=" << endl << T << endl;
  Vector6d calib_params_test;

  //外参优化
  Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
  calib_params_test[0] = euler[0];
  calib_params_test[1] = euler[1];
  calib_params_test[2] = euler[2];
  calib_params_test[3] =T[0];
  calib_params_test[4] = T[0];
  calib_params_test[5] = T[0];

  // 估计出的内参补偿
  cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
  cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, imageSize, 1, imageSize, 0),
  imageSize, CV_16SC2, map1, map2);
  cv::remap(calibra.image_, imageCalibration_pnp, map1, map2, cv::INTER_LINEAR);
  assert(imageCalibration_pnp.data);//如果数据为空就终止执行
  cv::resize(imageCalibration_pnp, imageCalibration_show, cv::Size(1920, 1080));


  //外参补偿并投影
  opt_img = calibra.getProjectionImg(calib_params_test,imageCalibration_pnp);
  cv::resize(opt_img,opt_img_show,cv::Size(1280,720));
  cv::imshow("Optimization result_pnp", opt_img_show);
  cv::imwrite(opt_pnp_file, opt_img);

  pnpfile <<" camera_matrix_: "<<camera_matrix_<< std::endl;
  pnpfile <<" dist_coeffs_: "<<dist_coeffs_<< std::endl;
  pnpfile <<" R: "<<R << std::endl;
  pnpfile <<" t:"<<T << std::endl;

  pnpfile.close();

}

/**********************************************/
/*计算外参**************************************/
/**********************************************/
int CameraIMU_ExtPara_Cal(LidarIMU_ExtPara &paraL, CameraIMU_ExtPara &paraC, Vector6d &calib_params, Eigen::Matrix3d &R, Eigen::Vector3d &vcs, Eigen::Matrix3d &Csc0)
{
  //Lidar和IMU之间的旋转矩阵
  Eigen::Matrix3d   Cbs, Cbc, C1, C2;
  Cbs = Eigen::AngleAxisd(paraL.yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(paraL.pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(paraL.roll, Eigen::Vector3d::UnitX());

  C1 =  Eigen::AngleAxisd(paraL.deltaYaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(paraL.deltaPitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(paraL.deltaRoll, Eigen::Vector3d::UnitX());

  //Camer和Lidar之间
  R =   Eigen::AngleAxisd(calib_params[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(calib_params[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(calib_params[2], Eigen::Vector3d::UnitX());

  //Camera和IMU
  Cbc = Eigen::AngleAxisd(paraC.yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(paraC.pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(paraC.roll, Eigen::Vector3d::UnitX());

  //Camera-Lisar初始矩阵
  Csc0 =  Cbc *  Cbs.transpose();    

  //计算Camera和IMU之间小角度
  C2 = C1 * Cbs.transpose() * R.transpose() * Cbc;
  // vcs = C2.eulerAngles(2, 1, 0);
  vcs(2) = atan2(C2(2,1), C2(2,2));
  vcs(1) = -asin(C2(2,0));
  vcs(0) = atan2(C2(1,0), C2(0,0));
  vcs = vcs * 180 / M_PI;

  return 0;
}


/**********************************************/
/*主函数入口************************************/
/*********************************************/
int main(int argc, char **argv) {
  ros::init(argc, argv, "lidarCamCalib");
  ros::NodeHandle nh;
  ros::Rate loop_rate(0.1);

  //Yaml参数读取
  Config_OutDoor      outConfig;
  LidarIMU_ExtPara    paraExtLidarIMU;
  CameraIMU_ExtPara   paraExtCameraIMU;
  Yaml_Para_Deal(nh, outConfig, paraExtLidarIMU, paraExtCameraIMU);

  //读取Opencv Camera内参
  Camera_OpencvPara_Deal(opencv_file);

  //读取Lidar-IMU外参
  Lidar_IMU_ExtPara_Deal(lidarIMUExtPara_file, paraExtLidarIMU);

  //构造函数：读取image和pcd
  // Calibration calibra(image_file, pcd_file, calib_config_file,use_ada_voxel);
  Calibration calibra(image_file, pcd_file,use_ada_voxel, outConfig);

  //相机内参
  cv::Mat camera_matrix_=cv::Mat::eye(3, 3, CV_64F);
  cv::Mat dist_coeffs_=cv::Mat::zeros(5, 1, CV_64F);
  Calibration_Para_Deal(calibra, camera_matrix_, dist_coeffs_);

  // 图像畸变补偿用于点云投影
  cv::Size imageSize;
  Image_PinHole_Distort(camera_matrix_, imageSize, dist_coeffs_, calibra.image_, imageCalibration);
	assert(imageCalibration.data);
  calibra.image_ = imageCalibration.clone();

  //iamge边缘提取
  Image_Edge_Deal(calibra);

  //点云边缘提取
  Scan_Edge_Deal(calibra);

  imageCalibration = calibra.image_.clone();

  //初始相对姿态的旋转矩阵
  Eigen::Vector3d init_euler_angle = calibra.init_rotation_matrix_.eulerAngles(2, 1, 0);

  //初始相对位移
  Eigen::Vector3d init_transation = calibra.init_translation_vector_;

  //初始相对位姿
  Vector6d calib_params;
  calib_params << init_euler_angle(0), init_euler_angle(1), init_euler_angle(2),
                  init_transation(0), init_transation(1), init_transation(2);

  //初次映射
  Eigen::Matrix3d R;
  Eigen::Vector3d T;
  Scan_Camera_Projection_First(calibra, imageCalibration, calib_params, R, T);


  //粗校准
  time_t t1 = clock();
  if (use_rough_calib) {
    //粗校准次数 源代码设置：50（0.1*50），
    //后来改成20（0.1*20）次对最终结果没影响。
    //每个姿态角修正循环50次，为了调试方便，现改成5    
    //VPnn 算法中dis_threshold设置为25,粗校准偏移量不修正，只修正姿态
    roughCalib(calibra, calib_params, DEG2RAD(0.1),50); 
  }
  time_t t2 = clock();
  std::cout << "粗校准时间:" << (double)(t2 - t1) / (CLOCKS_PER_SEC) << "s" << std::endl;

  //粗校准后的效果
  cv::Mat test_img = calibra.getProjectionImg(calib_params,imageCalibration);
  cv::Mat test_img_show;
  cv::resize(test_img,test_img_show,cv::Size(1280,720));
  cv::imshow("粗校准外参后", test_img_show);
  cv::imwrite(rough_file, test_img);
  cv::waitKey(1000);

  //精校准
  std::vector<VPnPData> vpnp_list;
  std::ofstream pnpfile(pnp_file); 
  bool opt_flag = true;
  cv::Mat opt_img = calibra.getProjectionImg(calib_params,imageCalibration);
  cv::Mat opt_img_show;
  OptCalib(calibra, calib_params, imageCalibration, vpnp_list, opt_flag, R, T);

  //结果转到IMU系
  Eigen::Vector3d vcs;
  Eigen::Matrix3d Csc0;
  CameraIMU_ExtPara_Cal(paraExtLidarIMU, paraExtCameraIMU, calib_params, R, vcs, Csc0);

  //精校准结果
  OptResult(calibra, calib_params, R, T, t1, opt_img, opt_img_show, vcs, Csc0);


  // //优化内参
  // vector<Point3f> points_3D;
  // vector<Point2f> points_2D;
  // OptInner(points_3D, points_2D, camera_matrix_, dist_coeffs_, vpnp_list, imageSize);


  // //计算外参
  //  OptOuter(calibra, points_3D, points_2D, camera_matrix_, dist_coeffs_, R, T, imageSize,opt_img, opt_img_show,pnpfile);

   while (ros::ok()) {
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      calibra.colorCloud(calib_params, 1, imageCalibration,
                        calibra.raw_lidar_cloud_, rgb_cloud);
      pcl::toROSMsg(*rgb_cloud, pub_cloud);
      pub_cloud.header.frame_id = "livox";
      calibra.rgb_cloud_pub_.publish(pub_cloud);
      sensor_msgs::ImagePtr img_msg =
          cv_bridge::CvImage(std_msgs::Header(), "bgr8", imageCalibration)
              .toImageMsg();
      calibra.image_pub_.publish(img_msg);
      std::cout << "push enter to publish again" << std::endl;
      getchar();
      /* code */
    }

  return 0;
}