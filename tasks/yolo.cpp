#include "yolo.hpp"
#include <iostream>
#include "yolos/yolov5.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
{
  // 完全不读配置文件，直接创建YOLOV5实例
  yolo_ = std::make_unique<YOLOV5>("", debug);  // 配置路径传空字符串
  std::cout << "[INFO] 跳过YAML，直接初始化YOLOV5" << std::endl;
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim