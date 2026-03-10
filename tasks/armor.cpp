#include "armor.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace auto_aim
{
Lightbar::Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id)
  // 按Lightbar结构体声明顺序初始化（消除潜在reorder警告）
  : id(id),
    color(none),
    center(rotated_rect.center),
    top(cv::Point2f(0, 0)),
    bottom(cv::Point2f(0, 0)),
    top2bottom(cv::Point2f(0, 0)),
    points(std::vector<cv::Point2f>()),
    angle(0.0),
    angle_error(0.0),
    length(0.0),
    width(0.0),
    ratio(0.0),
    rotated_rect(rotated_rect)
{
  std::vector<cv::Point2f> corners(4);
  rotated_rect.points(&corners[0]);
  std::sort(corners.begin(), corners.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  center = rotated_rect.center;
  top = (corners[0] + corners[1]) / 2;
  bottom = (corners[2] + corners[3]) / 2;
  top2bottom = bottom - top;

  points.emplace_back(top);
  points.emplace_back(bottom);

  width = cv::norm(corners[0] - corners[1]);
  angle = std::atan2(top2bottom.y, top2bottom.x);
  angle_error = std::abs(angle - CV_PI / 2);
  length = cv::norm(top2bottom);
  ratio = length / width;
}

// 传统构造函数（基于左右灯条）
Armor::Armor(const Lightbar & left, const Lightbar & right)
  // 严格按Armor结构体声明顺序初始化所有成员
  : color(none),
    left(left),
    right(right),
    center(cv::Point2f(0, 0)),
    center_norm(cv::Point2f(0, 0)),
    points(std::vector<cv::Point2f>()),
    ratio(0.0),
    side_ratio(0.0),
    rectangular_error(0.0),
    type(small),
    name(not_armor),
    priority(first),
    class_id(-1),
    box(cv::Rect()),
    pattern(cv::Mat()),
    confidence(0.0),
    duplicated(false),
    xyz_in_gimbal(Eigen::Vector3d::Zero()),
    xyz_in_world(Eigen::Vector3d::Zero()),
    ypr_in_gimbal(Eigen::Vector3d::Zero()),
    ypr_in_world(Eigen::Vector3d::Zero()),
    ypd_in_world(Eigen::Vector3d::Zero()),
    yaw_raw(0.0)
{
  color = left.color;
  center = (left.center + right.center) / 2;

  points.emplace_back(left.top);
  points.emplace_back(right.top);
  points.emplace_back(right.bottom);
  points.emplace_back(left.bottom);

  auto left2right = right.center - left.center;
  auto width = cv::norm(left2right);
  auto max_lightbar_length = std::max(left.length, right.length);
  auto min_lightbar_length = std::min(left.length, right.length);
  ratio = width / max_lightbar_length;
  side_ratio = max_lightbar_length / min_lightbar_length;

  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(left.angle - roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
}

// 神经网络构造函数
Armor::Armor(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints)
  // 按Armor结构体声明顺序初始化
  : color(none),
    left(Lightbar()),
    right(Lightbar()),
    center(cv::Point2f(0, 0)),
    center_norm(cv::Point2f(0, 0)),
    points(armor_keypoints),  // points声明在box前，先初始化
    ratio(0.0),
    side_ratio(0.0),
    rectangular_error(0.0),
    type(small),
    name(not_armor),
    priority(first),
    class_id(class_id),      // class_id声明在box前
    box(box),                // box声明在confidence前
    pattern(cv::Mat()),
    confidence(confidence),  // confidence声明在最后
    duplicated(false),
    xyz_in_gimbal(Eigen::Vector3d::Zero()),
    xyz_in_world(Eigen::Vector3d::Zero()),
    ypr_in_gimbal(Eigen::Vector3d::Zero()),
    ypr_in_world(Eigen::Vector3d::Zero()),
    ypd_in_world(Eigen::Vector3d::Zero()),
    yaw_raw(0.0)
{
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  ratio = max_length / max_width;

  // 修复-Wsign-compare：将class_id转为size_t再比较
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < armor_properties.size()) {
    auto [color, name, type] = armor_properties[static_cast<std::size_t>(class_id)];
    this->color = color;
    this->name = name;
    this->type = type;
  } else {
    this->color = blue;      // Default
    this->name = not_armor;  // Default
    this->type = small;      // Default
  }
}

// 神经网络ROI构造函数
Armor::Armor(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
  cv::Point2f offset)
  // 按声明顺序初始化
  : color(none),
    left(Lightbar()),
    right(Lightbar()),
    center(cv::Point2f(0, 0)),
    center_norm(offset),  // offset赋值给center_norm
    points(armor_keypoints),
    ratio(0.0),
    side_ratio(0.0),
    rectangular_error(0.0),
    type(small),
    name(not_armor),
    priority(first),
    class_id(class_id),
    box(box),
    pattern(cv::Mat()),
    confidence(confidence),
    duplicated(false),
    xyz_in_gimbal(Eigen::Vector3d::Zero()),
    xyz_in_world(Eigen::Vector3d::Zero()),
    ypr_in_gimbal(Eigen::Vector3d::Zero()),
    ypr_in_world(Eigen::Vector3d::Zero()),
    ypd_in_world(Eigen::Vector3d::Zero()),
    yaw_raw(0.0)
{
  std::transform(
    armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  std::transform(
    points.begin(), points.end(), points.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  ratio = max_length / max_width;

  // 修复-Wsign-compare
  if (class_id >= 0 && static_cast<std::size_t>(class_id) < armor_properties.size()) {
    auto [color, name, type] = armor_properties[static_cast<std::size_t>(class_id)];
    this->color = color;
    this->name = name;
    this->type = type;
  } else {
    this->color = blue;      // Default
    this->name = not_armor;  // Default
    this->type = small;      // Default
  }
}

// YOLOV5构造函数
Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints)
  // 按声明顺序初始化
  : color(none),
    left(Lightbar()),
    right(Lightbar()),
    center(cv::Point2f(0, 0)),
    center_norm(cv::Point2f(0, 0)),
    points(armor_keypoints),
    ratio(0.0),
    side_ratio(0.0),
    rectangular_error(0.0),
    type(small),
    name(not_armor),
    priority(first),
    class_id(-1),
    box(box),
    pattern(cv::Mat()),
    confidence(confidence),
    duplicated(false),
    xyz_in_gimbal(Eigen::Vector3d::Zero()),
    xyz_in_world(Eigen::Vector3d::Zero()),
    ypr_in_gimbal(Eigen::Vector3d::Zero()),
    ypr_in_world(Eigen::Vector3d::Zero()),
    ypd_in_world(Eigen::Vector3d::Zero()),
    yaw_raw(0.0)
{
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  ratio = max_length / max_width;
  color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
  name = num_id == 0  ? ArmorName::sentry
         : num_id > 5 ? ArmorName(num_id)
                      : ArmorName(num_id - 1);
  type = num_id == 1 ? ArmorType::big : ArmorType::small;
}

// YOLOV5+ROI构造函数
Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset)
  // 按声明顺序初始化
  : color(none),
    left(Lightbar()),
    right(Lightbar()),
    center(cv::Point2f(0, 0)),
    center_norm(offset),
    points(armor_keypoints),
    ratio(0.0),
    side_ratio(0.0),
    rectangular_error(0.0),
    type(small),
    name(not_armor),
    priority(first),
    class_id(-1),
    box(box),
    pattern(cv::Mat()),
    confidence(confidence),
    duplicated(false),
    xyz_in_gimbal(Eigen::Vector3d::Zero()),
    xyz_in_world(Eigen::Vector3d::Zero()),
    ypr_in_gimbal(Eigen::Vector3d::Zero()),
    ypr_in_world(Eigen::Vector3d::Zero()),
    ypd_in_world(Eigen::Vector3d::Zero()),
    yaw_raw(0.0)
{
  std::transform(
    armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  std::transform(
    points.begin(), points.end(), points.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  ratio = max_length / max_width;
  color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
  name = num_id == 0 ? ArmorName::sentry : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
  type = num_id == 1 ? ArmorType::big : ArmorType::small;
}

}  // namespace auto_aim