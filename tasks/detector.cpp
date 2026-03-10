#include <numeric>
#include "detector.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <cstdio>   // 用于 sprintf
#include <chrono>   // 用于时间格式化

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
Detector::Detector(const std::string & config_path, bool debug)
: classifier_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  threshold_ = yaml["threshold"].as<double>();
  max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3;  // degree to rad
  min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
  max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
  min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
  min_armor_ratio_ = yaml["min_armor_ratio"].as<double>();
  max_armor_ratio_ = yaml["max_armor_ratio"].as<double>();
  max_side_ratio_ = yaml["max_side_ratio"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  max_rectangular_error_ = yaml["max_rectangular_error"].as<double>() / 57.3;  // degree to rad

  save_path_ = "patterns";
  std::filesystem::create_directory(save_path_);
}

std::list<Armor> Detector::detect(const cv::Mat & bgr_img, int frame_count)
{
  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  // 进行二值化
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
  if (debug_) cv::imshow("binary_img", binary_img);

  // 获取轮廓点
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // 获取灯条
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour);
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  // 将灯条从左到右排序
  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // 获取装甲板
  std::list<Armor> armors;
  for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
    for (auto right = std::next(left); right != lightbars.end(); right++) {
      if (left->color != right->color) continue;

      auto armor = Armor(*left, *right);
      if (!check_geometry(armor)) continue;

      armor.pattern = get_pattern(bgr_img, armor);
      classifier_.classify(armor);
      if (!check_name(armor)) continue;

      armor.type = get_type(armor);
      if (!check_type(armor)) continue;

      armor.center_norm = get_center_norm(bgr_img, armor.center);
      armors.emplace_back(armor);
    }
  }

  // 检查装甲板是否存在共用灯条的情况
  for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
    for (auto armor2 = std::next(armor1); armor2 != armors.end(); armor2++) {
      if (
        armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
        armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
        continue;
      }

      // 装甲板重叠，保留 roi 小的
      if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
        auto area1 = armor1->pattern.cols * armor1->pattern.rows;
        auto area2 = armor2->pattern.cols * armor2->pattern.rows;
        if (area1 < area2)
          armor2->duplicated = true;
        else
          armor1->duplicated = true;
      }

      // 装甲板相连，保留置信度大的
      if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
        if (armor1->confidence < armor2->confidence)
          armor1->duplicated = true;
        else
          armor2->duplicated = true;
      }
    }
  }

  armors.remove_if([&](const Armor & a) { return a.duplicated; });

  if (debug_) show_result(binary_img, bgr_img, lightbars, armors, frame_count);

  return armors;
}

bool Detector::detect(Armor & armor, const cv::Mat & bgr_img)
{
  // 取得四个角点
  auto tl = armor.points[0];
  auto tr = armor.points[1];
  auto br = armor.points[2];
  auto bl = armor.points[3];
  // 计算向量和调整后的点
  auto lt2b = bl - tl;
  auto rt2b = br - tr;
  auto tl1 = (tl + bl) / 2 - lt2b;
  auto bl1 = (tl + bl) / 2 + lt2b;
  auto br1 = (tr + br) / 2 + rt2b;
  auto tr1 = (tr + br) / 2 - rt2b;
  auto tl2tr = tr1 - tl1;
  auto bl2br = br1 - bl1;
  auto tl2 = (tl1 + tr) / 2 - 0.75 * tl2tr;
  auto tr2 = (tl1 + tr) / 2 + 0.75 * tl2tr;
  auto bl2 = (bl1 + br) / 2 - 0.75 * bl2br;
  auto br2 = (bl1 + br) / 2 + 0.75 * bl2br;
  // 构造新的四个角点
  std::vector<cv::Point> points = {tl2, tr2, br2, bl2};
  auto armor_rotaterect = cv::minAreaRect(points);
  cv::Rect boundingBox = armor_rotaterect.boundingRect();
  // 检查 boundingBox 是否超出图像边界
  if (
    boundingBox.x < 0 || boundingBox.y < 0 || boundingBox.x + boundingBox.width > bgr_img.cols ||
    boundingBox.y + boundingBox.height > bgr_img.rows) {
    return false;
  }

  // 在图像上裁剪出这个矩形区域（ROI）
  cv::Mat armor_roi = bgr_img(boundingBox);
  if (armor_roi.empty()) {
    return false;
  }

  // 彩色图转灰度图
  cv::Mat gray_img;
  cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);
  // 进行二值化
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
  // cv::imshow("binary_img", binary_img);
  // 获取轮廓点
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  // 获取灯条
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour);
    // lightbar_points_corrector(lightbar, gray_img); //关闭 PCA
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  if (lightbars.size() < 2) return false;

  // 将灯条从左到右排序
  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // 计算与 tl_roi, bl_roi 和 br_roi, tr_roi 距离最近的灯条
  Lightbar * closest_left_lightbar = nullptr;
  Lightbar * closest_right_lightbar = nullptr;
  float min_distance_tl_bl = std::numeric_limits<float>::max();
  float min_distance_br_tr = std::numeric_limits<float>::max();
  for (auto & lightbar : lightbars) {
    float distance_tl_bl =
      cv::norm(tl - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(bl - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y)));
    if (distance_tl_bl < min_distance_tl_bl) {
      min_distance_tl_bl = distance_tl_bl;
      closest_left_lightbar = &lightbar;
    }
    float distance_br_tr =
      cv::norm(br - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(tr - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y)));
    if (distance_br_tr < min_distance_br_tr) {
      min_distance_br_tr = distance_br_tr;
      closest_right_lightbar = &lightbar;
    }
  }

  // tools::logger()->debug(
  // "min_distance_br_tr + min_distance_tl_bl is {}", min_distance_br_tr + min_distance_tl_bl);
  // std::vector<cv::Point2f> points2f{
  //   closest_left_lightbar->top, closest_left_lightbar->bottom, closest_right_lightbar->bottom,
  //   closest_right_lightbar->top};
  // tools::draw_points(armor_roi, points2f, {0, 0, 255}, 2);
  // cv::imshow("armor_roi", armor_roi);

  if (
    closest_left_lightbar && closest_right_lightbar &&
    min_distance_br_tr + min_distance_tl_bl < 15) {
    // 将四个点从 armor_roi 坐标系转换到原始图像坐标系
    armor.points[0] = closest_left_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[1] = closest_right_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[2] = closest_right_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[3] = closest_left_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    return true;
  }

  return false;
}

bool Detector::check_geometry(const Lightbar & lightbar) const
{
  auto angle_ok = lightbar.angle_error < max_angle_error_;
  auto ratio_ok = lightbar.ratio > min_lightbar_ratio_ && lightbar.ratio < max_lightbar_ratio_;
  auto length_ok = lightbar.length > min_lightbar_length_;
  return angle_ok && ratio_ok && length_ok;
}

bool Detector::check_geometry(const Armor & armor) const
{
  auto ratio_ok = armor.ratio > min_armor_ratio_ && armor.ratio < max_armor_ratio_;
  auto side_ratio_ok = armor.side_ratio < max_side_ratio_;
  auto rectangular_error_ok = armor.rectangular_error < max_rectangular_error_;
  return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

bool Detector::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于分类器的迭代
  if (name_ok && !confidence_ok) save(armor);

  // 出现 5 号 则显示 debug 信息。但不过滤。
  if (armor.name == ArmorName::five) tools::logger()->debug("See pattern 5");

  return name_ok && confidence_ok;
}

bool Detector::check_type(const Armor & armor) const
{
  auto name_ok = armor.type == ArmorType::small
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name == ArmorName::one || armor.name == ArmorName::base);

  // 保存异常的图案，用于分类器的迭代
  if (!name_ok) {
    // 替换 fmt::format -> 字符串拼接
    tools::logger()->debug(
      "see strange armor: " + ARMOR_TYPES[armor.type] + " " + ARMOR_NAMES[armor.name]);
    save(armor);
  }

  return name_ok;
}

Color Detector::get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const {
    int red_sum = 0, blue_sum = 0, green_sum = 0;
    int count = 0;

    for (const auto & point : contour) {
        cv::Vec3b pixel = bgr_img.at<cv::Vec3b>(point);
        blue_sum += pixel[0];
        green_sum += pixel[1]; // 注意：BGR 格式，绿色在中间
        red_sum += pixel[2];
        count++;
    }

    // 计算平均值
    float r = red_sum / count;
    float g = green_sum / count;
    float b = blue_sum / count;

    // 【核心逻辑】判断颜色纯度
    // 如果是白光，R≈G≈B。如果是蓝光，B >> R 且 B >> G。
    
    if (b > r * 1.5 && b > g * 1.5) { 
        return Color::blue; 
    }
    if (r > b * 1.5 && r > g * 1.5) { 
        return Color::red; 
    }

    // 如果都不满足，说明是白光/黄光（天花板灯），直接返回“非装甲板颜色”
    return Color::none; // 你需要在 enum 里加一个 none，或者在调用处判断
}

cv::Mat Detector::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  // 延长灯条获得装甲板角点
  // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
  auto tl = armor.left.center - armor.left.top2bottom * 1.125;
  auto bl = armor.left.center + armor.left.top2bottom * 1.125;
  auto tr = armor.right.center - armor.right.top2bottom * 1.125;
  auto br = armor.right.center + armor.right.top2bottom * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  return bgr_img(roi);
}

ArmorType Detector::get_type(const Armor & armor)
{
  /// 优先根据当前 armor.ratio 判断
  /// TODO: 25 赛季是否还需要根据比例判断大小装甲？能否根据图案直接判断？

  if (armor.ratio > 3.0) {
    // tools::logger()->debug(
    //   "[Detector] get armor type by ratio: BIG {} {:.2f}", ARMOR_NAMES[armor.name], armor.ratio);
    return ArmorType::big;
  }

  if (armor.ratio < 2.5) {
    // tools::logger()->debug(
    //   "[Detector] get armor type by ratio: SMALL {} {:.2f}", ARMOR_NAMES[armor.name], armor.ratio);
    return ArmorType::small;
  }

  // tools::logger()->debug("[Detector] get armor type by name: {}", ARMOR_NAMES[armor.name]);

  // 英雄、基地只能是大装甲板
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
    return ArmorType::big;
  }

  // 其他所有（工程、哨兵、前哨站、步兵）都是小装甲板
  /// TODO: 基地顶装甲是小装甲板
  return ArmorType::small;
}

cv::Point2f Detector::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void Detector::save(const Armor & armor) const
{
  // 使用 C 风格时间格式化
  auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  char buffer[30];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
  auto file_name = std::string(buffer);
  
  auto img_path = save_path_ + "/" + ARMOR_NAMES[armor.name] + "_" + file_name + ".jpg";
  cv::imwrite(img_path, armor.pattern);
}

void Detector::show_result(
  const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
  const std::list<Armor> & armors, int frame_count) const
{
  auto detection = bgr_img.clone();
  
  // 替换 fmt::format -> std::to_string
  tools::draw_text(detection, "[" + std::to_string(frame_count) + "]", cv::Point(10, 30), cv::Scalar(255, 255, 255));

  for (const auto & lightbar : lightbars) {
    // 替换 fmt::format -> sprintf (用于浮点数格式化)
    char info_buf[100];
    sprintf(info_buf, "%.1f %.1f %.1f %d", 
            lightbar.angle_error * 57.3, lightbar.ratio, lightbar.length, (int)lightbar.color);
    std::string info(info_buf);
    
    tools::draw_text(detection, info, lightbar.top, cv::Scalar(0, 255, 255));
    tools::draw_points(detection, lightbar.points, cv::Scalar(0, 255, 255), 3);
  }

    for (const auto & armor : armors) {
    // --- 1. 准备格式化文字 ---
    char info_buf[150];
    sprintf(info_buf, "%s (%.2f)", 
            ARMOR_NAMES[armor.name].c_str(), armor.confidence);
    std::string info(info_buf);

    // --- 2. 提取四个角点 (左上，右上，右下，左下) ---
    // 注意：armor.points 的顺序通常是 0:TL, 1:TR, 2:BR, 3:BL
    std::vector<cv::Point> pts = {
        armor.points[0], 
        armor.points[1], 
        armor.points[2], 
        armor.points[3]
    };

    // --- 3. 画闭合的绿色大框 (核心修改！) ---
    // true 表示闭合多边形，线宽为 2
    cv::polylines(detection, pts, true, cv::Scalar(0, 255, 0), 2);

    // --- 4. (可选) 填充半透明绿色，让框更醒目 ---
    // 创建蒙版
    std::vector<std::vector<cv::Point>> poly = {pts};
    cv::Mat mask = cv::Mat::zeros(detection.size(), CV_8UC1);
    cv::fillPoly(mask, poly, cv::Scalar(255));
    
    // 融合颜色 (alpha=0.2，轻微绿色填充)
    cv::Scalar color = cv::Scalar(0, 255, 0);
    for (int y = 0; y < detection.rows; ++y) {
        for (int x = 0; x < detection.cols; ++x) {
            if (mask.at<uchar>(y, x)) {
                cv::Vec3b &pixel = detection.at<cv::Vec3b>(y, x);
                pixel[0] = pixel[0] * 0.8 + color[0] * 0.2; // B
                pixel[1] = pixel[1] * 0.8 + color[1] * 0.2; // G
                pixel[2] = pixel[2] * 0.8 + color[2] * 0.2; // R
            }
        }
    }
    // 注意：上面的像素级循环比较慢，如果卡顿，可以注释掉填充部分，只保留 polylines

    // --- 5. 在框的上方写文字 ---
    // 计算文字位置：取左上和右上的中点，再往上移 10 像素
    cv::Point text_pos = (armor.points[0] + armor.points[1]) / 2;
    text_pos.y -= 10;

    // 画黑色背景底框
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(info, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    cv::Point tl_bg = cv::Point(text_pos.x - textSize.width/2 - 2, text_pos.y - textSize.height - 2);
    cv::Point br_bg = cv::Point(text_pos.x + textSize.width/2 + 2, text_pos.y + baseline + 2);
    cv::rectangle(detection, tl_bg, br_bg, cv::Scalar(0, 0, 0), -1); // 黑色填充

    // 画白色文字
    cv::putText(detection, info, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }

  cv::Mat binary_img2;
  cv::resize(binary_img, binary_img2, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::resize(detection, detection, {}, 0.5, 0.5);     // 显示时缩小图片尺寸

  // cv::imshow("threshold", binary_img2);
  if (debug_) {
      cv::imshow("detection", detection);
      cv::waitKey(1);
  }
}

void Detector::lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const
{
  // 配置参数
  constexpr float MAX_BRIGHTNESS = 25;  // 归一化最大亮度值
  constexpr float ROI_SCALE = 0.07;     // ROI 扩展比例
  constexpr float SEARCH_START = 0.4;   // 搜索起始位置比例（原 0.8/2）
  constexpr float SEARCH_END = 0.6;     // 搜索结束位置比例（原 1.2/2）

  // 扩展并裁剪 ROI
  cv::Rect roi_box = lightbar.rotated_rect.boundingRect();
  roi_box.x -= roi_box.width * ROI_SCALE;
  roi_box.y -= roi_box.height * ROI_SCALE;
  roi_box.width += 2 * roi_box.width * ROI_SCALE;
  roi_box.height += 2 * roi_box.height * ROI_SCALE;

  // 边界约束
  roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);

  // 归一化 ROI
  cv::Mat roi = gray_img(roi_box);
  const float mean_val = cv::mean(roi)[0];
  roi.convertTo(roi, CV_32F);
  cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

  // 计算质心
  const cv::Moments moments = cv::moments(roi);
  const cv::Point2f centroid(
    moments.m10 / moments.m00 + roi_box.x, moments.m01 / moments.m00 + roi_box.y);

  // 生成稀疏点云（优化性能）
  std::vector<cv::Point2f> points;
  for (int i = 0; i < roi.rows; ++i) {
    for (int j = 0; j < roi.cols; ++j) {
      const float weight = roi.at<float>(i, j);
      if (weight > 1e-3) {          // 忽略极小值提升性能
        points.emplace_back(j, i);  // 坐标相对于 ROI 区域
      }
    }
  }

  // PCA 计算对称轴方向
  if (points.empty()) return; // 防止空点云导致 PCA 崩溃
  
  cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
  axis /= cv::norm(axis);
  if (axis.y > 0) axis = -axis;  // 统一方向

  const auto find_corner = [&](int direction) -> cv::Point2f {
    const float dx = axis.x * direction;
    const float dy = axis.y * direction;
    const float search_length = lightbar.length * (SEARCH_END - SEARCH_START);

    std::vector<cv::Point2f> candidates;

    // 横向采样多个候选线
    const int half_width = (lightbar.width - 2) / 2;
    for (int i_offset = -half_width; i_offset <= half_width; ++i_offset) {
      // 计算搜索起点
      cv::Point2f start_point(
        centroid.x + lightbar.length * SEARCH_START * dx + i_offset,
        centroid.y + lightbar.length * SEARCH_START * dy);

      // 沿轴搜索亮度跳变点
      cv::Point2f corner = start_point;
      float max_diff = 0;
      bool found = false;

      for (float step = 0; step < search_length; ++step) {
        const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);

        // 边界检查
        if (
          cur_point.x < 0 || cur_point.x >= gray_img.cols || cur_point.y < 0 ||
          cur_point.y >= gray_img.rows) {
          break;
        }

        // 计算亮度差（使用双线性插值提升精度）
        const auto prev_val = gray_img.at<uchar>(cv::Point2i(cur_point - cv::Point2f(dx, dy)));
        const auto cur_val = gray_img.at<uchar>(cv::Point2i(cur_point));
        const float diff = prev_val - cur_val;

        if (diff > max_diff && prev_val > mean_val) {
          max_diff = diff;
          corner = cur_point - cv::Point2f(dx, dy);  // 跳变发生在上一位置
          found = true;
        }
      }

      if (found) {
        candidates.push_back(corner);
      }
    }

    // 返回候选点均值
    return candidates.empty()
             ? cv::Point2f(-1, -1)
             : std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0)) /
                 static_cast<float>(candidates.size());
  };

  // 并行检测顶部和底部
  lightbar.top = find_corner(1);
  lightbar.bottom = find_corner(-1);
}

}  // namespace auto_aim