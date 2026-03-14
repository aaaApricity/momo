#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <iostream>
#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
// 构造函数：完全硬编码配置，移除 Detector 初始化
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug)  // 【修改1】删掉 detector_("", false)，彻底不初始化 Detector
{
  try {
    // ========== 1. 硬编码所有配置参数（替代YAML读取） ==========
    model_path_ = "yolov5.xml";          // 模型文件路径（确保该文件存在）
    device_ = "CPU";                     // 推理设备
    binary_threshold_ = 150.0;           // 传统方法二值化阈值（仅保留，无实际作用）
    min_confidence_ = 0.8;               // 置信度过滤阈值
    
    // ROI参数（硬编码，use_roi=false时不生效）
    int x = 420, y = 50, width = 600, height = 600;
    use_roi_ = false;                    // 不启用ROI裁剪
    use_traditional_ = false;            // 【修改2】关闭传统方法，不再调用 Detector
    
    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);

    // ========== 2. 创建保存目录 ==========
    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);
    std::cout << "[INFO] YOLOV5配置硬编码成功！" << std::endl;

    // ========== 3. 模型加载（检查文件存在性） ==========
    if (!std::filesystem::exists(model_path_)) {
      std::cerr << "[FATAL] 模型文件不存在：" << model_path_ << std::endl;
      throw std::runtime_error("Model file not found: " + model_path_);
    }

    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, 640, 640, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale(255.0);

    model = ppp.build();
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    std::cout << "[INFO] YOLOV5模型加载成功！" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "[FATAL] YOLOV5初始化失败：" << e.what() << std::endl;
    throw;
  }
}

// 检测函数：处理单帧图像，返回装甲板列表
std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    // logger为空时兜底输出
    if (tools::logger()) {
      tools::logger()->warn("Empty img!, camera drop!");
    } else {
      std::cerr << "[WARNING] Empty img!, camera drop!" << std::endl;
    }
    return std::list<Armor>();
  }

  // 处理ROI裁剪（use_roi=false时直接用原图）
  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // 图像缩放至模型输入尺寸（640x640）
  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // 构建模型输入张量
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // 模型推理
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 获取推理输出并后处理
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

// 解析推理输出，提取装甲板信息
std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  // 遍历推理输出的每一行（每个检测框）
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    // 过滤低置信度结果
    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    // 解析颜色和类别得分
    cv::Mat color_scores = output.row(r).colRange(9, 13);
    cv::Mat classes_scores = output.row(r).colRange(13, 22);
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    // 解析装甲板角点坐标（反缩放至原图尺寸）
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    // 计算装甲板包围框
    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (size_t i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    // 保存检测结果
    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  // 非极大值抑制（NMS）去重
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // 过滤无效装甲板（移除 Detector 调用）
  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 【修改3】注释掉 Detector 调用，彻底移除依赖
    // if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  // 调试模式下绘制检测结果
  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

// 检查装甲板名称有效性
bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

// 检查装甲板类型有效性
bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);
  return name_ok;
}

// 归一化装甲板中心坐标
cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

// 绘制检测结果（调试模式）
void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  
  // 绘制帧号（logger为空时跳过）
  if (tools::logger()) {
    tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  }

  // 绘制每个装甲板的信息
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    
    tools::draw_points(detection, armor.points, {0, 255, 0});
    if (tools::logger()) {
      tools::draw_text(detection, info, armor.center, {0, 255, 0});
    }
  }

  // 绘制ROI框（启用时）
  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }

  // 缩小显示窗口并展示
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

// 保存可疑装甲板图像（用于迭代优化）
void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, ARMOR_NAMES[armor.name], file_name);
  cv::imwrite(img_path, tmp_img_);
}

// Sigmoid激活函数（计算置信度）
double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(-x));
}

// 后处理函数（复用parse逻辑）
std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim