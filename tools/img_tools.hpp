#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace tools {
void draw_text(
    cv::Mat& img, const std::string& text, const cv::Point& pos,
    const cv::Scalar& color, double font_scale = 0.5, int thickness = 1);

void draw_points(
    cv::Mat& img, const std::vector<cv::Point2f>& points,
    const cv::Scalar& color, int thickness = 3);
}