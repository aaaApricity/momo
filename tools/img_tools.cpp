#include "tools/img_tools.hpp"

namespace tools {
void draw_text(
    cv::Mat& img, const std::string& text, const cv::Point& pos,
    const cv::Scalar& color, double font_scale, int thickness) {
    cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

void draw_points(
    cv::Mat& img, const std::vector<cv::Point2f>& points,
    const cv::Scalar& color, int thickness) {
    for(const auto& p : points) {
        cv::circle(img, p, thickness, color, -1);
    }
}
}