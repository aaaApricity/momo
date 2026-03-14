#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <list>
#include <filesystem>
#include <sstream>  // 新增：用于字符串拼接

// 引入头文件
#include "yolov5.hpp"
#include "tasks/classifier.hpp"
#include "tools/logger.hpp"
#include "detector.hpp"

// 装甲板信息转字符串
std::string getArmorInfo(const auto_aim::Armor& armor) {
    std::string info = 
        "颜色：" + auto_aim::COLORS[armor.color] + " | " +
        "类型：" + auto_aim::ARMOR_TYPES[static_cast<int>(armor.type)] + " | " +
        "名称：" + auto_aim::ARMOR_NAMES[static_cast<int>(armor.name)] + " | " +
        "置信度：" + std::to_string(armor.confidence).substr(0, 4);
    return info;
}

// ========== Detector风格画框函数（移除fmt依赖） ==========
void drawLikeDetector(cv::Mat& frame, const std::list<auto_aim::Armor>& armors) {
    for (const auto& armor : armors) {
        // 1. 绘制装甲板四点框（绿色，和Detector一致）
        std::vector<cv::Point> pts;
        for (const auto& p : armor.points) {
            pts.emplace_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }
        cv::polylines(frame, pts, true, cv::Scalar(0, 255, 0), 2);

        // 2. 拼接检测信息（改用原生C++，替代fmt::format）
        std::ostringstream info_stream;
        // 保留两位小数的置信度 + 装甲板名称 + 类型
        info_stream.precision(2);  // 设置小数精度
        info_stream << std::fixed << armor.confidence << " " 
                    << auto_aim::ARMOR_NAMES[armor.name] << " " 
                    << auto_aim::ARMOR_TYPES[armor.type];
        std::string info = info_stream.str();

        // 3. 绘制文字（和Detector样式一致：黄色文字、小字体）
        cv::putText(
            frame,
            info,
            cv::Point(static_cast<int>(armor.points[0].x), static_cast<int>(armor.points[0].y) - 10),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,                // 字体大小（和Detector一致）
            cv::Scalar(0, 255, 255),  // 黄色文字（和Detector一致）
            1                   // 文字线宽
        );
    }
}

int main(int argc, char** argv) {
    // ========== 配置参数 ==========
    const std::string CONFIG_PATH = "/home/a/Desktop/try/config.yaml";
    const std::string VIDEO_PATH = "/home/a/Desktop/try/demo/demo.avi";

    // ========== 初始化YOLO检测器（关闭debug，避免多窗口） ==========
    auto_aim::YOLOV5 yolo_detector(CONFIG_PATH, false);
    auto_aim::Classifier classifier(CONFIG_PATH);

    // ========== 打开视频 ==========
    cv::VideoCapture cap;
    if (std::filesystem::exists(VIDEO_PATH)) {
        cap.open(VIDEO_PATH);
        std::cout << "[INFO] 打开视频：" << VIDEO_PATH << std::endl;
    } else {
        cap.open(0);
        std::cout << "[INFO] 打开摄像头..." << std::endl;
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] 无法打开视频/摄像头！" << std::endl;
        return -1;
    }

    // ========== 初始化窗口 ==========
    const std::string WINDOW_NAME = "YOLO + Detector 画框";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 1280, 720);

    // ========== 主循环 ==========
    cv::Mat frame;
    int frame_count = 0;
    bool is_paused = false;

    std::cout << "[INFO] 按ESC退出，按P暂停/继续" << std::endl;
    while (true) {
        if (!is_paused) {
            if (!cap.read(frame)) break;  // 读取视频帧
            frame_count++;

            // 1. YOLO检测装甲板
            std::list<auto_aim::Armor> armors = yolo_detector.detect(frame, frame_count);

            // 2. 用Detector风格绘制装甲板框（核心：只画框，不改原图）
            drawLikeDetector(frame, armors);

            // 3. 打印检测结果（可选）
            std::cout << "\n[FRAME " << frame_count << "] 检测到：" << armors.size() << " 个装甲板" << std::endl;
            int idx = 0;
            for (const auto& armor : armors) {
                idx++;
                std::cout << "  装甲板" << idx << "：" << getArmorInfo(armor) << std::endl;
            }
        }

        // 显示画面（唯一窗口，无冲突）
        if (!frame.empty()) {
            cv::imshow(WINDOW_NAME, frame);
        }

        // 按键控制（唯一waitKey，确保画面刷新）
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) {          // ESC退出
            break;
        } else if (key == 'p' || key == 'P') {  // P暂停/继续
            is_paused = !is_paused;
        }
    }

    // ========== 释放资源 ==========
    cap.release();
    cv::destroyAllWindows();
    std::cout << "[INFO] 程序退出" << std::endl;
    return 0;
}