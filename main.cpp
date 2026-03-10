#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// 引入你的检测器头文件（确保路径正确）
#include "detector.hpp"

int main() {
    // ========== 1. 初始化参数 ==========
    // 视频路径
    std::string video_path = "/home/a/Desktop/try/demo/demo.avi";
    // 配置文件路径（关键！需包含所有参数的配置文件，如yaml/json，你需自行创建）
    std::string config_path = "/home/a/Desktop/try/config.yaml";
    // 调试模式开启（会自动显示检测结果窗口）
    bool debug = true;

    // ========== 2. 初始化检测器（加载配置参数） ==========
    auto_aim::Detector detector(config_path, debug);

    // ========== 3. 打开视频 ==========
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开视频文件！路径：" << video_path << std::endl;
        return -1;
    }

    // ========== 4. 逐帧检测装甲板 ==========
    cv::Mat frame;
    int frame_count = 0; // 帧计数（用于调试显示/保存）

    while (cap.read(frame)) {
        frame_count++;
        // 检查帧是否为空
        if (frame.empty()) {
            std::cout << "视频读取完毕或帧为空！" << std::endl;
            break;
        }

        // ========== 核心：调用检测器检测装甲板 ==========
        // frame 是彩色图（BGR格式），对应头文件的 bgr_img
        std::list<auto_aim::Armor> armors = detector.detect(frame, frame_count);

        // 打印检测结果（调试用）
        std::cout << "第 " << frame_count << " 帧：检测到 " << armors.size() << " 个装甲板" << std::endl;

        // ========== 退出逻辑（ESC键） ==========
        if (cv::waitKey(30) == 27) { // 30ms控制播放速度，ESC键退出
            std::cout << "用户按下ESC键，退出视频播放！" << std::endl;
            break;
        }
    }

    // ========== 5. 释放资源 ==========
    cap.release();
    cv::destroyAllWindows();

    return 0;
}