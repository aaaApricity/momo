#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <list>

// 引入检测器和假分类器头文件
#include "detector.hpp"
#include "classifier.hpp"

// 装甲板名称枚举转字符串（适配你的假分类器映射规则）
std::string armorNameToString(auto_aim::ArmorName name) {
    switch (name) {
        case auto_aim::ArmorName::one:    return "1号";
        case auto_aim::ArmorName::two:    return "2号";
        case auto_aim::ArmorName::three:  return "3号";
        case auto_aim::ArmorName::four:   return "4号";
        case auto_aim::ArmorName::five:   return "5号";
        default:                          return "未知";
    }
}

// 绘制装甲板识别框、角点、分类结果的函数
void drawArmor(cv::Mat& frame, const auto_aim::Armor& armor) {
    // 1. 绘制装甲板红色矩形框（基于armor.box）
    cv::rectangle(frame, armor.box, cv::Scalar(0, 0, 255), 2);
    
    // 2. 绘制装甲板绿色角点（基于armor.points）
    for (const auto& point : armor.points) {
        cv::circle(frame, point, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // 3. 绘制分类结果文字（白色+黑色描边，提升可读性）
    std::string text = armorNameToString(armor.name) + " (置信度:" + std::to_string(armor.confidence).substr(0,4) + ")";
    cv::Point textPos(armor.box.x, armor.box.y - 10); // 文字在框上方
    cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2); // 黑色描边
    cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1); // 白色填充
}

int main() {
    // ========== 1. 初始化参数 ==========
    std::string video_path = "/home/a/Desktop/try/demo/demo.avi";  // 替换为你的视频路径
    std::string config_path = "/home/a/Desktop/try/config.yaml";  // 替换为你的配置文件路径
    bool debug = true;

    // ========== 2. 初始化检测器和假分类器 ==========
    auto_aim::Detector detector(config_path, debug);
    auto_aim::Classifier classifier(config_path);  // 假分类器仅需传入配置路径（实际未使用）

    // ========== 3. 打开视频 ==========
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] 无法打开视频文件！路径：" << video_path << std::endl;
        return -1;
    }

    // ========== 4. 暂停控制+绘制核心变量 ==========
    cv::Mat frame, draw_frame;  // draw_frame：独立绘制帧，避免原帧被覆盖
    int frame_count = 0;
    bool is_paused = false;
    int key = 0;

    // 创建并置顶窗口（确保按键监听生效）
    cv::namedWindow("Armor Detection", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Armor Detection", cv::WND_PROP_TOPMOST, 1);
    cv::resizeWindow("Armor Detection", 1280, 720);  // 固定窗口大小

    // ========== 5. 逐帧处理主循环 ==========
    while (true) {
        // 未暂停时读取并处理视频帧
        if (!is_paused) {
            if (!cap.read(frame)) {  // 视频读取完毕
                std::cout << "[INFO] 视频播放完毕！" << std::endl;
                break;
            }
            frame_count++;

            // 关键：复制原帧到独立绘制帧（避免检测器修改原帧导致绘制失效）
            draw_frame = frame.clone();

            // ========== 检测装甲板 ==========
            std::list<auto_aim::Armor> armors = detector.detect(frame, frame_count);

            // ========== 假分类器处理 + 绘制 ==========
            if (armors.empty()) {
                std::cout << "第 " << frame_count << " 帧：未检测到装甲板" << std::endl;
            } else {
                int armor_idx = 0;
                for (auto& armor : armors) {
                    armor_idx++;
                    // 调用假分类器：随机生成1-5号装甲板种类
                    classifier.classify(armor);
                    // 绘制识别框+角点+分类结果
                    drawArmor(draw_frame, armor);

                    // 打印检测+分类结果
                    std::cout << "第 " << frame_count << " 帧 → 装甲板" << armor_idx 
                              << "：种类=" << armorNameToString(armor.name) 
                              << "，置信度=" << armor.confidence << std::endl;
                }
                std::cout << "第 " << frame_count << " 帧：共检测到 " << armors.size() << " 个装甲板" << std::endl;
            }
        }

        // ========== 显示绘制后的帧（始终显示，避免黑屏） ==========
        if (!draw_frame.empty()) {
            cv::imshow("Armor Detection", draw_frame);
        }

        // ========== 按键监听（P暂停/继续，ESC退出） ==========
        key = cv::waitKey(1) & 0xFF;  // 强制刷新窗口
        if (key == 27) {  // ESC键退出
            std::cout << "[INFO] 手动退出程序！" << std::endl;
            break;
        } else if (key == 'p' || key == 'P') {  // P键暂停/继续
            is_paused = !is_paused;
            std::cout << (is_paused ? "[INFO] 已暂停（按P继续）" : "[INFO] 继续播放") << std::endl;
        }
    }

    // ========== 6. 释放资源 ==========
    cap.release();
    cv::destroyAllWindows();
    return 0;
}