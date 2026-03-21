#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <list>
#include <filesystem>
#include <sstream>
#include <Eigen/Dense>

// ========== 使用ROS2标准基础类型（无Stamped） ==========
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"  // 替换PointStamped
#include "std_msgs/msg/float64.hpp"     // 替换Float64Stamped

#include "solver.hpp"
#include "yolov5.hpp"
#include "tasks/classifier.hpp"
#include "tools/logger.hpp"
#include "detector.hpp"

// XYZ转YPD工具函数
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz)
{
    double yaw = atan2(xyz.y(), xyz.x());
    double pitch = atan2(-xyz.z(), sqrt(xyz.x()*xyz.x() + xyz.y()*xyz.y()));
    double distance = xyz.norm();
    return Eigen::Vector3d(yaw, pitch, distance);
}

// 终端打印完整信息
std::string getArmorInfo(const auto_aim::Armor& armor)
{
    std::ostringstream oss;
    oss.precision(6);
    oss << "颜色:" << auto_aim::COLORS[armor.color]
        << " 类型:" << auto_aim::ARMOR_TYPES[static_cast<int>(armor.type)]
        << " 名称:" << auto_aim::ARMOR_NAMES[static_cast<int>(armor.name)]
        << " 置信度:" << armor.confidence
        << " 距离:" << armor.xyz_in_gimbal.z() << "m"
        << " 偏航角:" << (armor.ypr_in_gimbal[0] * 180 / CV_PI) << "°"
        << "\n    直角坐标(XYZ/云台系)：" 
        << armor.xyz_in_gimbal.x() << "m, " 
        << armor.xyz_in_gimbal.y() << "m, " 
        << armor.xyz_in_gimbal.z() << "m"
        << "\n    球坐标(YPD)：" 
        << (xyz2ypd(armor.xyz_in_gimbal)[0] * 180 / CV_PI) << "°, "
        << (xyz2ypd(armor.xyz_in_gimbal)[1] * 180 / CV_PI) << "°, "
        << xyz2ypd(armor.xyz_in_gimbal)[2] << "m"
        << "\n    朝向角(YPR/云台系)：" 
        << (armor.ypr_in_gimbal[0] * 180 / CV_PI) << "°, "
        << (armor.ypr_in_gimbal[1] * 180 / CV_PI) << "°, "
        << (armor.ypr_in_gimbal[2] * 180 / CV_PI) << "°";
    return oss.str();
}

// 画面显示
void drawLikeDetector(cv::Mat& frame, const std::list<auto_aim::Armor>& armors)
{
    for (const auto& armor : armors)
    {
        std::vector<cv::Point> pts;
        for (const auto& p : armor.points)
        {
            pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        cv::polylines(frame, pts, true, cv::Scalar(0, 255, 0), 2);

        cv::Point2f center(0, 0);
        for (const auto& p : armor.points) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x /= 4;
        center.y /= 4;

        // 基础信息
        std::ostringstream base_stream;
        base_stream.precision(1);
        base_stream << std::fixed
                    << armor.confidence << " "
                    << auto_aim::ARMOR_NAMES[armor.name] << " "
                    << armor.xyz_in_gimbal.z() << "m";
        cv::putText(frame, base_stream.str(),
                    cv::Point(static_cast<int>(center.x) - 60, static_cast<int>(center.y) - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);

        // YPD
        Eigen::Vector3d ypd = xyz2ypd(armor.xyz_in_gimbal);
        std::ostringstream ypd_stream;
        ypd_stream.precision(1);
        ypd_stream << std::fixed
                   << "YPD:" << (ypd[0] * 180 / CV_PI) << "°," 
                   << (ypd[1] * 180 / CV_PI) << "°," 
                   << ypd[2] << "m";
        cv::putText(frame, ypd_stream.str(),
                    cv::Point(static_cast<int>(center.x) - 80, static_cast<int>(center.y) + 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

        // YPR
        std::ostringstream ypr_stream;
        ypr_stream.precision(1);
        ypr_stream << std::fixed
                   << "YPR:" << (armor.ypr_in_gimbal[0] * 180 / CV_PI) << "°,"
                   << (armor.ypr_in_gimbal[1] * 180 / CV_PI) << "°,"
                   << (armor.ypr_in_gimbal[2] * 180 / CV_PI) << "°";
        cv::putText(frame, ypr_stream.str(),
                    cv::Point(static_cast<int>(center.x) - 80, static_cast<int>(center.y) + 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1);
    }
}

int main(int argc, char** argv)
{
    // ========== 初始化ROS2节点 ==========
    rclcpp::init(argc, argv);
    auto ros_node = rclcpp::Node::make_shared("armor_data_publisher");
    
    // 创建ROS2发布者（使用标准基础类型）
    auto pub_xyz = ros_node->create_publisher<geometry_msgs::msg::Point>("/armor/xyz", 10);
    auto pub_yaw = ros_node->create_publisher<std_msgs::msg::Float64>("/armor/yaw", 10);
    auto pub_distance = ros_node->create_publisher<std_msgs::msg::Float64>("/armor/distance", 10);

    // ========== 原有代码 ==========
    const std::string CONFIG_PATH = "/home/a/Desktop/try/config.yaml";
    const std::string VIDEO_PATH = "/home/a/Desktop/try/demo/demo.avi";

    auto_aim::YOLOV5 yolo_detector(CONFIG_PATH, false);
    auto_aim::Classifier classifier(CONFIG_PATH);
    auto_aim::Solver pnp_solver(CONFIG_PATH);

    const int TARGET_FPS = 60;
    const int FRAME_DELAY = 1000 / TARGET_FPS;
    int frame_delay_actual = FRAME_DELAY;
    cv::TickMeter tm;

    cv::VideoCapture cap;
    if (std::filesystem::exists(VIDEO_PATH))
    {
        cap.open(VIDEO_PATH);
        std::cout << "[INFO] 打开视频：" << VIDEO_PATH << std::endl;
    }
    else
    {
        cap.open(0);
        std::cout << "[INFO] 打开摄像头" << std::endl;
    }

    if (!cap.isOpened())
    {
        std::cerr << "[ERROR] 无法打开视频/摄像头！" << std::endl;
        return -1;
    }

    const std::string WINDOW_NAME = "YOLO + PnP 60FPS";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 1280, 720);

    cv::Mat frame;
    int frame_count = 0;
    bool is_paused = false;

    while (rclcpp::ok())
    {
        if (!is_paused)
        {
            tm.reset();
            tm.start();

            // 修复：空帧时重试3次，仍失败则退出循环
            int retry = 0;
            while (!cap.read(frame) && retry < 3)
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); // 重置到第一帧
                retry++;
                std::cout << "[WARN] 视频帧读取失败，重试第" << retry << "次..." << std::endl;
            }
            if (retry >= 3)
            {
                std::cerr << "[ERROR] 视频文件损坏，无法读取帧！" << std::endl;
                break; // 退出循环，避免空帧死循环
            }

            frame_count++;
            auto armors = yolo_detector.detect(frame, frame_count);

            // PnP解算 + ROS2发布
            for (auto& armor : armors)
            {
                pnp_solver.solve(armor);

                // 坐标系修正
                armor.xyz_in_gimbal.x() = -armor.xyz_in_gimbal.x();
                armor.xyz_in_gimbal.y() = -armor.xyz_in_gimbal.y();
                armor.xyz_in_gimbal.z() =  armor.xyz_in_gimbal.z();

                armor.ypr_in_gimbal[0] = -armor.ypr_in_gimbal[0];
                armor.ypr_in_gimbal[1] = -armor.ypr_in_gimbal[1];

                // ========== 发布ROS2数据（无Stamped） ==========
                // 1. 发布XYZ坐标
                geometry_msgs::msg::Point xyz_msg;
                xyz_msg.x = armor.xyz_in_gimbal.x();
                xyz_msg.y = armor.xyz_in_gimbal.y();
                xyz_msg.z = armor.xyz_in_gimbal.z();
                pub_xyz->publish(xyz_msg);

                // 2. 发布Yaw角度
                std_msgs::msg::Float64 yaw_msg;
                yaw_msg.data = armor.ypr_in_gimbal[0] * 180 / CV_PI;
                pub_yaw->publish(yaw_msg);

                // 3. 发布距离
                std_msgs::msg::Float64 dist_msg;
                dist_msg.data = armor.xyz_in_gimbal.z();
                pub_distance->publish(dist_msg);
            }

            drawLikeDetector(frame, armors);

            std::cout << "\n[FRAME " << frame_count << "] 检测到：" << armors.size() << " 个装甲板" << std::endl;
            int idx = 0;
            for (const auto& armor : armors)
            {
                idx++;
                std::cout << "  装甲板" << idx << "：" << getArmorInfo(armor) << std::endl;
            }

            tm.stop();
            double process_time = tm.getTimeMilli();
            frame_delay_actual = std::max(1, FRAME_DELAY - static_cast<int>(process_time));
        }

        if (!frame.empty())
        {
            cv::imshow(WINDOW_NAME, frame);
        }

        int key = cv::waitKey(frame_delay_actual) & 0xFF;
        if (key == 27) break;
        else if (key == 'p' || key == 'P') is_paused = !is_paused;

        // ROS2节点自旋
        rclcpp::spin_some(ros_node);
    }

    // 资源清理
    cap.release();
    cv::destroyAllWindows();
    rclcpp::shutdown();
    return 0;
}