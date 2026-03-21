#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
using namespace std::chrono_literals;

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("test_node");
  auto pub = node->create_publisher<std_msgs::msg::Float64>("/armor/distance", 10);

  rclcpp::Rate rate(10);
  float value = 3.0f;
  while (rclcpp::ok()) {
    std_msgs::msg::Float64 msg;
    msg.data = value;
    pub->publish(msg);
    RCLCPP_INFO(node->get_logger(), "发布距离: %.2f", value);
    value += 0.1;
    if (value > 5) value = 3.0;
    rate.sleep();
  }
  rclcpp::shutdown();
  return 0;
}
