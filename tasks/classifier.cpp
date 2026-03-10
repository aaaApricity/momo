
#include "classifier.hpp"
#include <iostream>
#include <random>

namespace auto_aim {

// 假分类器：不加载模型，直接返回固定值或随机值
Classifier::Classifier(const std::string& config_path) {
    std::cout << "[WARNING] Using Dummy Classifier (OpenVINO disabled)." << std::endl;
    std::cout << "[WARNING] Armor number recognition is simulated." << std::endl;
}

void Classifier::classify(Armor& armor) {
    // 模拟分类结果
    // 这里我们简单地根据颜色或随机给一个名字，避免程序崩溃
    // 假设 50% 概率是 "one" (1号), 50% 概率是 "two" (2号)，或者其他
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 4); // 0~4 代表几种可能的数字

    int random_id = dis(gen);
    
    // 简单映射 (根据你的 ArmorName 枚举调整)
    // 假设: 0=one, 2=two, 3=three, 4=four, 5=five
    if (random_id == 0) armor.name = ArmorName::one;
    else if (random_id == 1) armor.name = ArmorName::two;
    else if (random_id == 2) armor.name = ArmorName::three;
    else if (random_id == 3) armor.name = ArmorName::four;
    else armor.name = ArmorName::five;

    armor.confidence = 0.95; // 给一个高置信度，让它通过检查
}

} // namespace auto_aim
