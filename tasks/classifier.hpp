#pragma once

#include "armor.hpp"
#include <string>
// #include <openvino/openvino.hpp>  <-- 注释掉或删除这行

namespace auto_aim {

class Classifier {
public:
    Classifier(const std::string& config_path);
    void classify(Armor& armor);

private:
    // 移除所有 ov:: 相关的成员变量
    // ov::Core core;
    // ov::CompiledModel compiled_model;
};

} // namespace auto_aim