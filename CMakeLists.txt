cmake_minimum_required(VERSION 3.11)
project(auto_aim)

# ========== 核心修复：强制C++17 & 解决字符编码/兼容性 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fpermissive -std=c++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")  # 解决ABI兼容问题
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -ldl -lm")

# ========== 1. OpenVINO配置 ==========
set(OPENVINO_ROOT "/home/a/Desktop/SuperPower/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64")
set(OPENVINO_INC "${OPENVINO_ROOT}/runtime/include")
set(OPENVINO_LIB "${OPENVINO_ROOT}/runtime/lib/intel64")

include_directories(${OPENVINO_INC})
link_directories(${OPENVINO_LIB})

if(NOT EXISTS "${OPENVINO_INC}/openvino/openvino.hpp")
    message(FATAL_ERROR "❌ OpenVINO头文件缺失 → ${OPENVINO_INC}/openvino/openvino.hpp")
endif()
if(NOT EXISTS "${OPENVINO_LIB}/libopenvino.so")
    message(FATAL_ERROR "❌ OpenVINO核心库缺失 → ${OPENVINO_LIB}/libopenvino.so")
else()
    message(STATUS "✅ OpenVINO路径验证成功 ✅")
endif()

# ========== 2. 自动创建auto_aim软链接 ==========
set(AUTO_AIM_DIR "${PROJECT_SOURCE_DIR}/tasks/auto_aim")
if(NOT EXISTS ${AUTO_AIM_DIR})
    message(STATUS "⚠️ 自动创建auto_aim软链接 → 适配tasks/auto_aim/armor.hpp引用")
    file(CREATE_LINK ${PROJECT_SOURCE_DIR}/tasks ${AUTO_AIM_DIR} SYMBOLIC)
endif()

# ========== 3. 依赖库（修复fmt/yaml-cpp查找） ==========
find_package(OpenCV 4 REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "❌ 安装OpenCV：sudo apt install libopencv-dev")
endif()

# 修复fmt库查找
find_package(fmt REQUIRED CONFIG)
if(NOT TARGET fmt::fmt)
    message(WARNING "⚠️ fmt库未找到，手动指定路径")
    include_directories(/usr/include/fmt)
    link_directories(/usr/lib/x86_64-linux-gnu)
    set(FMT_LIB fmt)
else()
    set(FMT_LIB fmt::fmt)
endif()

# 修复yaml-cpp查找
find_package(yaml-cpp REQUIRED)
if(NOT yaml-cpp_FOUND)
    include_directories(/usr/include/yaml-cpp)
    link_directories(/usr/lib/x86_64-linux-gnu)
    set(YAML_LIB yaml-cpp)
else()
    set(YAML_LIB yaml-cpp)
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# ========== 4. 源文件列表 ==========
set(CORE_SOURCES
    main.cpp
    yolov5.cpp
    tools/img_tools.cpp
    tools/logger.cpp
    tasks/armor.cpp
    tasks/classifier.cpp
    tasks/detector.cpp
)

# 自动跳过缺失文件
foreach(FILE IN LISTS CORE_SOURCES)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${FILE}")
        message(WARNING "⚠️ 跳过缺失文件：${FILE}")
        list(REMOVE_ITEM CORE_SOURCES ${FILE})
    endif()
endforeach()

# ========== 5. 头文件包含 ==========
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tasks
    ${PROJECT_SOURCE_DIR}/tasks/auto_aim
    ${PROJECT_SOURCE_DIR}/tools
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# ========== 6. 生成可执行文件 & 链接库 ==========
add_executable(main ${CORE_SOURCES})

target_link_libraries(
    main
    PRIVATE
    ${OpenCV_LIBS}
    ${FMT_LIB}
    ${YAML_LIB}
    Eigen3::Eigen
    openvino
    openvino_intel_cpu_plugin
)

message(STATUS "✅ CMake配置完成 → 可执行文件：${CMAKE_BINARY_DIR}/main")