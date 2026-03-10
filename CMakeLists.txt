cmake_minimum_required(VERSION 3.11)
project(auto_aim)

# ========== 第一步：设置C++标准 & 编译选项（新增优化） ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项：Debug模式带调试信息，Release模式优化性能，开启所有警告
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -O0")  # 调试模式：无优化+调试信息
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")    # 发布模式：最高优化
endif()

# ========== 第二步：查找依赖库（优化yaml-cpp查找逻辑） ==========
# 1. Eigen3（强制查找）
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "❌ 未找到Eigen3库，请执行：sudo apt install libeigen3-dev")
endif()

# 2. OpenCV（强制查找）
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "❌ 未找到OpenCV库，请执行：sudo apt install libopencv-dev")
endif()

# 3. yaml-cpp（优先原生find_package，失败则用手动路径，兼容所有环境）
find_package(yaml-cpp QUIET)  # 静默查找，不报错
if(yaml-cpp_FOUND)
    message(STATUS "✅ 自动找到yaml-cpp库：${yaml-cpp_VERSION}")
else()
    # 手动适配你的环境路径（仅当自动查找失败时生效）
    set(YAML_CPP_INCLUDE_DIR "/usr/include/yaml-cpp")
    set(YAML_CPP_LIBRARIES "/usr/lib/x86_64-linux-gnu/libyaml-cpp.so")
    
    # 验证路径有效性
    if(NOT EXISTS ${YAML_CPP_INCLUDE_DIR})
        message(FATAL_ERROR "❌ yaml-cpp头文件不存在：${YAML_CPP_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS ${YAML_CPP_LIBRARIES})
        message(FATAL_ERROR "❌ yaml-cpp库文件不存在：${YAML_CPP_LIBRARIES}")
    endif()
    message(STATUS "✅ 手动加载yaml-cpp库：${YAML_CPP_LIBRARIES}")
endif()

# ========== 第三步：验证源文件/目录存在性（新增，提前报错） ==========
# 检查tasks目录
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tasks)
    message(FATAL_ERROR "❌ tasks目录不存在：${PROJECT_SOURCE_DIR}/tasks")
endif()
# 检查tools目录（如果用到则验证）
if(EXISTS ${PROJECT_SOURCE_DIR}/tools)
    message(STATUS "✅ 找到tools目录：${PROJECT_SOURCE_DIR}/tools")
else()
    message(WARNING "⚠️ tools目录不存在，若未用到可删除tools/相关代码")
    # 若不需要tools，注释下面两行：
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_TOOLS")  # 可选：定义宏跳过tools相关代码
    list(REMOVE_ITEM SOURCES tools/logger.cpp tools/img_tools.cpp)
endif()

# 检查关键源文件
foreach(FILE IN ITEMS tasks/detector.cpp tasks/armor.cpp tasks/classifier.cpp main.cpp)
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/${FILE})
        message(FATAL_ERROR "❌ 源文件不存在：${PROJECT_SOURCE_DIR}/${FILE}")
    endif()
endforeach()

# ========== 第四步：包含头文件目录（修复tasks/缺失） ==========
include_directories(
    ${PROJECT_SOURCE_DIR}          # 项目根目录
    ${PROJECT_SOURCE_DIR}/tasks    # 新增：tasks目录（detector.hpp/armor.hpp）
    ${PROJECT_SOURCE_DIR}/tools    # tools目录（若存在）
    ${EIGEN3_INCLUDE_DIR}          # Eigen头文件
    ${OpenCV_INCLUDE_DIRS}         # OpenCV头文件
    ${YAML_CPP_INCLUDE_DIR}        # yaml-cpp头文件
)

# ========== 第五步：生成可执行文件 ==========
# 源文件列表（单独定义，更易维护）
set(SOURCES
    main.cpp
    tasks/detector.cpp
    tasks/armor.cpp
    tasks/classifier.cpp
    # 若tools目录不存在，注释下面两行：
    tools/logger.cpp
    tools/img_tools.cpp
)

add_executable(main ${SOURCES})

# ========== 第六步：链接依赖库 ==========
target_link_libraries(
    main
    ${OpenCV_LIBS}
    $<IF:$<TARGET_EXISTS:yaml-cpp>,yaml-cpp,${YAML_CPP_LIBRARIES}>  # 兼容自动/手动查找
    Eigen3::Eigen
)

# ========== 调试信息（增强） ==========
message(STATUS "✅ 项目名称：${PROJECT_NAME}")
message(STATUS "✅ C++标准：${CMAKE_CXX_STANDARD}")
message(STATUS "✅ Eigen3 头文件路径：${EIGEN3_INCLUDE_DIR}")
message(STATUS "✅ OpenCV 头文件路径：${OpenCV_INCLUDE_DIRS}")
message(STATUS "✅ OpenCV 库路径：${OpenCV_LIBS}")
message(STATUS "✅ yaml-cpp 头文件路径：${YAML_CPP_INCLUDE_DIR}")
message(STATUS "✅ 编译模式：${CMAKE_BUILD_TYPE}")
message(STATUS "✅ 可执行文件输出路径：${CMAKE_BINARY_DIR}/main")