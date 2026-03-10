#pragma once
#include <iostream>
#include <string>
#include <sstream>

namespace tools {

class Logger {
public:
    void debug(const std::string& msg) {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
    
    // 支持可变参数模板
    template<typename... Args>
    void debug(const std::string& fmt, Args... args) {
        std::ostringstream oss;
        format_impl(oss, fmt, args...);
        std::cout << "[DEBUG] " << oss.str() << std::endl;
    }
    
    void info(const std::string& msg) {
        std::cout << "[INFO] " << msg << std::endl;
    }
    
    void warn(const std::string& msg) {
        std::cout << "[WARN] " << msg << std::endl;
    }
    
    void error(const std::string& msg) {
        std::cerr << "[ERROR] " << msg << std::endl;
    }

private:
    template<typename T>
    void format_impl(std::ostringstream& oss, const std::string& fmt, const T& val) {
        size_t pos = fmt.find("{}");
        if (pos != std::string::npos) {
            oss << fmt.substr(0, pos) << val << fmt.substr(pos + 2);
        } else {
            oss << fmt;
        }
    }
    
    template<typename T, typename... Args>
    void format_impl(std::ostringstream& oss, const std::string& fmt, const T& val, Args... args) {
        size_t pos = fmt.find("{}");
        if (pos != std::string::npos) {
            oss << fmt.substr(0, pos) << val;
            format_impl(oss, fmt.substr(pos + 2), args...);
        }
    }
};

inline Logger* logger() {
    static Logger instance;
    return &instance;
}

}  // namespace tools