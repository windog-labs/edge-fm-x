#pragma once

#include <edge-fm/core.h>
#include <string>

namespace edge_fm {

/**
 * @brief Check a condition and throw an exception if it fails
 * 
 * @tparam ExceptionType The exception type to throw (defaults to ConfigurationError)
 * @param condition The condition to check
 * @param err_msg The error message to include in the exception
 */
template<typename ExceptionType = ConfigurationError>
inline void check(bool condition, const std::string& err_msg) {
    if (!condition) {
        throw ExceptionType(err_msg);
    }
}

} // namespace edge_fm
