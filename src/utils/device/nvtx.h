#pragma once

#include <string>
#include <cstdint>
#include <nvtx3/nvToolsExt.h>
#include "utils/non_copyable.h"

namespace edge_fm {

enum class NVTXColor : uint32_t {
    RED = 0xFF0000FF,
    GREEN = 0xFF00FF00,
    BLUE = 0xFFFF0000,
    YELLOW = 0xFF00FFFF,
    CYAN = 0xFFFFFF00,
    MAGENTA = 0xFFFF00FF,
    WHITE = 0xFFFFFFFF,
    BLACK = 0xFF000000,
    ORANGE = 0xFF0080FF,
    PURPLE = 0xFF800080
};

class NVTX {
public:
    static inline void range_push(const std::string& name) {
        nvtxRangePushA(name.c_str());
    }

    static inline void range_push(const std::string& name, NVTXColor color) {
        nvtxEventAttributes_t eventAttrib = {};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = static_cast<uint32_t>(color);
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name.c_str();
        nvtxRangePushEx(&eventAttrib);
    }

    static inline void range_pop() {
        nvtxRangePop();
    }

    class Range : public NonCopyable {
    public:
        explicit Range(const std::string& name) {
            if (!name.empty()) {
                NVTX::range_push(name);
                pushed_ = true;
            }
        }

        Range(const std::string& name, NVTXColor color) {
            if (!name.empty()) {
                NVTX::range_push(name, color);
                pushed_ = true;
            }
        }

        ~Range() {
            if (pushed_) {
                NVTX::range_pop();
            }
        }

    private:
        bool pushed_ = false;
    };
};

} // namespace edge_fm
