#pragma once

namespace edge_fm {

/**
 * @brief Base class that prohibits copying but allows moving
 * 
 * Classes inheriting from NonCopyable cannot be copied, but can be moved.
 * This is useful for classes that manage resources but should be movable.
 */
class NonCopyable {
protected:
    NonCopyable() = default;
    ~NonCopyable() = default;

    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(NonCopyable&&) = default;
};

/**
 * @brief Base class that prohibits both copying and moving
 * 
 * Classes inheriting from NonCopyableNonMovable cannot be copied or moved.
 * This is useful for singleton classes or classes that should never be moved or copied.
 */
class NonCopyableNonMovable {
protected:
    NonCopyableNonMovable() = default;
    ~NonCopyableNonMovable() = default;

    NonCopyableNonMovable(const NonCopyableNonMovable&) = delete;
    NonCopyableNonMovable& operator=(const NonCopyableNonMovable&) = delete;
    
    NonCopyableNonMovable(NonCopyableNonMovable&&) = delete;
    NonCopyableNonMovable& operator=(NonCopyableNonMovable&&) = delete;
};

} // namespace edge_fm

