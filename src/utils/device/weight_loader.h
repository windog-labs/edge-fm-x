#pragma once

#include <edge-fm/core.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <functional>
#include <optional>
#include "utils/non_copyable.h"
#include "engine/engine.h"

namespace edge_fm {

/**
 * @brief 权重加载器（单例模式）
 * 
 * 专门用于从 safetensors 文件加载模型权重，并使用全局缓存避免重复加载。
 * 线程安全的单例实现。
 */
class WeightLoader : public NonCopyableNonMovable {
public:
    /**
     * @brief 获取单例实例
     * 
     * @return WeightLoader& 单例引用
     */
    static WeightLoader& instance();
    
    /**
     * @brief 清空指定 stage 的权重缓存（用于创建新 engine 前重置，避免 Fused 层修改后的脏缓存影响后续加载）
     */
    void clear_stage(ModelStage cache_key);

    /**
     * @brief 获取缓存的权重
     * 
     * @param cache_key 缓存键（ModelStage 类型）
     * @return const std::unordered_map<std::string, Tensor>& 权重映射表的引用
     * 
     * @throws ConfigurationError 如果 cache_key 对应的权重不存在
     */
    const std::unordered_map<std::string, Tensor>& get(ModelStage cache_key) const;
    
    /**
     * @brief 从 safetensors 文件加载权重
     * 
     * @param cache_key 缓存键（ModelStage 类型）
     * @param safetensors_file safetensors 文件路径
     * @param device 设备类型
     * @param device_id 设备 ID
     * @param overwrite_if_exists 如果权重已存在是否覆盖
     * @param weight_filter 可选的权重名称过滤器。如果提供，只加载匹配过滤器的权重。
     *                      过滤器函数接收权重名称（std::string），返回 true 表示加载，false 表示跳过。
     *                      如果为 std::nullopt（默认），则加载所有权重。
     * @param key_mapper 可选的存储键映射。若提供，缓存时使用 key_mapper(原名) 作为键（用于 VLM 的 model.model.xxx -> model.xxx）。
     * @throws ConfigurationError 如果文件加载失败
     * @throws DeviceError 如果设备操作失败
     * 
     * @note 加载的权重会存储到全局缓存中，使用 cache_key 作为键
     * @note 如果同一个文件已经被加载过，会直接返回，不会重复加载
     * @note 如果同一个 cache_key 对应的缓存已存在，新加载的权重会合并到现有缓存中（同名权重会被覆盖）
     * 
     * @example
     * // 加载所有权重（默认行为）
     * loader.load_weights_from_file(ModelStage::Prefill, "model.safetensors", Device::GPU, 0);
     * 
     * // 只加载 MLP 层的权重
     * auto mlp_filter = [](const std::string& name) {
     *     return name.find(".mlp.") != std::string::npos;
     * };
     * loader.load_weights_from_file(ModelStage::Prefill, "model.safetensors", Device::GPU, 0, false, mlp_filter);
     */
    void load_weights_from_file(
        ModelStage cache_key,
        const std::string& safetensors_file,
        Device device,
        int32_t device_id,
        bool overwrite_if_exists = false,
        const std::optional<std::function<bool(const std::string&)>>& weight_filter = std::nullopt,
        const std::optional<std::function<std::string(const std::string&)>>& key_mapper = std::nullopt);

    /**
     * @brief 获取修改权重的互斥锁
     * 
     * 用于需要修改权重缓存的场景（如 Fused 层创建 fused 权重并删除原始权重）。
     * 调用者应该在使用 const_cast 修改权重 map 之前获取此锁。
     * 
     * @return std::mutex& 互斥锁的引用
     * 
     * @note 此方法用于特殊场景，普通 Layer 不应该使用此方法
     */
    std::mutex& get_modification_mutex() { return mutex_; }

private:
    WeightLoader() = default;
    ~WeightLoader() = default;
    
    mutable std::mutex mutex_;
    std::unordered_map<ModelStage, std::vector<std::string>> stage_stfiles_;
    std::unordered_map<ModelStage, std::unordered_map<std::string, Tensor>> cache_;
};

} // namespace edge_fm

