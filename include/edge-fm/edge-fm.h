#pragma once
#include <edge-fm/core.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace edge_fm {

class EdgeFM {
public:
    /**
     * @brief Construct an EdgeFM inference engine
     * 
     * @param config_path Path to the JSON configuration file
     * 
     * @note The configuration file should contain model paths, runtime settings,
     *       KV cache configuration, and sampling parameters.
     * 
     * @throws std::runtime_error if the configuration file cannot be loaded or parsed,
     *         or if model initialization fails
     */
    explicit EdgeFM(const std::string& config_path);

    // Destructor
    ~EdgeFM() noexcept;

    /**
     * @brief Generate response tokens from the given request
     * 
     * @param request Input request containing token IDs and optional image embeddings
     * 
     * @return Response object containing generated token IDs
     * 
     * @note This method performs inference using the loaded model. The request must
     *       be valid and contain at least one token ID.
     * 
     * @throws std::runtime_error if the request is invalid, model is not loaded,
     *         or inference fails (e.g., GPU memory不足, timeout)
     */
    Response generate(const Request& request) const;

    /**
     * @brief Return stage timing from the most recent generate() call.
     *
     * Keys are stable strings such as `prefill_ms`, `decode_ms`,
     * `total_stage_ms`, `decode_step_avg_ms`, `generated_tokens_total`,
     * and `decode_steps`.
     */
    std::unordered_map<std::string, double> last_generate_metrics() const;

    /**
     * @brief Build or tune backend-specific execution artifacts for the current model/configuration.
     *
     * CUDA-like backends may benchmark fixed layer ops and fused ops, while
     * whole-graph backends may emit or refresh compiled artifacts such as HBM.
     * This method does not change user-facing model structure, weights, or
     * quantization.
     */
    void tune();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_fm
