#include "engine/compact_vocab.h"

#include <edge-fm/core.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path make_case_dir(const std::string& name) {
    auto dir = std::filesystem::temp_directory_path() / ("edgefm_compact_vocab_" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir / "model");
    return dir;
}

void write_json(const std::filesystem::path& path, const nlohmann::json& payload) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open " + path.string());
    }
    out << payload.dump(2);
}

void set_default_config_dir_once() {
    static bool configured = false;
    if (configured) {
        return;
    }

    auto dir = std::filesystem::temp_directory_path() / "edgefm_compact_vocab_default_config";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    write_json(dir / "engine_default.json", nlohmann::json::object());
    setenv("EDGE_FM_CONFIG_DIR", dir.string().c_str(), 1);
    configured = true;
}

nlohmann::json base_engine_config(const std::filesystem::path& dir, int vocab_size) {
    write_json(dir / "model" / "config.json", {
        {"model_type", "qwen2"},
        {"vocab_size", vocab_size},
        {"hidden_size", 8},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"intermediate_size", 16},
        {"torch_dtype", "float16"},
    });
    return {
        {"model_name", "qwen2_5"},
        {"prefill_model_path", (dir / "model").string()},
        {"runtime", {{"device", "cuda"}, {"device_id", 0}}},
        {"kvcache", {
            {"dtype", "fp16"},
            {"attention_type", "mha"},
            {"requests", {{{"request_id", 0}, {"prefix_token_ids", nlohmann::json::array()}, {"max_tokens", 8}}}},
        }},
        {"sampling", {{"temperature", 0.0}, {"seed", 42}}},
    };
}

edge_fm::EngineConfig write_engine_config(const std::filesystem::path& dir, const nlohmann::json& payload) {
    set_default_config_dir_once();
    const auto path = dir / "engine_config.json";
    write_json(path, payload);
    return edge_fm::EngineConfig(path.string());
}

void test_non_identity_remap_and_restore() {
    auto dir = make_case_dir("non_identity");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 6},
        {"compact_vocab_size", 3},
        {"old_to_new", {-1, 2, -1, 0, 1, -1}},
        {"new_to_old", {3, 4, 1}},
        {"special_token_ids", {1, 3}},
    });

    auto payload = base_engine_config(dir, 3);
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);
    edge_fm::CompactVocab compact(config);

    require(compact.enabled(), "compact vocab should be enabled");
    require(compact.compact_vocab_size() == 3, "compact vocab size mismatch");

    std::vector<int32_t> external = {3, 4, 1};
    std::vector<int32_t> internal = compact.remap_token_ids(external, "request token_ids");
    require((internal == std::vector<int32_t>{0, 1, 2}), "non-identity remap mismatch");

    std::vector<int32_t> restored = compact.restore_token_ids(internal, "response token_ids");
    require(restored == external, "restore mismatch");

    std::vector<int32_t> stop = compact.remap_required_token_ids({1, 3}, "stop_token_ids");
    require((stop == std::vector<int32_t>{2, 0}), "stop remap mismatch");
}

void test_disabled_mapping_is_noop() {
    auto dir = make_case_dir("disabled");
    auto payload = base_engine_config(dir, 8);
    auto config = write_engine_config(dir, payload);
    edge_fm::CompactVocab compact(config);

    require(!compact.enabled(), "compact vocab should be disabled by default");
    std::vector<int32_t> ids = {0, 7, 3};
    require(compact.remap_token_ids(ids, "request token_ids") == ids, "disabled remap must be noop");
    require(compact.restore_token_ids(ids, "response token_ids") == ids, "disabled restore must be noop");
}

void test_identity_mapping_request_remap() {
    auto dir = make_case_dir("identity");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 4},
        {"old_to_new", {0, 1, 2, 3}},
        {"new_to_old", {0, 1, 2, 3}},
        {"special_token_ids", {0, 3}},
    });

    auto payload = base_engine_config(dir, 4);
    payload["sampling"]["stop_token_ids"] = {3};
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);
    edge_fm::CompactVocab compact(config);

    edge_fm::Request request(0, {0, 1, 2});
    request.set_stop_token_ids({3});
    edge_fm::Request remapped = compact.remap_request(request, edge_fm::Device::CPU, 0, edge_fm::MemoryOwnership::OwnCpuMalloc);

    require((remapped.token_ids() == std::vector<int32_t>{0, 1, 2}), "identity request tokens changed");
    require((remapped.stop_token_ids() == std::vector<int32_t>{3}), "identity request stop tokens changed");
    require(
        (compact.restore_token_ids(remapped.token_ids(), "response token_ids") == std::vector<int32_t>{0, 1, 2}),
        "identity restore changed tokens");
}

void test_non_identity_request_stop_remap() {
    auto dir = make_case_dir("request_stop");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 6},
        {"compact_vocab_size", 3},
        {"old_to_new", {-1, 2, -1, 0, 1, -1}},
        {"new_to_old", {3, 4, 1}},
        {"special_token_ids", {1, 3}},
    });

    auto payload = base_engine_config(dir, 3);
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);
    edge_fm::CompactVocab compact(config);

    edge_fm::Request request(0, {3, 4, 1});
    request.set_stop_token_ids({1});
    edge_fm::Request remapped = compact.remap_request(request, edge_fm::Device::CPU, 0, edge_fm::MemoryOwnership::OwnCpuMalloc);

    require((remapped.token_ids() == std::vector<int32_t>{0, 1, 2}), "request token remap mismatch");
    require((remapped.stop_token_ids() == std::vector<int32_t>{2}), "request stop token remap mismatch");
    require(
        (compact.restore_token_ids(remapped.token_ids(), "response token_ids") == std::vector<int32_t>{3, 4, 1}),
        "request restore mismatch");
}

void test_unknown_input_rejected() {
    auto dir = make_case_dir("unknown");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 2},
        {"old_to_new", {0, -1, 1, -1}},
        {"new_to_old", {0, 2}},
        {"special_token_ids", {0}},
    });

    auto payload = base_engine_config(dir, 2);
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);
    edge_fm::CompactVocab compact(config);

    bool threw = false;
    try {
        (void)compact.remap_token_ids({0, 1, 2}, "request token_ids");
    } catch (const edge_fm::InvalidRequestError&) {
        threw = true;
    }
    require(threw, "unmapped input token must throw InvalidRequestError");
}

void test_mapping_shape_validation() {
    auto dir = make_case_dir("shape");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 2},
        {"old_to_new", {0, -1, 1}},
        {"new_to_old", {0, 2}},
        {"special_token_ids", {0}},
    });

    auto payload = base_engine_config(dir, 2);
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);

    bool threw = false;
    try {
        edge_fm::CompactVocab compact(config);
        (void)compact;
    } catch (const edge_fm::ConfigurationError&) {
        threw = true;
    }
    require(threw, "old_to_new length mismatch must throw ConfigurationError");
}

void test_special_tokens_must_be_kept() {
    auto dir = make_case_dir("special");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 2},
        {"old_to_new", {0, -1, 1, -1}},
        {"new_to_old", {0, 2}},
        {"special_token_ids", {1}},
    });

    auto payload = base_engine_config(dir, 2);
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);

    bool threw = false;
    try {
        edge_fm::CompactVocab compact(config);
        (void)compact;
    } catch (const edge_fm::ConfigurationError&) {
        threw = true;
    }
    require(threw, "unmapped special token must throw ConfigurationError");
}

void test_config_stop_tokens_must_be_kept() {
    auto dir = make_case_dir("config_stop");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 2},
        {"old_to_new", {0, -1, 1, -1}},
        {"new_to_old", {0, 2}},
        {"special_token_ids", {0}},
    });

    auto payload = base_engine_config(dir, 2);
    payload["sampling"]["stop_token_ids"] = {3};
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);

    bool threw = false;
    try {
        edge_fm::CompactVocab compact(config);
        (void)compact;
    } catch (const edge_fm::ConfigurationError&) {
        threw = true;
    }
    require(threw, "unmapped config stop token must throw ConfigurationError");
}

void test_config_eos_tokens_must_be_kept() {
    auto dir = make_case_dir("config_eos");
    write_json(dir / "compact_vocab.json", {
        {"format", "edgefm.compact_vocab.v1"},
        {"original_vocab_size", 4},
        {"compact_vocab_size", 2},
        {"old_to_new", {0, -1, 1, -1}},
        {"new_to_old", {0, 2}},
        {"special_token_ids", {0}},
    });

    auto payload = base_engine_config(dir, 2);
    write_json(dir / "model" / "config.json", {
        {"model_type", "qwen2"},
        {"vocab_size", 2},
        {"hidden_size", 8},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"intermediate_size", 16},
        {"torch_dtype", "float16"},
        {"eos_token_id", 1},
    });
    payload["compact_vocab"] = {
        {"enabled", true},
        {"mapping_path", "compact_vocab.json"},
        {"reject_unknown_input_ids", true},
    };
    auto config = write_engine_config(dir, payload);

    bool threw = false;
    try {
        edge_fm::CompactVocab compact(config);
        (void)compact;
    } catch (const edge_fm::ConfigurationError&) {
        threw = true;
    }
    require(threw, "unmapped config eos token must throw ConfigurationError");
}

} // namespace

int main() {
    try {
        test_disabled_mapping_is_noop();
        test_identity_mapping_request_remap();
        test_non_identity_remap_and_restore();
        test_non_identity_request_stop_remap();
        test_unknown_input_rejected();
        test_mapping_shape_validation();
        test_special_tokens_must_be_kept();
        test_config_stop_tokens_must_be_kept();
        test_config_eos_tokens_must_be_kept();
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
