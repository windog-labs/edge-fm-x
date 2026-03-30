#include "models/model_description.h"

#include <edge-fm/core.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unordered_set>

namespace edge_fm {

namespace {

uint64_t fnv1a_64(const std::string& data) {
    constexpr uint64_t kOffset = 14695981039346656037ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    for (unsigned char c : data) {
        hash ^= static_cast<uint64_t>(c);
        hash *= kPrime;
    }
    return hash;
}

std::string to_hex(uint64_t value) {
    std::ostringstream oss;
    oss << std::hex << value;
    return oss.str();
}

void check_supported_op(const std::string& op_type) {
    static const std::unordered_set<std::string> kSupportedOps = {
        "Embedding",
        "RMSNorm",
        "Linear",
        "QKVLinear",
        "Attention",
        "ResidualAdd",
        "SiLUAndMul",
        "LMHead",
        "InjectEmbedding",
    };
    if (kSupportedOps.find(op_type) == kSupportedOps.end()) {
        throw ConfigurationError("Unsupported model_description op_type: " + op_type);
    }
}

void expect(
    bool condition,
    const std::string& message)
{
    if (!condition) {
        throw ConfigurationError(message);
    }
}

struct ModelOpView {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string weight_key;
    nlohmann::json attrs = nlohmann::json::object();
};

ModelOpView parse_model_op(const nlohmann::json& json) {
    ModelOpView node;
    node.name = json.value("name", std::string(""));
    node.op_type = json.value("op_type", std::string(""));
    node.inputs = json.value("inputs", std::vector<std::string>{});
    node.outputs = json.value("outputs", std::vector<std::string>{});
    node.weight_key = json.value("weight_key", std::string(""));
    node.attrs = json.contains("attrs") && json["attrs"].is_object()
        ? json["attrs"]
        : nlohmann::json::object();
    return node;
}

std::vector<ModelOpView> parse_model_ops(const nlohmann::json& model_description) {
    expect(
        model_description.is_object(),
        "model_description must be a JSON object");
    expect(
        model_description.contains("nodes") && model_description["nodes"].is_array(),
        "model_description.nodes must be an array");

    std::vector<ModelOpView> nodes;
    for (const auto& node_json : model_description["nodes"]) {
        nodes.push_back(parse_model_op(node_json));
    }
    return nodes;
}

void validate_model_description_basic(const nlohmann::json& model_description) {
    expect(
        model_description.is_object(),
        "model_description must be a JSON object");
    expect(
        !model_description.value("model_type", std::string("")).empty(),
        "model_description.model_type must not be empty");

    const auto nodes = parse_model_ops(model_description);
    expect(!nodes.empty(), "model_description must contain at least one op");

    std::unordered_set<std::string> node_names;
    std::unordered_set<std::string> produced_tensors;
    for (const auto& node : nodes) {
        check_supported_op(node.op_type);
        expect(!node.name.empty(), "model_description op name must not be empty");
        expect(
            node_names.insert(node.name).second,
            "Duplicate model_description op name: " + node.name);
        expect(!node.outputs.empty(), "model_description op outputs must not be empty");
        for (const auto& output : node.outputs) {
            expect(
                !output.empty(),
                "model_description op output must not be empty for op '" + node.name + "'");
            expect(
                produced_tensors.insert(output).second,
                "Duplicate model_description tensor output: " + output);
        }
    }
}

std::string get_role(const ModelOpView& node) {
    if (!node.attrs.is_object()) {
        return "";
    }
    return node.attrs.value("role", std::string(""));
}

int32_t get_layer_id(const ModelOpView& node, int32_t default_value = -1) {
    if (!node.attrs.is_object()) {
        return default_value;
    }
    return node.attrs.value("layer_id", default_value);
}

void validate_weight_key(
    const ModelOpView& node,
    const std::string& expected,
    const std::string& label)
{
    if (node.weight_key.empty()) {
        return;
    }
    expect(
        node.weight_key == expected,
        "Unexpected weight_key for " + label + ". Expected '" + expected +
            "', got '" + node.weight_key + "'");
}

void expect_node(
    const ModelOpView& node,
    const std::string& op_type,
    int32_t layer_id,
    const std::string& role,
    const std::string& weight_key)
{
    expect(
        node.op_type == op_type,
        "Unexpected model_description op_type. Expected '" + op_type +
            "', got '" + node.op_type + "' at node '" + node.name + "'");
    if (layer_id >= 0) {
        expect(
            get_layer_id(node) == layer_id,
            "Unexpected layer_id at node '" + node.name + "'");
    }
    if (!role.empty()) {
        expect(
            get_role(node) == role,
            "Unexpected role at node '" + node.name + "'");
    }
    if (!weight_key.empty()) {
        validate_weight_key(node, weight_key, node.name);
    }
}

std::string default_backend_for_op(const std::string& op_type) {
    if (op_type == "Attention") {
        return "flashinfer";
    }
    if (op_type == "Linear" || op_type == "QKVLinear" || op_type == "LMHead") {
        return "cublasLt";
    }
    return "builtin";
}

} // namespace

nlohmann::json CompiledOp::to_json() const {
    return nlohmann::json{
        {"name", name},
        {"op_type", op_type},
        {"backend", backend},
        {"attrs", attrs},
    };
}

CompiledOp CompiledOp::from_json(const nlohmann::json& json) {
    CompiledOp op;
    op.name = json.value("name", std::string(""));
    op.op_type = json.value("op_type", std::string(""));
    op.backend = json.value("backend", std::string(""));
    op.attrs = json.contains("attrs") && json["attrs"].is_object()
        ? json["attrs"]
        : nlohmann::json::object();
    return op;
}

nlohmann::json ExecutionPlan::to_json() const {
    nlohmann::json prefill_json = nlohmann::json::array();
    nlohmann::json decode_json = nlohmann::json::array();
    for (const auto& op : prefill_ops) {
        prefill_json.push_back(op.to_json());
    }
    for (const auto& op : decode_ops) {
        decode_json.push_back(op.to_json());
    }
    return nlohmann::json{
        {"model_description_hash", model_description_hash},
        {"uses_inject_embedding", uses_inject_embedding},
        {"uses_fused_qkv", uses_fused_qkv},
        {"uses_fused_gate_up", uses_fused_gate_up},
        {"uses_mrope", uses_mrope},
        {"prefill_ops", std::move(prefill_json)},
        {"decode_ops", std::move(decode_json)},
    };
}

ExecutionPlan ExecutionPlan::from_json(const nlohmann::json& json) {
    ExecutionPlan plan;
    plan.model_description_hash = json.value(
        "model_description_hash",
        json.value("graph_hash", std::string("")));
    plan.uses_inject_embedding = json.value("uses_inject_embedding", false);
    plan.uses_fused_qkv = json.value("uses_fused_qkv", true);
    plan.uses_fused_gate_up = json.value("uses_fused_gate_up", true);
    plan.uses_mrope = json.value("uses_mrope", false);
    if (json.contains("prefill_ops") && json["prefill_ops"].is_array()) {
        for (const auto& op_json : json["prefill_ops"]) {
            plan.prefill_ops.push_back(CompiledOp::from_json(op_json));
        }
    }
    if (json.contains("decode_ops") && json["decode_ops"].is_array()) {
        for (const auto& op_json : json["decode_ops"]) {
            plan.decode_ops.push_back(CompiledOp::from_json(op_json));
        }
    }
    return plan;
}

std::string hash_model_description(const nlohmann::json& model_description) {
    validate_model_description_basic(model_description);
    return to_hex(fnv1a_64(model_description.dump()));
}

ExecutionPlan compile_model_description(
    const nlohmann::json& model_description,
    const nlohmann::json& model_config)
{
    validate_model_description_basic(model_description);
    expect(
        model_description.value("model_type", std::string("")) == "decoder_only_transformer",
        "Only decoder_only_transformer model descriptions are supported");

    const int32_t num_layers = model_config.value("num_hidden_layers", 0);
    expect(num_layers > 0, "num_hidden_layers is required in model config");

    const auto nodes = parse_model_ops(model_description);
    size_t expected_nodes = 2 + static_cast<size_t>(num_layers) * 10;
    bool uses_inject_embedding = false;
    if (nodes.size() == expected_nodes + 1 && nodes[1].op_type == "InjectEmbedding") {
        uses_inject_embedding = true;
        expected_nodes += 1;
    }
    expect(
        nodes.size() == expected_nodes,
        "Unexpected model description size for decoder_only_transformer");

    size_t index = 0;
    expect_node(
        nodes.at(index++),
        "Embedding",
        -1,
        "",
        "model.embed_tokens.weight");

    if (uses_inject_embedding) {
        expect_node(nodes.at(index++), "InjectEmbedding", -1, "", "");
    }

    for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        const std::string prefix = "model.layers." + std::to_string(layer_id);
        expect_node(
            nodes.at(index++),
            "RMSNorm",
            layer_id,
            "input",
            prefix + ".input_layernorm.weight");
        expect_node(
            nodes.at(index++),
            "QKVLinear",
            layer_id,
            "",
            prefix + ".self_attn");
        expect_node(nodes.at(index++), "Attention", layer_id, "", "");
        expect_node(
            nodes.at(index++),
            "Linear",
            layer_id,
            "o_proj",
            prefix + ".self_attn.o_proj");
        expect_node(nodes.at(index++), "ResidualAdd", layer_id, "attn", "");
        expect_node(
            nodes.at(index++),
            "RMSNorm",
            layer_id,
            "post_attention",
            prefix + ".post_attention_layernorm.weight");
        expect_node(
            nodes.at(index++),
            "Linear",
            layer_id,
            "gate_up_fused",
            prefix + ".mlp");
        expect_node(nodes.at(index++), "SiLUAndMul", layer_id, "", "");
        expect_node(
            nodes.at(index++),
            "Linear",
            layer_id,
            "down_proj",
            prefix + ".mlp.down_proj");
        expect_node(nodes.at(index++), "ResidualAdd", layer_id, "mlp", "");
    }

    expect_node(nodes.at(index++), "RMSNorm", -1, "final", "model.norm.weight");
    expect(
        nodes.at(index).op_type == "LMHead",
        "Expected LMHead as the final op in model description");

    ExecutionPlan plan;
    plan.model_description_hash = hash_model_description(model_description);
    plan.uses_inject_embedding = uses_inject_embedding;
    plan.uses_fused_qkv = true;
    plan.uses_fused_gate_up = true;
    plan.uses_mrope = false;

    if (model_config.contains("rope_scaling") && model_config["rope_scaling"].is_object()) {
        const auto& rope_scaling = model_config["rope_scaling"];
        const std::string rope_type = rope_scaling.value(
            "type",
            rope_scaling.value("rope_type", std::string("")));
        plan.uses_mrope = (rope_type == "mrope");
    }

    for (const auto& node : nodes) {
        CompiledOp prefill_op;
        prefill_op.name = node.name;
        prefill_op.op_type = node.op_type;
        prefill_op.backend = default_backend_for_op(node.op_type);
        prefill_op.attrs = node.attrs;
        plan.prefill_ops.push_back(prefill_op);

        if (node.op_type == "InjectEmbedding") {
            continue;
        }
        CompiledOp decode_op = prefill_op;
        plan.decode_ops.push_back(std::move(decode_op));
    }

    return plan;
}

} // namespace edge_fm
