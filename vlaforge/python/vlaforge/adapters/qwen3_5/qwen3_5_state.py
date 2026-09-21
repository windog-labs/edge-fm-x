"""Explicit prefill/decode state for hybrid Qwen3.5 text models.

The upstream ``DynamicCache`` is intentionally not used at the deployment
boundary. Linear-attention layers expose convolution and recurrent state;
full-attention layers expose fixed-capacity key/value state. The implementation
calls the original upstream modules and mirrors their tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class QwenStateTensor:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class QwenStateSpec:
    tensors: tuple[QwenStateTensor, ...]
    layer_types: tuple[str, ...]

    def zeros(self, *, device: torch.device | str, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.zeros(item.shape, dtype=item.dtype, device=device)
            for item in self.tensors
        )


def state_spec(
    model: nn.Module,
    *,
    batch_size: int = 1,
    max_sequence_length: int,
) -> QwenStateSpec:
    config = model.config.text_config
    dtype = next(model.parameters()).dtype
    rows = []
    for index, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            rows.append(QwenStateTensor(
                f"layer_{index}_conv",
                (batch_size, config.linear_num_key_heads * config.linear_key_head_dim * 3,
                 config.linear_conv_kernel_dim),
                dtype,
            ))
            rows.append(QwenStateTensor(
                f"layer_{index}_recurrent",
                (batch_size, config.linear_num_value_heads,
                 config.linear_key_head_dim, config.linear_value_head_dim),
                dtype,
            ))
        elif layer_type == "full_attention":
            rows.append(QwenStateTensor(
                f"layer_{index}_keys",
                (batch_size, config.num_key_value_heads, max_sequence_length,
                 config.head_dim),
                dtype,
            ))
            rows.append(QwenStateTensor(
                f"layer_{index}_values",
                (batch_size, config.num_key_value_heads, max_sequence_length,
                 config.head_dim),
                dtype,
            ))
        else:
            raise ValueError(f"unsupported Qwen3.5 layer type: {layer_type}")
    return QwenStateSpec(tuple(rows), tuple(config.layer_types))


def _functional_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream torch chunked rule with the in-place output writes made functional."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    initial_dtype = query.dtype
    query = qwen.l2norm(query, dim=-1, eps=1e-6)
    key = qwen.l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        item.transpose(1, 2).contiguous().to(torch.float32)
        for item in (query, key, value, beta, g)
    )
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
        for item in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(upper, 0)
    for index in range(1, chunk_size):
        row = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(
        batch_size, num_heads, k_head_dim, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    outputs = []
    lower = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    for index in range(total_sequence_length // chunk_size):
        q_chunk, k_chunk, v_chunk = query[:, :, index], key[:, :, index], value[:, :, index]
        chunk_attn = q_chunk @ k_chunk.transpose(-1, -2) * decay_mask[:, :, index]
        v_prime = k_cumdecay[:, :, index] @ state
        v_new = v_chunk - v_prime
        attn_inter = (q_chunk * g[:, :, index, :, None].exp()) @ state
        outputs.append(attn_inter + chunk_attn @ v_new)
        state = (
            state * g[:, :, index, -1, None, None].exp()
            + (k_chunk * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None])
            .transpose(-1, -2) @ v_new
        )
    output = torch.cat(outputs, dim=2)[:, :, :sequence_length]
    output = output.transpose(1, 2).contiguous().to(initial_dtype)
    return output, state


def _functional_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream single-token recurrent rule without an in-place output buffer."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    initial_dtype = query.dtype
    query = qwen.l2norm(query, dim=-1, eps=1e-6)
    key = qwen.l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = (
        item.transpose(1, 2).contiguous().to(torch.float32)
        for item in (query, key, value, beta, g)
    )
    query = query * (1 / (query.shape[-1] ** 0.5))
    q_t, k_t, v_t = query[:, :, 0], key[:, :, 0], value[:, :, 0]
    g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta[:, :, 0].unsqueeze(-1)
    state = initial_state.to(value) * g_t
    kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    output = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    output = output.unsqueeze(2).transpose(1, 2).contiguous().to(initial_dtype)
    return output, state


def _linear_prefill(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    state_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

    batch_size, sequence_length, _ = hidden_states.shape
    mixed_qkv = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    z = layer.in_proj_z(hidden_states).reshape(
        batch_size, sequence_length, -1, layer.head_v_dim
    )
    b = layer.in_proj_b(hidden_states)
    a = layer.in_proj_a(hidden_states)
    conv_state = F.pad(mixed_qkv, (layer.conv_kernel_size - mixed_qkv.shape[-1], 0))
    mixed_qkv = F.silu(layer.conv1d(mixed_qkv)[:, :, :sequence_length])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )
    query = query.reshape(batch_size, sequence_length, -1, layer.head_k_dim)
    key = key.reshape(batch_size, sequence_length, -1, layer.head_k_dim)
    value = value.reshape(batch_size, sequence_length, -1, layer.head_v_dim)
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
    core_attn_out, recurrent_state = _functional_chunk_gated_delta_rule(
        query, key, value, g, beta, chunk_size=64
    )
    recurrent_state = recurrent_state.to(state_dtype or hidden_states.dtype)
    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z).reshape(
        batch_size, sequence_length, -1
    )
    return layer.out_proj(core_attn_out), conv_state, recurrent_state


def _linear_decode(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    state_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, _ = hidden_states.shape
    if sequence_length != 1:
        raise ValueError("explicit Qwen decode requires one token")
    mixed_qkv = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    z = layer.in_proj_z(hidden_states).reshape(
        batch_size, sequence_length, -1, layer.head_v_dim
    )
    b = layer.in_proj_b(hidden_states)
    a = layer.in_proj_a(hidden_states)
    combined = torch.cat([conv_state, mixed_qkv], dim=-1).to(layer.conv1d.weight.dtype)
    new_conv_state = combined[:, :, -layer.conv_kernel_size:]
    mixed_qkv = F.conv1d(
        combined,
        layer.conv1d.weight,
        layer.conv1d.bias,
        padding=0,
        groups=layer.conv_dim,
    )
    mixed_qkv = F.silu(mixed_qkv[:, :, -sequence_length:]).to(hidden_states.dtype)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )
    query = query.reshape(batch_size, sequence_length, -1, layer.head_k_dim)
    key = key.reshape(batch_size, sequence_length, -1, layer.head_k_dim)
    value = value.reshape(batch_size, sequence_length, -1, layer.head_v_dim)
    beta = b.sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
    core_attn_out, recurrent_state = _functional_recurrent_gated_delta_rule(
        query, key, value, g, beta, initial_state=recurrent_state
    )
    recurrent_state = recurrent_state.to(state_dtype or hidden_states.dtype)
    core_attn_out = core_attn_out.reshape(-1, layer.head_v_dim)
    z = z.reshape(-1, layer.head_v_dim)
    core_attn_out = layer.norm(core_attn_out, z).reshape(
        batch_size, sequence_length, -1
    )
    return layer.out_proj(core_attn_out), new_conv_state, recurrent_state


def _project_full_attention(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.head_dim)
    query_states, gate = torch.chunk(
        layer.q_proj(hidden_states).view(*input_shape, -1, layer.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(*input_shape, -1)
    query_states = layer.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = layer.k_norm(layer.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, *position_embeddings
    )
    return query_states, key_states, value_states, gate


def _full_prefill(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    query, key, value, gate = _project_full_attention(
        layer, hidden_states, position_embeddings
    )
    query_length = query.shape[-2]
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        layer.config._attn_implementation, eager_attention_forward
    )
    output, _ = interface(
        layer, query, key, value, None, dropout=0.0, scaling=layer.scaling,
        is_causal=True,
    )
    output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    output = layer.o_proj(output * torch.sigmoid(gate))
    padding = max_sequence_length - query_length
    return (
        output,
        F.pad(key, (0, 0, 0, padding)),
        F.pad(value, (0, 0, 0, padding)),
    )


def _full_decode(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    cache_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    query, key, value, gate = _project_full_attention(
        layer, hidden_states, position_embeddings
    )
    cache_keys = torch.index_copy(cache_keys, 2, cache_position, key)
    cache_values = torch.index_copy(cache_values, 2, cache_position, value)
    slots = torch.arange(cache_keys.shape[-2], device=query.device)
    mask = (slots.view(1, -1) <= cache_position.view(1, 1)).view(
        1, 1, 1, cache_keys.shape[-2]
    )
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        layer.config._attn_implementation, eager_attention_forward
    )
    output, _ = interface(
        layer, query, cache_keys, cache_values, mask,
        dropout=0.0, scaling=layer.scaling,
    )
    output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
    return (
        layer.o_proj(output * torch.sigmoid(gate)),
        cache_keys,
        cache_values,
    )


def _full_decode_exact(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class Cache:
        def __init__(self, keys: torch.Tensor, values: torch.Tensor):
            self.keys = keys
            self.values = values

        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: object = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del layer_idx, cache_kwargs
            self.keys = torch.cat((self.keys, key_states), dim=-2)
            self.values = torch.cat((self.values, value_states), dim=-2)
            return self.keys, self.values

    cache = Cache(cache_keys, cache_values)
    output, _ = layer(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=cache,
    )
    return output, cache.keys, cache.values


class Qwen3_5ExplicitPrefill(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        max_sequence_length: int,
        position_ids: torch.Tensor,
        text_position_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        rope_deltas: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.register_buffer("position_ids", position_ids.clone(), persistent=False)
        self.register_buffer("text_position_ids", text_position_ids.clone(), persistent=False)
        self.register_buffer("rope_deltas", rope_deltas.clone(), persistent=False)
        self.image_grid_values = tuple(
            int(value) for value in image_grid_thw.reshape(-1).tolist()
        )
        self.state = state_spec(
            model, max_sequence_length=max_sequence_length
        )
        self.vision = Qwen3_5VisionFeatures(model.model.visual, image_grid_thw)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        language = self.model.model.language_model
        inputs_embeds = language.embed_tokens(input_ids)
        image_embeds = self.vision(pixel_values).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        hidden_states = inputs_embeds
        position_embeddings = language.rotary_emb(hidden_states, self.position_ids)
        states = []
        state_index = 0
        for index, (layer_type, layer) in enumerate(
            zip(self.state.layer_types, language.layers, strict=True)
        ):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            if layer_type == "linear_attention":
                mixer, conv_state, recurrent_state = _linear_prefill(
                    layer.linear_attn,
                    normed,
                    state_dtype=next(self.model.parameters()).dtype,
                )
                states.extend((conv_state, recurrent_state))
                state_index += 2
            else:
                mixer, keys, values = _full_prefill(
                    layer.self_attn,
                    normed,
                    position_embeddings,
                    self.max_sequence_length,
                )
                states.extend((keys, values))
                state_index += 2
            hidden_states = residual + mixer
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        hidden_states = language.norm(hidden_states)
        logits = self.model.lm_head(hidden_states[:, -1:, :])
        if state_index != len(self.state.tensors):
            raise ValueError("prefill state count differs from the declared contract")
        return tuple(
            (logits.contiguous(), self.rope_deltas.contiguous(),
             *(value.contiguous() for value in states))
        )


class Qwen3_5VisionFeatures(nn.Module):
    """Original Qwen3.5 vision blocks with bilinear constants precomputed."""

    def __init__(self, visual: nn.Module, image_grid_thw: torch.Tensor):
        super().__init__()
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen

        self.visual = visual
        grid = image_grid_thw.to(dtype=torch.int64, device="cpu")
        bilinear_indices, bilinear_weights = qwen.get_vision_bilinear_indices_and_weights(
            grid,
            num_grid_per_side=visual.num_grid_per_side,
            spatial_merge_size=visual.config.spatial_merge_size,
            kwargs={},
        )
        position_ids = qwen.get_vision_position_ids(
            grid,
            spatial_merge_size=visual.spatial_merge_size,
            kwargs={},
        )
        cu_seqlens = qwen.get_vision_cu_seqlens(grid, kwargs={})
        self.register_buffer("bilinear_indices", bilinear_indices, persistent=False)
        self.register_buffer("bilinear_weights", bilinear_weights, persistent=False)
        self.register_buffer("position_ids", position_ids, persistent=False)
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.visual.patch_embed(hidden_states)
        bilinear_indices = self.bilinear_indices.to(hidden_states.device)
        bilinear_weights = self.bilinear_weights.to(hidden_states.device)
        position_ids = self.position_ids.to(hidden_states.device)
        cu_seqlens = self.cu_seqlens.to(hidden_states.device)
        pos_embeds = (
            self.visual.pos_embed(bilinear_indices)
            * bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary_pos_emb = self.visual.rotary_pos_emb(position_ids)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        for block in self.visual.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return self.visual.merger(hidden_states)


class Qwen3_5ExplicitDecode(nn.Module):
    def __init__(self, model: nn.Module, *, max_sequence_length: int):
        super().__init__()
        self.model = model
        self.max_sequence_length = max_sequence_length
        self.state = state_spec(
            model, max_sequence_length=max_sequence_length
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        *states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(states) != len(self.state.tensors):
            raise ValueError("decode state count differs from the declared contract")
        language = self.model.model.language_model
        hidden_states = language.embed_tokens(input_ids)
        position_embeddings = language.rotary_emb(hidden_states, position_ids)
        output_states = []
        state_index = 0
        for layer_type, layer in zip(self.state.layer_types, language.layers, strict=True):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            if layer_type == "linear_attention":
                mixer, conv_state, recurrent_state = _linear_decode(
                    layer.linear_attn,
                    normed,
                    states[state_index],
                    states[state_index + 1],
                    state_dtype=next(self.model.parameters()).dtype,
                )
                output_states.extend((conv_state, recurrent_state))
            else:
                mixer, keys, values = _full_decode(
                    layer.self_attn,
                    normed,
                    position_embeddings,
                    states[state_index],
                    states[state_index + 1],
                    cache_position,
                )
                output_states.extend((keys, values))
            state_index += 2
            hidden_states = residual + mixer
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        hidden_states = language.norm(hidden_states)
        return tuple(
            (self.model.lm_head(hidden_states).contiguous(),
             *(value.contiguous() for value in output_states))
        )


class Qwen3_5Generate(nn.Module):
    """Static fixed-length generation over the explicit prefill/decode state."""

    def __init__(
        self,
        prefill: Qwen3_5ExplicitPrefill,
        decode: Qwen3_5ExplicitDecode,
        *,
        prompt_length: int,
        new_tokens: int,
    ):
        super().__init__()
        if prompt_length < 1 or new_tokens < 1:
            raise ValueError("generation profile requires positive lengths")
        self.prefill = prefill
        self.decode = decode
        self.prompt_length = prompt_length
        self.new_tokens = new_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.prefill(input_ids, attention_mask, pixel_values)
        logits, rope_deltas, *states = outputs
        trimmed_states = []
        state_index = 0
        for layer_type in self.layer_types:
            if layer_type == "linear_attention":
                trimmed_states.extend(
                    (states[state_index], states[state_index + 1])
                )
            else:
                trimmed_states.extend(
                    (
                        states[state_index][..., : self.prompt_length, :],
                        states[state_index + 1][..., : self.prompt_length, :],
                    )
                )
            state_index += 2
        states = trimmed_states
        tokens = [logits[:, -1:, :].argmax(dim=-1)]
        for step in range(1, self.new_tokens):
            cache_position = torch.tensor(
                [self.prompt_length + step - 1],
                dtype=torch.int64,
                device=input_ids.device,
            )
            position_ids = (
                cache_position.view(1, 1, 1)
                + rope_deltas.view(1, 1, 1)
            )
            decoded = self.decode(
                tokens[-1], position_ids, cache_position, *states
            )
            logits, *states = decoded
            tokens.append(logits[:, -1:, :].argmax(dim=-1))
        return torch.cat(tokens, dim=1).contiguous()


class Qwen3_5GenerateExact(nn.Module):
    """Fixed generation with exact upstream dynamic growth for full-attention K/V."""

    def __init__(
        self,
        prefill: Qwen3_5ExplicitPrefill,
        *,
        prompt_length: int,
        new_tokens: int,
    ):
        super().__init__()
        self.prefill = prefill
        self.model = prefill.model
        self.prompt_length = prompt_length
        self.new_tokens = new_tokens
        self.layer_types = tuple(prefill.state.layer_types)
        self.state_dtype = next(prefill.model.parameters()).dtype

    def _decode_once(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        states: list[torch.Tensor],
        layer_outputs: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        language = self.model.model.language_model
        hidden_states = language.embed_tokens(input_ids)
        position_embeddings = language.rotary_emb(hidden_states, position_ids)
        next_states = []
        index = 0
        for layer_type, layer in zip(self.layer_types, language.layers, strict=True):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            if layer_type == "linear_attention":
                mixer, conv_state, recurrent_state = _linear_decode(
                    layer.linear_attn,
                    normed,
                    states[index],
                    states[index + 1],
                    state_dtype=self.state_dtype,
                )
                next_states.extend((conv_state, recurrent_state))
            else:
                mixer, keys, values = _full_decode_exact(
                    layer.self_attn,
                    normed,
                    position_embeddings,
                    states[index],
                    states[index + 1],
                )
                next_states.extend((keys, values))
            index += 2
            hidden_states = residual + mixer
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
            if layer_outputs is not None:
                layer_outputs.append(hidden_states)
        hidden_states = language.norm(hidden_states)
        return self.model.lm_head(hidden_states), next_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.prefill(input_ids, attention_mask, pixel_values)
        logits, rope_deltas, *states = outputs
        trimmed_states = []
        state_index = 0
        for layer_type in self.layer_types:
            if layer_type == "linear_attention":
                trimmed_states.extend(
                    (states[state_index], states[state_index + 1])
                )
            else:
                trimmed_states.extend(
                    (
                        states[state_index][..., : self.prompt_length, :],
                        states[state_index + 1][..., : self.prompt_length, :],
                    )
                )
            state_index += 2
        states = trimmed_states
        tokens = [logits[:, -1:, :].argmax(dim=-1)]
        for step in range(1, self.new_tokens):
            cache_position = torch.tensor(
                [self.prompt_length + step - 1],
                dtype=torch.int64,
                device=input_ids.device,
            )
            position_ids = (
                cache_position.view(1, 1, 1) + rope_deltas.view(1, 1, 1)
            )
            logits, states = self._decode_once(
                tokens[-1], position_ids, states
            )
            tokens.append(logits[:, -1:, :].argmax(dim=-1))
        return torch.cat(tokens, dim=1).contiguous(), logits


class Qwen3_5ExplicitDecodeStep(nn.Module):
    """Decode one token using a tensor step index and the prefill rope delta."""

    def __init__(
        self,
        decode: Qwen3_5ExplicitDecode,
        *,
        prompt_length: int,
    ):
        super().__init__()
        self.decode = decode
        self.prompt_length = prompt_length

    def forward(
        self,
        input_ids: torch.Tensor,
        rope_deltas: torch.Tensor,
        step: torch.Tensor,
        *states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache_position = self.prompt_length + step - 1
        position_ids = (
            cache_position.reshape(1, 1, 1) + rope_deltas.reshape(1, 1, 1)
        )
        return self.decode(
            input_ids, position_ids, cache_position, *states
        )


class QwenSeedTokens(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length

    def forward(
        self, first_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        remainder = torch.zeros(
            (1, self.sequence_length - 1),
            dtype=first_token.dtype,
            device=first_token.device,
        )
        tokens = torch.cat((first_token, remainder), dim=1).contiguous()
        step = torch.ones((1,), dtype=torch.int64, device=first_token.device)
        valid = torch.ones((1,), dtype=torch.bool, device=first_token.device)
        return tokens, step, valid


class QwenAppendToken(nn.Module):
    def forward(
        self,
        tokens: torch.Tensor,
        token: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        return torch.index_copy(tokens, 1, index, token).contiguous()


class QwenAdvanceStep(nn.Module):
    def forward(self, step: torch.Tensor) -> torch.Tensor:
        return (step + 1).contiguous()


class QwenSelectToken(nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1:, :].argmax(dim=-1).contiguous()


class QwenMakeDecodeState(nn.Module):
    def __init__(self, prompt_length: int):
        super().__init__()
        self.prompt_length = prompt_length

    def forward(
        self, rope_deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = torch.full(
            (1,),
            self.prompt_length,
            dtype=torch.int64,
            device=rope_deltas.device,
        )
        position_ids = (
            cache_position.reshape(1, 1, 1) + rope_deltas.reshape(1, 1, 1)
        ).expand(3, 1, 1).contiguous()
        return position_ids, cache_position


class QwenAdvanceDecode(nn.Module):
    def forward(
        self,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            (position_ids + 1).contiguous(),
            (cache_position + 1).contiguous(),
            (step + 1).contiguous(),
        )


def flatten_states(states: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(states)
