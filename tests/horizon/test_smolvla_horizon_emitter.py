from pathlib import Path


def test_smolvla_adapter_passes_attention_mask_and_position_ids():
    project_root = Path(__file__).resolve().parent.parent.parent
    source = (project_root / "src" / "backends" / "horizon_module_emitter.cpp").read_text(
        encoding="utf-8"
    )

    assert "attention_mask=prefix_attention_mask.to(torch.bool)" in source
    assert "position_ids=prefix_position_ids.to(torch.long)" in source
    assert "attention_mask=denoise_attention_mask.to(torch.bool)" in source
    assert "position_ids=suffix_position_ids.to(torch.long)" in source
    assert "from lerobot.configs.policies import PreTrainedConfig" in source
    assert "config = PreTrainedConfig.from_pretrained(str(PREFILL_MODEL_PATH))" in source
    assert "torch.stack([k.contiguous(), v.contiguous()], dim=0).to(torch.float32)" in source
    assert "past_key_values = [_unpack_kv(kv, kv_dtype) for kv in packed_layers]" in source
    assert "return _extract_suffix_hidden(outputs).to(torch.float32)" in source
    assert 'EDGE_FM_SMOLVLA_EXPORT_DTYPE\\", \\"float32\\"' in source
    assert "self.policy.to(dtype=torch.float32)" in source
    assert "sum() * 0.0" not in source[source.index("std::string smolvla_horizon_module_source") :]
