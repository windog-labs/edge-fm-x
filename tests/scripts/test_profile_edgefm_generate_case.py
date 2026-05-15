from scripts.profile import profile_edgefm_generate_case as profile


def test_add_edgefm_json_contract_adds_owner_a_decode_breakdown():
    result = {
        "generated_counts": [4, 4],
        "stage_metrics": [
            {
                "prefill_ms": 10.0,
                "decode_ms": 8.0,
                "decode_step_avg_ms": 2.0,
                "tokens_per_second": 200.0,
                "decode_tokens_per_second": 150.0,
                "executed_generated_tokens_total": 4.0,
                "returned_generated_tokens_total": 4.0,
                "cuda_graph_enabled": 0.0,
                "lm_head_top1_enabled": 0.0,
                "lm_head_top1_decode_steps": 0.0,
                "decode_model_ms": 5.0,
                "decode_sampler_ms": 2.0,
                "decode_finalize_ms": 1.0,
                "decode_graph_replay_ms": 0.0,
            },
            {
                "prefill_ms": 12.0,
                "decode_ms": 10.0,
                "decode_step_avg_ms": 2.5,
                "tokens_per_second": 180.0,
                "decode_tokens_per_second": 140.0,
                "executed_generated_tokens_total": 4.0,
                "returned_generated_tokens_total": 4.0,
                "cuda_graph_enabled": 0.0,
                "lm_head_top1_enabled": 0.0,
                "lm_head_top1_decode_steps": 0.0,
                "decode_model_ms": 6.0,
                "decode_sampler_ms": 3.0,
                "decode_finalize_ms": 1.0,
                "decode_graph_replay_ms": 0.0,
            },
        ],
    }

    out = profile.add_edgefm_json_contract(result, expected_generated_tokens=4)

    breakdown = out["owner_a_decode_breakdown"]
    assert breakdown["decode_model_including_lm_head_ms"] == 5.5
    assert breakdown["decode_sampler_ms"] == 2.5
    assert breakdown["decode_finalize_ms"] == 1.0
    assert breakdown["full_logits_default"] is True
    assert breakdown["lm_head_top1_status"] == "available_default_off"


def test_format_owner_a_breakdown_names_full_logits_decision_gate():
    text = profile.format_owner_a_decode_breakdown({
        "decode_model_including_lm_head_ms": 5.5,
        "decode_sampler_ms": 2.5,
        "decode_finalize_ms": 1.0,
        "decode_graph_replay_ms": 0.0,
        "decode_ms": 9.0,
        "model_pct": 61.111,
        "sampler_pct": 27.777,
        "finalize_pct": 11.111,
        "full_logits_default": True,
        "lm_head_top1_status": "available_default_off",
    })

    assert "decode model+lm_head" in text
    assert "full_logits_default=True" in text
    assert "lm_head_top1=available_default_off" in text


def test_owner_a_breakdown_reports_enabled_lm_head_top1():
    breakdown = profile.build_owner_a_decode_breakdown([
        {
            "decode_ms": 4.0,
            "decode_model_ms": 3.0,
            "decode_sampler_ms": 0.0,
            "decode_finalize_ms": 1.0,
            "decode_graph_replay_ms": 0.0,
            "lm_head_top1_enabled": 1.0,
            "lm_head_top1_decode_steps": 4.0,
        }
    ])

    assert breakdown["full_logits_default"] is False
    assert breakdown["lm_head_top1_status"] == "enabled_experimental"
    assert breakdown["lm_head_top1_decode_steps"] == 4.0
