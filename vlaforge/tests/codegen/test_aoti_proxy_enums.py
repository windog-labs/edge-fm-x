from __future__ import annotations

import os

import pytest


@pytest.mark.cuda_aoti
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_CUDA_AOTI") != "1",
    reason="set VLAFORGE_RUN_CUDA_AOTI=1 for actual AOTI proxy enum regression",
)
def test_proxy_package_preserves_dtype_layout_and_full_values(tmp_path):
    import torch
    from vlaforge.deployment.aoti_export import prepare_backend_options
    from vlaforge.deployment.aoti_package import normalize_proxy_enums
    from vlaforge.deployment.aoti_profile import aoti_configs

    if not torch.cuda.is_available():
        pytest.skip("actual CUDA required")

    class Probe(torch.nn.Module):
        def forward(self, values):
            torch.ops.aten._assert_tensor_metadata.default(
                values,
                dtype=torch.float32,
                device=torch.device("cuda:0"),
                layout=torch.strided,
            )
            index = torch.arange(
                values.numel(), device=values.device, dtype=torch.int64
            )
            return values.to(torch.bfloat16), index

    values = torch.tensor([0.0, -0.0, 0.12345, -0.71], device="cuda:0")
    exported = torch.export.export(Probe(), (values,))
    options, _ = prepare_backend_options(aoti_configs("aten-preserving"))
    original, translated = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    torch._inductor.aoti_compile_and_package(
        exported, package_path=str(original), inductor_configs=options
    )
    audit = normalize_proxy_enums(original, translated)
    assert {item["kind"] for item in audit["rewrites"]} >= {
        "as_scalar_type",
        "as_layout",
    }
    runner = torch._inductor.aoti_load_package(
        str(translated), run_single_threaded=True
    )
    for offset in (0.0, 0.125, -1.5):
        inputs = values + offset if offset else values
        actual, expected = runner(inputs), Probe()(inputs)
        torch.cuda.synchronize()
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert left.dtype == right.dtype
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
