from vlaforge.adapters.diffusiondrive_real import (
    DIFFUSIONDRIVE_UPSTREAM_REVISION,
    build_real_diffusiondrive_program,
)
from vlaforge.analysis import verify
from vlaforge.compiler import compile_module


def test_real_diffusiondrive_program_is_generic_and_cache_scoped():
    module = build_real_diffusiondrive_program()
    assert verify(module, raise_on_error=False) == ()
    assert module.name == "diffusiondrive_real_cuda_l4"
    assert module.states == ()
    assert tuple(port.name for port in module.outputs) == (
        "candidate_trajectories",
        "candidate_scores",
        "trajectory",
        "bev_semantic_map",
        "agent_states",
        "agent_labels",
    )
    invocation = module.invocations[0]
    assert invocation.metadata["source_revision"] == (
        DIFFUSIONDRIVE_UPSTREAM_REVISION
    )
    assert invocation.metadata["core_op_delta"] == 0

    compilation = compile_module(
        module, default_device="cuda:0", state_device="cuda:0"
    )
    assert compilation.plan.arena is not None
    assert compilation.plan.arena.device == "cuda:0"
    assert len(compilation.certificate.caches) == 1
    cache = compilation.certificate.caches[0]
    assert cache.region == "condition_encoder"
    assert cache.input_ids == (0, 1, 2)
    assert cache.state_ids == ()
