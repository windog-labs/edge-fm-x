"""OpenPI binding of the shared checked pure-Tensor output-stage interface."""

from vlaforge.adapters.shared.output_stage import OutputStageInput, attach_checked_output_stage


def attach_output_stage(module, region, *, state_input="state",
                        source_output="normalized_action_chunk",
                        output_name="native_action_chunk"):
    """Bind (state, normalized, incoming acceptance) without dropping a guard."""
    return attach_checked_output_stage(module, region,
        inputs=(OutputStageInput("module-input", state_input), OutputStageInput("source-output", source_output),
                OutputStageInput("acceptance")), source_output=source_output, output_name=output_name)
