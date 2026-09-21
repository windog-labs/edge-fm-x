"""Tensor-carry lowering of a fixed, audited diffusers DPMSolver++ profile.

The arithmetic graphs come from the installed official scheduler primitives.
Only CPU scalar constants become device-indexed tables; no solver equation is
reimplemented here. Other scheduler profiles must provide their own lowering.
"""

from __future__ import annotations

import copy
import importlib.metadata


def make_dpm_solver_step(scheduler, steps: int, *, device):
    import torch
    from diffusers import DPMSolverMultistepScheduler

    if importlib.metadata.version("diffusers") != "0.27.2":
        raise ValueError("the audited primitive graph requires diffusers 0.27.2")
    if type(scheduler) is not DPMSolverMultistepScheduler or type(steps) is not int or steps < 3:
        raise ValueError("expected a fixed DPMSolverMultistepScheduler with at least three steps")
    expected = {
        "prediction_type": "sample", "algorithm_type": "dpmsolver++",
        "solver_order": 2, "solver_type": "midpoint", "thresholding": False,
        "lower_order_final": True, "euler_at_final": False, "final_sigmas_type": "zero",
    }
    if any(getattr(scheduler.config, key) != value for key, value in expected.items()):
        raise ValueError("scheduler configuration is outside the audited deterministic profile")
    fixed = type(scheduler).from_config(dict(scheduler.config))
    fixed.set_timesteps(steps)

    class Primitive(torch.nn.Module):
        def __init__(self, index, order):
            super().__init__()
            self.scheduler = copy.deepcopy(fixed)
            self.scheduler._step_index = index
            self.order = order

        def forward(self, sample, current, previous):
            sample = sample.float()
            current = self.scheduler.convert_model_output(current, sample=sample)
            if self.order == 1:
                result = self.scheduler.dpm_solver_first_order_update(current, sample=sample)
            else:
                result = self.scheduler.multistep_dpm_solver_second_order_update(
                    [previous, current], sample=sample,
                )
            return result.to(current.dtype)

    def indexed_primitive(order):
        graphs = [torch.fx.symbolic_trace(Primitive(
            index if order == 1 else min(max(index, 1), steps - 2), order,
        )) for index in range(steps)]
        template = graphs[0]
        signatures = [tuple((node.op, str(node.target), str(node.args), str(node.kwargs))
                            for node in graph.graph.nodes) for graph in graphs]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("official primitive graph topology changes across the fixed schedule")
        graph = template.graph
        first = next(iter(graph.nodes))
        with graph.inserting_before(first):
            index_node = graph.placeholder("device_step_index")
        constant_names = [node.target for node in graph.nodes if node.op == "get_attr"]
        for name in dict.fromkeys(constant_names):
            values = [getattr(item, name) for item in graphs]
            if any(value.device.type != "cpu" or value.ndim != 0 or not torch.isfinite(value)
                   for value in values):
                raise ValueError("primitive constants must be finite CPU scalars")
            table_name = f"{name}_by_step"
            template.register_buffer(table_name, torch.stack(values).to(device))
            for node in tuple(graph.nodes):
                if node.op != "get_attr" or node.target != name:
                    continue
                with graph.inserting_before(node):
                    table = graph.get_attr(table_name)
                    selected = graph.call_method("index_select", (table, 0, index_node))
                    scalar = graph.call_method("reshape", (selected, ()))
                if torch.device(device).type == "cuda":
                    # CUDA's CPU-scalar fast path keeps FP32 opmath even when
                    # the other operand/output is BF16. A CUDA 0D tensor instead
                    # gets rounded to BF16 before multiplication. Preserve the
                    # original fast-path rounding without an in-loop H2D scalar.
                    for user in tuple(node.users):
                        if user.op != "call_method" or user.target != "mul" or user.args[0] is not node or len(user.args) != 2:
                            raise ValueError("unrecognized CPU-scalar primitive arithmetic")
                        operand = user.args[1]
                        with graph.inserting_before(user):
                            dtype = graph.call_function(getattr, (operand, "dtype"))
                            opmath = graph.call_method("float", (operand,))
                            product = graph.call_method("mul", (scalar, opmath))
                            rounded = graph.call_method("to", (product, dtype))
                        user.replace_all_uses_with(rounded)
                        graph.erase_node(user)
                node.replace_all_uses_with(scalar)
                graph.erase_node(node)
            delattr(template, name)
        graph.lint()
        template.recompile()
        return template

    first_order, second_order = indexed_primitive(1), indexed_primitive(2)

    class TensorCarrySolver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_order = first_order
            self.second_order = second_order
            self.register_buffer("timesteps", fixed.timesteps.to(device))
            self.register_buffer("sigmas", fixed.sigmas.to(device))

        def forward(self, sample, current, older, previous, valid_count, step_index):
            first = self.first_order(step_index, sample, current, previous)
            second = self.second_order(step_index, sample, current, previous)
            first_selected = (valid_count < 1) | (step_index == steps - 1)
            result = torch.where(first_selected.reshape(1, 1, 1), first, second)
            # Order two consumes previous/current; older is retained as explicit
            # scheduler state so the pair corresponds to model_outputs exactly.
            return result, previous.clone(), current.clone(), torch.clamp(valid_count + 1, max=2), step_index + 1

    return TensorCarrySolver().eval()
