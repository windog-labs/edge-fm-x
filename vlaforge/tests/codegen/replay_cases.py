"""Small model-independent Tensor programs for generated replay validation."""

from __future__ import annotations

import math

from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, StateSlot, Value
from vlaforge.ir.types import TensorType


def make_case(kind: str, replay: str, device: str = "cuda:0"):
    import torch

    position = TensorType((1,), "i64")
    flag = TensorType((1,), "bool")
    action = TensorType((2, 3), "f32")
    cache = TensorType((1, 4, 8), "f32")
    hidden = TensorType((1, 8), "f32")
    payload = action if kind == "continuous" else cache
    condition_type = action if kind == "continuous" else hidden

    class Denoise(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.full((2, 3), 0.1))

        def forward(self, values, condition, step):
            return values.sin() + condition * self.scale + step.to(values.dtype) * 0.01

    class Euler(torch.nn.Module):
        def forward(self, values, velocity, step):
            return values - velocity * 0.2, step + 1

    class Decode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.full((1, 8), 0.01))

        def forward(self, cache, token, position, condition):
            return (
                condition
                + cache.sum(dim=1) * self.scale
                + token.to(cache.dtype).reshape(1, 1) * 0.02
                + position.to(cache.dtype).reshape(1, 1) * 0.03
            )

    class Append(torch.nn.Module):
        def forward(self, cache, hidden, position):
            mask = (torch.arange(4, device=cache.device) == position).reshape(1, 4, 1)
            return (
                cache + hidden[:, None, :] * mask,
                hidden.argmax(dim=-1),
                position + 1,
            )

    def annotate(name, model, inputs, outputs):
        return tensor_region(
            name, inputs=(Value(key, value) for key, value in inputs), outputs=outputs
        )(model.eval().to(device))

    if kind == "continuous":
        first = annotate(
            "tensor_transform",
            Denoise(),
            (("values", action), ("condition", action), ("step", position)),
            (action,),
        )
        second = annotate(
            "tensor_update",
            Euler(),
            (("values", action), ("velocity", action), ("step", position)),
            (action, position),
        )
        states = ()
        ports = (InputPort("seed", action, device=device),)
        steps = 4
    else:
        first = annotate(
            "tensor_transform",
            Decode(),
            (
                ("cache", cache),
                ("token", position),
                ("position", position),
                ("condition", hidden),
            ),
            (hidden,),
        )
        second = annotate(
            "tensor_update",
            Append(),
            (("cache", cache), ("hidden", hidden), ("position", position)),
            (cache, position, position),
        )
        states = (StateSlot("history", cache, retention=2),)
        ports = (InputPort("token", position, device=device),)
        steps = 3
    builder = InvocationBuilder(
        "generic_" + kind,
        inputs=(
            *ports,
            InputPort("position", position, device=device),
            InputPort("condition", condition_type, device=device),
            InputPort("accepted", flag),
        ),
        outputs=(OutputPort("result", payload, group="result", device=device),),
        states=states,
    )
    pos = builder.input("position")
    condition = builder.input("condition")
    accepted = builder.input("accepted")
    if kind == "continuous":
        seed = builder.input("seed")

        def body(index, values, step):
            (velocity,) = builder.call(first, values, condition, step)
            return builder.call(second, values, velocity, step)

        result, _ = builder.iterate((seed, pos), body, steps=steps, replay=replay)
    else:
        token = builder.input("token")
        state = builder.state("history")

        def body(index, cache, token, position):
            (hidden,) = builder.call(first, cache, token, position, condition)
            return builder.call(second, cache, hidden, position)

        result, _, _ = builder.iterate(
            (state, token, pos), body, steps=steps, replay=replay
        )
        builder.update_state("history", result)
    program = builder.finish({"result": result}, accepted=accepted)
    state = torch.zeros(payload.shape, device=device)
    samples = []
    expected = []
    examples = None
    for run in range(4):
        if run == 3:
            state = torch.zeros_like(state)
        inputs = {
            "position": torch.zeros((1,), dtype=torch.int64, device=device),
            "condition": torch.arange(
                math.prod(condition_type.shape), dtype=torch.float32, device=device
            ).reshape(condition_type.shape)
            * 0.03
            + 0.01 * run,
            "accepted": torch.tensor([run != 1], dtype=torch.bool),
        }
        if kind == "continuous":
            inputs["seed"] = (
                torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3) * 0.05
                + run * 0.01
            )
            current, step = inputs["seed"], inputs["position"]
            for _ in range(steps):
                args = (current, inputs["condition"], step)
                velocity = first(*args)
                update_args = (current, velocity, step)
                if examples is None:
                    examples = {"tensor_transform": args, "tensor_update": update_args}
                current, step = second(*update_args)
        else:
            inputs["token"] = torch.tensor([run + 1], dtype=torch.int64, device=device)
            current, token, pos = state, inputs["token"], inputs["position"]
            for _ in range(steps):
                args = (current, token, pos, inputs["condition"])
                hidden_value = first(*args)
                update_args = (current, hidden_value, pos)
                if examples is None:
                    examples = {"tensor_transform": args, "tensor_update": update_args}
                current, token, pos = second(*update_args)
            if run != 1:
                state = current
        samples.append(inputs)
        expected.append(current if run != 1 else expected[-1])
    return program, examples, samples, expected, steps
