"""Deterministic interface examples, not pretrained model implementations."""

from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, StateSlot, Value
from vlaforge.ir.types import ScalarType, TensorType


VECTOR = TensorType((2,), "f32")
TOKEN = TensorType((1,), "i64")
FLAG = TensorType((1,), "bool")
INDEX = ScalarType("index")


@tensor_region("encode", inputs=(Value("observation", VECTOR),), outputs=(VECTOR,))
def encode(observation):
    return tuple(2 * x for x in observation)


@tensor_region(
    "flow_step",
    inputs=(
        Value("context", VECTOR),
        Value("sample", VECTOR),
        Value("velocity", VECTOR),
        Value("step", INDEX),
    ),
    outputs=(VECTOR, VECTOR),
)
def flow_step(context, sample, velocity, step):
    next_velocity = tuple(c + 0.125 * x for c, x in zip(context, sample))
    next_sample = tuple(
        x - 0.25 * v + 0.0625 * old
        for x, v, old in zip(sample, next_velocity, velocity)
    )
    return next_sample, next_velocity


def continuous_program(*, cache=True, steps=4):
    builder = InvocationBuilder(
        "continuous_example",
        inputs=(
            InputPort("observation", VECTOR),
            InputPort("noise", VECTOR),
            InputPort("accepted", FLAG),
        ),
        outputs=(OutputPort("action", VECTOR),),
    )
    observation = builder.input("observation")
    noise = builder.input("noise")
    accepted = builder.input("accepted")
    (context,) = builder.call(encode, observation, cache=cache)
    sample, _ = builder.iterate(
        (noise, noise),
        lambda step, sample, velocity: builder.call(
            flow_step,
            context,
            sample,
            velocity,
            step,
        ),
        steps=steps,
    )
    return builder.finish({"action": sample}, accepted=accepted)


@tensor_region(
    "merge_context",
    inputs=(Value("observation", VECTOR), Value("history", VECTOR)),
    outputs=(VECTOR,),
)
def merge_context(observation, history):
    return tuple(x + h for x, h in zip(observation, history))


@tensor_region(
    "token_step",
    inputs=(
        Value("context", VECTOR),
        Value("token", TOKEN),
        Value("kv", VECTOR),
        Value("step", INDEX),
    ),
    outputs=(TOKEN, VECTOR),
)
def token_step(context, token, kv, step):
    return (
        ((token[0] + step + 1) % 64,),
        tuple(k + c * 0.125 for k, c in zip(kv, context)),
    )


def autoregressive_program(*, steps=3):
    builder = InvocationBuilder(
        "autoregressive_example",
        inputs=(
            InputPort("observation", VECTOR),
            InputPort("token", TOKEN),
            InputPort("accepted", FLAG),
        ),
        outputs=(OutputPort("token", TOKEN), OutputPort("features", VECTOR)),
        states=(StateSlot("history", VECTOR),),
    )
    observation = builder.input("observation")
    token = builder.input("token")
    accepted = builder.input("accepted")
    history = builder.state("history")
    (context,) = builder.call(merge_context, observation, history, cache=True)
    token, kv = builder.iterate(
        (token, history),
        lambda step, token, kv: builder.call(token_step, context, token, kv, step),
        steps=steps,
    )
    builder.update_state("history", kv)
    return builder.finish({"token": token, "features": kv}, accepted=accepted)
