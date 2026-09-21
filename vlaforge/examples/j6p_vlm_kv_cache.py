"""J6P VLM deployment-side orchestration with two complete HBM graphs.

This example is deliberately provider-neutral.  It shows the Python control
plane that a J6P BPU provider must satisfy; it does not call individual layers
or custom operators and it does not pretend that the current repository already
contains a validated J6P VLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


def _bytes_for_identity(value: Any) -> bytes:
    """Get stable bytes without importing a tensor framework in the example."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"bytes:" + bytes(value)
    if hasattr(value, "detach"):
        import torch

        value = value.detach().cpu().contiguous()
        # Byte view also supports BF16, which numpy() cannot encode directly.
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    elif hasattr(value, "tobytes"):
        raw = value.tobytes()
    else:
        return b"json:" + json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    metadata = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return b"tensor:" + len(metadata).to_bytes(8, "big") + metadata + raw


def input_identity(values: Mapping[str, Any]) -> str:
    """Diagnostic input digest, not permission to reuse mutated decode state."""
    digest = hashlib.sha256()
    for name in sorted(values):
        for part in (name.encode(), _bytes_for_identity(values[name])):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StateTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.layout:
            raise ValueError("state name, dtype and layout must be nonempty")
        if any(type(size) is not int or size < 1 for size in self.shape):
            raise ValueError("state dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class CompleteHbmManifest:
    """ABI metadata for one complete compiled HBM graph."""

    stage: str
    hbm: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_inputs: tuple[StateTensorSpec, ...]
    state_outputs: tuple[StateTensorSpec, ...]
    supports_custom_ops: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"prefill", "decode"}:
            raise ValueError("stage must be prefill or decode")
        if self.supports_custom_ops is not False:
            raise ValueError("J6P manifests must reject custom operators")
        for names in (self.input_names, self.output_names):
            if not names or any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("HBM port names must be nonempty and unique")
        for states, ports in ((self.state_inputs, self.input_names),
                              (self.state_outputs, self.output_names)):
            names = [item.name for item in states]
            if len(set(names)) != len(names) or not set(names) <= set(ports):
                raise ValueError("state ports must be unique declared HBM ports")
        if self.stage == "prefill" and self.state_inputs:
            raise ValueError("fresh-request prefill does not accept history state")
        if not self.state_outputs or (self.stage == "decode" and not self.state_inputs):
            raise ValueError("explicit state ports are required")
        if not self.hbm.is_file():
            raise FileNotFoundError(self.hbm)

    @classmethod
    def from_json(cls, path: Path) -> "CompleteHbmManifest":
        data = json.loads(path.read_text())
        def states(field):
            return tuple(StateTensorSpec(
                name=item["name"], shape=tuple(item["shape"]),
                dtype=item["dtype"], layout=item["layout"],
            ) for item in data[field])
        hbm = Path(data["hbm"])
        if not hbm.is_absolute():
            hbm = path.parent / hbm
        return cls(
            stage=str(data["stage"]),
            hbm=hbm,
            input_names=tuple(str(name) for name in data["inputs"]),
            output_names=tuple(str(name) for name in data["outputs"]),
            state_inputs=states("state_inputs"),
            state_outputs=states("state_outputs"),
            supports_custom_ops=data["custom_ops"],
        )


def state_order(source: tuple[StateTensorSpec, ...], target: tuple[StateTensorSpec, ...],
                bindings: Mapping[str, str]) -> tuple[int, ...]:
    """Map destination port -> source port; only direct compatible binding."""
    if set(bindings) != {item.name for item in target}:
        raise ValueError("every decode state input needs an explicit binding")
    sources = {item.name: (index, item) for index, item in enumerate(source)}
    if (len(bindings) != len(source) or len(set(bindings.values())) != len(source)
            or set(bindings.values()) != set(sources)):
        raise ValueError("state bindings must use every source exactly once")
    order = []
    for port in target:
        index, incoming = sources[bindings[port.name]]
        if (incoming.shape, incoming.dtype, incoming.layout) != (port.shape, port.dtype, port.layout):
            raise ValueError("state ABI mismatch: export matching ports or explicitly repack in the provider")
        order.append(index)
    return tuple(order)


@dataclass(frozen=True, slots=True)
class PrefillResult:
    """Result of vision plus text prefill.

    ``first_token`` is selected by the compiled provider.  Keeping argmax or
    sampling out of this Python loop makes the J6P timing boundary explicit.
    """

    first_token: int
    rope_deltas: Any
    state: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class DecodeResult:
    next_token: int
    state: tuple[Any, ...]


class J6PCompleteHbmProvider(Protocol):
    """The narrow seam implemented by a real J6P provider.

    ``prefill_manifest`` and ``decode_manifest`` each describe a complete HBM
    model.  They are not per-layer artifacts and are never invoked as custom
    operators from the other graph.
    """

    model_id: str
    prompt_length: int
    max_sequence_length: int
    prefill_manifest: CompleteHbmManifest
    decode_manifest: CompleteHbmManifest
    prefill_to_decode: Mapping[str, str]
    decode_to_decode: Mapping[str, str]

    def reset(self) -> None:
        """Discard device-side state and staged outputs for this Session."""

    def prefill(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        pixel_values: Any,
        image_grid_thw: Any,
        mm_token_type_ids: Any,
    ) -> PrefillResult:
        """Run vision and text prefill and return the initial explicit state."""

    def decode(
        self,
        *,
        token: Any,
        cache_position: int,
        rope_deltas: Any,
        state: tuple[Any, ...],
    ) -> DecodeResult:
        """One complete HBM call; build RoPE/mask from cache_position.

        State buffers remain valid until completion; return an owned Python int
        token, not a view of a reused device output buffer. The provider checks
        the physical tensors against its manifest, including quantization ABI.
        """


@dataclass(frozen=True, slots=True)
class PreparedVLMInput:
    input_ids: Any
    attention_mask: Any
    pixel_values: Any
    image_grid_thw: Any
    mm_token_type_ids: Any
    prompt_length: int


def prepare_qwen35(processor: Any, image: Any, prompt: str) -> PreparedVLMInput:
    """Reuse the checkpoint processor; no padding or manual vision transforms."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}], tokenize=False, add_generation_prompt=True,
    )
    values = processor(text=[text], images=[image], return_tensors="pt")
    ids = values["input_ids"]
    mask = values["attention_mask"]
    if len(ids.shape) != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError("example requires a single unpadded prompt")
    if not bool((mask == 1).all()):
        raise ValueError("padded prompts require a separately validated prefill graph")
    return PreparedVLMInput(
        input_ids=ids, attention_mask=mask, pixel_values=values["pixel_values"],
        image_grid_thw=values["image_grid_thw"],
        mm_token_type_ids=values["mm_token_type_ids"], prompt_length=int(ids.shape[1]),
    )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    tokens: tuple[int, ...]
    cache_key: str
    prompt_length: int


class J6PVLMDeployment:
    """Fresh-request orchestration; failure invalidates state, not rollback."""

    def __init__(self, session: J6PCompleteHbmProvider):
        self.session = session
        self._committed_key: str | None = None
        self._prefill_order = state_order(
            session.prefill_manifest.state_outputs, session.decode_manifest.state_inputs,
            session.prefill_to_decode,
        )
        self._decode_order = state_order(
            session.decode_manifest.state_outputs, session.decode_manifest.state_inputs,
            session.decode_to_decode,
        )

    def reset(self) -> None:
        self._committed_key = None
        self.session.reset()

    def _validate_profile(self, inputs: PreparedVLMInput, new_tokens: int) -> None:
        if type(inputs.prompt_length) is not int or inputs.prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if inputs.prompt_length != self.session.prompt_length:
            raise ValueError("prompt_length differs from the compiled profile")
        if type(new_tokens) is not int or new_tokens < 1:
            raise ValueError("new_tokens must be positive")
        if inputs.prompt_length + new_tokens - 1 > self.session.max_sequence_length:
            raise ValueError(
                "prompt plus generated tokens exceed the compiled cache capacity"
            )

    def generate(self, inputs: PreparedVLMInput, *, new_tokens: int) -> GenerationResult:
        """Start a new image+prompt episode and greedily decode a fixed length."""
        self._validate_profile(inputs, new_tokens)
        self.reset()
        cache_key = input_identity(
            {
                "model_id": self.session.model_id,
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "pixel_values": inputs.pixel_values,
                "image_grid_thw": inputs.image_grid_thw,
                "mm_token_type_ids": inputs.mm_token_type_ids,
                "prompt_length": inputs.prompt_length,
                "new_tokens": new_tokens,
            }
        )
        try:
            prefill = self.session.prefill(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                mm_token_type_ids=inputs.mm_token_type_ids,
            )
            self._validate_state(prefill.state, self.session.prefill_manifest.state_outputs)
            self._validate_token(prefill.first_token)
            tokens = [prefill.first_token]
            state = tuple(prefill.state[i] for i in self._prefill_order)
            # Prefill already emitted token 1; N outputs require N-1 decode calls.
            for step in range(1, new_tokens):
                decoded = self.session.decode(
                    token=tokens[-1],
                    cache_position=inputs.prompt_length + step - 1,
                    rope_deltas=prefill.rope_deltas,
                    state=state,
                )
                self._validate_state(decoded.state, self.session.decode_manifest.state_outputs)
                self._validate_token(decoded.next_token)
                tokens.append(decoded.next_token)
                state = tuple(decoded.state[i] for i in self._decode_order)
            self._committed_key = cache_key
            return GenerationResult(tuple(tokens), cache_key, inputs.prompt_length)
        except BaseException:
            # A failed prefill/decode must not leave a state that can be reused.
            self.reset()
            raise

    @staticmethod
    def _validate_token(token: int) -> None:
        if type(token) is not int or token < 0:
            raise ValueError("provider must return an owned nonnegative integer token")

    @staticmethod
    def _validate_state(state: Sequence[Any], schema: tuple[StateTensorSpec, ...]) -> None:
        expected = len(schema)
        if len(state) != expected:
            raise ValueError(f"provider returned {len(state)} states; expected {expected}")
        if any(value is None for value in state):
            raise ValueError("provider returned an empty cache tensor")


class J6PCompleteHbmProviderImpl:
    """Integration point for two complete, standard-op J6P HBM graphs."""

    def __init__(
        self,
        *,
        prefill_manifest: CompleteHbmManifest,
        decode_manifest: CompleteHbmManifest,
        model_id: str,
        prompt_length: int,
        max_sequence_length: int,
        prefill_to_decode: Mapping[str, str],
        decode_to_decode: Mapping[str, str],
    ):
        if prefill_manifest.stage != "prefill":
            raise ValueError("prefill_manifest must describe the prefill graph")
        if decode_manifest.stage != "decode":
            raise ValueError("decode_manifest must describe the decode graph")
        self.prefill_manifest = prefill_manifest
        self.decode_manifest = decode_manifest
        self.model_id = model_id
        if (type(prompt_length) is not int or type(max_sequence_length) is not int
                or not 1 <= prompt_length <= max_sequence_length):
            raise ValueError("invalid compiled sequence profile")
        self.prompt_length = prompt_length
        self.max_sequence_length = max_sequence_length
        self.prefill_to_decode = dict(prefill_to_decode)
        self.decode_to_decode = dict(decode_to_decode)

    def reset(self) -> None:
        raise NotImplementedError("connect reset to the J6P Session/provider implementation")

    def prefill(self, **_: Any) -> PrefillResult:
        raise NotImplementedError(
            "load the compiled vision+prefill HBM and bind its typed I/O here"
        )

    def decode(self, **_: Any) -> DecodeResult:
        raise NotImplementedError(
            "load the compiled one-token decode HBM and bind explicit state here"
        )


def load_complete_hbm_provider(
    *,
    prefill_manifest: CompleteHbmManifest,
    decode_manifest: CompleteHbmManifest,
    model_id: str,
    prompt_length: int,
    max_sequence_length: int,
    prefill_to_decode: Mapping[str, str],
    decode_to_decode: Mapping[str, str],
) -> J6PCompleteHbmProviderImpl:
    """Load metadata for two complete HBM graphs after a J6P build exists."""
    return J6PCompleteHbmProviderImpl(
        prefill_manifest=prefill_manifest,
        decode_manifest=decode_manifest,
        model_id=model_id,
        prompt_length=prompt_length,
        max_sequence_length=max_sequence_length,
        prefill_to_decode=prefill_to_decode,
        decode_to_decode=decode_to_decode,
    )


def main() -> None:
    """Show the provider contract without claiming a J6P run."""
    print(
        "This example supplies the Python control plane. A validated J6P "
        "prefill/decode provider is required before generate() can run."
    )


if __name__ == "__main__":
    main()
