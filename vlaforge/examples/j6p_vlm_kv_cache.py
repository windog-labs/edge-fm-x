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
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
        if hasattr(value, "numpy"):
            return value.numpy().tobytes()
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def input_identity(values: Mapping[str, Any]) -> str:
    """Hash all prompt/image/profile inputs used to create a prefill state."""
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(_bytes_for_identity(values[name]))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StateTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class CompleteHbmManifest:
    """ABI metadata for one complete compiled HBM graph."""

    stage: str
    hbm: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    state_schema: tuple[StateTensorSpec, ...]
    supports_custom_ops: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {"prefill", "decode"}:
            raise ValueError("stage must be prefill or decode")
        if self.supports_custom_ops:
            raise ValueError("J6P manifests must reject custom operators")
        if not self.hbm.is_file():
            raise FileNotFoundError(self.hbm)

    @classmethod
    def from_json(cls, path: Path) -> "CompleteHbmManifest":
        data = json.loads(path.read_text())
        state_schema = tuple(
            StateTensorSpec(
                name=str(item["name"]),
                shape=tuple(int(size) for size in item["shape"]),
                dtype=str(item["dtype"]),
            )
            for item in data.get("state_schema", ())
        )
        hbm = Path(data["hbm"])
        if not hbm.is_absolute():
            hbm = path.parent / hbm
        return cls(
            stage=str(data["stage"]),
            hbm=hbm,
            input_names=tuple(str(name) for name in data["inputs"]),
            output_names=tuple(str(name) for name in data["outputs"]),
            state_schema=state_schema,
            supports_custom_ops=bool(data.get("custom_ops", False)),
        )


@dataclass(frozen=True, slots=True)
class PrefillResult:
    """Result of vision plus text prefill.

    ``first_token`` is selected by the compiled provider.  Keeping argmax or
    sampling out of this Python loop makes the J6P timing boundary explicit.
    """

    first_token: Any
    rope_deltas: Any
    state: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class DecodeResult:
    next_token: Any
    state: tuple[Any, ...]


class J6PCompleteHbmProvider(Protocol):
    """The narrow seam implemented by a real J6P provider.

    ``prefill_manifest`` and ``decode_manifest`` each describe a complete HBM
    model.  They are not per-layer artifacts and are never invoked as custom
    operators from the other graph.
    """

    model_id: str
    max_sequence_length: int
    prefill_manifest: CompleteHbmManifest
    decode_manifest: CompleteHbmManifest

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
        step: int,
        rope_deltas: Any,
        state: tuple[Any, ...],
    ) -> DecodeResult:
        """Run exactly one token using the state returned by prefill/decode."""


@dataclass(frozen=True, slots=True)
class PreparedVLMInput:
    input_ids: Any
    attention_mask: Any
    pixel_values: Any
    image_grid_thw: Any
    mm_token_type_ids: Any
    prompt_length: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    tokens: tuple[Any, ...]
    cache_key: str
    prompt_length: int


class J6PVLMDeployment:
    """Small transactional control plane for one fixed-profile VLM Session."""

    def __init__(self, session: J6PCompleteHbmProvider):
        self.session = session
        self._committed_key: str | None = None

    def reset(self) -> None:
        self.session.reset()
        self._committed_key = None

    def _validate_profile(self, inputs: PreparedVLMInput, new_tokens: int) -> None:
        if inputs.prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if new_tokens < 1:
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
            self._validate_state(prefill.state)
            tokens = [prefill.first_token]
            state = prefill.state
            # Qwen3.5ExplicitDecodeStep uses step=1 for cache_position=prompt_length.
            for step in range(1, new_tokens):
                decoded = self.session.decode(
                    token=tokens[-1],
                    step=step,
                    rope_deltas=prefill.rope_deltas,
                    state=state,
                )
                self._validate_state(decoded.state)
                tokens.append(decoded.next_token)
                state = decoded.state
            self._committed_key = cache_key
            return GenerationResult(tuple(tokens), cache_key, inputs.prompt_length)
        except Exception:
            # A failed prefill/decode must not leave a state that can be reused.
            self.reset()
            raise

    def _validate_state(self, state: Sequence[Any]) -> None:
        expected = len(self.session.decode_manifest.state_schema)
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
        max_sequence_length: int,
    ):
        if prefill_manifest.stage != "prefill":
            raise ValueError("prefill_manifest must describe the prefill graph")
        if decode_manifest.stage != "decode":
            raise ValueError("decode_manifest must describe the decode graph")
        self.prefill_manifest = prefill_manifest
        self.decode_manifest = decode_manifest
        self.model_id = model_id
        self.max_sequence_length = max_sequence_length

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


def load_j6p_qwen35_08b_provider(
    *,
    prefill_manifest: CompleteHbmManifest,
    decode_manifest: CompleteHbmManifest,
) -> J6PCompleteHbmProviderImpl:
    """Load metadata for two complete HBM graphs after a J6P build exists."""
    return J6PCompleteHbmProviderImpl(
        prefill_manifest=prefill_manifest,
        decode_manifest=decode_manifest,
        model_id="Qwen3.5-0.8B",
        max_sequence_length=384,
    )


def main() -> None:
    """Show the provider contract without claiming a J6P run."""
    print(
        "This example supplies the Python control plane. A validated J6P "
        "prefill/decode provider is required before generate() can run."
    )


if __name__ == "__main__":
    main()
