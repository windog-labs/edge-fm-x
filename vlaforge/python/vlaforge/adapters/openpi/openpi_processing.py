"""Pinned OpenPI RGB/state/text processing without constructing model weights."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

from vlaforge.adapters.openpi.openpi_assets import verify_openpi_tokenizer
from vlaforge.adapters.openpi.openpi_phased import _native_processor_context


@dataclass(frozen=True)
class PreparedOpenPIObservation:
    observation: object
    noise: object


class OpenPIProcessor:
    """Use the official transforms and GPU preprocessing on original observations.

    The raw input contains decoded RGB arrays, native state, text and saved
    noise. File/video decoding is outside this interface. GPU preprocessing is
    invoked by the native path; the official model invokes the same operation
    internally, so it is not duplicated in the official path.
    """

    def __init__(self, capture, *, device):
        data, stats, transforms, self.model_config = _native_processor_context(capture)
        self.tokenizer_identity = verify_openpi_tokenizer()
        self.device = device
        self._input = transforms.compose([
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ])
        self._output = transforms.compose([
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ])

    def prepare(self, observation, noise):
        import jax
        import numpy as np
        import torch

        model_types = importlib.import_module('openpi.models.model')
        transformed = self._input(jax.tree.map(lambda value: value, dict(observation)))
        batched = jax.tree.map(lambda value: torch.from_numpy(np.array(value)).to(self.device)[None], transformed)
        official = model_types.Observation.from_dict(batched)
        noise = torch.as_tensor(noise, device=self.device)
        if noise.ndim == 2:
            noise = noise[None]
        expected = (official.state.shape[0], self.model_config.action_horizon, self.model_config.action_dim)
        if noise.dtype != torch.float32 or tuple(noise.shape) != expected:
            raise ValueError('saved noise differs from the pinned action profile')
        return PreparedOpenPIObservation(official, noise)

    def session_inputs(self, prepared):
        preprocessing = importlib.import_module('openpi.models_pytorch.preprocessing_pytorch')
        observation = preprocessing.preprocess_observation_pytorch(prepared.observation, train=False)
        result = {f'image_{index}': value.contiguous() for index, value in enumerate(observation.images.values())}
        result.update({f'image_mask_{index}': value.contiguous() for index, value in enumerate(observation.image_masks.values())})
        result.update(language_tokens=observation.tokenized_prompt.contiguous(),
                      language_mask=observation.tokenized_prompt_mask.contiguous(),
                      state=observation.state.contiguous(), noise=prepared.noise.contiguous())
        return result

    def native_actions(self, prepared, normalized):
        """Return the official native action scale; robot calibration is separate."""
        return self._output({'state': prepared.observation.state[0].detach().cpu().numpy(),
                             'actions': normalized[0].detach().cpu().numpy()})['actions']
