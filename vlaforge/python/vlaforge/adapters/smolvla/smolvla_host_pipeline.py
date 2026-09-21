"""Original recorded SmolVLA observations through official/native host pipelines."""

import hashlib
import json
from pathlib import Path
import sys

from vlaforge.adapters.smolvla.smolvla_processing import prepare_smolvla_statistics, resolve_smolvla_statistics
from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext
from vlaforge.validation.host_pipeline import digest
from vlaforge.validation.native_session import NativeTensorSession


def _read(path):
    return json.loads(Path(path).read_text())


def load_raw_records(path, expected_sha256):
    """Load hash-bound numerical NPZ arrays and explicit text without pickle."""
    import numpy as np

    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError('original observation manifest identity differs')
    manifest = _read(path)
    if manifest['schema'] != 'edgefm.recorded_lerobot_inputs/1' or not manifest['records']:
        raise ValueError('original recorded LeRobot observations required')
    samples = []
    for index, item in enumerate(manifest['records']):
        source = (path.parent / item['path']).resolve(strict=True)
        if item['index'] != index or not source.is_relative_to(path.parent.resolve()) or digest(source) != item['sha256']:
            raise ValueError('original observation path, order or file digest differs')
        with np.load(source, allow_pickle=False) as storage:
            arrays = {name: storage[name].copy() for name in storage.files}
        if set(arrays) != set(item['arrays']):
            raise ValueError('original observation array set differs')
        for name, value in arrays.items():
            actual = {'shape': list(value.shape), 'dtype': str(value.dtype),
                      'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
            if actual != item['arrays'][name] or value.dtype.kind not in 'buif' or not np.isfinite(value).all():
                raise ValueError('original observation array identity differs: ' + name)
        if not isinstance(item['text'], dict) or any(not isinstance(value, str) for value in item['text'].values()):
            raise ValueError('original text must be explicit strings')
        samples.append((arrays, dict(item['text'])))
    return manifest, samples


class SmolVLAHostPipeline:
    def __init__(self, config):
        root = Path(config['original_root'])
        for name, expected in config['original_files_sha256'].items():
            if digest(root / name) != expected:
                raise ValueError('SmolVLA original evidence differs: ' + name)
        setup = _read(root / 'local-setup.json')
        upstream = root / 'upstream'
        for name, expected in setup['upstream_files'].items():
            if digest(upstream / name) != expected:
                raise ValueError('pinned LeRobot source differs: ' + name)
        vlm = Path(setup['execution_base']) / 'assets/smolvla/SmolVLM2-500M-Video-Instruct'
        for name, expected in setup['vlm_files'].items():
            if digest(vlm / name) != expected:
                raise ValueError('pinned vision/language assets differ: ' + name)
        sys.path.insert(0, str(upstream / 'src'))
        import lerobot
        if not Path(lerobot.__file__).resolve().is_relative_to(upstream.resolve()):
            raise ValueError('another LeRobot implementation was imported')
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla import processor_smolvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep, UnnormalizerProcessorStep
        from lerobot.processor.converters import batch_to_transition, transition_to_batch, policy_action_to_transition, transition_to_policy_action
        from safetensors.torch import load_file
        from vlaforge.adapters.smolvla.smolvla_migration import verify_recovered_profile

        self.capture = _read(root / 'capture-001/capture.json')
        self.processed = _read(root / 'input-pack/manifest.json')
        policy = root / 'policy'
        if digest(policy / 'model.safetensors') != self.processed['checkpoint_sha256']:
            raise ValueError('checkpoint differs from the original input profile')
        for name, expected in self.processed['processor_files'].items():
            if digest(policy / name) != expected:
                raise ValueError('checkpoint processor identity differs')
        self.robot_type = self.processed['normalization']['selection']['robot_type']
        if verify_recovered_profile(policy, robot_type=self.robot_type) != self.processed['statistics_recovery']:
            raise ValueError('recovered statistics profile differs')
        self.raw, self.samples = load_raw_records(config['raw_inputs'], config['raw_inputs_sha256'])
        if self.raw['source_processed_manifest_sha256'] != digest(root / 'input-pack/manifest.json'):
            raise ValueError('raw observations were selected from another input profile')
        if len(self.samples) != len(self.processed['records']):
            raise ValueError('raw/reference observation counts differ')
        for raw, original in zip(self.raw['records'], self.processed['records'], strict=True):
            if any(raw[key] != original[key] for key in ('dataset_index', 'episode_index', 'frame_index')):
                raise ValueError('raw/reference frame order differs')
        self.camera_mapping = self.processed['camera_mapping']
        profile, _ = prepare_smolvla_statistics(policy, mode='strict-statistics', namespace=None, robot_type=self.robot_type)
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(str(policy), config_filename='policy_preprocessor.json',
            overrides={'rename_observations_processor': {'rename_map': self.camera_mapping},
                       'tokenizer_processor': {'tokenizer_name': str(vlm)}, 'device_processor': {'device': 'cpu'},
                       'normalizer_processor': {'stats': profile.stats}},
            to_transition=batch_to_transition, to_output=transition_to_batch)
        normalizers = [step for step in self.preprocessor.steps if isinstance(step, NormalizerProcessorStep)]
        if len(normalizers) != 1:
            raise ValueError('official normalizer structure changed')
        profile.require_consumed(normalizers[0])
        self.statistics = profile
        declaration = _read(policy / 'policy_postprocessor.json')['steps'][0]
        stats_path = (policy / declaration['state_file']).resolve(strict=True)
        if not stats_path.is_relative_to(policy.resolve()):
            raise ValueError('postprocessor statistics escape checkpoint')
        self.action_statistics = resolve_smolvla_statistics(load_file(str(stats_path)), namespace=None,
            robot_type=self.robot_type, feature_shapes={'action': tuple(declaration['config']['features']['action']['shape'])},
            source_sha256=digest(stats_path))
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(str(policy), config_filename='policy_postprocessor.json',
            overrides={'unnormalizer_processor': {'stats': self.action_statistics.stats}},
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        unnormalizers = [step for step in self.postprocessor.steps if isinstance(step, UnnormalizerProcessorStep)]
        if len(unnormalizers) != 1:
            raise ValueError('official unnormalizer structure changed')
        self.action_statistics.require_consumed(unnormalizers[0])
        self.batch_keys = {f'image_{index}': name for index, name in enumerate(self.capture['camera_keys'])}
        self.batch_keys.update({f'image_mask_{index}': name + '_padding_mask' for index, name in enumerate(self.capture['camera_keys'])})
        self.batch_keys.update(state='observation.state', instruction_tokens='observation.language.tokens',
                               instruction_mask='observation.language.attention_mask')
        self.native, self.model, self.loops = None, None, []
        self.backend = config['backend']
        NumericalContext.from_dict(_read(root / 'complete-001/direct.json')['numerical_context']).require_current()
        if self.backend == 'official-pytorch':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            options = PreTrainedConfig.from_pretrained(policy, local_files_only=True)
            options.vlm_model_name, options.device, options.num_steps = str(vlm), 'cuda:0', self.capture['num_steps']
            self.model = SmolVLAPolicy.from_pretrained(policy, config=options, local_files_only=True, strict=True).eval()
        elif self.backend == 'generated-session':
            if digest(config['library_receipt']) != config['library_receipt_sha256']:
                raise ValueError('native build receipt identity differs')
            receipt = _read(config['library_receipt'])
            if receipt['status'] != 'built_not_executed' or digest(config['cuda_runtime']) != config['cuda_runtime_sha256']:
                raise ValueError('native build or CUDA runtime differs')
            bundle = Path(receipt['bundle'])
            manifest = load_bundle_manifest(bundle / 'bundle.json')
            manifest.verify_files(bundle)
            selections = [item for item in manifest.generated_sources if item.role == 'loop_execution']
            if len(selections) != 1:
                raise ValueError('explicit native loop selection required')
            self.loops = _read(bundle / selections[0].path)['loops']
            if any(loop['policy'] != config['policy'] for loop in self.loops):
                raise ValueError('compiled loop policy differs')
            module = parse_canonical_json((bundle / manifest.semantic_ir.path).read_text())
            self.native = NativeTensorSession(receipt['library'], module, library_sha256=receipt['library_sha256'],
                bundle=bundle, bundle_sha256=receipt['bundle_sha256'], cuda_runtime=config['cuda_runtime'])
        else:
            raise ValueError('unsupported SmolVLA host backend')

    def prepare(self, sample):
        import torch

        arrays, text = sample
        observation = {name: torch.from_numpy(arrays[name]).float() / 255 for name in self.camera_mapping}
        observation.update({'observation.state': torch.from_numpy(arrays['observation.state'].copy()), **text})
        batch = self.preprocessor(observation)
        batch = {name: value.to('cuda:0').contiguous() for name, value in batch.items() if isinstance(value, torch.Tensor)}
        for name in self.camera_mapping.values():
            batch[name + '_padding_mask'] = torch.ones((1,), dtype=torch.bool, device='cuda:0')
        noise = torch.from_numpy(arrays['noise']).to('cuda:0').contiguous()
        return batch, noise

    def _inputs(self, prepared):
        batch, noise = prepared
        return {**{name: batch[key] for name, key in self.batch_keys.items()}, 'noise': noise}

    def infer(self, prepared):
        if self.native:
            return self.native.run(self._inputs(prepared))
        batch, noise = prepared
        normalized = self.model.predict_action_chunk(batch, noise=noise)
        physical = self.postprocessor(normalized)
        return {'action_chunk': normalized.detach().cpu().contiguous(), 'native_action_chunk': physical.detach().cpu().contiguous()}

    def qualify_inputs(self, originals):
        import torch

        if len(originals) != len(self.samples):
            raise ValueError('raw/reference sample counts differ')
        checks = []
        for index, (sample, previous) in enumerate(zip(self.samples, originals, strict=True)):
            values = self._inputs(self.prepare(sample))
            if set(values) != set(previous['inputs']):
                raise ValueError('original-input port mapping differs')
            for name, value in values.items():
                actual = value.cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
                if actual != Path(previous['inputs'][name]).read_bytes():
                    raise ValueError('official raw preprocessing differs: ' + name)
            checks.append({'sample': index, 'input_tensors': len(values), 'all_byte_exact': True})
        return checks

    def evidence(self):
        result = {'backend': self.backend, 'python_host': True, 'no_python_deployment': False,
                  'checkpoint_sha256': self.capture['checkpoint_sha256'], 'input_identities': self.raw['records'],
                  'statistics': self.statistics.to_dict(), 'action_statistics': self.action_statistics.to_dict(),
                  'loops': self.loops, 'physical_robot_units_verified': False}
        if self.native:
            result.update(native=self.native.evidence, replay=[self.native.replay_info(loop['task_id'])
                for loop in self.loops if loop['policy'] == 'required'])
        return result

    def close(self):
        if self.native:
            self.native.close()


def create(config):
    return SmolVLAHostPipeline(config)
