"""Official OpenPI and generated Session implementations of the host pipeline.

Both configurations use the same pinned processor and complete typed outputs.
Model-specific configuration remains here; the timing/storage runner has no
model or backend dispatch. The Python host is explicit for both paths.
"""

import json
from pathlib import Path

from vlaforge.adapters.openpi.openpi_frontend import OpenPIConfig, load_openpi
from vlaforge.adapters.openpi.openpi_inputs import load_openpi_input_pack
from vlaforge.adapters.openpi.openpi_processing import OpenPIProcessor
from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext
from vlaforge.validation.host_pipeline import digest
from vlaforge.validation.native_session import NativeTensorSession


def _read(path):
    return json.loads(Path(path).read_text())


class OpenPIHostPipeline:
    def __init__(self, config):
        self.native = None
        self.loaded = None
        capture_path = Path(config['capture'])
        if digest(capture_path) != config['capture_sha256']:
            raise ValueError('OpenPI capture identity differs')
        capture = _read(capture_path)
        NumericalContext.from_dict(capture['numerical_context']).require_current()
        self.capture = capture
        self.processor = OpenPIProcessor(capture, device='cuda:0')
        index_path = Path(config['inputs'])
        if digest(index_path) != config['inputs_sha256']:
            raise ValueError('OpenPI input index identity differs')
        index = _read(index_path)
        self.samples = []
        self.identities = []
        for item in index['samples']:
            path = (index_path.parent / item['manifest']).resolve(strict=True)
            if not path.is_relative_to(index_path.parent.resolve()) or digest(path) != item['sha256']:
                raise ValueError('OpenPI input manifest identity differs')
            observation, noise, identity = load_openpi_input_pack(path, source_root=capture['processor_config']['source_root'])
            if identity['frame_index'] != item['frame_index'] or identity['episode_index'] != item['episode_index']:
                raise ValueError('OpenPI observation identity differs')
            self.samples.append((observation, noise))
            self.identities.append({'path': str(path), 'sha256': item['sha256'], 'observation': identity})
        self.backend = config['backend']
        self.loops = []
        if self.backend == 'official-pytorch':
            options = capture['processor_config']
            self.loaded = load_openpi(OpenPIConfig(source_root=Path(options['source_root']),
                checkpoint_dir=Path(options['checkpoint_dir']), config_name=capture['config_name'],
                checkpoint_sha256=capture['checkpoint_provenance']['checkpoint']['sha256'],
                device='cuda:0', num_steps=capture['num_steps']))
        elif self.backend == 'generated-session':
            path = Path(config['library_receipt'])
            if digest(path) != config['library_receipt_sha256']:
                raise ValueError('native library build receipt identity differs')
            receipt = _read(path)
            if receipt['status'] != 'built_not_executed':
                raise ValueError('native library build is not complete')
            bundle = Path(receipt['bundle'])
            manifest = load_bundle_manifest(bundle / 'bundle.json')
            manifest.verify_files(bundle)
            loop_records = [item for item in manifest.generated_sources if item.role == 'loop_execution']
            if len(loop_records) != 1:
                raise ValueError('explicit loop execution selection required')
            self.loops = _read(bundle / loop_records[0].path)['loops']
            if any(item['policy'] != config['policy'] for item in self.loops):
                raise ValueError('native loop execution policy differs')
            if digest(config['cuda_runtime']) != config['cuda_runtime_sha256']:
                raise ValueError('explicit CUDA runtime identity differs')
            module = parse_canonical_json((bundle / manifest.semantic_ir.path).read_text())
            self.native = NativeTensorSession(receipt['library'], module,
                library_sha256=receipt['library_sha256'], bundle=bundle,
                bundle_sha256=receipt['bundle_sha256'], cuda_runtime=config['cuda_runtime'])
        else:
            raise ValueError('unsupported OpenPI host backend')

    def prepare(self, sample):
        return self.processor.prepare(*sample)

    def infer(self, prepared):
        import numpy as np
        import torch

        if self.native:
            return self.native.run(self.processor.session_inputs(prepared))
        normalized = self.loaded.model.sample_actions('cuda:0', prepared.observation,
            noise=prepared.noise, num_steps=self.capture['num_steps'])
        physical = self.processor.native_actions(prepared, normalized)
        return {'normalized_action_chunk': normalized.detach().cpu().contiguous(),
                'native_action_chunk': torch.from_numpy(np.array(physical))[None].contiguous()}

    def qualify_inputs(self, original_samples):
        """Verify raw observation preprocessing against the frozen tensor inputs."""
        import torch

        if len(original_samples) != len(self.samples):
            raise ValueError('prepared/raw sample counts differ')
        result = []
        for index, (sample, previous) in enumerate(zip(self.samples, original_samples, strict=True)):
            inputs = self.processor.session_inputs(self.prepare(sample))
            if set(inputs) != set(previous['inputs']):
                raise ValueError('prepared/raw input port names differ')
            for name, value in inputs.items():
                raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                if raw != Path(previous['inputs'][name]).read_bytes():
                    raise ValueError('raw observation preprocessing changed: ' + name)
            result.append({'sample': index, 'input_tensors': len(inputs), 'all_byte_exact': True})
        return result

    def evidence(self):
        result = {'backend': self.backend, 'python_host': True, 'no_python_deployment': False,
                  'input_identities': self.identities, 'tokenizer': self.processor.tokenizer_identity,
                  'checkpoint': self.capture['checkpoint_provenance'], 'loops': self.loops}
        if self.native:
            result.update(native=self.native.evidence,
                          replay=[self.native.replay_info(item['task_id']) for item in self.loops if item['policy'] == 'required'])
        return result

    def close(self):
        if self.native:
            self.native.close()


def create(config):
    return OpenPIHostPipeline(config)
