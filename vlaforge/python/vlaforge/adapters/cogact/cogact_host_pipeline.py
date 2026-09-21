"""Recorded CogACT inputs with official components and complete native outputs."""

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
from unittest.mock import patch

from vlaforge.adapters.cogact.cogact_partitioned import prepare_cogact_observation
from vlaforge.adapters.cogact.cogact_real import load_verified_candidate, verify_candidate_assets
from vlaforge.deployment import load_bundle_manifest
from vlaforge.ir.serializer import parse_canonical_json
from vlaforge.numerical_context import NumericalContext
from vlaforge.validation.host_pipeline import digest
from vlaforge.validation.native_session import NativeTensorSession


def read(path):
    return json.loads(Path(path).read_text())


def load_recorded_samples(observations, series):
    import numpy as np
    import torch
    from PIL import Image

    manifest, reference = read(observations / 'manifest.json'), read(series / 'report.json')
    if (manifest['schema'] != 'vlaforge.cogact.real-observation-series/1'
            or reference['status'] != 'real_series_partition_verified'
            or digest(observations / 'manifest.json') != reference['series_manifest_sha256']
            or manifest['count'] != len(manifest['frames']) or manifest['count'] != len(reference['runs'])):
        raise ValueError('complete matching recorded observation/reference series required')
    samples = []
    for index, (item, original) in enumerate(zip(manifest['frames'], reference['runs'], strict=True)):
        directory = (observations / item['directory']).resolve(strict=True)
        if not directory.is_relative_to(observations.resolve()) or original['index'] != index or original['frame'] != item['frame']:
            raise ValueError('recorded observation path or frame order differs')
        record = read(directory / 'input.json')
        if (record['schema'] != 'vlaforge.cogact.public-observation/1' or record['simulation']
                or not isinstance(record['instruction'], str) or not record['instruction'].strip()
                or record['frame'] != item['frame'] or record['instruction'] != item['instruction']
                or record['source_lock_sha256'] != manifest['source_lock_sha256']
                or record['image_sha256'] != item['image_sha256']
                or digest(directory / 'observation.png') != item['image_sha256']):
            raise ValueError('raw recorded image/instruction identity differs')
        with Image.open(directory / 'observation.png') as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        if (list(rgb.shape) != record['image_shape'] or rgb.dtype != np.uint8
                or hashlib.sha256(rgb.tobytes()).hexdigest() != record['rgb_sha256']
                or record['rgb_sha256'] != item['rgb_sha256']):
            raise ValueError('decoded original pixels differ')
        path = series / f'sample-{index}' / 'inputs.npz'
        if digest(path) != original['input_sha256']:
            raise ValueError('original saved input/RNG package differs')
        with np.load(path, allow_pickle=False) as package:
            tensors = {name: torch.from_numpy(package[name].copy()).contiguous() for name in package.files}
        expected = {'tokens': torch.int64, 'dino': torch.float32, 'siglip': torch.float32,
                    'initial_noise': torch.float32, 'step_noise': torch.float32,
                    'rng_states': torch.uint8, 'rng_before': torch.uint8}
        if set(tensors) != set(expected) or any(tensors[name].dtype != dtype or not torch.isfinite(tensors[name]).all()
                                               for name, dtype in expected.items()):
            raise ValueError('original input/RNG dtype or finite-value profile differs')
        if (tensors['initial_noise'].shape != (1, 16, 7) or tensors['step_noise'].shape != (10, 2, 16, 7)
                or tensors['rng_before'].ndim != 1 or tensors['rng_states'].shape != (12, tensors['rng_before'].numel())
                or not torch.equal(tensors['rng_before'], tensors['rng_states'][0])):
            raise ValueError('complete original RNG tape required')
        samples.append({'rgb': rgb, 'instruction': record['instruction'], 'unnorm_key': record['unnorm_key'],
                        'original': tensors, 'identity': record, 'input_sha256': original['input_sha256']})
    return manifest, samples


def make_image_transform(model_config):
    """Build upstream transforms from timm registry metadata, without weights."""
    import timm
    from torchvision.transforms import Compose, Resize
    from prismatic.models.backbones.vision.base_vision import LetterboxPad
    from prismatic.models.backbones.vision.dinosiglip_vit import DINOSigLIP_VISION_BACKBONES, DinoSigLIPImageTransform

    if model_config.vision_backbone_id != 'dinosiglip-vit-so-224px':
        raise ValueError('the archived native input profile requires the declared 224px backbone')
    transforms, configurations = {}, {}
    for name, identifier in DINOSigLIP_VISION_BACKBONES[model_config.vision_backbone_id].items():
        config = timm.data.resolve_data_config(pretrained_cfg=timm.get_pretrained_cfg(identifier).to_dict())
        config['input_size'] = (3, 224, 224)
        transform = timm.data.create_transform(**config, is_training=False)
        if not isinstance(transform, Compose) or not isinstance(transform.transforms[0], Resize):
            raise ValueError('upstream timm image transform structure changed')
        if name == 'siglip':
            transform = Compose([Resize(224, interpolation=transform.transforms[0].interpolation), *transform.transforms[1:]])
        if model_config.image_resize_strategy == 'resize-naive':
            transform = Compose([Resize((224, 224), interpolation=transform.transforms[0].interpolation), *transform.transforms[1:]])
        elif model_config.image_resize_strategy == 'letterbox':
            transform = Compose([LetterboxPad(tuple(int(value * 255) for value in config['mean'])), *transform.transforms])
        elif model_config.image_resize_strategy != 'resize-crop':
            raise ValueError('unsupported declared image resize strategy')
        transforms[name], configurations[name] = transform, config
    return DinoSigLIPImageTransform(transforms['dino'], transforms['siglip']), configurations


@contextmanager
def count_official_draws():
    """Count actual Python random calls in the exclusively owned worker thread."""
    import torch

    owner = threading.get_ident()
    calls = []
    def observe(name, function):
        def invoke(*args, **kwargs):
            if threading.get_ident() != owner:
                raise RuntimeError('another thread called the observed global RNG')
            value = function(*args, **kwargs)
            calls.append({'operation': name, 'shape': list(value.shape), 'dtype': str(value.dtype)})
            return value
        return invoke
    with patch.object(torch, 'randn', observe('randn', torch.randn)), patch.object(torch, 'randn_like', observe('randn_like', torch.randn_like)):
        yield calls


def official_model_samples(model, values):
    """Run only the original generation and action sampler."""
    import torch
    from prismatic.models.vlms.prismatic import PrismaticVLM

    with count_official_draws() as draws:
        with torch.autocast('cuda', dtype=model.vlm.llm_backbone.half_precision_dtype,
                            enabled=model.vlm.enable_mixed_precision_training):
            output = super(PrismaticVLM, model.vlm).generate(input_ids=values['tokens'],
                pixel_values={'dino': values['dino'], 'siglip': values['siglip']}, max_new_tokens=1,
                output_hidden_states=True, return_dict_in_generate=True)
        cognition = output.hidden_states[0][-1][:, -1, :]
        if cognition.shape != (1, 4096):
            raise ValueError('official complete cognition profile changed')
        dtype = next(model.action_model.net.parameters()).dtype
        cognition = cognition.unsqueeze(1).to(dtype)
        noise = torch.randn(1, model.future_action_window_size + 1, model.action_model.in_channels,
                            device=cognition.device).to(dtype)
        noise = torch.cat([noise, noise], 0)
        uncondition = model.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(1, 1, -1)
        samples = model.action_model.ddim_diffusion.ddim_sample_loop(model.action_model.net.forward_with_cfg,
            noise.shape, noise, clip_denoised=False, model_kwargs={'z': torch.cat([cognition, uncondition], 0), 'cfg_scale': 1.5},
            progress=False, device=cognition.device, eta=0.0)
    expected = [{'operation': 'randn', 'shape': [1, 16, 7], 'dtype': 'torch.float32'}] + [
        {'operation': 'randn_like', 'shape': [2, 16, 7], 'dtype': 'torch.float32'}] * 10
    if draws != expected:
        raise ValueError('official sampler random call sequence changed')
    return samples, draws


def official_postprocess_outputs(model, samples, draws, unnorm_key):
    """Apply the original action postprocessing and D2H conversion."""
    import numpy as np
    import torch

    raw = samples.chunk(2, dim=0)[0][0].detach().cpu().contiguous()
    normalized = np.clip(raw.numpy(), -1, 1)
    normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0, 1)
    stats = model.get_action_stats(unnorm_key)
    high, low = np.array(stats['q99']), np.array(stats['q01'])
    mask = stats.get('mask', np.ones_like(stats['q01'], dtype=bool))
    native = np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)
    return {'raw_action_chunk': raw, 'normalized_action_chunk': torch.from_numpy(normalized).contiguous(),
            'native_action_chunk': torch.from_numpy(native).contiguous(),
            'rng_after': torch.cuda.get_rng_state(0).contiguous(),
            'draws_consumed': torch.tensor([len(draws)], dtype=torch.int64)}


def official_complete_outputs(model, values, unnorm_key):
    """Call original generation, DDIM and complete CPU output conversion."""
    samples, draws = official_model_samples(model, values)
    return official_postprocess_outputs(model, samples, draws, unnorm_key)


class CogACTHostPipeline:
    def __init__(self, config):
        import torch

        for name, expected in config['files_sha256'].items():
            if digest(name) != expected:
                raise ValueError('CogACT original evidence changed: ' + name)
        for root, records in config['source_files_sha256'].items():
            for name, expected in records.items():
                if digest(Path(root) / name) != expected:
                    raise ValueError('pinned official dependency source changed: ' + name)
            sys.path.insert(0, root)
        required = {'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128', 'transformers': '4.40.1',
                    'timm': '0.9.10', 'tokenizers': '0.19.1', 'numpy': '1.26.4'}
        if {name: importlib.metadata.version(name) for name in required} != required:
            raise ValueError('pinned CogACT numerical environment differs')
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from prismatic.conf import ModelConfig
        from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
        from transformers import AutoTokenizer
        import prismatic
        import vla

        if (not Path(prismatic.__file__).resolve().is_relative_to(Path(config['openvla_source']).resolve())
                or not Path(vla.__file__).resolve().is_relative_to(Path(config['official_source']).resolve())):
            raise ValueError('official model imports resolved outside pinned sources')
        self.context = NumericalContext.from_dict(config['numerical_context'])
        self.context.require_current()
        self.manifest, self.samples = load_recorded_samples(Path(config['observations']), Path(config['reference_series']))
        self.model, self.native, self.loops = None, None, []
        self.backend = config['backend']
        checkpoint, llm, vision = (Path(config[name]) for name in ('checkpoint', 'llm_candidate', 'vision_assets'))
        model_config = ModelConfig.get_choice_class(read(checkpoint / 'config.json')['vla']['base_vlm'])()
        if model_config.llm_backbone_id != 'llama2-7b-pure':
            raise ValueError('declared public dependency LLM profile changed')
        if self.backend == 'official-pytorch':
            self.model, self.weights = load_verified_candidate(checkpoint, llm, vision)
            self.model = self.model.to('cuda:0').eval()
            self.model.action_model.create_ddim(10)
            self.tokenizer = self.model.vlm.llm_backbone.tokenizer
            self.prompt = self.model.vlm.get_prompt_builder
            self.transform = self.model.vlm.vision_backbone.image_transform
            self.transform_configs = None
        elif self.backend == 'generated-session':
            candidate = verify_candidate_assets(checkpoint, llm, vision)
            self.weights = {'dependency_candidate': candidate, 'installed_source_weights_checked_by_bound_bundle': True}
            self.tokenizer = AutoTokenizer.from_pretrained(llm, local_files_only=True,
                model_max_length=model_config.llm_max_length, padding_side='right')
            self.prompt = lambda: PurePromptBuilder(model_family='llama2')
            self.transform, self.transform_configs = make_image_transform(model_config)
            if digest(config['library_receipt']) != config['library_receipt_sha256']:
                raise ValueError('native build receipt identity differs')
            receipt = read(config['library_receipt'])
            if receipt['status'] != 'built_not_executed' or digest(config['cuda_runtime']) != config['cuda_runtime_sha256']:
                raise ValueError('native build or CUDA runtime differs')
            bundle = Path(receipt['bundle'])
            manifest = load_bundle_manifest(bundle / 'bundle.json')
            manifest.verify_files(bundle)
            selections = [item for item in manifest.generated_sources if item.role == 'loop_execution']
            if len(selections) != 1:
                raise ValueError('explicit native loop selection required')
            self.loops = read(bundle / selections[0].path)['loops']
            if any(loop['policy'] != config['policy'] for loop in self.loops):
                raise ValueError('compiled loop policy differs')
            module = parse_canonical_json((bundle / manifest.semantic_ir.path).read_text())
            self.native = NativeTensorSession(receipt['library'], module, library_sha256=receipt['library_sha256'],
                bundle=bundle, bundle_sha256=receipt['bundle_sha256'], cuda_runtime=config['cuda_runtime'])
        else:
            raise ValueError('unsupported CogACT host backend')

    def prepare(self, sample):
        import torch
        from PIL import Image

        values = prepare_cogact_observation(self.tokenizer, self.prompt(), self.transform,
            Image.fromarray(sample['rgb']), sample['instruction'], device=torch.device('cuda:0'))
        values = {name: value.contiguous() for name, value in values.items()}
        values.update({name: value.to('cuda:0').contiguous() for name, value in sample['original'].items()
                       if name not in ('tokens', 'dino', 'siglip')})
        if self.model:
            torch.cuda.set_rng_state(sample['original']['rng_before'], 0)
        return values, sample['unnorm_key']

    def infer(self, prepared):
        values, unnorm_key = prepared
        if self.native:
            return self.native.run(values)
        return official_complete_outputs(self.model, values, unnorm_key)

    def qualify_inputs(self, originals):
        import torch

        if len(originals) != len(self.samples):
            raise ValueError('raw/reference sample counts differ')
        result = []
        for index, (sample, original) in enumerate(zip(self.samples, originals, strict=True)):
            values, _ = self.prepare(sample)
            if set(values) != set(original['inputs']):
                raise ValueError('original input port mapping differs')
            for name, value in values.items():
                if value.cpu().reshape(-1).view(torch.uint8).numpy().tobytes() != Path(original['inputs'][name]).read_bytes():
                    raise ValueError('raw preprocessing differs from frozen input: ' + name)
            with torch.random.fork_rng(devices=[0]):
                expected = sample['original']
                torch.cuda.set_rng_state(expected['rng_before'], 0)
                initial = torch.randn(1, 16, 7, device='cuda:0')
                states = [expected['rng_before'], torch.cuda.get_rng_state(0)]
                if not torch.equal(initial.cpu(), expected['initial_noise']):
                    raise ValueError('original initial RNG draw differs')
                template = torch.cat([initial, initial], dim=0)
                for step in range(10):
                    if not torch.equal(torch.randn_like(template).cpu(), expected['step_noise'][step]):
                        raise ValueError('original per-step RNG draw differs')
                    states.append(torch.cuda.get_rng_state(0))
                if not torch.equal(torch.stack(states), expected['rng_states']):
                    raise ValueError('complete original RNG state tape differs')
            result.append({'sample': index, 'input_tensors': len(values), 'all_byte_exact': True, 'all_rng_states_exact': True})
        return result

    def evidence(self):
        result = {'backend': self.backend, 'weights': self.weights, 'python_host': True, 'no_python_deployment': False,
            'input_identities': [sample['identity'] for sample in self.samples], 'loops': self.loops,
            'transform_configs': self.transform_configs, 'original_meta_config_verified': False,
            'physical_robot_units_verified': False, 'autonomous_cpp_rng': False,
            'official_scope': 'original generation and DDIM engines, qualified processor, original NumPy output formula; actual random calls counted',
            'rng_scope': 'official actual draws restored from saved state; Session consumes the saved external tape'}
        if self.native:
            result.update(native=self.native.evidence, replay=[self.native.replay_info(loop['task_id'])
                for loop in self.loops if loop['policy'] == 'required'])
        return result

    def close(self):
        if self.native:
            self.native.close()


def create(config):
    return CogACTHostPipeline(config)
